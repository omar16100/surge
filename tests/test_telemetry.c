/* Wire-level fault injection: no SMC connection, GPU work, or fan writes. */
#include "telemetry_internal.h"
#include "tinytest.h"
#include <math.h>
#include <string.h>

typedef struct {
    double cpu, gpu[2], rpm;
    int gpu_read_error, firmware_error, metadata_error, index_error, count_error;
    int recover, alternate;
    unsigned sample, calls, forbidden;
} fake_smc;
static const char *keys[] = {"Tp00", "Tg00", "Tg01", "F0Ac", "TgZZ"};

static void be32(uint8_t *b, uint32_t n) {
    for (unsigned i = 0; i < 4; i++) b[i] = (uint8_t)(n >> (24 - 8*i));
}

static void flt(uint8_t *b, double d) {
    float f = (float)d;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    for (unsigned i = 0; i < 4; i++) b[i] = (uint8_t)(bits >> (8*i));
}

static int fake_call(void *ctx, const sg_smc_wire *in, sg_smc_wire *out) {
    fake_smc *f = ctx;
    f->calls++;
    memset(out, 0, sizeof *out);
    if (in->command != 5 && in->command != 8 && in->command != 9) {
        f->forbidden++; return -1;
    }
    if (in->command == 8) {
        if (in->index >= 5 || (f->index_error && in->index == 2)) return -1;
        out->key = sg_smc_fourcc(keys[in->index]);
        return 0;
    }
    if (in->key == sg_smc_fourcc("#KEY")) {
        if (f->count_error) return -1;
        out->info = (sg_smc_info){.size=4, .type=sg_smc_fourcc("ui32")};
        be32(out->bytes, 5);
        return 0;
    }
    int k = -1;
    for (unsigned i = 0; i < 5; i++) if (in->key == sg_smc_fourcc(keys[i])) k = (int)i;
    if (k < 0) return -1;
    if (in->command == 9) {
        if (f->metadata_error && k == 2) return -1;
        out->info = (sg_smc_info){.size=4, .type=sg_smc_fourcc(k == 4 ? "ui32" : "flt ")};
        return 0;
    }
    if (k == 0) f->sample++;
    if (k == 2 && f->gpu_read_error) return -1;
    if (k == 2 && f->firmware_error) { out->result = 0x84; return 0; }
    double value = k == 0 ? f->cpu : k == 3 ? f->rpm : f->gpu[k - 1];
    if (f->recover && f->sample == 1 && (k == 1 || k == 2)) value = 0;
    if (f->alternate) {
        if (f->sample == 1 && k == 0) value = 0;
        if (f->sample > 1 && (k == 1 || k == 2)) value = 0;
    }
    flt(out->bytes, value);
    return 0;
}

static sg_thermal_sample sample_components(fake_smc *f, unsigned retries, unsigned components) {
    sg_thermal *t = NULL;
    sg_err e = sg_thermal_open_transport(fake_call, f, &t);
    tt_assert(!sg_failed(e) && t, "open fake SMC: %s", e.msg ? e.msg : "");
    sg_thermal_sample out = {0};
    if (t) {
        e = components == SG_THERMAL_ALL ? sg_thermal_read(t, retries, &out) :
            sg_thermal_read_components(t, retries, components, &out);
        tt_assert(!sg_failed(e), "sample fake SMC");
        sg_thermal_close(t);
    }
    tt_assert(f->forbidden == 0, "telemetry issued a non-read command");
    return out;
}

static sg_thermal_sample sample(fake_smc *f, unsigned retries) {
    return sample_components(f, retries, SG_THERMAL_ALL);
}

static fake_smc healthy(void) {
    return (fake_smc){.cpu=45, .gpu={40, 60}, .rpm=3600};
}

int main(void) {
    fake_smc f = healthy();
    sg_thermal_sample s = sample(&f, 3);
    tt_assert(s.gpu.sensors == 2 && s.gpu.valid == 2 && s.gpu.avg_c == 50,
              "runtime type filtering and arithmetic mean");
    tt_assert(s.gpu.min_c == 40 && s.gpu.max_c == 60, "GPU min/max");
    tt_assert(s.attempts == 1 && !s.discovery_errors, "healthy read must not retry");
    tt_assert(s.fan_count == 1 && s.fans[0].rpm == 3600 && !strcmp(s.fans[0].key, "F0Ac"), "fan identity/RPM");
    tt_assert(s.wall_time > 0 && s.sample_ms >= 0, "fresh sample timestamps");

    f = healthy(); f.cpu = 0;
    s = sample_components(&f, 3, SG_THERMAL_GPU);
    tt_assert(s.attempts == 1 && s.gpu.valid == 2 && !s.cpu.valid,
              "GPU retry policy preserves unavailable CPU coverage");
    f = healthy(); f.cpu = 0;
    s = sample(&f, 3);
    tt_assert(s.attempts == 3 && !s.cpu.valid, "default policy requires both components");
    f = healthy(); f.gpu[0] = 0;
    s = sample_components(&f, 3, SG_THERMAL_GPU);
    tt_assert(s.attempts == 3 && s.gpu.valid == 1, "GPU policy retries partial GPU coverage");
    f = healthy(); f.gpu[0] = 0;
    s = sample_components(&f, 3, SG_THERMAL_CPU);
    tt_assert(s.attempts == 1 && s.cpu.valid == 1 && s.gpu.valid == 1,
              "CPU retry policy preserves partial GPU coverage");

    f = healthy(); f.gpu[0] = 0; f.gpu[1] = -8.75;
    s = sample(&f, 3);
    tt_assert(!s.gpu.valid && s.gpu.invalid == 2 && isnan(s.gpu.avg_c), "empty valid sensor set must not become zero");
    tt_assert(s.attempts == 3 && !strcmp(sg_temperature_status(&s.gpu), "no_valid_readings"), "bounded retry and reason");
    tt_assert(s.cpu.avg_c == 45 && s.fans[0].rpm == 3600, "GPU outage must preserve CPU/fans");

    f = healthy(); f.gpu[0] = NAN; f.gpu[1] = INFINITY;
    s = sample(&f, 1);
    tt_assert(s.gpu.invalid == 2 && isnan(s.gpu.avg_c), "nonfinite values rejected");
    f = healthy(); f.gpu[0] = 151; f.gpu[1] = 0;
    s = sample(&f, 1);
    tt_assert(s.gpu.invalid == 2 && !sg_temperature_complete(&s.gpu), "out of range values rejected");

    f = healthy(); f.gpu_read_error = 1;
    s = sample(&f, 1);
    tt_assert(s.gpu.read_errors == 1 && s.gpu.avg_c == 40 && !strcmp(sg_temperature_status(&s.gpu), "partial"), "partial coverage must remain visible");
    tt_assert(!sg_temperature_complete(&s.gpu), "gate must reject partial coverage");
    f = healthy(); f.firmware_error = 1;
    s = sample(&f, 1);
    tt_assert(s.gpu.read_errors == 1, "firmware error must not look like a zero reading");

    f = healthy(); f.recover = 1;
    s = sample(&f, 3);
    tt_assert(s.attempts == 2 && s.gpu.avg_c == 50, "transient outage should recover with fresh reads");
    f = healthy(); f.alternate = 1;
    s = sample(&f, 3);
    tt_assert(s.attempts == 3 && s.cpu.avg_c == 45 && isnan(s.gpu.avg_c), "must not merge temperatures from different attempts");

    f = healthy(); f.metadata_error = 1;
    s = sample(&f, 1);
    tt_assert(s.discovery_errors == 1, "incomplete metadata discovery must not be hidden");
    f = healthy(); f.index_error = 1;
    s = sample(&f, 1);
    tt_assert(s.discovery_errors == 1, "incomplete key enumeration must not be hidden");
    f = healthy(); f.rpm = 0;
    s = sample(&f, 1);
    tt_assert(s.fans[0].rpm == 0, "zero fan RPM is a real stopped fan");

    sg_thermal *t = (sg_thermal *)1;
    f = healthy(); f.count_error = 1;
    tt_assert(sg_failed(sg_thermal_open_transport(fake_call, &f, &t)) && !t, "key count failure must clear handle");
    tt_assert(sg_failed(sg_thermal_read(NULL, 1, &s)), "null handle rejected");
    f = healthy(); t = NULL;
    tt_assert(!sg_failed(sg_thermal_open_transport(fake_call, &f, &t)), "open for argument tests");
    tt_assert(sg_failed(sg_thermal_read_components(t, 3, 0, &s)), "empty component mask rejected");
    tt_assert(sg_failed(sg_thermal_read_components(t, 3, 4, &s)), "unknown component mask rejected");
    sg_thermal_close(t);
    tt_assert(!sg_temperature_complete(NULL), "null temperature rejected");
    sg_temperature empty = {.avg_c=NAN};
    tt_assert(!strcmp(sg_temperature_status(&empty), "no_sensors"), "absent sensors have a distinct reason");

    uint8_t b[4] = {0x13, 0x88, 0, 0}; double v;
    tt_assert(sg_smc_number(sg_smc_fourcc("fpe2"), 2, b, &v) && v == 1250, "fan fixed point codec");
    b[0] = 0x32; b[1] = 0x80;
    tt_assert(sg_smc_number(sg_smc_fourcc("sp78"), 2, b, &v) && v == 50.5, "temperature fixed point codec");
    b[0] = 0xff; b[1] = 0x80;
    tt_assert(sg_smc_number(sg_smc_fourcc("sp78"), 2, b, &v) && v == -0.5, "negative sentinel codec");
    tt_assert(!sg_smc_number(sg_smc_fourcc("flt "), 2, b, &v), "truncated float rejected");
    tt_assert(!sg_smc_number(sg_smc_fourcc("????"), 4, b, &v), "unknown encoding rejected");
    return tt_report();
}
