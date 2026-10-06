/* Read-only AppleSMC temperature/fan sampling. Wire protocol adapted from
 * fanpro (Copyright (c) 2026 Omar Shabab, MIT; see LICENSE).
 * Only commands 5 (read), 8 (enumerate), and 9 (metadata) are ever issued.
 * CPU Tp/Te/Ts and GPU Tg keys match the existing macmon metric's scope.
 * These undocumented namespaces are verified on M3 Ultra, not a universal
 * guarantee across Apple hardware. Missing/partial discovery stays visible. */
#include "telemetry_internal.h"
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

_Static_assert(sizeof(sg_smc_wire) == 80, "SMC wire size");
_Static_assert(offsetof(sg_smc_wire, info) == 28, "SMC metadata offset");
_Static_assert(offsetof(sg_smc_wire, bytes) == 48, "SMC payload offset");

typedef struct { uint32_t key; sg_smc_info info; int kind; } sg_sensor;
struct sg_thermal {
    sg_smc_transport call;
    void *ctx;
    io_connect_t connection;
    sg_sensor *sensors;
    uint32_t count, discovery_errors;
};

uint32_t sg_smc_fourcc(const char s[4]) {
    return (uint32_t)(uint8_t)s[0] << 24 | (uint32_t)(uint8_t)s[1] << 16 |
           (uint32_t)(uint8_t)s[2] << 8 | (uint8_t)s[3];
}

bool sg_smc_number(uint32_t type, uint32_t n, const uint8_t *b, double *out) {
    if (!b || !out || n > 4 || n == 0) return false;
    uint32_t v = 0;
    for (uint32_t i = 0; i < n; i++) v = (v << 8) | b[i];
    if (type == sg_smc_fourcc("flt ") && n == 4) {
        uint32_t bits = (uint32_t)b[0] | (uint32_t)b[1] << 8 |
                        (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
        float f;
        memcpy(&f, &bits, sizeof f);
        *out = f;
    } else if ((type == sg_smc_fourcc("ui8 ") && n == 1) ||
               (type == sg_smc_fourcc("ui16") && n == 2) ||
               (type == sg_smc_fourcc("ui32") && n == 4)) {
        *out = v;
    } else if (type == sg_smc_fourcc("fpe2") && n == 2) {
        *out = (double)v / 4;
    } else if (type == sg_smc_fourcc("sp78") && n == 2) {
        *out = (v & 0x8000 ? (double)v - 65536 : (double)v) / 256;
    } else return false;
    return true;
}

static int sg_smc_call(sg_thermal *t, const sg_smc_wire *in, sg_smc_wire *out) {
    memset(out, 0, sizeof *out);
    return t->call(t->ctx, in, out) || out->result;
}

static bool sg_smc_read(sg_thermal *t, uint32_t key, sg_smc_info info, double *out) {
    sg_smc_wire in = {.key = key, .command = 5, .info = info}, result;
    if (sg_smc_call(t, &in, &result)) return false;
    return sg_smc_number(info.type, info.size, result.bytes, out);
}

static int sg_smc_iokit(void *ctx, const sg_smc_wire *in, sg_smc_wire *out) {
    io_connect_t conn = *(io_connect_t *)ctx;
    size_t size = sizeof *out;
    kern_return_t kr = IOConnectCallStructMethod(conn, 2, in, sizeof *in, out, &size);
    return kr != KERN_SUCCESS || size != sizeof *out;
}

sg_err sg_thermal_open_transport(sg_smc_transport call, void *ctx, sg_thermal **out) {
    if (!out) return (sg_err){"null telemetry output"};
    *out = NULL;
    if (!call) return (sg_err){"null SMC transport"};
    sg_thermal *t = calloc(1, sizeof *t);
    if (!t) return (sg_err){"telemetry allocation failed"};
    t->call = call; t->ctx = ctx;
    sg_smc_wire in = {.key = sg_smc_fourcc("#KEY"), .command = 9}, result;
    double key_count;
    if (sg_smc_call(t, &in, &result) ||
        !sg_smc_read(t, in.key, result.info, &key_count) || key_count < 1 ||
        key_count > 65536 || floor(key_count) != key_count) {
        free(t);
        return (sg_err){"SMC key enumeration unavailable"};
    }
    t->sensors = calloc((size_t)key_count, sizeof *t->sensors);
    if (!t->sensors) { free(t); return (sg_err){"sensor allocation failed"}; }
    unsigned fans = 0;
    for (uint32_t i = 0; i < (uint32_t)key_count; i++) {
        in = (sg_smc_wire){.command = 8, .index = i};
        if (sg_smc_call(t, &in, &result)) { t->discovery_errors++; continue; }
        uint32_t key = result.key;
        unsigned a = key >> 24, b = (key >> 16) & 255;
        int kind = a == 'T' && b == 'g' ? 2 :
                   a == 'T' && (b == 'p' || b == 'e' || b == 's') ? 1 :
                   a == 'F' && ((b >= '0' && b <= '9') || (b >= 'A' && b <= 'F')) &&
                   (key & 65535) == ('A' << 8 | 'c') ? 3 : 0;
        if (!kind) continue;
        bool duplicate = false;
        for (uint32_t j = 0; j < t->count; j++) if (t->sensors[j].key == key) duplicate = true;
        if (duplicate) continue;
        in = (sg_smc_wire){.key = key, .command = 9};
        if (sg_smc_call(t, &in, &result)) { t->discovery_errors++; continue; }
        sg_smc_info info = result.info;
        /* Temperature keys with non-temperature encodings are not sensors.
         * Metadata, not a transient value, controls discovery. */
        bool temp_type = (info.type == sg_smc_fourcc("flt ") && info.size == 4) ||
                         (info.type == sg_smc_fourcc("sp78") && info.size == 2);
        if (kind != 3 && !temp_type) continue;
        if (kind == 3 && ++fans > SG_THERMAL_MAX_FANS) { t->discovery_errors++; continue; }
        t->sensors[t->count++] = (sg_sensor){key, info, kind};
    }
    *out = t;
    return SG_OK;
}

sg_err sg_thermal_open(sg_thermal **out) {
    if (!out) return (sg_err){"null telemetry output"};
    *out = NULL;
    io_iterator_t iter = IO_OBJECT_NULL;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AppleSMC"), &iter))
        return (sg_err){"AppleSMC service unavailable"};
    io_connect_t conn = IO_OBJECT_NULL;
    io_service_t service;
    while ((service = IOIteratorNext(iter))) {
        kern_return_t kr = IOServiceOpen(service, mach_task_self(), 0, &conn);
        IOObjectRelease(service);
        if (kr == KERN_SUCCESS && conn != IO_OBJECT_NULL) break;
        conn = IO_OBJECT_NULL;
    }
    IOObjectRelease(iter);
    if (conn == IO_OBJECT_NULL) return (sg_err){"cannot open AppleSMC for reading"};
    sg_err e = sg_thermal_open_transport(sg_smc_iokit, &conn, out);
    if (sg_failed(e)) { IOServiceClose(conn); return e; }
    (*out)->connection = conn;
    (*out)->ctx = &(*out)->connection;
    return SG_OK;
}

void sg_thermal_close(sg_thermal *t) {
    if (!t) return;
    if (t->connection != IO_OBJECT_NULL) IOServiceClose(t->connection);
    free(t->sensors);
    free(t);
}

bool sg_temperature_complete(const sg_temperature *t) {
    return t && t->sensors > 0 && t->valid == t->sensors && isfinite(t->avg_c) && t->avg_c > 0;
}

const char *sg_temperature_status(const sg_temperature *t) {
    if (!t || !t->sensors) return "no_sensors";
    if (sg_temperature_complete(t)) return "ok";
    if (t->valid) return "partial";
    return t->read_errors ? "read_error" : "no_valid_readings";
}

static void sg_temp_add(sg_temperature *t, bool read_ok, double v) {
    t->sensors++;
    if (!read_ok) { t->read_errors++; return; }
    /* Zero and negative SMC sentinels were observed across all 64 M3 Ultra
     * GPU keys at once. They cannot mean a cold, powered-on SoC. */
    if (!isfinite(v) || v <= 0 || v > 150) { t->invalid++; return; }
    if (!t->valid) { t->avg_c = 0; t->min_c = v; t->max_c = v; }
    t->avg_c += v;
    if (v < t->min_c) t->min_c = v;
    if (v > t->max_c) t->max_c = v;
    t->valid++;
}

static double sg_thermal_time(clockid_t clock) {
    struct timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

sg_err sg_thermal_read(sg_thermal *t, unsigned retries, sg_thermal_sample *out) {
    return sg_thermal_read_components(t, retries, SG_THERMAL_ALL, out);
}

sg_err sg_thermal_read_components(sg_thermal *t, unsigned retries,
                                  unsigned components, sg_thermal_sample *out) {
    if (!t || !out || retries < 1 || retries > 10 ||
        !components || (components & ~SG_THERMAL_ALL))
        return (sg_err){"invalid telemetry read arguments"};
    double begin = sg_thermal_time(CLOCK_MONOTONIC);
    for (unsigned attempt = 1; attempt <= retries; attempt++) {
        if (attempt > 1) {
            struct timespec delay = {.tv_nsec = 50000000};
            while (nanosleep(&delay, &delay) && errno == EINTR) {}
        }
        *out = (sg_thermal_sample){.attempts = attempt, .discovery_errors = t->discovery_errors};
        out->cpu.avg_c = out->cpu.min_c = out->cpu.max_c = NAN;
        out->gpu.avg_c = out->gpu.min_c = out->gpu.max_c = NAN;
        for (uint32_t i = 0; i < t->count; i++) {
            sg_sensor *s = &t->sensors[i];
            double value = NAN;
            bool ok = sg_smc_read(t, s->key, s->info, &value);
            if (s->kind == 1 || s->kind == 2) {
                sg_temp_add(s->kind == 1 ? &out->cpu : &out->gpu, ok, value);
            } else {
                sg_thermal_fan *fan = &out->fans[out->fan_count++];
                for (unsigned j = 0; j < 4; j++) fan->key[j] = (char)(s->key >> (24 - j * 8));
                fan->rpm = ok && value >= 0 && value <= 100000 ? value : NAN;
            }
        }
        if (out->cpu.valid) out->cpu.avg_c /= out->cpu.valid;
        if (out->gpu.valid) out->gpu.avg_c /= out->gpu.valid;
        bool cpu_ready = !(components & SG_THERMAL_CPU) || sg_temperature_complete(&out->cpu);
        bool gpu_ready = !(components & SG_THERMAL_GPU) || sg_temperature_complete(&out->gpu);
        if (cpu_ready && gpu_ready) break;
    }
    out->wall_time = sg_thermal_time(CLOCK_REALTIME);
    out->sample_ms = 1000 * (sg_thermal_time(CLOCK_MONOTONIC) - begin);
    return SG_OK;
}
