#define _POSIX_C_SOURCE 200809L
#include "telemetry_json.h"
#include <math.h>
#include <time.h>

static void json_number(FILE *stream, double n) {
    if (isfinite(n)) fprintf(stream, "%.6f", n); else fputs("null", stream);
}

static void coverage(FILE *stream, const sg_temperature *t) {
    fprintf(stream, "{\"status\":\"%s\",\"sensors\":%u,\"valid\":%u,\"invalid\":%u,\"read_errors\":%u,\"min_c\":",
           sg_temperature_status(t), t->sensors, t->valid, t->invalid, t->read_errors);
    json_number(stream, t->min_c);
    fputs(",\"max_c\":", stream); json_number(stream, t->max_c);
    fputc('}', stream);
}

void sg_thermal_json(FILE *stream, const sg_thermal_sample *s) {
    time_t seconds = (time_t)s->wall_time;
    struct tm utc;
    char stamp[32];
    gmtime_r(&seconds, &utc);
    strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%S", &utc);
    fprintf(stream, "{\"schema_version\":1,\"source\":\"surge-smc\",\"timestamp\":\"%s.%06uZ\",\"sample_ms\":%.3f,\"attempts\":%u,\"discovery_errors\":%u,\"temp\":{\"cpu_temp_avg\":",
           stamp, (unsigned)((s->wall_time - seconds) * 1e6), s->sample_ms, s->attempts, s->discovery_errors);
    json_number(stream, s->cpu.avg_c);
    fputs(",\"gpu_temp_avg\":", stream); json_number(stream, s->gpu.avg_c);
    fputs("},\"sensors\":{\"cpu\":", stream); coverage(stream, &s->cpu);
    fputs(",\"gpu\":", stream); coverage(stream, &s->gpu);
    fputs("},\"fans\":[", stream);
    for (uint32_t i = 0; i < s->fan_count; i++) {
        fprintf(stream, "%s{\"key\":\"%s\",\"rpm\":", i ? "," : "", s->fans[i].key);
        json_number(stream, s->fans[i].rpm);
        fputc('}', stream);
    }
    fputs("]}\n", stream);
}
