/* surge-monitor: native temperature/fan JSONL, or a strict scalar GPU gate. */
#define _POSIX_C_SOURCE 200809L
#include "surge.h"
#include "telemetry_json.h"
#include <signal.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void usage(FILE *f) {
    fputs("usage: surge-monitor [--samples N] [--interval-ms N] [--retries N]\n"
          "       [--retry-components cpu|gpu|all]\n"
          "       surge-monitor --gpu-temp [--retries N]\n"
          "JSONL by default; samples=1 (0 streams), interval=1000 ms (minimum 10), retries=3.\n"
          "Retry components default to all; --gpu-temp requires only GPU coverage.\n"
          "Missing values are null. --gpu-temp prints only a complete fresh GPU\n"
          "temperature; exit 3 if unavailable/partial, 1 on I/O error, 2 on usage.\n"
          "No sudo, subprocesses, GPU work, or fan writes.\n", f);
}

static int number(const char *text, unsigned long limit, unsigned long *out) {
    if (!text[0] || text[0] < '0' || text[0] > '9') return 0;
    char *end;
    errno = 0;
    unsigned long n = strtoul(text, &end, 10);
    if (errno || *end || n > limit) return 0;
    *out = n;
    return 1;
}

int main(int argc, char **argv) {
    unsigned long samples = 1, interval = 1000, retries = 3;
    bool scalar = false;
    unsigned components = SG_THERMAL_ALL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) { usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { puts("surge-monitor 1 (native SMC)"); return 0; }
        if (!strcmp(argv[i], "--gpu-temp")) { scalar = true; continue; }
        if (!strcmp(argv[i], "--retry-components")) {
            if (++i == argc) { usage(stderr); return 2; }
            if (!strcmp(argv[i], "cpu")) components = SG_THERMAL_CPU;
            else if (!strcmp(argv[i], "gpu")) components = SG_THERMAL_GPU;
            else if (!strcmp(argv[i], "all")) components = SG_THERMAL_ALL;
            else { usage(stderr); return 2; }
            continue;
        }
        unsigned long *dest, max;
        if (!strcmp(argv[i], "--samples")) { dest = &samples; max = 1000000000; }
        else if (!strcmp(argv[i], "--interval-ms")) { dest = &interval; max = 3600000; }
        else if (!strcmp(argv[i], "--retries")) { dest = &retries; max = 10; }
        else { usage(stderr); return 2; }
        if (++i == argc || !number(argv[i], max, dest)) { usage(stderr); return 2; }
    }
    if (!retries || interval < 10 || (scalar && samples != 1)) { usage(stderr); return 2; }
    if (scalar) components = SG_THERMAL_GPU;
    signal(SIGPIPE, SIG_IGN);
    sg_thermal *t = NULL;
    sg_err e = sg_thermal_open(&t);
    if (sg_failed(e)) { fprintf(stderr, "surge-monitor: %s\n", e.msg); return 1; }
    int rc = 0;
    for (unsigned long i = 0; !samples || i < samples; i++) {
        sg_thermal_sample s;
        e = sg_thermal_read_components(t, (unsigned)retries, components, &s);
        if (sg_failed(e)) { fprintf(stderr, "surge-monitor: %s\n", e.msg); rc = 1; break; }
        if (scalar) {
            if (s.discovery_errors || !sg_temperature_complete(&s.gpu)) {
                fprintf(stderr, "surge-monitor: GPU temperature %s (%u/%u sensors, %u discovery errors)\n",
                        sg_temperature_status(&s.gpu), s.gpu.valid, s.gpu.sensors, s.discovery_errors);
                rc = 3;
            } else printf("%.6f\n", s.gpu.avg_c);
        } else sg_thermal_json(stdout, &s);
        if (fflush(stdout) == EOF) { rc = 1; break; }
        if (samples && i + 1 == samples) break;
        /* Interval is measured between samples, including the bounded retry time. */
        double remain_ms = (double)interval - s.sample_ms;
        if (remain_ms > 0) {
            struct timespec delay = {.tv_sec = (time_t)(remain_ms / 1000),
                .tv_nsec = (long)(fmod(remain_ms, 1000) * 1000000)};
            while (nanosleep(&delay, &delay) && errno == EINTR) {}
        }
    }
    sg_thermal_close(t);
    return rc;
}
