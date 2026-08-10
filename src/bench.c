/* bench.c - pure-C bench math: decode-by-slope, mlx-style average, and the
 * leaderboard-row formatters (Task B1).
 *
 * PURE C. No Metal, no GPU, no Foundation -- safe to build and test while
 * the GPU is busy running the 256K comparison loop. Lives in LIB_SRC (the
 * Makefile's src wildcard) and links into every pure-C test binary.
 *
 * See surge.h for the full contract; this file is the implementation only.
 */

#include "surge.h"
#include <math.h>
#include <stdio.h>

double sg_bench_slope(const double *t_wall_cum, uint32_t n, uint32_t warmup) {
    if (!t_wall_cum || warmup >= n) return 0.0;
    uint32_t count = n - warmup;
    if (count < 2) return 0.0;

    /* Mean-centered, two-pass ordinary least squares, x = wall time,
     * y = token index: slope = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2).
     *
     * This is deliberately NOT the textbook one-pass normal-equation form
     * (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2): that form is
     * catastrophically unstable once t_wall_cum holds realistic epoch-scale
     * values. surge.h documents t_wall_cum as "any epoch", and a caller
     * that passes raw gettimeofday()-style seconds (~1.7e9) hits exactly
     * that case: sum_xx and sum_x^2 both land within a few ULPs of
     * N*epoch^2, so the subtraction that is supposed to recover the tiny
     * (O(N) not O(N*epoch^2)) variance signal instead returns noise --
     * verified to produce a NEGATIVE slope on an exact 5 tok/s series at
     * epoch ~1.8e9 with the one-pass form. Centering on the mean first
     * removes the large common offset before any squaring happens, so the
     * two-pass version is accurate at any offset the caller reasonably
     * passes in. */
    double mean_x = 0.0, mean_y = 0.0;
    for (uint32_t i = warmup; i < n; i++) {
        mean_x += t_wall_cum[i];
        mean_y += (double)i;
    }
    double N = (double)count;
    mean_x /= N;
    mean_y /= N;

    double sum_dxdy = 0.0, sum_dxdx = 0.0;
    for (uint32_t i = warmup; i < n; i++) {
        double dx = t_wall_cum[i] - mean_x;
        double dy = (double)i - mean_y;
        sum_dxdy += dx * dy;
        sum_dxdx += dx * dx;
    }
    if (sum_dxdx == 0.0) return 0.0;   /* every timestamp in range identical */
    return sum_dxdy / sum_dxdx;
}

double sg_bench_avg_tps(const double *t_wall_cum, uint32_t n, uint32_t warmup) {
    if (!t_wall_cum || n == 0 || warmup >= n) return 0.0;
    /* Signed so "warmup == n-1" (zero tokens counted) cannot underflow a
     * uint32_t subtraction. */
    int64_t count = (int64_t)(n - 1) - (int64_t)warmup;
    if (count < 1) return 0.0;
    double dt = t_wall_cum[n - 1] - t_wall_cum[warmup];
    if (dt <= 0.0) return 0.0;
    return (double)count / dt;
}

uint32_t sg_bench_default_warmup(uint32_t n_gen) {
    long w = lround(0.02 * (double)n_gen);
    if (w < 1) w = 1;
    return (uint32_t)w;
}

void sg_bench_finalize_status(sg_bench_row *row) {
    if (!row) return;
    bool ok = (row->gemm_tflops > 20.5) && row->ingestion_ok;
    snprintf(row->status, sizeof row->status, "%s", ok ? "DONE" : "VOID");
    fprintf(stderr, "bench: status=%s (gemm_tflops=%.2f ingestion_ok=%s)\n",
            row->status, row->gemm_tflops, row->ingestion_ok ? "true" : "false");
}

void sg_bench_format_md_row(const sg_bench_row *row, char *buf, size_t cap) {
    if (!buf || cap == 0) return;
    if (!row) { buf[0] = '\0'; return; }

    char prefill[32];
    if (row->prefill_tps < 0.0) {
        snprintf(prefill, sizeof prefill, "-");
    } else {
        snprintf(prefill, sizeof prefill, "%.0f", row->prefill_tps);
    }

    long wall_min = lround(row->wall_s / 60.0);

    /* %.63s / %.15s bound the read to the field's array capacity even if a
     * caller ever left one of these char arrays without a NUL (defense in
     * depth on top of the "always NUL-terminate" contract in surge.h). */
    snprintf(buf, cap, "| %.63s | %.63s | %s | %.2f | %.1f GiB | %u/%u | %ld min | %.15s |",
             row->model, row->engine, prefill, row->decode_tps_slope,
             row->peak_ram_gib, row->recall_hits, row->recall_total,
             wall_min, row->status);
}

/* Escapes s (JSON string escaping: \\, \", and control bytes as \u00XX)
 * into out, truncating at cap-1 bytes so the field's own array bound
 * (rather than an assumed NUL) is what limits how far this reads --
 * matching the %.Ns bound in sg_bench_format_md_row above. A truncated
 * escape is still a well-formed JSON string, just shorter than the
 * source. s == NULL is treated as empty. */
static void bench_json_escape(const char *s, size_t max_src, char *out, size_t cap) {
    if (!out || cap == 0) return;
    size_t o = 0;
    for (size_t i = 0; s && i < max_src && s[i] != '\0' && o + 1 < cap; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            if (o + 2 >= cap) break;
            out[o++] = '\\';
            out[o++] = (char)c;
        } else if (c < 0x20) {
            if (o + 6 >= cap) break;
            snprintf(out + o, 7, "\\u%04x", c);
            o += 6;
        } else {
            out[o++] = (char)c;
        }
    }
    out[o] = '\0';
}

void sg_bench_format_json(const sg_bench_row *row, char *buf, size_t cap) {
    if (!buf || cap == 0) return;
    if (!row) { buf[0] = '\0'; return; }

    /* Sized for the worst case (every source byte becomes a 6-byte \u00XX
     * escape): model/engine (63 usable bytes) need up to 379, status (15)
     * up to 91, log_id (95) up to 571. Rounded up with margin. */
    char model_esc[400], engine_esc[400], status_esc[128], log_esc[600];
    bench_json_escape(row->model, sizeof row->model, model_esc, sizeof model_esc);
    bench_json_escape(row->engine, sizeof row->engine, engine_esc, sizeof engine_esc);
    bench_json_escape(row->status, sizeof row->status, status_esc, sizeof status_esc);
    bench_json_escape(row->log_id, sizeof row->log_id, log_esc, sizeof log_esc);

    snprintf(buf, cap,
        "{\"model\":\"%s\",\"engine\":\"%s\",\"prefill_tps\":%.6g,"
        "\"decode_tps_slope\":%.6g,\"decode_tps_avg\":%.6g,"
        "\"peak_ram_gib\":%.6g,\"gpu_alloc_gib\":%.6g,"
        "\"recall_hits\":%u,\"recall_total\":%u,\"assoc_hits\":%u,"
        "\"n_prompt_tok\":%llu,\"n_gen\":%u,\"wall_s\":%.6g,"
        "\"gemm_tflops\":%.6g,\"ingestion_ok\":%s,"
        "\"status\":\"%s\",\"log_id\":\"%s\"}",
        model_esc, engine_esc, row->prefill_tps,
        row->decode_tps_slope, row->decode_tps_avg,
        row->peak_ram_gib, row->gpu_alloc_gib,
        row->recall_hits, row->recall_total, row->assoc_hits,
        (unsigned long long)row->n_prompt_tok, row->n_gen, row->wall_s,
        row->gemm_tflops, row->ingestion_ok ? "true" : "false",
        status_esc, log_esc);
}
