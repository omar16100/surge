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
#include <ctype.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/task_info.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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

/* --------------------------------------------------------------------
 * Task B3: prompt ingestion + truncation guard.
 * -------------------------------------------------------------------- */

sg_err sg_bench_read_file(const char *path, char **out, size_t *len) {
    if (out) *out = NULL;
    if (len) *len = 0;
    if (!path || !out || !len) return (sg_err){"bench: read_file: invalid arguments"};

    int fd = open(path, O_RDONLY);
    if (fd < 0) return (sg_err){"bench: read_file: failed to open file"};

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return (sg_err){"bench: read_file: fstat failed"};
    }
    if (st.st_size <= 0) {
        close(fd);
        return (sg_err){"bench: read_file: empty file"};
    }
    uint64_t size = (uint64_t)st.st_size;
    if (size > SG_BENCH_MAX_FILE_BYTES) {
        close(fd);
        return (sg_err){"bench: read_file: file exceeds 3 GiB limit"};
    }

    char *buf = malloc((size_t)size + 1);
    if (!buf) {
        close(fd);
        return (sg_err){"bench: read_file: out of memory"};
    }

    size_t total = 0;
    while (total < (size_t)size) {
        ssize_t n = read(fd, buf + total, (size_t)size - total);
        if (n < 0) {
            close(fd);
            free(buf);
            return (sg_err){"bench: read_file: read failed"};
        }
        if (n == 0) break;   /* file shrank under us -- treat as short read */
        total += (size_t)n;
    }
    close(fd);
    if (total != (size_t)size) {
        free(buf);
        return (sg_err){"bench: read_file: short read"};
    }
    buf[size] = '\0';

    *out = buf;
    *len = (size_t)size;
    fprintf(stderr, "bench: read_file: %s (%zu bytes)\n", path, (size_t)size);
    return SG_OK;
}

void sg_bench_check_ingestion(uint64_t n_ids, uint32_t max_ctx, uint64_t expect_min,
                              uint64_t expect_max, bool *ok) {
    if (!ok) return;
    bool within_ctx = n_ids <= (uint64_t)max_ctx;
    bool within_expect = n_ids >= expect_min && n_ids <= expect_max;
    *ok = within_ctx && within_expect;
    fprintf(stderr,
            "bench: check_ingestion: n_ids=%llu max_ctx=%u expect=[%llu,%llu] ok=%s\n",
            (unsigned long long)n_ids, max_ctx, (unsigned long long)expect_min,
            (unsigned long long)expect_max, *ok ? "true" : "false");
}

/* --------------------------------------------------------------------
 * Task B4: NIAH recall scorer.
 * -------------------------------------------------------------------- */

/* The exact anchor phrase prefix this project's NIAH prompt generator
 * writes for every needle. Extraction keys on this literal, not on a bare
 * digit-run heuristic, so the prompt's trailing question line (which
 * repeats the city names with no codes attached) can never be mistaken
 * for a needle and filler digit runs elsewhere never inflate ground
 * truth -- see the contract comment in surge.h. */
static const char *const NIAH_ANCHOR = "IMPORTANT RECORD: the secret access code for ";

sg_err sg_bench_extract_needles(const char *prompt, sg_bench_needle *out, uint32_t cap,
                                 uint32_t *n_out) {
    if (n_out) *n_out = 0;
    if (!prompt || !out || cap == 0 || !n_out) {
        return (sg_err){"bench: extract_needles: invalid arguments"};
    }

    size_t anchor_len = strlen(NIAH_ANCHOR);
    uint32_t count = 0;
    const char *p = prompt;

    while ((p = strstr(p, NIAH_ANCHOR)) != NULL) {
        const char *anchor_pos = p;

        /* Always resume the NEXT strstr search exactly one byte past
         * where THIS anchor occurrence started, whether this candidate
         * turns out valid or not, and regardless of how far city/code
         * parsing below consumes. Jumping straight past a failed (or a
         * successful) match instead would risk stepping over a DIFFERENT
         * anchor occurrence that starts inside the text this candidate
         * consumed -- e.g. a malformed "city" scan that swallows the
         * literal "IMPORTANT" of a following, well-formed occurrence.
         * One-byte steps make that impossible; the extra strstr calls are
         * negligible at this file's scale (a handful of needles). */
        p = anchor_pos + 1;

        const char *city_start = anchor_pos + anchor_len;

        /* City: a capitalized word (first letter uppercase, then any run
         * of letters, no spaces), bounded to the field's capacity so an
         * unexpectedly long token can never overflow the fixed-size
         * out[].city array. */
        if (!isupper((unsigned char)city_start[0])) continue;
        const char *q = city_start;
        size_t city_len = 0;
        while (isalpha((unsigned char)*q) && city_len < SG_BENCH_NEEDLE_CITY_MAX - 1) {
            q++;
            city_len++;
        }

        /* Literal " is " between city and code. */
        const char *r = q;
        if (r[0] != ' ' || r[1] != 'i' || r[2] != 's' || r[3] != ' ') continue;
        r += 4;

        /* Code: a run of digits, bounded to the field's capacity, then a
         * literal '.' immediately after the last digit -- required by the
         * exact anchor phrase, so a code with no trailing '.' (i.e. not
         * this needle format at all) is rejected rather than partially
         * matched. This project's needle codes are always 8+ digits (see
         * surge.h); a shorter digit run is rejected too, since it cannot
         * be a real needle in this format. */
        const char *code_start = r;
        size_t code_len = 0;
        while (isdigit((unsigned char)r[code_len]) && code_len < SG_BENCH_NEEDLE_CODE_MAX - 1) {
            code_len++;
        }
        if (code_len < 8 || r[code_len] != '.') continue;

        if (count >= cap) {
            fprintf(stderr,
                    "bench: extract_needles: cap %u reached, stopping scan (more matches "
                    "may remain unscanned)\n", cap);
            break;
        }
        memcpy(out[count].city, city_start, city_len);
        out[count].city[city_len] = '\0';
        memcpy(out[count].code, code_start, code_len);
        out[count].code[code_len] = '\0';
        count++;
    }

    *n_out = count;
    fprintf(stderr, "bench: extract_needles: found %u needle pair(s) (cap=%u)\n", count, cap);
    return SG_OK;
}

/* Naive bounded substring search: does hay[0..hay_len) contain needle (a
 * NUL-terminated C string) anywhere? Unlike strstr, this never reads past
 * hay_len, so it is safe to call on a LINE slice of a larger buffer that
 * is not itself NUL-terminated at the line boundary. */
static bool bench_mem_find(const char *hay, size_t hay_len, const char *needle) {
    size_t needle_len = strlen(needle);
    if (needle_len == 0 || needle_len > hay_len) return false;
    for (size_t i = 0; i + needle_len <= hay_len; i++) {
        if (memcmp(hay + i, needle, needle_len) == 0) return true;
    }
    return false;
}

void sg_bench_score_niah(const char *gen, const sg_bench_needle *needles, uint32_t n_needles,
                          uint32_t *retrieval_hits, uint32_t *assoc_hits) {
    if (retrieval_hits) *retrieval_hits = 0;
    if (assoc_hits) *assoc_hits = 0;
    if (!gen || !needles || n_needles == 0) return;

    size_t gen_len = strlen(gen);
    uint32_t r_hits = 0, a_hits = 0;

    for (uint32_t i = 0; i < n_needles; i++) {
        const char *code = needles[i].code;
        const char *city = needles[i].city;
        if (code[0] == '\0') continue;

        bool retrieved = strstr(gen, code) != NULL;
        if (retrieved) r_hits++;
        if (!retrieved || city[0] == '\0') continue;

        /* Walk gen's lines by tracking [line_start, j) offsets -- no
         * strtok, no copy, no write into gen. A trailing '\r' is trimmed
         * from each line so CRLF-terminated gen output still associates
         * correctly. */
        bool assoc = false;
        size_t line_start = 0;
        for (size_t j = 0; j <= gen_len && !assoc; j++) {
            if (j != gen_len && gen[j] != '\n') continue;
            size_t line_len = j - line_start;
            if (line_len > 0 && gen[line_start + line_len - 1] == '\r') line_len--;
            if (bench_mem_find(gen + line_start, line_len, code) &&
                bench_mem_find(gen + line_start, line_len, city)) {
                assoc = true;
            }
            line_start = j + 1;
        }
        if (assoc) a_hits++;
    }

    if (retrieval_hits) *retrieval_hits = r_hits;
    if (assoc_hits) *assoc_hits = a_hits;
    fprintf(stderr, "bench: score_niah: retrieval=%u/%u assoc=%u/%u\n",
            r_hits, n_needles, a_hits, n_needles);
}

/* --------------------------------------------------------------------
 * Task B2: peak-memory probe (process phys_footprint + the pure-C tracker).
 * sg_gpu_current_alloc_bytes, the Metal half of the probe, lives in
 * src/metal.m -- this file stays Metal-free, per the header comment.
 * -------------------------------------------------------------------- */

uint64_t sg_proc_phys_footprint(void) {
    task_vm_info_data_t info;
    memset(&info, 0, sizeof info);
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                 (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "bench: proc_phys_footprint: task_info failed (kr=%d)\n", (int)kr);
        return 0;
    }
    /* task_info's count is IN/OUT: on a kernel that only understands an
     * older, shorter task_vm_info revision, it can return success with
     * fewer natural_t words written than this SDK's full struct declares,
     * leaving the tail (phys_footprint included) uninitialized past
     * whatever the kernel actually filled in. TASK_VM_INFO_REV0_COUNT is
     * the one revision that predates phys_footprint entirely ("doesn't
     * include phys_footprint", mach/task_info.h); reject anything short of
     * REV1 rather than read that field out of the memset(0) padding. */
    if (count < TASK_VM_INFO_REV1_COUNT) {
        fprintf(stderr, "bench: proc_phys_footprint: kernel returned rev0 task_vm_info "
                        "(count=%u, no phys_footprint field)\n", (unsigned)count);
        return 0;
    }
    return (uint64_t)info.phys_footprint;
}

void sg_mem_tracker_reset(sg_mem_tracker *t) {
    if (!t) return;
    t->peak = 0;
}

void sg_mem_tracker_sample(sg_mem_tracker *t, uint64_t current_alloc, uint64_t phys_footprint) {
    if (!t) return;
    uint64_t m = current_alloc > phys_footprint ? current_alloc : phys_footprint;
    if (m > t->peak) t->peak = m;
}

uint64_t sg_mem_tracker_peak(const sg_mem_tracker *t) {
    return t ? t->peak : 0;
}
