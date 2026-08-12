/* test_bench.c - src/bench.c: decode-by-slope, mlx-style average, and the
 * leaderboard-row formatters (Task B1). Pure C, no Metal, no GPU; runs
 * under a plain `make check` and under `make debug` (ASan/UBSan,
 * SURGE_NO_METAL).
 *
 * Covers the B1 gate verbatim:
 *   (a) exact-linear series slope within 1e-9 relative of 5.0 tok/s, and
 *       slope == avg to 1e-9.
 *   (b) jittered series: |slope - avg| < 3%.
 *   (c) warmup=1 provably drops index 0.
 *   (d) sg_bench_format_md_row byte-equals a checked-in golden string.
 *   (e) JSON round-trips numeric fields (parsed back out, not just
 *       substring-matched).
 *   (f) status auto-sets to VOID/DONE per the gemm/ingestion rule.
 */

#include "surge.h"
#include "tinytest.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* --------------------------------------------------------------------
 * Deterministic xorshift64 RNG for the jittered series (b). Fixed seed so
 * the test is reproducible run to run.
 * -------------------------------------------------------------------- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static double rng_unit(void) {   /* uniform in [0, 1) */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
}

/* --------------------------------------------------------------------
 * (a) exact-linear series at 5.0 tok/s.
 * -------------------------------------------------------------------- */
static void test_slope_exact_linear(void) {
    const uint32_t n = 1000;
    const double rate = 5.0;   /* tok/s */
    double t[1000];
    for (uint32_t i = 0; i < n; i++) t[i] = (double)i / rate;

    uint32_t warmup = 20;
    double slope = sg_bench_slope(t, n, warmup);
    double avg = sg_bench_avg_tps(t, n, warmup);

    double rel_slope = fabs(slope - rate) / rate;
    tt_assert(rel_slope < 1e-9, "slope=%.15g rate=%.15g rel=%.3e (want <1e-9)",
              slope, rate, rel_slope);

    double rel_avg = fabs(avg - rate) / rate;
    tt_assert(rel_avg < 1e-9, "avg=%.15g rate=%.15g rel=%.3e (want <1e-9)",
              avg, rate, rel_avg);

    double rel_slope_avg = fabs(slope - avg) / avg;
    tt_assert(rel_slope_avg < 1e-9, "slope=%.15g avg=%.15g rel diff=%.3e (want <1e-9)",
              slope, avg, rel_slope_avg);
}

/* --------------------------------------------------------------------
 * (a2) same exact-linear series, but shifted by a realistic epoch-scale
 * offset (~1.8e9, like raw gettimeofday() seconds). A naive one-pass
 * normal-equation OLS loses this to catastrophic cancellation (verified
 * separately in Python: it returns a NEGATIVE slope at base=1.8e9 on
 * exactly this series); the mean-centered implementation must not.
 * -------------------------------------------------------------------- */
static void test_slope_epoch_offset(void) {
    const uint32_t n = 300;
    const double rate = 5.0;
    const uint32_t warmup = 6;
    double bases[] = { 0.0, 1e6, 1e9, 1.8e9 };

    for (size_t b = 0; b < sizeof bases / sizeof bases[0]; b++) {
        double t[300];
        for (uint32_t i = 0; i < n; i++) t[i] = bases[b] + (double)i / rate;

        double slope = sg_bench_slope(t, n, warmup);
        double avg = sg_bench_avg_tps(t, n, warmup);

        double rel_slope = fabs(slope - rate) / rate;
        tt_assert(rel_slope < 1e-6,
                  "base=%.3g: slope=%.15g rate=%.15g rel=%.3e (want <1e-6)",
                  bases[b], slope, rate, rel_slope);
        double rel_avg = fabs(avg - rate) / rate;
        tt_assert(rel_avg < 1e-6,
                  "base=%.3g: avg=%.15g rate=%.15g rel=%.3e (want <1e-6)",
                  bases[b], avg, rate, rel_avg);
    }
}

/* --------------------------------------------------------------------
 * (b) jittered series: +/-15% multiplicative jitter on each inter-token
 * interval, still monotonic increasing. |slope - avg| < 3%.
 * -------------------------------------------------------------------- */
static void test_slope_jittered(void) {
    const uint32_t n = 2000;
    const double rate = 5.0;
    const double dt_base = 1.0 / rate;
    static double t[2000];

    double acc = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double eps = (rng_unit() * 2.0 - 1.0) * 0.15;   /* +/-15% */
        double dt = dt_base * (1.0 + eps);
        acc += dt;
        t[i] = acc;
    }

    uint32_t warmup = sg_bench_default_warmup(n);   /* exercise the helper too */
    double slope = sg_bench_slope(t, n, warmup);
    double avg = sg_bench_avg_tps(t, n, warmup);

    double rel = fabs(slope - avg) / avg;
    tt_assert(rel < 0.03, "jittered: slope=%.6f avg=%.6f rel diff=%.4f (want <0.03)",
              slope, avg, rel);
    /* both should still land near the true 5.0 tok/s rate */
    tt_assert(fabs(slope - rate) / rate < 0.05,
              "jittered slope=%.6f far from true rate %.6f", slope, rate);
}

/* --------------------------------------------------------------------
 * (c) warmup=1 provably drops index 0: corrupting t[0] must not move the
 * slope computed with warmup=1 (it reads t[1..n) only, byte-identical to
 * the uncorrupted array over that range), while warmup=0 (which reads the
 * corrupted point) clearly does move.
 * -------------------------------------------------------------------- */
static void test_warmup_excludes_index0(void) {
    const uint32_t n = 500;
    const double rate = 5.0;
    double t[500], t_bad[500];
    for (uint32_t i = 0; i < n; i++) t[i] = (double)i / rate;
    memcpy(t_bad, t, sizeof t);
    t_bad[0] = 999999.0;   /* drastic outlier, index 0 only */

    double slope_clean = sg_bench_slope(t, n, 1);
    double slope_bad_w1 = sg_bench_slope(t_bad, n, 1);
    tt_assert(slope_clean == slope_bad_w1,
              "warmup=1 must exclude index 0: clean=%.15g corrupted=%.15g",
              slope_clean, slope_bad_w1);

    /* sanity: the corruption DOES matter when index 0 is included, so the
     * above equality is a real exclusion and not a no-op. */
    double slope_bad_w0 = sg_bench_slope(t_bad, n, 0);
    tt_assert(fabs(slope_bad_w0 - rate) > 1e-3,
              "including the corrupted index 0 should visibly perturb the fit "
              "(got slope=%.6g, true rate=%.6g)", slope_bad_w0, rate);
}

/* --------------------------------------------------------------------
 * (d) golden md row.
 * -------------------------------------------------------------------- */
static void test_format_md_row_golden(void) {
    sg_bench_row row;
    memset(&row, 0, sizeof row);
    snprintf(row.model, sizeof row.model, "Qwen3.6-27B");
    snprintf(row.engine, sizeof row.engine, "surge Q8_0");
    row.prefill_tps = 46.0;
    row.decode_tps_slope = 3.64;
    row.decode_tps_avg = 3.60;
    row.peak_ram_gib = 75.5;
    row.recall_hits = 2;
    row.recall_total = 8;
    row.wall_s = 97.0 * 60.0;
    snprintf(row.status, sizeof row.status, "DONE");

    char buf[256];
    sg_bench_format_md_row(&row, buf, sizeof buf);
    const char *golden =
        "| Qwen3.6-27B | surge Q8_0 | 46 | 3.64 | 75.5 GiB | 2/8 | 97 min | DONE |";
    tt_assert(strcmp(buf, golden) == 0,
              "md row mismatch:\n  got:  %s\n  want: %s", buf, golden);

    /* prefill_tps < 0 -> "-" */
    row.prefill_tps = -1.0;
    sg_bench_format_md_row(&row, buf, sizeof buf);
    const char *golden_dash =
        "| Qwen3.6-27B | surge Q8_0 | - | 3.64 | 75.5 GiB | 2/8 | 97 min | DONE |";
    tt_assert(strcmp(buf, golden_dash) == 0,
              "md row (no prefill) mismatch:\n  got:  %s\n  want: %s", buf, golden_dash);

    /* VOID status row, distinct model/engine/wall, to catch any hard-coded
     * field in the formatter. */
    sg_bench_row row2;
    memset(&row2, 0, sizeof row2);
    snprintf(row2.model, sizeof row2.model, "Qwen3.6-35B-A3B");
    snprintf(row2.engine, sizeof row2.engine, "mlx-lm 8-bit");
    row2.prefill_tps = 136.0;
    row2.decode_tps_slope = 9.54;
    row2.peak_ram_gib = 61.8;
    row2.recall_hits = 2;
    row2.recall_total = 8;
    row2.wall_s = 32.0 * 60.0;
    snprintf(row2.status, sizeof row2.status, "VOID");
    sg_bench_format_md_row(&row2, buf, sizeof buf);
    const char *golden2 =
        "| Qwen3.6-35B-A3B | mlx-lm 8-bit | 136 | 9.54 | 61.8 GiB | 2/8 | 32 min | VOID |";
    tt_assert(strcmp(buf, golden2) == 0,
              "md row 2 mismatch:\n  got:  %s\n  want: %s", buf, golden2);

    /* NULL row -> empty string; zero cap -> untouched (no overflow). */
    buf[0] = 'X'; buf[1] = '\0';
    sg_bench_format_md_row(NULL, buf, sizeof buf);
    tt_assert(buf[0] == '\0', "NULL row -> empty string");
    buf[0] = 'X';
    sg_bench_format_md_row(&row, buf, 0);
    tt_assert(buf[0] == 'X', "zero cap must not write");
}

/* --------------------------------------------------------------------
 * (e) JSON round-trips numeric fields: parse the emitted JSON's values
 * back out with strtod/strtoull and compare to the source row.
 * -------------------------------------------------------------------- */
static double json_double_after(const char *json, const char *key) {
    const char *p = strstr(json, key);
    if (!p) return NAN;
    return strtod(p + strlen(key), NULL);
}
static unsigned long long json_uint_after(const char *json, const char *key) {
    const char *p = strstr(json, key);
    if (!p) return 0;
    return strtoull(p + strlen(key), NULL, 10);
}

static void test_format_json(void) {
    sg_bench_row row;
    memset(&row, 0, sizeof row);
    snprintf(row.model, sizeof row.model, "Qwen3.6-27B");
    snprintf(row.engine, sizeof row.engine, "surge Q8_0");
    row.prefill_tps = 46.0;
    row.decode_tps_slope = 3.64;
    row.decode_tps_avg = 3.60123;
    row.prefill_wall_s = 5700.0;
    row.prefill_rest_s = 5400.0;         /* Task B8 */
    row.prefill_compute_tps = 873.13;    /* Task B8 */
    row.peak_ram_gib = 75.5;
    row.gpu_alloc_gib = 70.25;
    row.recall_hits = 2;
    row.recall_total = 8;
    row.assoc_hits = 1;
    row.n_prompt_tok = 262144;
    row.n_gen = 300;
    row.wall_s = 5820.0;
    row.gemm_tflops = 21.37;
    row.ingestion_ok = true;
    snprintf(row.status, sizeof row.status, "DONE");
    snprintf(row.log_id, sizeof row.log_id, "ctx256k_qwen27b_surge_20260811_120000");

    char buf[1024];
    sg_bench_format_json(&row, buf, sizeof buf);

    tt_assert(strstr(buf, "\"model\":\"Qwen3.6-27B\"") != NULL, "model key: %s", buf);
    tt_assert(strstr(buf, "\"engine\":\"surge Q8_0\"") != NULL, "engine key: %s", buf);
    tt_assert(strstr(buf, "\"status\":\"DONE\"") != NULL, "status key: %s", buf);
    tt_assert(strstr(buf, "\"ingestion_ok\":true") != NULL, "ingestion_ok key: %s", buf);
    tt_assert(strstr(buf, "\"log_id\":\"ctx256k_qwen27b_surge_20260811_120000\"") != NULL,
              "log_id key: %s", buf);

    #define CHECK_D(key, want) do { \
        double got = json_double_after(buf, key); \
        double rel = fabs(got - (want)) / ((want) == 0.0 ? 1.0 : fabs(want)); \
        tt_assert(rel < 1e-4, key " round-trip: got %.9g want %.9g (rel %.3e)", \
                  got, (double)(want), rel); \
    } while (0)
    CHECK_D("\"prefill_tps\":", row.prefill_tps);
    CHECK_D("\"decode_tps_slope\":", row.decode_tps_slope);
    CHECK_D("\"decode_tps_avg\":", row.decode_tps_avg);
    CHECK_D("\"prefill_wall_s\":", row.prefill_wall_s);
    CHECK_D("\"prefill_rest_s\":", row.prefill_rest_s);         /* Task B8 */
    CHECK_D("\"prefill_compute_tps\":", row.prefill_compute_tps); /* Task B8 */
    CHECK_D("\"peak_ram_gib\":", row.peak_ram_gib);
    CHECK_D("\"gpu_alloc_gib\":", row.gpu_alloc_gib);
    CHECK_D("\"wall_s\":", row.wall_s);
    CHECK_D("\"gemm_tflops\":", row.gemm_tflops);
    #undef CHECK_D

    tt_assert(json_uint_after(buf, "\"recall_hits\":") == row.recall_hits, "recall_hits");
    tt_assert(json_uint_after(buf, "\"recall_total\":") == row.recall_total, "recall_total");
    tt_assert(json_uint_after(buf, "\"assoc_hits\":") == row.assoc_hits, "assoc_hits");
    tt_assert(json_uint_after(buf, "\"n_prompt_tok\":") == row.n_prompt_tok, "n_prompt_tok");
    tt_assert(json_uint_after(buf, "\"n_gen\":") == row.n_gen, "n_gen");

    /* NULL row -> empty string; zero cap -> untouched. */
    buf[0] = 'X'; buf[1] = '\0';
    sg_bench_format_json(NULL, buf, sizeof buf);
    tt_assert(buf[0] == '\0', "NULL row -> empty string");
    buf[0] = 'X';
    sg_bench_format_json(&row, buf, 0);
    tt_assert(buf[0] == 'X', "zero cap must not write");
}

/* A model/log_id string containing a double quote and a backslash must
 * still produce valid, round-trippable JSON (the escaper's own contract).
 * -------------------------------------------------------------------- */
static void test_format_json_escaping(void) {
    sg_bench_row row;
    memset(&row, 0, sizeof row);
    snprintf(row.model, sizeof row.model, "Qwen\"3.6\\27B");
    snprintf(row.engine, sizeof row.engine, "surge");
    snprintf(row.status, sizeof row.status, "DONE");
    snprintf(row.log_id, sizeof row.log_id, "id_with_tab\there");

    char buf[512];
    sg_bench_format_json(&row, buf, sizeof buf);

    tt_assert(strstr(buf, "\"model\":\"Qwen\\\"3.6\\\\27B\"") != NULL,
              "quote/backslash escaped: %s", buf);
    tt_assert(strstr(buf, "\\u0009") != NULL, "tab escaped as \\u0009: %s", buf);
    /* No raw (unescaped) quote should appear where it would break the
     * surrounding "..." string; a crude check: the model value's escaped
     * form above already proves the quote was escaped, so just confirm the
     * object still closes cleanly. */
    size_t len = strlen(buf);
    tt_assert(len > 2 && buf[0] == '{' && buf[len - 1] == '}',
              "still a well-formed JSON object: %s", buf);
}

/* --------------------------------------------------------------------
 * (f) status auto-set: VOID unless (gemm_tflops > 20.5 && ingestion_ok).
 * -------------------------------------------------------------------- */
static void test_finalize_status(void) {
    struct { double gemm; bool ok; const char *want; const char *note; } cases[] = {
        { 25.0, true,  "DONE", "good gemm + ingestion ok" },
        { 20.6, true,  "DONE", "just above threshold" },
        { 20.5, true,  "VOID", "exactly at threshold (not strictly >)" },
        { 20.5000001, true, "DONE", "just above threshold (float compare)" },
        { 10.0, true,  "VOID", "gemm below threshold" },
        { 25.0, false, "VOID", "ingestion failed despite good gemm" },
        { 5.0,  false, "VOID", "both bad" },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        sg_bench_row row;
        memset(&row, 0, sizeof row);
        row.gemm_tflops = cases[i].gemm;
        row.ingestion_ok = cases[i].ok;
        sg_bench_finalize_status(&row);
        tt_assert(strcmp(row.status, cases[i].want) == 0,
                  "%s: gemm=%.7f ok=%d -> got %s want %s",
                  cases[i].note, cases[i].gemm, cases[i].ok, row.status, cases[i].want);

        /* sg_bench_admitted is the shared predicate finalize_status and
         * surge-bench's pre-load short-circuit both use: it must agree with
         * the status the finalize just wrote, case for case, so the two paths
         * cannot desync. */
        bool admitted = sg_bench_admitted(&row);
        tt_assert(admitted == (strcmp(cases[i].want, "DONE") == 0),
                  "%s: admitted=%d disagrees with status %s",
                  cases[i].note, admitted, row.status);
    }
    /* NULL row must not crash (both entry points). */
    sg_bench_finalize_status(NULL);
    tt_assert(!sg_bench_admitted(NULL), "NULL row is not admitted");
}

/* --------------------------------------------------------------------
 * default warmup helper: max(1, round(0.02*n_gen)).
 * -------------------------------------------------------------------- */
static void test_default_warmup(void) {
    tt_assert(sg_bench_default_warmup(300) == 6,
              "default_warmup(300) = %u, want 6", sg_bench_default_warmup(300));
    tt_assert(sg_bench_default_warmup(100) == 2,
              "default_warmup(100) = %u, want 2", sg_bench_default_warmup(100));
    tt_assert(sg_bench_default_warmup(10) == 1,
              "default_warmup(10) = %u, want 1 (floored)", sg_bench_default_warmup(10));
    tt_assert(sg_bench_default_warmup(0) == 1,
              "default_warmup(0) = %u, want 1 (floored)", sg_bench_default_warmup(0));
    tt_assert(sg_bench_default_warmup(50) == 1,
              "default_warmup(50) = %u, want round(1.0)=1", sg_bench_default_warmup(50));
}

/* --------------------------------------------------------------------
 * edge cases: NULL / n=0 / warmup>=n must not crash and must return 0.0.
 * -------------------------------------------------------------------- */
static void test_edge_cases(void) {
    double t[5] = { 0.0, 0.2, 0.4, 0.6, 0.8 };
    tt_assert(sg_bench_slope(NULL, 10, 0) == 0.0, "slope NULL -> 0");
    tt_assert(sg_bench_slope(t, 0, 0) == 0.0, "slope n=0 -> 0");
    tt_assert(sg_bench_slope(t, 5, 5) == 0.0, "slope warmup==n -> 0");
    tt_assert(sg_bench_slope(t, 5, 10) == 0.0, "slope warmup>n -> 0");
    tt_assert(sg_bench_slope(t, 5, 4) == 0.0, "slope only 1 point after warmup -> 0");

    tt_assert(sg_bench_avg_tps(NULL, 10, 0) == 0.0, "avg NULL -> 0");
    tt_assert(sg_bench_avg_tps(t, 0, 0) == 0.0, "avg n=0 -> 0");
    tt_assert(sg_bench_avg_tps(t, 5, 5) == 0.0, "avg warmup==n -> 0");
    tt_assert(sg_bench_avg_tps(t, 5, 4) == 0.0, "avg warmup==n-1 -> 0 (zero tokens)");

    double flat[4] = { 1.0, 1.0, 1.0, 1.0 };
    tt_assert(sg_bench_slope(flat, 4, 0) == 0.0, "degenerate (flat) x -> slope 0");
    tt_assert(sg_bench_avg_tps(flat, 4, 0) == 0.0, "degenerate (flat) x -> avg 0");
}

int main(void) {
    tt_run("slope: exact-linear 5.0 tok/s", test_slope_exact_linear);
    tt_run("slope: robust to epoch-scale time offsets", test_slope_epoch_offset);
    tt_run("slope vs avg: jittered series <3%", test_slope_jittered);
    tt_run("warmup=1 provably drops index 0", test_warmup_excludes_index0);
    tt_run("md row: golden byte-equal", test_format_md_row_golden);
    tt_run("json: round-trips numeric fields", test_format_json);
    tt_run("json: escapes quote/backslash/control bytes", test_format_json_escaping);
    tt_run("finalize_status: VOID/DONE rule", test_finalize_status);
    tt_run("default warmup helper", test_default_warmup);
    tt_run("edge cases (NULL/n=0/warmup>=n)", test_edge_cases);
    return tt_report();
}
