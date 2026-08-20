/* bench_splitk.c - Task P2.3a: split-K decode-attention timing harness.
 *
 * P2.2 gated the split-K decode-attention kernels (k_attn_decode_splitk_partial
 * + k_attn_decode_splitk_combine) for CORRECTNESS: worst 1.027e-06 relative vs
 * both CPU oracles, 100x byte-identical. What P2.2 could NOT answer, because
 * the GPU was held by a benchmark for that entire task, is whether split-K is
 * FASTER than the incumbent k_attn_decode_f16, and at what n_splits. This file
 * is the instrument that answers that question. It is a DECISION gate, not a
 * parameter fit: P2.3 (wiring split-K into enc_attn) should not happen at all
 * if split-K does not beat the incumbent at the shapes that matter.
 *
 * NO TIMINGS HAVE BEEN MEASURED AS OF THIS FILE BEING WRITTEN. A Phase 0
 * benchmark held the GPU for the entire time this was authored (confirmed via
 * `pgrep -f "bench_niah|phase0|surge-bench"` before the first edit and after
 * the last), so this harness has only been WRITTEN and COMPILE-CHECKED
 * (`xcrun clang -fsyntax-only`), never linked against the Metal frameworks or
 * run. Every number this binary ever prints is measured live, by whoever runs
 * it after the GPU frees; nothing below is a claim about what those numbers
 * will be.
 *
 * WHY THIS IS A SEPARATE FILE, NOT PART OF `make check`. This is a
 * MEASUREMENT tool with no pass/fail assertion (unlike tests/test_metal_ops.c,
 * which gates correctness). `make check` builds $(TESTS:.c=.bin), and TESTS is
 * `$(wildcard tests/test_*.c)` in the Makefile, so naming this file
 * `bench_splitk.c` (not `test_*.c`) keeps it out of that list structurally,
 * not by convention someone has to remember. It is reachable only through the
 * explicit `make bench-splitk` target, which builds AND RUNS it; `make check`
 * and `make debug` never build or run it. See the Makefile's own comment
 * block above the BENCH_SPLITK variable for the build-rule side of this.
 *
 * METHODOLOGY.
 *   - Shapes: the real 27B decode shape (n_heads 24, n_kv_heads 4,
 *     head_dim 256) and the real 4B dense shape (n_heads 32, n_kv_heads 8,
 *     head_dim 128), both from task-P2.3a-brief.md.
 *   - q_stride is set to head_dim (dense) for both shapes, not 2*head_dim (the
 *     hybrid model's interleaved attention-gate layout). This is deliberate
 *     and does not bias the timing: both k_attn_decode_f16 and
 *     k_attn_decode_splitk_partial only ever read q[0 .. head_dim) (see
 *     kernels.metal's own comment on the hybrid layout, "the gate half is
 *     never read"), so the extra stride changes which bytes q_stride skips
 *     over, not how many the kernel touches. Using head_dim keeps the q
 *     buffer smaller with no effect on either kernel's memory traffic.
 *   - Sequence lengths: 8192, 32768, 131072, 262144 (spans the regimes from
 *     "fits in a heartbeat" to the 256K target this project is built for).
 *   - n_splits swept over {1,2,4,8,16,32,64,128,256,512,1024}, CLAMPED to the
 *     occupancy band documented in surge.h next to
 *     sg_gpu_run_attn_splitk_partial ("OCCUPANCY" paragraph) and restated in
 *     task-P2.3a-brief.md: roughly 4 <= n_splits <= seq / 256 (256 is SG_TG,
 *     the threadgroup width both metal.m and kernels.metal fix the split-K
 *     kernels to). Each raw sweep value is clamped into [4, seq/256] and
 *     adjacent duplicates (which clamping produces once several raw values
 *     collapse to the same boundary) are dropped, so the printed n_splits
 *     list for a given seq is always the DISTINCT, in-band values, never a
 *     value known in advance to leave most lanes or most threadgroups idle.
 *   - For each (shape, seq), the incumbent is measured ONCE (it does not
 *     depend on n_splits) and printed on EVERY n_splits row for that (shape,
 *     seq) group, so every row is a direct, self-contained A/B: the brief's
 *     requirement that split-K is "ALWAYS" timed against the incumbent on the
 *     identical shape in the same run, not compared to a number from a
 *     different invocation.
 *   - Split-K's timed unit is BOTH dispatches together
 *     (k_attn_decode_splitk_partial then k_attn_decode_splitk_combine),
 *     because that pair is what is needed to reach the same "final attention
 *     output for this token" the incumbent produces in one dispatch. Timing
 *     only the partial would understate split-K's real cost.
 *   - Warm-up: one untimed, discarded call before the timed loop, for both
 *     the incumbent and the split-K pair, so first-dispatch effects (pipeline
 *     state binding, page-in of the freshly filled K/V buffers) do not land
 *     inside the measured samples.
 *   - N repetitions (--reps, default DEFAULT_REPS below) are timed with
 *     clock_gettime(CLOCK_MONOTONIC) wrapped tightly around each synchronous
 *     commit-and-wait call (sg_gpu_run_attn_decode_f16 /
 *     sg_gpu_run_attn_splitk_partial / _combine all commit and
 *     waitUntilCompleted internally, per metal.m, so wall time here is
 *     dispatch + GPU execution + wait, the same convention
 *     src/cli_bench.c's own now_s() uses). Mean, min and max are computed;
 *     the printed table's *_us columns and the achieved-GB/s columns are the
 *     MEAN; min/max are folded into the per-seq band line only when they
 *     diverge enough to be worth a second look (see run_shape below).
 *
 * ACHIEVED GB/s, and exactly what it counts. Per the design spec
 * (docs/superpowers/specs/2026-08-08-surge-design.md, "per-kernel
 * achieved-GB/s instrumentation"): bytes_kv() below counts the K and V bytes
 * read from the fp16 KV cache, ONCE PER QUERY-HEAD THREADGROUP. That matches
 * how both kernels actually touch memory: k_attn_decode_f16 dispatches one
 * threadgroup per query head, and within it every key/value element in
 * [0, seq) is read by exactly one thread (see the strided loops in
 * kernels.metal); k_attn_decode_splitk_partial covers the same total [0, seq)
 * range, just cut into n_splits contiguous pieces across more threadgroups,
 * so the SAME total element count is read for a given (shape, seq) regardless
 * of n_splits. Using n_heads (not n_kv_heads) as the multiplier is
 * deliberate: under GQA, `repeat` query heads share one kv head's data, but
 * each is its own threadgroup with no cross-threadgroup cache contract, and
 * at the sequence lengths here (8192..262144) one kv head's K+V already spans
 * tens to hundreds of MiB, far past any GPU cache, so the repeat heads are
 * NOT expected to hit a shared cache line for each other's reads. This is
 * therefore the realistic total HBM traffic, not the smaller "unique KV data"
 * figure, and it is the SAME formula for both the incumbent and split-K, so
 * the achieved-GB/s columns are directly comparable and the gap to whatever
 * this machine's real memory bandwidth is stays attributable to occupancy,
 * not to the two kernels being held to different yardsticks. It excludes: the
 * q read and the final output write (n_heads*head_dim floats each, orders of
 * magnitude smaller than the KV term at every seq tested here), and split-K's
 * m/s/acc scratch traffic (n_heads*n_splits*(2+head_dim) floats, also small
 * next to the KV term in this sweep's range, and asymmetric between the two
 * kernels in a way that would stop the GB/s columns from being a clean A/B if
 * folded in).
 *
 * ALLOCATION HONESTY. K/V for the largest shape/seq pair here (262144 x
 * head_dim 256) is a bit over 1 GiB total (see the per-seq band line, printed
 * from the exact byte count, not estimated). Every K/V allocation goes
 * through galloc(), which returns success/failure without exiting the
 * process; a failure prints a SKIP line naming the shape, seq and the exact
 * byte count that could not be allocated, and the sweep continues to the next
 * seq rather than aborting or silently dropping the row. The q/out/m/s/acc
 * buffers are small (tens of MiB at the sweep's largest n_splits) and are
 * allocated ONCE per shape, sized for the LARGEST n_splits this file ever
 * tests (1024) and reused unchanged across every seq and every n_splits in
 * that shape's sweep: sg_gpu_run_attn_splitk_partial/_combine accept a buffer
 * that is bigger than params[] strictly requires (buf_big_enough in metal.m
 * is a >= check), so this is not a size rule violation, just fewer
 * allocations. If even that shared setup allocation fails, the whole shape is
 * skipped with a clear message (see run_shape).
 *
 * Purely additive: no kernel, no metal.m entry point, no sg_ref_*, no
 * existing gate is touched by this file.
 *
 * Usage: tests/bench_splitk.bin [--reps N] [--seqs A,B,C] [--gqa] [-h|--help]
 *   (built and run together by `make bench-splitk`; see the Makefile.)
 *
 * --seqs was added by task P2.3 to sweep sequence lengths the default list
 * does not reach. That list starts at 8192, but decode has to decide what to
 * do BELOW it, where the n_splits = clamp(seq/SG_TG, 4, 1024) closed form
 * stops handing each split a full SG_TG keys; a threshold there should be
 * measured like the rest of the curve rather than argued from the grid
 * arithmetic. The default sweep is unchanged when the flag is absent.
 *
 * --gqa was added by task P2.4 to time the GQA-shared partial
 * (k_attn_decode_splitk_partial_gqa) in place of the per-head one. Same
 * sweep, same shapes, same yardstick, same output bytes: the two runs differ
 * only in which kernel is dispatched, so subtracting the two tables is the
 * kernel-level A/B for that task, including the short-sequence end where the
 * GQA grid is `repeat` times smaller and may lose. Default off, i.e. every
 * pre-P2.4 invocation of this binary means exactly what it did before.
 */
#ifdef SURGE_NO_METAL

#include <stdio.h>
int main(void) {
    fprintf(stderr, "SKIP bench_splitk: built with -DSURGE_NO_METAL "
                    "(Metal and the ASan/UBSan run do not mix)\n");
    return 0;
}

#else

#include "../surge.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_REPS 20

/* SG_TG, mirrored from src/metal.m / src/kernels.metal: the fixed threadgroup
 * width both k_attn_decode_f16 and the split-K pair dispatch. Used only to
 * derive the occupancy band's upper bound (seq / SPLIT_TG_KEYS), exactly as
 * surge.h documents it next to sg_gpu_run_attn_splitk_partial. */
#define SPLIT_TG_KEYS 256u

/* The occupancy band's lower bound, from task-P2.3a-brief.md and surge.h's
 * OCCUPANCY paragraph: "roughly 4 <= n_splits <= seq / SG_TG" derived from
 * n_heads * n_splits >= 80 GPU cores at n_heads in {24, 32}. Both shapes swept
 * here share this one lower bound rather than each computing its own tighter
 * ceil(80 / n_heads), matching the brief's literal band. */
#define BAND_LO 4u

/* --------------------------------------------------------------------
 * Small helpers: wall clock, a deterministic fill, byte formatting.
 * -------------------------------------------------------------------- */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Numerical Recipes LCG, the same constants tests/test_metal_ops.c's lcg_next
 * uses, reimplemented locally (this file links against no test file) so the
 * fill data is deterministic across runs without needing true randomness --
 * only "finite, bounded, not degenerate" matters for a timing benchmark. */
static uint32_t g_lcg = 0;
static void lcg_seed(uint32_t seed) { g_lcg = seed; }
static float lcg_next(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)((int32_t)(g_lcg >> 8) - 8388608) / 8388608.0f;   /* [-1, 1) */
}

static uint32_t f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static void human_bytes(uint64_t n, char *buf, size_t bufsz) {
    double v = (double)n;
    const char *unit = "B";
    if (v >= 1024.0 * 1024.0 * 1024.0) { v /= 1024.0 * 1024.0 * 1024.0; unit = "GiB"; }
    else if (v >= 1024.0 * 1024.0) { v /= 1024.0 * 1024.0; unit = "MiB"; }
    else if (v >= 1024.0) { v /= 1024.0; unit = "KiB"; }
    snprintf(buf, bufsz, "%llu bytes (%.2f %s)", (unsigned long long)n, v, unit);
}

/* --------------------------------------------------------------------
 * GPU buffer helper: allocate without exiting on failure (unlike
 * tests/test_metal_ops.c's gb_new, which is FATAL by design for a
 * correctness gate; this is a benchmark, and a failed allocation must
 * produce a SKIP row, per the brief, not abort the whole sweep).
 * -------------------------------------------------------------------- */

typedef struct { void *b; void *h; uint64_t nbytes; } gbuf;

static bool galloc(sg_gpu *g, uint64_t nbytes, gbuf *out, char *errbuf, size_t errbuf_sz) {
    *out = (gbuf){NULL, NULL, nbytes};
    sg_err e = sg_gpu_alloc(g, nbytes, &out->b, &out->h);
    if (sg_failed(e)) {
        snprintf(errbuf, errbuf_sz, "%s", e.msg ? e.msg : "(no message)");
        out->b = NULL;
        return false;
    }
    return true;
}

static void gfree(gbuf *b) {
    if (b->b) sg_gpu_buf_free(b->b);
    b->b = NULL;
    b->h = NULL;
}

/* --------------------------------------------------------------------
 * Shapes, sequence lengths, n_splits sweep: task-P2.3a-brief.md verbatim.
 * -------------------------------------------------------------------- */

typedef struct { const char *name; uint32_t n_heads, n_kv, head_dim; } shape_t;

static const shape_t SHAPES[] = {
    { "27b_decode_24h_4kv_256d", 24, 4, 256 },
    { "4b_dense_32h_8kv_128d",   32, 8, 128 },
};
#define N_SHAPES ((int)(sizeof SHAPES / sizeof SHAPES[0]))

static const uint32_t SEQS[] = { 8192u, 32768u, 131072u, 262144u };
#define N_SEQS ((int)(sizeof SEQS / sizeof SEQS[0]))

/* Cap on a --seqs list (P2.3). A fixed stack array rather than a malloc: the
 * list is typed by hand on a command line, so a couple of dozen entries is
 * already far more than anyone will pass, and a bounded buffer cannot leak. */
#define MAX_SEQS 32

static const uint32_t SPLITS[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
#define N_SPLITS ((int)(sizeof SPLITS / sizeof SPLITS[0]))

/* The largest n_splits this file ever tests, at seq 262144 (262144 / 256 =
 * 1024): the shared m/s/acc scratch is sized for this once per shape, see
 * run_shape. */
#define MAX_N_SPLITS 1024u

static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* See the file header's ACHIEVED GB/s paragraph for the full derivation:
 * K + V, f16 (2 bytes/elem), read once per query-head threadgroup, the SAME
 * total for the incumbent and split-K at a given (shape, seq). */
static double bytes_kv(const shape_t *sh, uint32_t seq) {
    return 2.0 /* K and V */ * (double)sh->n_heads * (double)seq
         * (double)sh->head_dim * 2.0 /* bytes per f16 element */;
}

/* Decimal GB/s (1e9 bytes/s), matching how Apple states this machine's memory
 * bandwidth, so the achieved number is directly comparable to that spec. */
static double gbps(double bytes, double seconds) {
    if (seconds <= 0.0) return 0.0;
    return bytes / seconds / 1.0e9;
}

/* --------------------------------------------------------------------
 * Timed dispatch: one discarded warm-up, then `reps` timed calls.
 * -------------------------------------------------------------------- */

typedef struct {
    bool ok;
    double mean_s, min_s, max_s;
    char err[256];
} timing_t;

static timing_t time_incumbent(sg_gpu *g, gbuf *q, gbuf *k, gbuf *v, gbuf *out,
                               const uint32_t params[8], int reps) {
    timing_t r = {false, 0.0, 0.0, 0.0, {0}};

    sg_err e = sg_gpu_run_attn_decode_f16(g, q->b, k->b, v->b, out->b, params);
    if (sg_failed(e)) {
        snprintf(r.err, sizeof r.err, "%s", e.msg ? e.msg : "(no message)");
        return r;
    }

    double sum = 0.0, mn = 0.0, mx = 0.0;
    for (int i = 0; i < reps; i++) {
        double t0 = now_s();
        e = sg_gpu_run_attn_decode_f16(g, q->b, k->b, v->b, out->b, params);
        double t1 = now_s();
        if (sg_failed(e)) {
            snprintf(r.err, sizeof r.err, "%s", e.msg ? e.msg : "(no message)");
            return r;
        }
        double d = t1 - t0;
        sum += d;
        if (i == 0 || d < mn) mn = d;
        if (i == 0 || d > mx) mx = d;
    }
    r.ok = true;
    r.mean_s = sum / (double)reps;
    r.min_s = mn;
    r.max_s = mx;
    return r;
}

/* Task P2.4's --gqa flag: time k_attn_decode_splitk_partial_gqa (one
 * threadgroup per GQA GROUP, each K/V element read once for all the query heads
 * that share it) instead of the per-head k_attn_decode_splitk_partial. The two
 * produce the same bytes, so this only ever changes the timing; run the binary
 * twice, once with and once without, for the A/B.
 *
 * WHAT IT DOES TO THE GB/s COLUMN, which the file header's accounting has to
 * be read with: bytes_kv() counts n_heads * seq * head_dim * 2 dtypes * 2
 * bytes, i.e. bytes ISSUED by the per-head grid. That is the right count for
 * the incumbent and for the per-head partial, both of which really do issue
 * one read per query head. The GQA partial issues n_kv_heads/n_heads of it
 * (1/repeat), so its GB/s column is NOT achieved DRAM bandwidth; it stays on
 * the same yardstick as the other two on purpose, so the columns remain
 * comparable, and the traffic saving shows up as a shorter TIME rather than as
 * a bigger bandwidth number. */
static bool g_gqa = false;

/* --online (task P2.8): time k_attn_decode_splitk_partial_gqa_online, the
 * streaming form of the GQA partial. It has the same grid as --gqa (so the same
 * GB/s caveat above applies word for word) and does the same K/V reads, but it
 * writes NO score row: the four-pass kernel's two writes and three reads of
 * n_heads * len floats per split are replaced by a running (m, s, acc). The A/B
 * that matters for P2.8 is therefore --gqa against --gqa --online at the SAME
 * n_splits, which is what this harness makes possible on one binary.
 *
 * It implies the GQA grid rather than composing with the per-head one: there is
 * no per-head online kernel, so --online alone selects the online GQA kernel and
 * --gqa --online is the same request. UNLIKE the --gqa A/B, this one is NOT
 * byte-identical to the arm it is compared against (streaming reorders the
 * exponential sums), so a correctness claim here belongs to
 * tests/test_metal_ops.c's oracle comparison, not to this file. */
static bool g_online = false;

/* Times the chosen split-K partial THEN k_attn_decode_splitk_combine as one
 * unit (see the file header: that pair together is what reaches the same
 * "final attention output" the incumbent produces in a single dispatch). */
static sg_err run_partial(sg_gpu *g, gbuf *q, gbuf *k, gbuf *v,
                          gbuf *m, gbuf *s, gbuf *acc, const uint32_t params[8]) {
    if (g_online) {
        return sg_gpu_run_attn_splitk_partial_gqa_online(g, q->b, k->b, v->b,
                                                         m->b, s->b, acc->b, params);
    }
    if (g_gqa) {
        return sg_gpu_run_attn_splitk_partial_gqa(g, q->b, k->b, v->b,
                                                  m->b, s->b, acc->b, params);
    }
    return sg_gpu_run_attn_splitk_partial(g, q->b, k->b, v->b, m->b, s->b, acc->b, params);
}

static timing_t time_splitk(sg_gpu *g, gbuf *q, gbuf *k, gbuf *v,
                            gbuf *m, gbuf *s, gbuf *acc, gbuf *out,
                            const uint32_t params[8], int reps) {
    timing_t r = {false, 0.0, 0.0, 0.0, {0}};

    sg_err e = run_partial(g, q, k, v, m, s, acc, params);
    if (!sg_failed(e)) {
        e = sg_gpu_run_attn_splitk_combine(g, m->b, s->b, acc->b, out->b, params);
    }
    if (sg_failed(e)) {
        snprintf(r.err, sizeof r.err, "%s", e.msg ? e.msg : "(no message)");
        return r;
    }

    double sum = 0.0, mn = 0.0, mx = 0.0;
    for (int i = 0; i < reps; i++) {
        double t0 = now_s();
        e = run_partial(g, q, k, v, m, s, acc, params);
        if (!sg_failed(e)) {
            e = sg_gpu_run_attn_splitk_combine(g, m->b, s->b, acc->b, out->b, params);
        }
        double t1 = now_s();
        if (sg_failed(e)) {
            snprintf(r.err, sizeof r.err, "%s", e.msg ? e.msg : "(no message)");
            return r;
        }
        double d = t1 - t0;
        sum += d;
        if (i == 0 || d < mn) mn = d;
        if (i == 0 || d > mx) mx = d;
    }
    r.ok = true;
    r.mean_s = sum / (double)reps;
    r.min_s = mn;
    r.max_s = mx;
    return r;
}

/* --------------------------------------------------------------------
 * Printing. Every numeric field is pre-formatted into a string and printed
 * with %s, uniformly, so a row missing a value (an incumbent that failed, an
 * allocation that failed) prints "n/a"/"FAIL" in that slot without a
 * printf format/argument-type mismatch anywhere.
 * -------------------------------------------------------------------- */

static void print_header(void) {
    printf("%-26s %10s %9s %15s %15s %10s %13s %13s\n",
           "shape", "seq", "n_splits", "incumbent_us", "splitk_us",
           "speedup", "inc_GBps", "splitk_GBps");
    fflush(stdout);
}

static void print_data_row(const char *shape, uint32_t seq, uint32_t ns,
                           const char *inc_us, const char *split_us,
                           const char *speedup, const char *inc_gbps,
                           const char *split_gbps) {
    printf("%-26s %10u %9u %15s %15s %10s %13s %13s\n",
           shape, seq, ns, inc_us, split_us, speedup, inc_gbps, split_gbps);
    fflush(stdout);
}

static void print_skip_seq(const char *shape, uint32_t seq, const char *reason) {
    printf("%-26s seq=%-10u SKIP  %s\n", shape, seq, reason);
    fflush(stdout);
}

static void print_skip_ns(const char *shape, uint32_t seq, uint32_t ns, const char *reason) {
    printf("%-26s seq=%-10u n_splits=%-6u SKIP  %s\n", shape, seq, ns, reason);
    fflush(stdout);
}

/* --------------------------------------------------------------------
 * One shape's whole sweep: alloc q/out/m/s/acc once (reused across every
 * seq and n_splits in this shape), then per seq alloc K/V, time the
 * incumbent once, and sweep the clamped, deduped n_splits list.
 * -------------------------------------------------------------------- */

static void run_shape(sg_gpu *g, const shape_t *sh, int reps,
                      const uint32_t *seqs, int n_seqs) {
    char err[256];
    uint32_t q_stride = sh->head_dim;   /* dense; see file header */

    uint64_t q_bytes = (uint64_t)sh->n_heads * q_stride * sizeof(float);
    uint64_t out_bytes = (uint64_t)sh->n_heads * sh->head_dim * sizeof(float);
    uint64_t ms_bytes = (uint64_t)sh->n_heads * MAX_N_SPLITS * sizeof(float);
    uint64_t acc_bytes = (uint64_t)sh->n_heads * MAX_N_SPLITS * sh->head_dim * sizeof(float);

    gbuf q = {NULL, NULL, 0}, out_inc = {NULL, NULL, 0}, out_split = {NULL, NULL, 0};
    gbuf m = {NULL, NULL, 0}, s = {NULL, NULL, 0}, acc = {NULL, NULL, 0};

    bool setup_ok = galloc(g, q_bytes, &q, err, sizeof err)
                 && galloc(g, out_bytes, &out_inc, err, sizeof err)
                 && galloc(g, out_bytes, &out_split, err, sizeof err)
                 && galloc(g, ms_bytes, &m, err, sizeof err)
                 && galloc(g, ms_bytes, &s, err, sizeof err)
                 && galloc(g, acc_bytes, &acc, err, sizeof err);
    if (!setup_ok) {
        printf("%-26s SKIP (whole shape): cannot allocate q/out/m/s/acc scratch: %s\n",
               sh->name, err);
        fflush(stdout);
        gfree(&q); gfree(&out_inc); gfree(&out_split);
        gfree(&m); gfree(&s); gfree(&acc);
        return;
    }

    lcg_seed(0xB1D00000u ^ ((uint32_t)sh->n_heads << 16) ^ sh->head_dim);
    float *qh = (float *)q.h;
    for (uint64_t i = 0; i < (uint64_t)sh->n_heads * q_stride; i++) qh[i] = lcg_next();

    printf("\n=== shape %s: n_heads=%u n_kv_heads=%u head_dim=%u q_stride=%u ===\n",
           sh->name, sh->n_heads, sh->n_kv, sh->head_dim, q_stride);
    print_header();

    float scale = (float)(1.0 / sqrt((double)sh->head_dim));

    for (int si = 0; si < n_seqs; si++) {
        uint32_t seq = seqs[si];
        uint64_t kv_elems = (uint64_t)seq * sh->n_kv * sh->head_dim;
        uint64_t kv_bytes_each = kv_elems * sizeof(uint16_t);

        gbuf k = {NULL, NULL, 0}, v = {NULL, NULL, 0};
        bool kv_ok = galloc(g, kv_bytes_each, &k, err, sizeof err);
        if (kv_ok) kv_ok = galloc(g, kv_bytes_each, &v, err, sizeof err);
        if (!kv_ok) {
            char hb[64], reason[320];
            human_bytes(kv_bytes_each, hb, sizeof hb);
            snprintf(reason, sizeof reason, "K/V cache needs 2x %s: %s", hb, err);
            print_skip_seq(sh->name, seq, reason);
            gfree(&k); gfree(&v);
            continue;
        }

        lcg_seed(0xC0DE0000u ^ (uint32_t)si ^ ((uint32_t)sh->head_dim << 4));
        uint16_t *kh = (uint16_t *)k.h;
        uint16_t *vh = (uint16_t *)v.h;
        for (uint64_t i = 0; i < kv_elems; i++) kh[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        for (uint64_t i = 0; i < kv_elems; i++) vh[i] = sg_f32_to_f16(lcg_next() * 2.0f);

        uint32_t base_p[8] = { sh->n_heads, sh->n_kv, sh->head_dim, seq, q_stride,
                                f32_bits(scale), 0, 0 };

        timing_t inc = time_incumbent(g, &q, &k, &v, &out_inc, base_p, reps);
        double kvb = bytes_kv(sh, seq);

        char inc_us_s[24], inc_gbps_s[24];
        if (inc.ok) {
            snprintf(inc_us_s, sizeof inc_us_s, "%.2f", inc.mean_s * 1.0e6);
            snprintf(inc_gbps_s, sizeof inc_gbps_s, "%.1f", gbps(kvb, inc.mean_s));
        } else {
            snprintf(inc_us_s, sizeof inc_us_s, "FAIL");
            snprintf(inc_gbps_s, sizeof inc_gbps_s, "n/a");
            printf("%-26s seq=%-10u INCUMBENT FAILED: %s\n", sh->name, seq, inc.err);
            fflush(stdout);
        }

        uint32_t band_lo = BAND_LO;
        uint32_t band_hi = seq / SPLIT_TG_KEYS;
        if (band_hi < band_lo) band_hi = band_lo;

        char kvb_h[64];
        human_bytes(kv_bytes_each, kvb_h, sizeof kvb_h);
        printf("  -- seq %u: K (and V) each %s; occupancy band n_splits in [%u, %u] "
               "(n_heads*n_splits >= 80 cores, n_splits <= seq/%u) --\n",
               seq, kvb_h, band_lo, band_hi, SPLIT_TG_KEYS);
        fflush(stdout);

        double best_speedup = 0.0;
        uint32_t best_ns = 0;
        bool have_best = false;

        uint32_t last_ns = 0;
        bool have_last = false;
        for (int ki = 0; ki < N_SPLITS; ki++) {
            uint32_t ns = clamp_u32(SPLITS[ki], band_lo, band_hi);
            if (have_last && ns == last_ns) continue;
            have_last = true;
            last_ns = ns;

            uint32_t p[8];
            memcpy(p, base_p, sizeof p);
            p[6] = ns;

            timing_t sp = time_splitk(g, &q, &k, &v, &m, &s, &acc, &out_split, p, reps);

            char split_us_s[24], split_gbps_s[24], speedup_s[24];
            if (sp.ok) {
                snprintf(split_us_s, sizeof split_us_s, "%.2f", sp.mean_s * 1.0e6);
                snprintf(split_gbps_s, sizeof split_gbps_s, "%.1f", gbps(kvb, sp.mean_s));
                if (inc.ok) {
                    double speedup = inc.mean_s / sp.mean_s;
                    snprintf(speedup_s, sizeof speedup_s, "%.3fx", speedup);
                    if (!have_best || speedup > best_speedup) {
                        best_speedup = speedup;
                        best_ns = ns;
                        have_best = true;
                    }
                } else {
                    snprintf(speedup_s, sizeof speedup_s, "n/a");
                }
            } else {
                snprintf(split_us_s, sizeof split_us_s, "FAIL");
                snprintf(split_gbps_s, sizeof split_gbps_s, "n/a");
                snprintf(speedup_s, sizeof speedup_s, "n/a");
            }

            print_data_row(sh->name, seq, ns, inc_us_s, split_us_s, speedup_s,
                          inc_gbps_s, split_gbps_s);
            if (!sp.ok) {
                print_skip_ns(sh->name, seq, ns, sp.err);
            }
        }

        if (have_best) {
            printf("  -- seq %u best: n_splits=%u speedup=%.3fx --\n", seq, best_ns, best_speedup);
            fflush(stdout);
        }

        gfree(&k);
        gfree(&v);
    }

    gfree(&q); gfree(&out_inc); gfree(&out_split);
    gfree(&m); gfree(&s); gfree(&acc);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--reps N] [--seqs A,B,C] [--gqa] [-h|--help]\n"
        "  --reps N     timed repetitions per (shape, n_splits), after 1 discarded\n"
        "               warm-up call (default %d)\n"
        "  --seqs LIST  comma-separated sequence lengths to sweep instead of the\n"
        "               default %d..%d (at most %d of them, each >= 1). Added by\n"
        "               task P2.3 to measure the SHORT-sequence end of the curve,\n"
        "               which the default list (>= 8192) does not reach and which\n"
        "               decode's fallback threshold depends on.\n"
        "  --gqa        time k_attn_decode_splitk_partial_gqa (task P2.4, one\n"
        "               threadgroup per GQA GROUP) instead of the per-head\n"
        "               k_attn_decode_splitk_partial. Same output bytes, so the\n"
        "               A/B is this binary run twice, with and without the flag.\n"
        "  --online     time k_attn_decode_splitk_partial_gqa_online (task P2.8,\n"
        "               the streaming form of the GQA partial: no score row in\n"
        "               device memory). Implies the GQA grid. NOT byte-identical\n"
        "               to --gqa, so compare TIMES here and correctness in\n"
        "               tests/test_metal_ops.c.\n",
        prog, DEFAULT_REPS, (int)SEQS[0], (int)SEQS[N_SEQS - 1], MAX_SEQS);
}

int main(int argc, char **argv) {
    int reps = DEFAULT_REPS;
    const uint32_t *seqs = SEQS;
    int n_seqs = N_SEQS;
    uint32_t seq_buf[MAX_SEQS];

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            long v = strtol(argv[++i], NULL, 10);
            if (v < 1) {
                fprintf(stderr, "bad --reps value: %s\n", argv[i]);
                return 2;
            }
            reps = (int)v;
        } else if (strcmp(argv[i], "--seqs") == 0 && i + 1 < argc) {
            const char *list = argv[++i];
            int n = 0;
            const char *p = list;
            while (*p) {
                char *end = NULL;
                long v = strtol(p, &end, 10);
                if (end == p || v < 1 || v > (long)UINT32_MAX) {
                    fprintf(stderr, "bad --seqs value: %s\n", list);
                    return 2;
                }
                if (n == MAX_SEQS) {
                    fprintf(stderr, "--seqs takes at most %d values\n", MAX_SEQS);
                    return 2;
                }
                seq_buf[n++] = (uint32_t)v;
                p = end;
                if (*p == ',') p++;
                else if (*p != '\0') {
                    fprintf(stderr, "bad --seqs value: %s\n", list);
                    return 2;
                }
            }
            if (n == 0) {
                fprintf(stderr, "--seqs needs at least one value\n");
                return 2;
            }
            seqs = seq_buf;
            n_seqs = n;
        } else if (strcmp(argv[i], "--gqa") == 0) {
            g_gqa = true;
        } else if (strcmp(argv[i], "--online") == 0) {
            /* Implies --gqa: the online kernel IS a GQA-shared kernel, and the
             * GB/s caveat and grid note both belong to it too. */
            g_online = true;
            g_gqa = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    sg_gpu *g = NULL;
    sg_err e = sg_gpu_init(&g);
    if (sg_failed(e)) {
        fprintf(stderr, "SKIP bench_splitk: %s\n", e.msg);
        return 0;
    }

    printf("P2.3a split-K decode-attention timing harness\n");
    printf("reps=%d (+1 discarded warm-up); incumbent=k_attn_decode_f16, "
           "split-K=%s+combine dispatched together (the pair needed for "
           "one decode step's answer)\n", reps,
           g_online ? "k_attn_decode_splitk_partial_gqa_online"
                    : (g_gqa ? "k_attn_decode_splitk_partial_gqa"
                             : "k_attn_decode_splitk_partial"));
    if (g_gqa) {
        printf("P2.4 --gqa: the GQA partial issues 1/repeat of the KV reads the "
               "per-head one does, so its splitk_GBps column is on the per-head "
               "yardstick (see the file header), not achieved DRAM bandwidth\n");
    }
    if (g_online) {
        printf("P2.8 --online: streaming softmax, no score row in device memory "
               "(splitk_scratch is neither grown nor bound). Same K/V reads as "
               "--gqa, so compare against a --gqa run at the same --reps and "
               "--seqs; the numbers are NOT bit-comparable, only time-comparable\n");
    }
    printf("speedup = incumbent_mean_time / splitk_mean_time (> 1.0 means split-K is faster)\n");
    printf("GB/s is decimal (1e9 bytes/s) achieved KV-read bandwidth, the same "
           "yardstick for both kernels; see the file header for the exact accounting\n");
    fflush(stdout);

    for (int i = 0; i < N_SHAPES; i++) {
        run_shape(g, &SHAPES[i], reps, seqs, n_seqs);
    }

    sg_gpu_free(g);
    return 0;
}

#endif /* SURGE_NO_METAL */
