/* test_metal_ops.c - the Task 9 Metal kernels against the Task 7 CPU
 * reference ops, op by op, on the SAME fixture inputs.
 *
 * The oracle here is deliberately NOT the fixture's stored outputs: those
 * belong to tests/test_ref_ops.c, which is what pins src/ref.c to mlx. This
 * file asks a narrower and more useful question -- does the GPU compute what
 * ref.c computes? -- so a divergence lands on exactly one side and Task 10's
 * Metal-vs-ref decode gate has a per-op explanation for any drift.
 *
 * Two bars, both from the plan:
 *
 *   PARITY. Every op within 1e-4 of the ref op, measured RELATIVE to the
 *   scale of the reference vector (max |err| / max |want|, so a near-zero
 *   element does not manufacture a huge relative error out of an absolute
 *   one that is fine). The gap is not zero and cannot be: ref.c accumulates
 *   in double and Metal has no f64, so a length-n dot product differs by
 *   about n * 2^-24 relative. Measured worst case here is ~1e-7.
 *
 *   DETERMINISM. Every reduction kernel (matvec, softmax, attn_decode,
 *   delta_step) run 100 times on identical input produces BYTE-IDENTICAL
 *   output. This is the property Task 10's byte-exact gate rests on, and it
 *   is why kernels.metal folds its reductions with a fixed-shape tree
 *   instead of atomics or simd_sum.
 *
 * Portability: if sg_gpu_init fails (no Metal device), this prints a skip
 * notice and exits 0, so `make check` stays green on a machine without a
 * GPU. It is also compiled to a bare skip under -DSURGE_NO_METAL, which is
 * how `make debug` keeps Metal out of the ASan run.
 */
#ifdef SURGE_NO_METAL

#include <stdio.h>
int main(void) {
    fprintf(stderr, "SKIP test_metal_ops: built with -DSURGE_NO_METAL "
                    "(Metal and the ASan/UBSan run do not mix)\n");
    return 0;
}

#else

#include "tinytest.h"
#include "fixture.h"
#include "../surge.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static fixture g_ops, g_hyb;
static sg_gpu *g_gpu;

/* The plan's per-op bar. Relative to the reference vector's scale. */
#define TOL_REL 1e-4

/* Worst error seen across every op, printed at the end so the task report
 * can quote one number without grepping. */
static double g_worst_rel = 0.0;
static const char *g_worst_label = "(none)";

/* --------------------------------------------------------------------
 * GPU buffer helpers
 * -------------------------------------------------------------------- */

typedef struct { void *b; float *h; uint64_t n; } gbuf;

static gbuf gb_new(uint64_t nfloats) {
    gbuf r = {NULL, NULL, nfloats};
    void *host = NULL;
    sg_err e = sg_gpu_alloc(g_gpu, nfloats * sizeof(float), &r.b, &host);
    if (sg_failed(e)) { fprintf(stderr, "FATAL: %s\n", e.msg); exit(2); }
    r.h = (float *)host;
    return r;
}

static gbuf gb_from(const float *src, uint64_t nfloats) {
    gbuf r = gb_new(nfloats);
    memcpy(r.h, src, nfloats * sizeof(float));
    return r;
}

/* bf16 weights, `n` counted in uint16_t elements. gbuf.n is a FLOAT count
 * (gb_poison is the only reader), so store the float-equivalent size rather
 * than the element count, or poisoning such a buffer would run off the end. */
static gbuf gb_from_u16(const uint16_t *src, uint64_t n) {
    gbuf r = {NULL, NULL, (n + 1) / 2};
    void *host = NULL;
    sg_err e = sg_gpu_alloc(g_gpu, n * sizeof(uint16_t), &r.b, &host);
    if (sg_failed(e)) { fprintf(stderr, "FATAL: %s\n", e.msg); exit(2); }
    r.h = (float *)host;
    memcpy(host, src, n * sizeof(uint16_t));
    return r;
}

static void gb_free(gbuf *b) { sg_gpu_buf_free(b->b); b->b = NULL; b->h = NULL; }

/* Fill an output buffer with a pattern no kernel would produce, so a
 * kernel that writes only part of its output fails loudly instead of
 * passing on whatever the previous dispatch left there. */
static void gb_poison(gbuf *b) { memset(b->h, 0xA5, (size_t)b->n * sizeof(float)); }

static bool gpu_run(const char *kernel, gbuf *a, gbuf *b, gbuf *out, const uint32_t p[8]) {
    sg_err e = sg_gpu_run_op(g_gpu, kernel, a ? a->b : NULL, b ? b->b : NULL,
                             out ? out->b : NULL, p);
    if (sg_failed(e)) { tt_assert(0, "%s", e.msg); return false; }
    return true;
}

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }

/* --------------------------------------------------------------------
 * Comparison
 * -------------------------------------------------------------------- */

/* max |got - want| divided by the reference vector's own scale. Absolute
 * error alone is meaningless across ops whose outputs span 1e-3 to 1e3, and
 * per-element relative error is meaningless wherever `want` crosses zero;
 * this is the measure the plan's "1e-4 relative" is checked in. */
static double check_rel(const char *label, const float *got, const float *want, uint64_t n) {
    if (n == 0) { tt_assert(0, "%s: nothing to compare", label); return 0.0; }

    double scale = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double m = fabs((double)want[i]);
        if (m > scale) scale = m;
    }
    if (scale == 0.0) scale = 1.0;

    double worst = 0.0;
    uint64_t at = 0;
    for (uint64_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        if (isnan(d)) { at = i; worst = INFINITY; break; }
        if (d > worst) { worst = d; at = i; }
    }
    double rel = worst / scale;

    tt_assert(rel < TOL_REL,
              "%s: max |err| %.3e over scale %.3e = %.3e relative, tol %.0e "
              "(worst at %llu: got %.9g, want %.9g)",
              label, worst, scale, rel, TOL_REL,
              (unsigned long long)at, (double)got[at], (double)want[at]);
    if (rel < TOL_REL) {
        fprintf(stderr, "   %-46s rel %.3e  abs %.3e\n", label, rel, worst);
        if (rel > g_worst_rel) { g_worst_rel = rel; g_worst_label = label; }
    }
    return rel;
}

/* --------------------------------------------------------------------
 * Deterministic synthetic inputs, for the sizes the committed fixtures are
 * too small to reach (the fixtures top out at 64 columns; the reduction
 * tree only starts striding above 256).
 * -------------------------------------------------------------------- */

static uint32_t g_lcg = 0;
static void lcg_seed(uint32_t s) { g_lcg = s; }
static float lcg_next(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return (float)((int32_t)(g_lcg >> 8) - 8388608) / 8388608.0f;   /* [-1, 1) */
}

/* f32 -> bf16, round to nearest even, the same rounding numpy's bfloat16
 * cast uses. Inputs here are finite by construction. */
static uint16_t f32_to_bf16(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    uint32_t lsb = (b >> 16) & 1u;
    b += 0x7FFFu + lsb;
    return (uint16_t)(b >> 16);
}

/* --------------------------------------------------------------------
 * RoPE table: the ONE piece of the rotation that stays on the CPU.
 *
 * sg_ref_rope_partial computes the angle and its sine/cosine in double
 * because the f32-rounded angle is already 8e-3 wrong at position 262143
 * (see the long comment on that function). Metal has no double, so the only
 * way for the kernel to agree with ref at long context is to be handed the
 * cosines and sines instead of the angle. Computed here EXACTLY as ref
 * computes them, then rounded once to f32.
 * -------------------------------------------------------------------- */
static void rope_table(float *cs, uint32_t rope_dim, uint32_t pos, float theta) {
    uint32_t half = rope_dim / 2;
    for (uint32_t i = 0; i < half; i++) {
        double inv_freq = pow((double)theta, -2.0 * (double)i / (double)rope_dim);
        double ang = (double)pos * inv_freq;
        cs[i] = (float)cos(ang);
        cs[half + i] = (float)sin(ang);
    }
}

/* ====================================================================
 * Per-op parity
 * ==================================================================== */

static void metal_rmsnorm_matches_ref(void) {
    /* 1. ops.bin, with a weight. */
    uint64_t n = 0;
    const float *x = fx_f32(&g_ops, "rmsnorm.x", 0, &n);
    const float *w = fx_f32(&g_ops, "rmsnorm.w", n, NULL);
    float eps = fx_scalar(&g_ops, "rmsnorm.eps");

    float *want = xmalloc(n * sizeof *want);
    memcpy(want, x, n * sizeof *want);
    sg_ref_rmsnorm(want, w, (uint32_t)n, eps);

    gbuf a = gb_from(x, n), b = gb_from(w, n), o = gb_new(n);
    uint32_t p[8] = {(uint32_t)n, f32_bits(eps), 1, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_rmsnorm", &a, &b, &o, p)) check_rel("rmsnorm (ops.bin, weighted)", o.h, want, n);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(want);

    /* 2. hybrid_ops.bin, NO weight: the DeltaNet q/k path calls
     *    mx.fast.rms_norm(x, None, eps), which sg_ref_rmsnorm models with a
     *    NULL w and the kernel with params[2] == 0. */
    const float *x2 = fx_f32(&g_hyb, "rmsnorm_now.x", 0, &n);
    float *want2 = xmalloc(n * sizeof *want2);
    memcpy(want2, x2, n * sizeof *want2);
    sg_ref_rmsnorm(want2, NULL, (uint32_t)n, fx_scalar(&g_hyb, "rmsnorm_w.eps"));

    gbuf a2 = gb_from(x2, n), o2 = gb_new(n);
    uint32_t p2[8] = {(uint32_t)n, f32_bits(fx_scalar(&g_hyb, "rmsnorm_w.eps")), 0, 0, 0, 0, 0, 0};
    gb_poison(&o2);
    if (gpu_run("k_rmsnorm", &a2, NULL, &o2, p2)) check_rel("rmsnorm (no weight)", o2.h, want2, n);
    gb_free(&a2); gb_free(&o2);
    free(want2);

    /* 3. n = 4000 > threadgroup width, and deliberately NOT a multiple of it,
     *    so every thread folds 15 or 16 strided elements before the tree sees
     *    them and the ragged tail is exercised. The fixtures stop at 64. */
    const uint32_t big = 4000;
    float *xb = xmalloc(big * sizeof *xb), *wb = xmalloc(big * sizeof *wb);
    lcg_seed(20260809u);
    for (uint32_t i = 0; i < big; i++) { xb[i] = lcg_next() * 3.0f; wb[i] = lcg_next(); }
    float *wantb = xmalloc(big * sizeof *wantb);
    memcpy(wantb, xb, big * sizeof *wantb);
    sg_ref_rmsnorm(wantb, wb, big, 1e-6f);

    gbuf a3 = gb_from(xb, big), b3 = gb_from(wb, big), o3 = gb_new(big);
    uint32_t p3[8] = {big, f32_bits(1e-6f), 1, 0, 0, 0, 0, 0};
    gb_poison(&o3);
    if (gpu_run("k_rmsnorm", &a3, &b3, &o3, p3)) check_rel("rmsnorm (n=4000, strided)", o3.h, wantb, big);
    gb_free(&a3); gb_free(&b3); gb_free(&o3);
    free(xb); free(wb); free(wantb);
}

/* One RoPE case: rotate x on both sides at `pos` and compare. */
static void rope_case(const char *label, const float *x, uint32_t head_dim,
                      uint32_t rope_dim, uint32_t pos, float theta) {
    float *want = xmalloc(head_dim * sizeof *want);
    memcpy(want, x, head_dim * sizeof *want);
    sg_ref_rope_partial(want, head_dim, rope_dim, pos, theta);

    float *cs = xmalloc(rope_dim * sizeof *cs);
    rope_table(cs, rope_dim, pos, theta);

    gbuf a = gb_from(x, head_dim), b = gb_from(cs, rope_dim), o = gb_new(head_dim);
    uint32_t p[8] = {head_dim, rope_dim, 0, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_rope", &a, &b, &o, p)) check_rel(label, o.h, want, head_dim);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(want); free(cs);
}

static void metal_rope_matches_ref(void) {
    /* Full-width RoPE from ops.bin. */
    uint64_t n = 0;
    const float *x = fx_f32(&g_ops, "rope.x", 0, &n);
    uint32_t hd = fx_dim(&g_ops, "rope.head_dim");
    float theta = fx_scalar(&g_ops, "rope.theta");
    tt_assert(hd == n, "rope.x should be one head of %u (got %llu)", hd, (unsigned long long)n);
    rope_case("rope full, pos 0", x, hd, hd, 0, theta);
    rope_case("rope full, pos 1", x, hd, hd, 1, theta);
    rope_case("rope full, pos 4096", x, hd, hd, 4096, theta);

    /* Partial RoPE, the shape the real checkpoints use (rope_dim < head_dim,
     * so the kernel's untouched tail is exercised). */
    const float *xp = fx_f32(&g_hyb, "rope_partial.x", 0, &n);
    uint32_t hd2 = fx_dim(&g_hyb, "rope_partial.head_dim");
    uint32_t rd2 = fx_dim(&g_hyb, "rope_partial.rope_dim");
    float th2 = fx_scalar(&g_hyb, "rope_partial.theta");
    tt_assert(rd2 < hd2, "the partial-RoPE fixture must be partial (%u < %u)", rd2, hd2);
    const float *pos4 = fx_f32(&g_hyb, "rope_partial.pos", 4, NULL);
    for (uint32_t i = 0; i < 4; i++) {
        char lbl[64];
        snprintf(lbl, sizeof lbl, "rope partial, pos %u", (uint32_t)pos4[i]);
        rope_case(lbl, xp, hd2, rd2, (uint32_t)pos4[i], th2);
    }

    /* The real checkpoint's parameters (head_dim 256, rope_dim 64,
     * theta 1e7) out to position 262143. This is the case the CPU-side
     * double-precision cos/sin table exists for: computing the angle in f32
     * on the GPU would be 8e-3 off here, five orders of magnitude past the
     * parity bar. */
    const float *xr = fx_f32(&g_hyb, "rope_real.x", 0, &n);
    uint32_t hd3 = fx_dim(&g_hyb, "rope_real.head_dim");
    uint32_t rd3 = fx_dim(&g_hyb, "rope_real.rope_dim");
    float th3 = fx_scalar(&g_hyb, "rope_real.theta");
    const float *pos5 = fx_f32(&g_hyb, "rope_real.pos", 5, NULL);
    for (uint32_t i = 0; i < 5; i++) {
        char lbl[64];
        snprintf(lbl, sizeof lbl, "rope real, pos %u", (uint32_t)pos5[i]);
        rope_case(lbl, xr, hd3, rd3, (uint32_t)pos5[i], th3);
    }
}

static void metal_matvec_matches_ref(void) {
    /* bf16 from ops.bin: 8 x 64, the same weights test_ref_ops.c checks
     * against numpy. */
    uint64_t nw = 0;
    const uint16_t *w = fx_u16(&g_ops, "matvec_bf16.w", 0, &nw);
    uint64_t nx = 0;
    const float *x = fx_f32(&g_ops, "matvec_bf16.x", 0, &nx);
    uint32_t cols = (uint32_t)nx, rows = (uint32_t)(nw / nx);
    tt_assert((uint64_t)rows * cols == nw, "matvec_bf16.w is not rows x cols");

    float *want = xmalloc(rows * sizeof *want);
    sg_ref_matvec_bf16(w, x, want, rows, cols);

    gbuf a = gb_from_u16(w, nw), b = gb_from(x, cols), o = gb_new(rows);
    uint32_t p[8] = {rows, cols, 0, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_matvec_bf16", &a, &b, &o, p)) check_rel("matvec_bf16 (8x64)", o.h, want, rows);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(want);

    /* 96 x 1000: many output rows, and a column count that is neither a
     * multiple of the threadgroup width nor a power of two, which is the
     * only configuration that exercises the strided partial, the ragged
     * tail, and the full tree fold at once. */
    const uint32_t br = 96, bc = 1000;
    uint16_t *bw = xmalloc((size_t)br * bc * sizeof *bw);
    float *bx = xmalloc(bc * sizeof *bx);
    lcg_seed(770077u);
    for (uint32_t i = 0; i < br * bc; i++) bw[i] = f32_to_bf16(lcg_next() * 0.5f);
    for (uint32_t i = 0; i < bc; i++) bx[i] = lcg_next() * 2.0f;

    float *wantb = xmalloc(br * sizeof *wantb);
    sg_ref_matvec_bf16(bw, bx, wantb, br, bc);

    gbuf a2 = gb_from_u16(bw, (uint64_t)br * bc), b2 = gb_from(bx, bc), o2 = gb_new(br);
    uint32_t p2[8] = {br, bc, 0, 0, 0, 0, 0, 0};
    gb_poison(&o2);
    if (gpu_run("k_matvec_bf16", &a2, &b2, &o2, p2)) check_rel("matvec_bf16 (96x1000)", o2.h, wantb, br);
    gb_free(&a2); gb_free(&b2); gb_free(&o2);
    free(bw); free(bx); free(wantb);

    /* f32 weights, from the attention fixture's real q_proj. */
    uint32_t hidden = fx_dim(&g_hyb, "attn.hidden");
    uint64_t nq = 0;
    const float *qw = fx_f32(&g_hyb, "attn.q_proj_w", 0, &nq);
    uint32_t qrows = (uint32_t)(nq / hidden);
    const float *xd = fx_f32(&g_hyb, "attn.x_decode", hidden, NULL);

    float *wantf = xmalloc(qrows * sizeof *wantf);
    sg_ref_matvec_f32(qw, xd, wantf, qrows, hidden);

    gbuf a3 = gb_from(qw, nq), b3 = gb_from(xd, hidden), o3 = gb_new(qrows);
    uint32_t p3[8] = {qrows, hidden, 0, 0, 0, 0, 0, 0};
    gb_poison(&o3);
    if (gpu_run("k_matvec_f32", &a3, &b3, &o3, p3)) check_rel("matvec_f32 (q_proj 256x48)", o3.h, wantf, qrows);
    gb_free(&a3); gb_free(&b3); gb_free(&o3);
    free(wantf);
}

static void metal_softmax_matches_ref(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_ops, "softmax.x", 0, &n);
    float *want = xmalloc(n * sizeof *want);
    memcpy(want, x, n * sizeof *want);
    sg_ref_softmax(want, (uint32_t)n);

    gbuf a = gb_from(x, n), o = gb_new(n);
    uint32_t p[8] = {(uint32_t)n, 0, 0, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_softmax", &a, NULL, &o, p)) check_rel("softmax (ops.bin, n=32)", o.h, want, n);
    gb_free(&a); gb_free(&o);
    free(want);

    /* n = 3000 with a wide spread, so the max tree, the strided exp and the
     * sum tree all run several rounds deep. */
    const uint32_t big = 3000;
    float *xb = xmalloc(big * sizeof *xb);
    lcg_seed(31337u);
    for (uint32_t i = 0; i < big; i++) xb[i] = lcg_next() * 40.0f;
    float *wantb = xmalloc(big * sizeof *wantb);
    memcpy(wantb, xb, big * sizeof *wantb);
    sg_ref_softmax(wantb, big);

    gbuf a2 = gb_from(xb, big), o2 = gb_new(big);
    uint32_t p2[8] = {big, 0, 0, 0, 0, 0, 0, 0};
    gb_poison(&o2);
    if (gpu_run("k_softmax", &a2, NULL, &o2, p2)) check_rel("softmax (n=3000, wide range)", o2.h, wantb, big);
    gb_free(&a2); gb_free(&o2);
    free(xb); free(wantb);
}

static void metal_elementwise_match_ref(void) {
    /* swiglu, from ops.bin. */
    uint64_t n = 0;
    const float *gate = fx_f32(&g_ops, "swiglu.gate", 0, &n);
    const float *up = fx_f32(&g_ops, "swiglu.up", n, NULL);
    float *want = xmalloc(n * sizeof *want);
    memcpy(want, gate, n * sizeof *want);
    sg_ref_swiglu(want, up, (uint32_t)n);

    gbuf a = gb_from(gate, n), b = gb_from(up, n), o = gb_new(n);
    uint32_t p[8] = {(uint32_t)n, 0, 0, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_swiglu", &a, &b, &o, p)) check_rel("swiglu (ops.bin)", o.h, want, n);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(want);

    /* silu, from the mlx fixture. */
    uint64_t ns = 0;
    const float *sx = fx_f32(&g_hyb, "silu.x", 0, &ns);
    float *wants = xmalloc(ns * sizeof *wants);
    memcpy(wants, sx, ns * sizeof *wants);
    sg_ref_silu(wants, (uint32_t)ns);

    gbuf a2 = gb_from(sx, ns), o2 = gb_new(ns);
    uint32_t p2[8] = {(uint32_t)ns, 0, 0, 0, 0, 0, 0, 0};
    gb_poison(&o2);
    if (gpu_run("k_silu", &a2, NULL, &o2, p2)) check_rel("silu (hybrid fixture)", o2.h, wants, ns);
    gb_free(&a2); gb_free(&o2);
    free(wants);

    /* The attention output gate, x * sigmoid(gate), on the fixture's own
     * silu/sigmoid inputs (same length, both real mlx-generated vectors). */
    uint64_t ng = 0;
    const float *gx = fx_f32(&g_hyb, "sigmoid.x", 0, &ng);
    tt_assert(ng == ns, "silu.x and sigmoid.x should be the same length");
    float *wantg = xmalloc(ns * sizeof *wantg);
    memcpy(wantg, sx, ns * sizeof *wantg);
    sg_ref_gate_sigmoid(wantg, gx, (uint32_t)ns);

    gbuf a3 = gb_from(sx, ns), b3 = gb_from(gx, ns), o3 = gb_new(ns);
    uint32_t p3[8] = {(uint32_t)ns, 0, 0, 0, 0, 0, 0, 0};
    gb_poison(&o3);
    if (gpu_run("k_gate_sigmoid", &a3, &b3, &o3, p3)) check_rel("gate_sigmoid (output gate)", o3.h, wantg, ns);
    gb_free(&a3); gb_free(&b3); gb_free(&o3);
    free(wantg);
}

/* --------------------------------------------------------------------
 * DeltaNet decode step
 * -------------------------------------------------------------------- */

static void metal_conv1d_step_matches_ref(void) {
    uint32_t channels = fx_dim(&g_hyb, "conv1d.channels");
    uint32_t ksize = fx_dim(&g_hyb, "conv1d.ksize");
    uint64_t nx = 0;
    const float *x = fx_f32(&g_hyb, "conv1d.x", 0, &nx);
    const float *w = fx_f32(&g_hyb, "conv1d.w", (uint64_t)channels * ksize, NULL);
    uint32_t steps = (uint32_t)(nx / channels);
    uint32_t keep = ksize - 1;

    /* Both sides start from a zero state and walk the same token sequence,
     * so a state that is carried wrongly diverges on step 2 rather than
     * hiding inside one step's output. */
    float *ref_state = xcalloc(keep * channels, sizeof *ref_state);
    float *ref_out = xmalloc((size_t)steps * channels * sizeof *ref_out);
    for (uint32_t t = 0; t < steps; t++) {
        float *row = ref_out + (size_t)t * channels;
        memcpy(row, x + (size_t)t * channels, channels * sizeof *row);
        sg_ref_conv1d_causal(row, w, ref_state, ref_state, channels, ksize);
    }

    gbuf wb = gb_from(w, (uint64_t)channels * ksize);
    gbuf ob = gb_new((uint64_t)channels * ksize);   /* out[C] then state[(K-1)*C] */
    uint32_t p[8] = {channels, ksize, 0, 0, 0, 0, 0, 0};
    bool ok = true;
    for (uint32_t t = 0; t < steps && ok; t++) {
        gbuf xb = gb_from(x + (size_t)t * channels, channels);
        /* Only the output half is poisoned: the state half is live input. */
        memset(ob.h, 0xA5, channels * sizeof(float));
        ok = gpu_run("k_conv1d_step", &xb, &wb, &ob, p);
        if (ok) {
            char lbl[64];
            snprintf(lbl, sizeof lbl, "conv1d_step token %u", t);
            check_rel(lbl, ob.h, ref_out + (size_t)t * channels, channels);
        }
        gb_free(&xb);
    }
    if (ok) check_rel("conv1d_step carried state", ob.h + channels, ref_state, (uint64_t)keep * channels);
    gb_free(&wb); gb_free(&ob);
    free(ref_state); free(ref_out);

    /* Again at the real conv width: the GatedDeltaNet fixture's own 256 x 4
     * conv1d weight (2*key_dim + value_dim channels), driven by a seeded
     * token stream. The 6-channel case above fits in a fraction of one
     * threadgroup and would not notice a channel-indexing mistake that only
     * bites past the first 256. */
    uint32_t cd = fx_dim(&g_hyb, "gdn.conv_dim");
    uint32_t ck = fx_dim(&g_hyb, "gdn.conv_kernel");
    const float *cw = fx_f32(&g_hyb, "gdn.conv1d_w", (uint64_t)cd * ck, NULL);
    const uint32_t ntok = 4;
    float *toks = xmalloc((size_t)ntok * cd * sizeof *toks);
    lcg_seed(606060u);
    for (uint32_t i = 0; i < ntok * cd; i++) toks[i] = lcg_next() * 2.0f;

    float *rs = xcalloc((size_t)(ck - 1) * cd, sizeof *rs);
    float *ro = xmalloc((size_t)ntok * cd * sizeof *ro);
    for (uint32_t t = 0; t < ntok; t++) {
        float *row = ro + (size_t)t * cd;
        memcpy(row, toks + (size_t)t * cd, cd * sizeof *row);
        sg_ref_conv1d_causal(row, cw, rs, rs, cd, ck);
    }

    gbuf wb2 = gb_from(cw, (uint64_t)cd * ck);
    gbuf ob2 = gb_new((uint64_t)cd * ck);
    uint32_t p2[8] = {cd, ck, 0, 0, 0, 0, 0, 0};
    ok = true;
    for (uint32_t t = 0; t < ntok && ok; t++) {
        gbuf xb = gb_from(toks + (size_t)t * cd, cd);
        memset(ob2.h, 0xA5, cd * sizeof(float));
        ok = gpu_run("k_conv1d_step", &xb, &wb2, &ob2, p2);
        if (ok && t == ntok - 1) {
            check_rel("conv1d_step (256 channels, token 3)", ob2.h, ro + (size_t)t * cd, cd);
        }
        gb_free(&xb);
    }
    if (ok) {
        check_rel("conv1d_step (256 channels) carried state", ob2.h + cd, rs,
                  (uint64_t)(ck - 1) * cd);
    }
    gb_free(&wb2); gb_free(&ob2);
    free(toks); free(rs); free(ro);

    /* ksize == 1: the degenerate conv that carries no state at all. The two
     * loops in the kernel both have zero trips and only the current token's
     * tap survives, which is the one shape where an off-by-one in `keep`
     * would read or write outside the buffer instead of merely being wrong. */
    const uint32_t c1 = 8;
    float *w1 = xmalloc(c1 * sizeof *w1), *x1 = xmalloc(c1 * sizeof *x1);
    for (uint32_t i = 0; i < c1; i++) { w1[i] = lcg_next(); x1[i] = lcg_next(); }
    float *want1 = xmalloc(c1 * sizeof *want1);
    memcpy(want1, x1, c1 * sizeof *want1);
    sg_ref_conv1d_causal(want1, w1, NULL, NULL, c1, 1);

    gbuf xb1 = gb_from(x1, c1), wb1 = gb_from(w1, c1), ob1 = gb_new(c1);
    uint32_t p1[8] = {c1, 1, 0, 0, 0, 0, 0, 0};
    gb_poison(&ob1);
    if (gpu_run("k_conv1d_step", &xb1, &wb1, &ob1, p1)) {
        check_rel("conv1d_step (ksize 1, no state)", ob1.h, want1, c1);
    }
    gb_free(&xb1); gb_free(&wb1); gb_free(&ob1);
    free(w1); free(x1); free(want1);
}

static void metal_delta_step_matches_ref(void) {
    uint32_t dk = fx_dim(&g_hyb, "delta_step.dk");
    uint32_t dv = fx_dim(&g_hyb, "delta_step.dv");
    uint32_t steps = fx_dim(&g_hyb, "delta_step.steps");
    const float *q = fx_f32(&g_hyb, "delta_step.q", (uint64_t)steps * dk, NULL);
    const float *k = fx_f32(&g_hyb, "delta_step.k", (uint64_t)steps * dk, NULL);
    const float *v = fx_f32(&g_hyb, "delta_step.v", (uint64_t)steps * dv, NULL);
    const float *decay = fx_f32(&g_hyb, "delta_step.decay", steps, NULL);
    const float *beta = fx_f32(&g_hyb, "delta_step.beta", steps, NULL);
    const float *s_in = fx_f32(&g_hyb, "delta_step.state_in", (uint64_t)dv * dk, NULL);

    /* Reference: the same chained sequence through sg_ref_delta_step. */
    float *ref_s = xmalloc((size_t)dv * dk * sizeof *ref_s);
    memcpy(ref_s, s_in, (size_t)dv * dk * sizeof *ref_s);
    float *ref_out = xmalloc((size_t)steps * dv * sizeof *ref_out);
    for (uint32_t t = 0; t < steps; t++) {
        sg_ref_delta_step(ref_s, q + (size_t)t * dk, k + (size_t)t * dk,
                          v + (size_t)t * dv, beta[t], decay[t],
                          ref_out + (size_t)t * dv, dk, dv);
    }

    gbuf sb = gb_from(s_in, (uint64_t)dv * dk);
    gbuf qkv = gb_new(2ull * dk + dv);
    gbuf ob = gb_new(dv);
    bool ok = true;
    for (uint32_t t = 0; t < steps && ok; t++) {
        memcpy(qkv.h, q + (size_t)t * dk, dk * sizeof(float));
        memcpy(qkv.h + dk, k + (size_t)t * dk, dk * sizeof(float));
        memcpy(qkv.h + 2 * dk, v + (size_t)t * dv, dv * sizeof(float));
        uint32_t p[8] = {dk, dv, f32_bits(beta[t]), f32_bits(decay[t]), 0, 0, 0, 0};
        gb_poison(&ob);
        ok = gpu_run("k_delta_step", &sb, &qkv, &ob, p);
        if (ok) {
            char lbl[64];
            snprintf(lbl, sizeof lbl, "delta_step readout, token %u", t);
            check_rel(lbl, ob.h, ref_out + (size_t)t * dv, dv);
        }
    }
    if (ok) check_rel("delta_step state after 3 tokens", sb.h, ref_s, (uint64_t)dv * dk);
    gb_free(&sb); gb_free(&qkv); gb_free(&ob);
    free(ref_s); free(ref_out);
}

/* RMSNormGated: silu(z) * rms_norm(y, ssm_norm) per value head, the last
 * thing a DeltaNet layer does before out_proj. Inputs are the real fixture
 * weights driven by the fixture's own decode token, so the vectors have the
 * magnitudes the kernel will actually see. */
static void metal_rmsnorm_gated_matches_ref(void) {
    uint32_t hidden = fx_dim(&g_hyb, "gdn.hidden");
    uint32_t heads = fx_dim(&g_hyb, "gdn.num_v_heads");
    uint32_t dv = fx_dim(&g_hyb, "gdn.head_v_dim");
    uint32_t conv_dim = fx_dim(&g_hyb, "gdn.conv_dim");
    float eps = fx_scalar(&g_hyb, "gdn.rms_eps");
    uint32_t value_dim = heads * dv;

    const float *qkv_w = fx_f32(&g_hyb, "gdn.in_proj_qkv_w", (uint64_t)conv_dim * hidden, NULL);
    const float *z_w = fx_f32(&g_hyb, "gdn.in_proj_z_w", (uint64_t)value_dim * hidden, NULL);
    const float *norm_w = fx_f32(&g_hyb, "gdn.norm_w", dv, NULL);
    const float *x = fx_f32(&g_hyb, "gdn.x_decode", hidden, NULL);

    /* y comes from the v slice of in_proj_qkv, z from in_proj_z: exactly
     * where gdn_layer gets them. */
    float *qkv = xmalloc((size_t)conv_dim * sizeof *qkv);
    sg_ref_matvec_f32(qkv_w, x, qkv, conv_dim, hidden);
    const float *y = qkv + (conv_dim - value_dim);
    float *z = xmalloc((size_t)value_dim * sizeof *z);
    sg_ref_matvec_f32(z_w, x, z, value_dim, hidden);

    float *want = xmalloc((size_t)value_dim * sizeof *want);
    for (uint32_t h = 0; h < heads; h++) {
        float *yh = want + (size_t)h * dv;
        memcpy(yh, y + (size_t)h * dv, dv * sizeof *yh);
        sg_ref_rmsnorm(yh, norm_w, dv, eps);
        float *zh = xmalloc(dv * sizeof *zh);
        memcpy(zh, z + (size_t)h * dv, dv * sizeof *zh);
        sg_ref_swiglu(zh, yh, dv);           /* zh = silu(zh) * yh */
        memcpy(yh, zh, dv * sizeof *yh);
        free(zh);
    }

    gbuf a = gb_from(y, value_dim);
    gbuf b = gb_new((uint64_t)value_dim + dv);
    memcpy(b.h, z, value_dim * sizeof(float));
    memcpy(b.h + value_dim, norm_w, dv * sizeof(float));
    gbuf o = gb_new(value_dim);
    uint32_t p[8] = {dv, heads, f32_bits(eps), 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_rmsnorm_gated", &a, &b, &o, p)) check_rel("rmsnorm_gated (4 heads x 32)", o.h, want, value_dim);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(qkv); free(z); free(want);
}

/* --------------------------------------------------------------------
 * Attention decode
 * -------------------------------------------------------------------- */

/* Builds the decode-time inputs the way ref.c's attn_layer does: project the
 * fixture's prefill tokens through the fixture's real q/k/v weights, apply
 * qk-norm then partial RoPE, and keep the K/V caches. Returns malloc'd
 * buffers; `qg` is the LAST token's q_proj output (queries interleaved with
 * the output gate, 2*head_dim per head, which is why the kernel takes a
 * q_stride). */
typedef struct {
    uint32_t n_heads, n_kv, hd, seq;
    uint32_t q_stride;   /* 2*hd for a real q_proj, hd for a compacted q */
    float *qg;      /* n_heads * q_stride */
    float *kv;      /* seq*n_kv*hd (K) followed by seq*n_kv*hd (V) */
} attn_inputs;

static attn_inputs attn_build_from_fixture(void) {
    attn_inputs in;
    uint32_t hidden = fx_dim(&g_hyb, "attn.hidden");
    in.n_heads = fx_dim(&g_hyb, "attn.n_heads");
    in.n_kv = fx_dim(&g_hyb, "attn.n_kv_heads");
    in.hd = fx_dim(&g_hyb, "attn.head_dim");
    in.seq = fx_dim(&g_hyb, "attn.prefill_len");
    in.q_stride = 2 * in.hd;              /* mlx splits q_proj on the LAST axis */
    uint32_t rope_dim = fx_dim(&g_hyb, "attn.rope_dim");
    float theta = fx_scalar(&g_hyb, "attn.rope_theta");
    float eps = fx_scalar(&g_hyb, "attn.rms_eps");

    uint32_t qw_rows = in.n_heads * 2 * in.hd, kvw_rows = in.n_kv * in.hd;
    const float *qw = fx_f32(&g_hyb, "attn.q_proj_w", (uint64_t)qw_rows * hidden, NULL);
    const float *kw = fx_f32(&g_hyb, "attn.k_proj_w", (uint64_t)kvw_rows * hidden, NULL);
    const float *vw = fx_f32(&g_hyb, "attn.v_proj_w", (uint64_t)kvw_rows * hidden, NULL);
    const float *qn = fx_f32(&g_hyb, "attn.q_norm_w", in.hd, NULL);
    const float *kn = fx_f32(&g_hyb, "attn.k_norm_w", in.hd, NULL);
    const float *xs = fx_f32(&g_hyb, "attn.x_prefill", (uint64_t)in.seq * hidden, NULL);

    in.qg = xmalloc((size_t)qw_rows * sizeof(float));
    in.kv = xmalloc(2u * (size_t)in.seq * kvw_rows * sizeof(float));
    float *kcache = in.kv, *vcache = in.kv + (size_t)in.seq * kvw_rows;

    for (uint32_t t = 0; t < in.seq; t++) {
        const float *x = xs + (size_t)t * hidden;
        sg_ref_matvec_f32(qw, x, in.qg, qw_rows, hidden);
        sg_ref_matvec_f32(kw, x, kcache + (size_t)t * kvw_rows, kvw_rows, hidden);
        sg_ref_matvec_f32(vw, x, vcache + (size_t)t * kvw_rows, kvw_rows, hidden);

        for (uint32_t h = 0; h < in.n_heads; h++) {
            float *qh = in.qg + (size_t)h * 2 * in.hd;
            sg_ref_rmsnorm(qh, qn, in.hd, eps);      /* the gate half is untouched */
            sg_ref_rope_partial(qh, in.hd, rope_dim, t, theta);
        }
        for (uint32_t h = 0; h < in.n_kv; h++) {
            float *kh = kcache + (size_t)t * kvw_rows + (size_t)h * in.hd;
            sg_ref_rmsnorm(kh, kn, in.hd, eps);
            sg_ref_rope_partial(kh, in.hd, rope_dim, t, theta);
        }
    }
    return in;
}

/* ref.c's attn_layer scoring loop, extracted so the GPU has something to be
 * compared against that is built out of sg_ref_* ops and nothing else. */
static void attn_ref_context(const attn_inputs *in, float *ctx) {
    const float *kcache = in->kv;
    const float *vcache = in->kv + (size_t)in->seq * in->n_kv * in->hd;
    double scale = 1.0 / sqrt((double)in->hd);
    uint32_t repeat = in->n_heads / in->n_kv;
    float *scores = xmalloc(in->seq * sizeof *scores);

    for (uint32_t h = 0; h < in->n_heads; h++) {
        uint32_t hk = h / repeat;
        const float *qh = in->qg + (size_t)h * in->q_stride;
        for (uint32_t t = 0; t < in->seq; t++) {
            const float *kt = kcache + ((size_t)t * in->n_kv + hk) * in->hd;
            double dot = 0.0;
            for (uint32_t i = 0; i < in->hd; i++) dot += (double)qh[i] * (double)kt[i];
            scores[t] = (float)(dot * scale);
        }
        sg_ref_softmax(scores, in->seq);
        float *ch = ctx + (size_t)h * in->hd;
        for (uint32_t i = 0; i < in->hd; i++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < in->seq; t++) {
                acc += (double)scores[t] *
                       (double)vcache[((size_t)t * in->n_kv + hk) * in->hd + i];
            }
            ch[i] = (float)acc;
        }
    }
    free(scores);
}

static void attn_case(const char *label, const attn_inputs *in) {
    uint32_t out_n = in->n_heads * in->hd;
    uint64_t kv_n = 2ull * in->seq * in->n_kv * in->hd;
    float *want = xmalloc(out_n * sizeof *want);
    attn_ref_context(in, want);

    gbuf qb = gb_from(in->qg, (uint64_t)in->n_heads * in->q_stride);
    gbuf kvb = gb_from(in->kv, kv_n);
    gbuf ob = gb_new(out_n);
    uint32_t p[8] = {in->n_heads, in->n_kv, in->hd, in->seq, in->q_stride,
                     (uint32_t)((uint64_t)in->seq * in->n_kv * in->hd),
                     f32_bits((float)(1.0 / sqrt((double)in->hd))), 0};
    gb_poison(&ob);
    if (gpu_run("k_attn_decode", &qb, &kvb, &ob, p)) check_rel(label, ob.h, want, out_n);
    gb_free(&qb); gb_free(&kvb); gb_free(&ob);
    free(want);
}

static void metal_attn_decode_matches_ref(void) {
    attn_inputs in = attn_build_from_fixture();
    tt_assert(in.n_heads > in.n_kv,
              "the attention fixture should exercise GQA (%u heads, %u kv heads)",
              in.n_heads, in.n_kv);
    attn_case("attn_decode (fixture, GQA, seq 5)", &in);

    /* The output gate that follows, on the same real vectors: the gate is
     * the second half of each head's q_proj slice. */
    uint32_t out_n = in.n_heads * in.hd;
    float *ctx = xmalloc(out_n * sizeof *ctx);
    attn_ref_context(&in, ctx);
    float *gate = xmalloc(out_n * sizeof *gate);
    for (uint32_t h = 0; h < in.n_heads; h++) {
        memcpy(gate + (size_t)h * in.hd, in.qg + (size_t)h * 2 * in.hd + in.hd,
               in.hd * sizeof(float));
    }
    float *want = xmalloc(out_n * sizeof *want);
    memcpy(want, ctx, out_n * sizeof *want);
    sg_ref_gate_sigmoid(want, gate, out_n);

    gbuf a = gb_from(ctx, out_n), b = gb_from(gate, out_n), o = gb_new(out_n);
    uint32_t p[8] = {out_n, 0, 0, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_gate_sigmoid", &a, &b, &o, p)) check_rel("attn output gate (fixture)", o.h, want, out_n);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(ctx); free(gate); free(want);
    free(in.qg); free(in.kv);

    /* A decode step at a context length the fixtures cannot reach: 1200
     * cached tokens, 8 query heads over 2 kv heads, head_dim 128. Past 256
     * keys the score loop strides, which is the path a real 128k decode
     * spends all its time in. */
    attn_inputs big;
    big.n_heads = 8; big.n_kv = 2; big.hd = 128; big.seq = 1200;
    big.q_stride = 2 * big.hd;
    big.qg = xmalloc((size_t)big.n_heads * big.q_stride * sizeof(float));
    big.kv = xmalloc(2ull * big.seq * big.n_kv * big.hd * sizeof(float));
    lcg_seed(4242u);
    for (uint32_t i = 0; i < big.n_heads * big.q_stride; i++) big.qg[i] = lcg_next();
    for (uint64_t i = 0; i < 2ull * big.seq * big.n_kv * big.hd; i++) big.kv[i] = lcg_next();
    attn_case("attn_decode (8 heads, seq 1200, hd 128)", &big);

    /* The same decode with a COMPACTED query buffer (q_stride == head_dim,
     * no interleaved gate). Task 10 may well hand the kernel queries in that
     * layout, and a stride bug is invisible while stride is always 2*hd. */
    attn_inputs tight = big;
    tight.q_stride = tight.hd;
    tight.qg = xmalloc((size_t)tight.n_heads * tight.hd * sizeof(float));
    for (uint32_t h = 0; h < tight.n_heads; h++) {
        memcpy(tight.qg + (size_t)h * tight.hd, big.qg + (size_t)h * big.q_stride,
               tight.hd * sizeof(float));
    }
    attn_case("attn_decode (q_stride == head_dim)", &tight);
    free(tight.qg);
    free(big.qg); free(big.kv);
}

/* --------------------------------------------------------------------
 * sg_gpu_wrap: the no-copy path a checkpoint mmap goes through
 * -------------------------------------------------------------------- */

static void metal_wrap_handles_page_offset(void) {
    const uint32_t rows = 40, cols = 512;
    uint64_t wbytes = (uint64_t)rows * cols * sizeof(uint16_t);
    long page = sysconf(_SC_PAGESIZE);
    /* A tensor never starts on a page boundary inside a real checkpoint, so
     * put it 1040 bytes into the second page: not page-aligned, but 16-byte
     * aligned, which is what surge.h asks of the caller. */
    uint64_t skew = (uint64_t)page + 1040;
    size_t maplen = (size_t)(skew + wbytes + (uint64_t)page);
    uint8_t *region = mmap(NULL, maplen, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANON, -1, 0);
    if (region == MAP_FAILED) { fprintf(stderr, "FATAL: mmap failed\n"); exit(2); }

    uint16_t *w = (uint16_t *)(region + skew);
    float *x = xmalloc(cols * sizeof *x);
    lcg_seed(90210u);
    for (uint32_t i = 0; i < rows * cols; i++) w[i] = f32_to_bf16(lcg_next() * 0.75f);
    for (uint32_t i = 0; i < cols; i++) x[i] = lcg_next();

    float *want = xmalloc(rows * sizeof *want);
    sg_ref_matvec_bf16(w, x, want, rows, cols);

    void *wrapped = NULL;
    sg_err e = sg_gpu_wrap(g_gpu, w, wbytes, &wrapped);
    tt_assert(!sg_failed(e), "sg_gpu_wrap: %s", e.msg ? e.msg : "");
    if (!sg_failed(e)) {
        gbuf a = {wrapped, NULL, 0};
        gbuf b = gb_from(x, cols), o = gb_new(rows);
        uint32_t p[8] = {rows, cols, 0, 0, 0, 0, 0, 0};
        gb_poison(&o);
        if (gpu_run("k_matvec_bf16", &a, &b, &o, p)) {
            check_rel("matvec_bf16 over wrapped (page-offset) memory", o.h, want, rows);
        }
        /* The handle must also expose the same bytes back to the CPU at the
         * requested offset, which is how Task 10 will read a wrapped tensor. */
        const uint16_t *host = sg_gpu_buf_host(wrapped);
        tt_assert(host != NULL && memcmp(host, w, (size_t)wbytes) == 0,
                  "sg_gpu_buf_host should point at the wrapped bytes, offset applied");
        sg_gpu_buf_free(wrapped);
        gb_free(&b); gb_free(&o);
    }
    free(x); free(want);
    munmap(region, maplen);
}

/* --------------------------------------------------------------------
 * Determinism: 100 runs, byte-identical
 * -------------------------------------------------------------------- */

#define DET_RUNS 100

/* Runs the op DET_RUNS times, restoring `restore` (the in-place state, if
 * any) before each, and asserts every run's output bytes match the first. */
static void det_check(const char *label, const char *kernel, gbuf *a, gbuf *b, gbuf *o,
                      const uint32_t p[8], const float *restore, uint64_t restore_n) {
    size_t obytes = (size_t)o->n * sizeof(float);
    uint8_t *first = xmalloc(obytes);
    uint8_t *state_first = restore ? xmalloc((size_t)restore_n * sizeof(float)) : NULL;
    uint32_t mismatches = 0, state_mismatches = 0;

    for (uint32_t r = 0; r < DET_RUNS; r++) {
        if (restore) memcpy(a->h, restore, (size_t)restore_n * sizeof(float));
        gb_poison(o);
        if (!gpu_run(kernel, a, b, o, p)) { free(first); free(state_first); return; }
        if (r == 0) {
            memcpy(first, o->h, obytes);
            if (state_first) memcpy(state_first, a->h, (size_t)restore_n * sizeof(float));
        } else {
            if (memcmp(first, o->h, obytes) != 0) mismatches++;
            if (state_first &&
                memcmp(state_first, a->h, (size_t)restore_n * sizeof(float)) != 0) {
                state_mismatches++;
            }
        }
    }
    tt_assert(mismatches == 0, "%s: %u of %u runs differed byte-wise from the first",
              label, mismatches, DET_RUNS - 1);
    tt_assert(state_mismatches == 0, "%s: in-place state differed on %u of %u runs",
              label, state_mismatches, DET_RUNS - 1);
    if (mismatches == 0 && state_mismatches == 0) {
        fprintf(stderr, "   %-46s %d/%d runs byte-identical\n", label, DET_RUNS, DET_RUNS);
    }
    free(first);
    free(state_first);
}

static void metal_reductions_are_deterministic(void) {
    /* matvec_bf16, at a width where the fold tree runs all 8 levels. */
    const uint32_t rows = 64, cols = 1024;
    uint16_t *w = xmalloc((size_t)rows * cols * sizeof *w);
    float *x = xmalloc(cols * sizeof *x);
    lcg_seed(5150u);
    for (uint32_t i = 0; i < rows * cols; i++) w[i] = f32_to_bf16(lcg_next());
    for (uint32_t i = 0; i < cols; i++) x[i] = lcg_next() * 4.0f;
    {
        gbuf a = gb_from_u16(w, (uint64_t)rows * cols), b = gb_from(x, cols), o = gb_new(rows);
        uint32_t p[8] = {rows, cols, 0, 0, 0, 0, 0, 0};
        det_check("k_matvec_bf16 (64x1024)", "k_matvec_bf16", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
    }
    free(w); free(x);

    /* matvec_f32 and rmsnorm fold the same tree, so their determinism is not
     * a separate mechanism, but it is a separate kernel and cheap to pin. */
    {
        const uint32_t frows = 48, fcols = 700;
        float *fw = xmalloc((size_t)frows * fcols * sizeof *fw);
        float *fx = xmalloc(fcols * sizeof *fx);
        for (uint32_t i = 0; i < frows * fcols; i++) fw[i] = lcg_next();
        for (uint32_t i = 0; i < fcols; i++) fx[i] = lcg_next() * 3.0f;
        gbuf a = gb_from(fw, (uint64_t)frows * fcols), b = gb_from(fx, fcols), o = gb_new(frows);
        uint32_t p[8] = {frows, fcols, 0, 0, 0, 0, 0, 0};
        det_check("k_matvec_f32 (48x700)", "k_matvec_f32", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(fw); free(fx);
    }
    {
        const uint32_t rn = 2500;
        float *rx = xmalloc(rn * sizeof *rx), *rw = xmalloc(rn * sizeof *rw);
        for (uint32_t i = 0; i < rn; i++) { rx[i] = lcg_next() * 2.0f; rw[i] = lcg_next(); }
        gbuf a = gb_from(rx, rn), b = gb_from(rw, rn), o = gb_new(rn);
        uint32_t p[8] = {rn, f32_bits(1e-6f), 1, 0, 0, 0, 0, 0};
        det_check("k_rmsnorm (n=2500)", "k_rmsnorm", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(rx); free(rw);
    }
    {
        const uint32_t heads = 8, dv = 512;
        float *y = xmalloc((size_t)heads * dv * sizeof *y);
        for (uint32_t i = 0; i < heads * dv; i++) y[i] = lcg_next();
        gbuf a = gb_from(y, (uint64_t)heads * dv);
        gbuf b = gb_new((uint64_t)heads * dv + dv);
        for (uint32_t i = 0; i < heads * dv + dv; i++) b.h[i] = lcg_next();
        gbuf o = gb_new((uint64_t)heads * dv);
        uint32_t p[8] = {dv, heads, f32_bits(1e-6f), 0, 0, 0, 0, 0};
        det_check("k_rmsnorm_gated (8 heads x 512)", "k_rmsnorm_gated", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(y);
    }

    /* softmax over a long row. */
    const uint32_t n = 3000;
    float *sx = xmalloc(n * sizeof *sx);
    for (uint32_t i = 0; i < n; i++) sx[i] = lcg_next() * 25.0f;
    {
        gbuf a = gb_from(sx, n), o = gb_new(n);
        uint32_t p[8] = {n, 0, 0, 0, 0, 0, 0, 0};
        det_check("k_softmax (n=3000)", "k_softmax", &a, NULL, &o, p, NULL, 0);
        gb_free(&a); gb_free(&o);
    }
    free(sx);

    /* attn_decode, the kernel with both a device-memory score row and two
     * reductions in it. */
    {
        attn_inputs in;
        in.n_heads = 8; in.n_kv = 2; in.hd = 128; in.seq = 1200;
        in.q_stride = 2 * in.hd;
        in.qg = xmalloc((size_t)in.n_heads * in.q_stride * sizeof(float));
        in.kv = xmalloc(2ull * in.seq * in.n_kv * in.hd * sizeof(float));
        for (uint32_t i = 0; i < in.n_heads * in.q_stride; i++) in.qg[i] = lcg_next();
        for (uint64_t i = 0; i < 2ull * in.seq * in.n_kv * in.hd; i++) in.kv[i] = lcg_next();

        gbuf qb = gb_from(in.qg, (uint64_t)in.n_heads * in.q_stride);
        gbuf kvb = gb_from(in.kv, 2ull * in.seq * in.n_kv * in.hd);
        gbuf ob = gb_new((uint64_t)in.n_heads * in.hd);
        uint32_t p[8] = {in.n_heads, in.n_kv, in.hd, in.seq, in.q_stride,
                         (uint32_t)((uint64_t)in.seq * in.n_kv * in.hd),
                         f32_bits((float)(1.0 / sqrt((double)in.hd))), 0};
        det_check("k_attn_decode (8 heads, seq 1200)", "k_attn_decode", &qb, &kvb, &ob, p, NULL, 0);
        gb_free(&qb); gb_free(&kvb); gb_free(&ob);
        free(in.qg); free(in.kv);
    }

    /* delta_step, which is deterministic only if each row of S is owned by
     * exactly one thread; the state is restored before every run so all 100
     * runs see identical input. */
    {
        const uint32_t dk = 128, dv = 128;
        float *s0 = xmalloc((size_t)dv * dk * sizeof *s0);
        for (uint32_t i = 0; i < dv * dk; i++) s0[i] = lcg_next() * 0.1f;
        gbuf sb = gb_from(s0, (uint64_t)dv * dk);
        gbuf qkv = gb_new(2ull * dk + dv);
        for (uint32_t i = 0; i < 2 * dk + dv; i++) qkv.h[i] = lcg_next();
        gbuf ob = gb_new(dv);
        uint32_t p[8] = {dk, dv, f32_bits(0.37f), f32_bits(0.93f), 0, 0, 0, 0};
        det_check("k_delta_step (128x128)", "k_delta_step", &sb, &qkv, &ob, p, s0, (uint64_t)dv * dk);
        gb_free(&sb); gb_free(&qkv); gb_free(&ob);
        free(s0);
    }
}

/* --------------------------------------------------------------------
 * Argument checking: a params/buffer mismatch must be an error return, not
 * an out-of-bounds GPU read (which faults the whole Metal context).
 * -------------------------------------------------------------------- */

static void metal_rejects_bad_arguments(void) {
    gbuf a = gb_new(64), b = gb_new(64), o = gb_new(64);
    uint32_t p[8] = {64, 0, 0, 0, 0, 0, 0, 0};

    sg_err e = sg_gpu_run_op(g_gpu, "k_not_a_kernel", a.b, b.b, o.b, p);
    tt_assert(sg_failed(e), "an unknown kernel name should be rejected");

    /* Aliasing: the output must never be an input. A kernel that overwrote
     * an input row another threadgroup had not read yet would be wrong AND
     * nondeterministic, which is the one thing this layer must not be. */
    e = sg_gpu_run_op(g_gpu, "k_swiglu", a.b, b.b, a.b, p);
    tt_assert(sg_failed(e), "out aliasing input a should be rejected");
    e = sg_gpu_run_op(g_gpu, "k_swiglu", a.b, b.b, b.b, p);
    tt_assert(sg_failed(e), "out aliasing input b should be rejected");

    /* Size arithmetic that would wrap: rows * cols * 4 overflows 64 bits for
     * these params, and a wrapped requirement would be SMALL and let an
     * undersized buffer through. */
    uint32_t overflow[8] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matvec_f32", a.b, b.b, o.b, overflow);
    tt_assert(sg_failed(e), "params whose byte count overflows should be rejected");

    uint32_t too_big[8] = {4096, 0, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_softmax", a.b, NULL, o.b, too_big);
    tt_assert(sg_failed(e), "n larger than the buffers should be rejected");

    uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_swiglu", a.b, b.b, o.b, zero);
    tt_assert(sg_failed(e), "a zero-length dispatch should be rejected");

    uint32_t odd_rope[8] = {64, 7, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_rope", a.b, b.b, o.b, odd_rope);
    tt_assert(sg_failed(e), "an odd rope_dim should be rejected (mlx rejects it too)");

    uint32_t bad_gqa[8] = {6, 4, 8, 2, 8, 96, f32_bits(0.35f), 0};
    e = sg_gpu_run_op(g_gpu, "k_attn_decode", a.b, b.b, o.b, bad_gqa);
    tt_assert(sg_failed(e), "n_heads not divisible by n_kv_heads should be rejected");

    /* A v_cache offset past the end of the kv buffer is the params mistake
     * that would read unmapped device memory rather than merely give a wrong
     * answer. */
    uint32_t bad_voff[8] = {4, 2, 8, 2, 8, 100000, f32_bits(0.35f), 0};
    e = sg_gpu_run_op(g_gpu, "k_attn_decode", a.b, b.b, o.b, bad_voff);
    tt_assert(sg_failed(e), "a v_cache offset past the end of b should be rejected");

    gb_free(&a); gb_free(&b); gb_free(&o);

    /* sg_gpu_wrap must refuse a pointer it cannot bind: setBuffer:offset:
     * has 4-byte granularity and the kernels cast the region to float*. */
    uint8_t *odd = xmalloc(64);
    void *handle = (void *)0x1;
    e = sg_gpu_wrap(g_gpu, odd + 1, 32, &handle);
    tt_assert(sg_failed(e) && handle == NULL,
              "sg_gpu_wrap should reject a pointer that is not 4-byte aligned");
    e = sg_gpu_wrap(g_gpu, odd, 0, &handle);
    tt_assert(sg_failed(e), "sg_gpu_wrap should reject a zero-length region");
    e = sg_gpu_wrap(g_gpu, odd, UINT64_MAX, &handle);
    tt_assert(sg_failed(e), "sg_gpu_wrap should reject a length that overflows");
    free(odd);

    /* The alias guard has to work on HOST RANGES, not on MTLBuffer identity.
     * newBufferWithBytesNoCopy hands back a different object every call, so
     * two wraps of the same array are two handles over one set of bytes and
     * an identity test would call them disjoint. Found by review; the mmap'd
     * weights of a real checkpoint are exactly the memory this happens to. */
    float *shared = xmalloc(4096 * sizeof *shared);
    for (int i = 0; i < 4096; i++) shared[i] = (float)i;
    void *w1 = NULL, *w2 = NULL;
    e = sg_gpu_wrap(g_gpu, shared, 16 * sizeof *shared, &w1);
    tt_assert(!sg_failed(e), "wrap 1: %s", e.msg ? e.msg : "ok");
    e = sg_gpu_wrap(g_gpu, shared, 16 * sizeof *shared, &w2);
    tt_assert(!sg_failed(e), "wrap 2: %s", e.msg ? e.msg : "ok");
    if (w1 && w2) {
        uint32_t mv[8] = {4, 4, 0, 0, 0, 0, 0, 0};
        e = sg_gpu_run_op(g_gpu, "k_matvec_f32", w1, w1, w2, mv);
        tt_assert(sg_failed(e),
                  "two separate wraps of one host range must count as overlapping");
    }
    /* Non-overlapping slices of the SAME array must still be allowed, or the
     * fix would have turned a missed alias into a false positive. */
    void *w3 = NULL, *w4 = NULL;
    e = sg_gpu_wrap(g_gpu, shared, 16 * sizeof *shared, &w3);
    tt_assert(!sg_failed(e), "wrap 3: %s", e.msg ? e.msg : "ok");
    e = sg_gpu_wrap(g_gpu, shared + 2048, 4 * sizeof *shared, &w4);
    tt_assert(!sg_failed(e), "wrap 4: %s", e.msg ? e.msg : "ok");
    if (w3 && w4) {
        uint32_t mv[8] = {4, 4, 0, 0, 0, 0, 0, 0};
        e = sg_gpu_run_op(g_gpu, "k_matvec_f32", w3, w3, w4, mv);
        tt_assert(!sg_failed(e), "disjoint slices of one array must be allowed: %s",
                  e.msg ? e.msg : "ok");
    }
    sg_gpu_buf_free(w1); sg_gpu_buf_free(w2);
    sg_gpu_buf_free(w3); sg_gpu_buf_free(w4);
    free(shared);
}

/* -------------------------------------------------------------------- */

int main(void) {
    sg_err e = sg_gpu_init(&g_gpu);
    if (sg_failed(e)) {
        fprintf(stderr, "SKIP test_metal_ops: %s\n", e.msg);
        return 0;
    }

    if (!fx_open(&g_ops, "tests/fixtures/ops.bin")) {
        fprintf(stderr, "FATAL: cannot open tests/fixtures/ops.bin "
                        "(run from the repo root)\n");
        return 2;
    }
    if (!fx_open(&g_hyb, "tests/fixtures/hybrid_ops.bin")) {
        fprintf(stderr, "FATAL: cannot open tests/fixtures/hybrid_ops.bin\n");
        return 2;
    }

    tt_run("metal_rmsnorm_matches_ref", metal_rmsnorm_matches_ref);
    tt_run("metal_rope_matches_ref", metal_rope_matches_ref);
    tt_run("metal_matvec_matches_ref", metal_matvec_matches_ref);
    tt_run("metal_softmax_matches_ref", metal_softmax_matches_ref);
    tt_run("metal_elementwise_match_ref", metal_elementwise_match_ref);
    tt_run("metal_conv1d_step_matches_ref", metal_conv1d_step_matches_ref);
    tt_run("metal_delta_step_matches_ref", metal_delta_step_matches_ref);
    tt_run("metal_rmsnorm_gated_matches_ref", metal_rmsnorm_gated_matches_ref);
    tt_run("metal_attn_decode_matches_ref", metal_attn_decode_matches_ref);
    tt_run("metal_wrap_handles_page_offset", metal_wrap_handles_page_offset);
    tt_run("metal_reductions_are_deterministic", metal_reductions_are_deterministic);
    tt_run("metal_rejects_bad_arguments", metal_rejects_bad_arguments);

    fprintf(stderr, "worst relative error across all ops: %.3e (%s)\n",
            g_worst_rel, g_worst_label);

    fx_close(&g_ops);
    fx_close(&g_hyb);
    sg_gpu_free(g_gpu);
    return tt_report();
}

#endif /* SURGE_NO_METAL */
