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

/* Task M5.3's own GEMM gates (task-M5.3-brief.md): bf16/f32 vs a host f64
 * reference, and the GEMM-vs-matvec row-consistency check for bf16/f32, must
 * be under 1e-5 relative; anything touching Q8_0 (dequant is only 8 bits of
 * mantissa to begin with) gets sg_ref_matvec_q8's own, looser 2e-2. */
#define GEMM_TOL    1e-5
#define GEMM_Q8_TOL 2e-2

/* Worst error seen across every op, printed at the end so the task report
 * can quote one number without grepping. Owned storage, not a `const char *`
 * that just remembers the winning check_rel_tol call's `label` argument:
 * several call sites (rope_case, attn_case, the M5.3 GEMM cases below) build
 * that label in a stack buffer with snprintf and it does not outlive the
 * function that built it, so latching the raw pointer would leave this
 * dangling by the time main() prints it. */
static double g_worst_rel = 0.0;
static char g_worst_label[64] = "(none)";

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

/* Raw bytes (a Q8_0 tensor) into a device buffer. gbuf.n is a FLOAT count
 * (gb_poison is its only reader), so round the byte count up to a float count,
 * the same way gb_from_u16 does for uint16 weights. */
static gbuf gb_from_u8(const void *src, uint64_t nbytes) {
    gbuf r = {NULL, NULL, (nbytes + 3) / 4};
    void *host = NULL;
    sg_err e = sg_gpu_alloc(g_gpu, nbytes, &r.b, &host);
    if (sg_failed(e)) { fprintf(stderr, "FATAL: %s\n", e.msg); exit(2); }
    r.h = (float *)host;
    memcpy(host, src, (size_t)nbytes);
    return r;
}

static void gb_free(gbuf *b) { sg_gpu_buf_free(b->b); b->b = NULL; b->h = NULL; }

/* Half-precision (2 bytes/element) device buffer, for the M5.2 fp16-KV
 * kernels: k_kv_store_f16's output and k_attn_decode_f16's K/V inputs. */
typedef struct { void *b; uint16_t *h; uint64_t n; } gbuf16;

static gbuf16 gb16_new(uint64_t n) {
    gbuf16 r = {NULL, NULL, n};
    void *host = NULL;
    sg_err e = sg_gpu_alloc(g_gpu, n * sizeof(uint16_t), &r.b, &host);
    if (sg_failed(e)) { fprintf(stderr, "FATAL: %s\n", e.msg); exit(2); }
    r.h = (uint16_t *)host;
    return r;
}

static void gb16_free(gbuf16 *b) { sg_gpu_buf_free(b->b); b->b = NULL; b->h = NULL; }

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

/* max |got - want| divided by the reference vector's own scale, checked
 * against an explicit tolerance. Absolute error alone is meaningless across
 * ops whose outputs span 1e-3 to 1e3, and per-element relative error is
 * meaningless wherever `want` crosses zero; this is the measure the plan's
 * "relative" bars (1e-4 for the per-op gate below, and M5.3's own 1e-5 /
 * 2e-2 GEMM gates) are checked in. */
static double check_rel_tol(const char *label, const float *got, const float *want,
                            uint64_t n, double tol) {
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

    tt_assert(rel < tol,
              "%s: max |err| %.3e over scale %.3e = %.3e relative, tol %.0e "
              "(worst at %llu: got %.9g, want %.9g)",
              label, worst, scale, rel, tol,
              (unsigned long long)at, (double)got[at], (double)want[at]);
    if (rel < tol) {
        fprintf(stderr, "   %-46s rel %.3e  abs %.3e\n", label, rel, worst);
        if (rel > g_worst_rel) {
            g_worst_rel = rel;
            snprintf(g_worst_label, sizeof g_worst_label, "%s", label);
        }
    }
    return rel;
}

static double check_rel(const char *label, const float *got, const float *want, uint64_t n) {
    return check_rel_tol(label, got, want, n, TOL_REL);
}

/* Bit-exact equality, element by element (memcmp per float so a NaN compares by
 * its bit pattern rather than as unequal-to-itself). The M5.5 chunk kernels are
 * a literal sequential replay of the per-token decode ops, so their equivalence
 * to those ops is BIT-IDENTICAL by construction, not merely within tolerance;
 * assert exactly that. Also reports the worst abs difference on failure. */
static void check_bit_identical(const char *label, const float *got, const float *want,
                                uint64_t n) {
    uint64_t mism = 0, at = 0;
    double worst = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        if (memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            if (mism == 0) at = i;
            mism++;
            double d = fabs((double)got[i] - (double)want[i]);
            if (d > worst) worst = d;
        }
    }
    tt_assert(mism == 0,
              "%s: %llu/%llu elements differ (first at %llu: got %.9g want %.9g, worst abs %.3e)",
              label, (unsigned long long)mism, (unsigned long long)n,
              (unsigned long long)at, mism ? (double)got[at] : 0.0,
              mism ? (double)want[at] : 0.0, worst);
    if (mism == 0) {
        fprintf(stderr, "   %-52s bit-identical (%llu elems)\n", label,
                (unsigned long long)n);
    }
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

/* Raw uint32 from the same LCG lcg_next draws from, for building byte
 * patterns (Q8_0 blocks) directly. Shares g_lcg, so it advances the same
 * stream. */
static uint32_t lcg_u32(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return g_lcg;
}

/* Build a synthetic Q8_0 tensor of rows x cols (cols a multiple of 32) into
 * dst, which must hold rows*(cols/32)*34 bytes. Each block is a 2-byte f16
 * scale (little-endian) then 32 int8. The scale bit-patterns are positive
 * finite normals in ~[2^-4, 2^3] (exponent field 11..18, never inf/NaN); the
 * int8 are any bit pattern. ref.c and the kernel decode these bytes
 * identically (bytewise f16 -> f32, scale*int8), so this is a valid oracle
 * for either side. The fixtures top out at 64 columns, so this is the only
 * way to exercise the reduction tree past its 256-wide first stride. */
static void build_q8(uint8_t *dst, uint32_t rows, uint32_t cols) {
    uint32_t blocks = cols / 32u;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            uint8_t *blk = dst + ((size_t)r * blocks + b) * 34u;
            uint16_t sbits = (uint16_t)(0x2C00u + (uint16_t)((lcg_u32() >> 11) & 0x1FFFu));
            blk[0] = (uint8_t)(sbits & 0xFFu);
            blk[1] = (uint8_t)(sbits >> 8);
            for (uint32_t i = 0; i < 32u; i++) blk[2u + i] = (uint8_t)(lcg_u32() >> 24);
        }
    }
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

static void metal_matvec_q8_matches_ref(void) {
    /* Q8_0 from ops.bin: the SAME 8x64 fixture test_ref_ops.c pins against
     * numpy. 8 rows, 64 cols = 2 blocks per row, 34 bytes each. The oracle is
     * sg_ref_matvec_q8 over the identical raw bytes, so any divergence is the
     * GPU dequant/reduction, nothing else. */
    uint64_t cols = 0, rows64 = 0;
    const float *x = fx_f32(&g_ops, "matvec_q8.x", 0, &cols);
    (void)fx_f32(&g_ops, "matvec_q8.out", 0, &rows64);   /* rows from the ref output */
    uint32_t rows = (uint32_t)rows64;
    tt_assert(rows == 8 && cols == 64, "matvec_q8 should be 8x64, got %ux%llu",
              rows, (unsigned long long)cols);
    uint64_t wbytes = (uint64_t)rows * (cols / 32) * 34;
    const void *w = fx_u8(&g_ops, "matvec_q8.w", wbytes);

    float *want = xmalloc(rows * sizeof *want);
    sg_ref_matvec_q8(w, x, want, rows, (uint32_t)cols);

    gbuf a = gb_from_u8(w, wbytes), b = gb_from(x, cols), o = gb_new(rows);
    uint32_t p[8] = {rows, (uint32_t)cols, 0, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_matvec_q8", &a, &b, &o, p)) check_rel("matvec_q8 (8x64)", o.h, want, rows);
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(want);

    /* A larger Q8_0 case the fixture cannot reach: 40 x 1056, 33 blocks per
     * row, a column count that is a multiple of 32 but not a power of two, so
     * the strided partial, the ragged tail and the full tree fold all run.
     * Synthetic blocks that ref.c and the kernel decode identically. */
    const uint32_t br = 40, bc = 1056;
    uint64_t bwbytes = (uint64_t)br * (bc / 32) * 34;
    uint8_t *bw = xmalloc((size_t)bwbytes);
    float *bx = xmalloc(bc * sizeof *bx);
    lcg_seed(0x9E37u);
    build_q8(bw, br, bc);
    for (uint32_t i = 0; i < bc; i++) bx[i] = lcg_next() * 2.0f;

    float *wantb = xmalloc(br * sizeof *wantb);
    sg_ref_matvec_q8(bw, bx, wantb, br, bc);

    gbuf a2 = gb_from_u8(bw, bwbytes), b2 = gb_from(bx, bc), o2 = gb_new(br);
    uint32_t p2[8] = {br, bc, 0, 0, 0, 0, 0, 0};
    gb_poison(&o2);
    if (gpu_run("k_matvec_q8", &a2, &b2, &o2, p2)) check_rel("matvec_q8 (40x1056)", o2.h, wantb, br);
    gb_free(&a2); gb_free(&b2); gb_free(&o2);
    free(bw); free(bx); free(wantb);
}

/* ====================================================================
 * Task M5.3: tiled GEMM, Y[N, M] = X[N, K] @ W[M, K]^T
 * ==================================================================== */

/* Host f64 GEMM reference, written fresh here rather than reusing
 * sg_ref_matvec_bf16/f32 (Task 7's own already-gated CPU reference): gate 1
 * asks whether the KERNEL agrees with plain double-precision math, and
 * routing that through ref.c's own implementation would make the check
 * "does the GPU agree with ref.c" twice over instead of once independently.
 * bf16 -> f32 widening is exact (the bf16 bits ARE the top half of the f32
 * ones), so doing it in float before promoting to double loses nothing. */
static void host_gemm_bf16_f64(const uint16_t *w, const float *x, float *y,
                               uint32_t n, uint32_t m, uint32_t k) {
    for (uint32_t ni = 0; ni < n; ni++) {
        const float *xr = x + (size_t)ni * k;
        for (uint32_t mi = 0; mi < m; mi++) {
            const uint16_t *wr = w + (size_t)mi * k;
            double acc = 0.0;
            for (uint32_t ki = 0; ki < k; ki++) {
                uint32_t bits = (uint32_t)wr[ki] << 16;
                float wf;
                memcpy(&wf, &bits, sizeof wf);
                acc += (double)wf * (double)xr[ki];
            }
            y[(size_t)ni * m + mi] = (float)acc;
        }
    }
}

static void host_gemm_f32_f64(const float *w, const float *x, float *y,
                              uint32_t n, uint32_t m, uint32_t k) {
    for (uint32_t ni = 0; ni < n; ni++) {
        const float *xr = x + (size_t)ni * k;
        for (uint32_t mi = 0; mi < m; mi++) {
            const float *wr = w + (size_t)mi * k;
            double acc = 0.0;
            for (uint32_t ki = 0; ki < k; ki++) acc += (double)wr[ki] * (double)xr[ki];
            y[(size_t)ni * m + mi] = (float)acc;
        }
    }
}

typedef struct { uint32_t n, m, k; } gemm_case;

/* Covers every one of the brief's {1, 7, 32, 256, 5120} in each of N, M and
 * K at least once (see the per-column comment), plus the explicitly
 * requested all-large case. All-5120-cubed would be ~134 billion FMAs and is
 * not what the brief asks for; the brief's own example of "all-large" is
 * 32x256x5120, so that is the one large-K*large-M*large-N case run here. */
static const gemm_case GEMM_CASES[] = {
    {1,    1,   1},      /* every dim at its smallest */
    {7,   32, 256},      /* N=7 */
    {32, 256,   7},      /* K=7, M=256 x N=32 combo */
    {256,  7,  32},      /* M=7, N=256 */
    {5120, 1,   1},      /* N=5120, cheap M/K */
    {1, 5120,   1},      /* M=5120, cheap N/K */
    {1,    1,5120},      /* K=5120, cheap N/M */
    {32, 256,5120},      /* the brief's "at least one all-large case" */
};
#define N_GEMM_CASES ((int)(sizeof GEMM_CASES / sizeof GEMM_CASES[0]))

/* Gate 1: k_matmul_bf16 and k_matmul_f32 vs the host f64 reference above,
 * across every (N, M, K) in GEMM_CASES. */
static void metal_matmul_matches_host_f64(void) {
    for (int c = 0; c < N_GEMM_CASES; c++) {
        uint32_t n = GEMM_CASES[c].n, m = GEMM_CASES[c].m, k = GEMM_CASES[c].k;
        char lbl[96];

        {
            uint16_t *w = xmalloc((size_t)m * k * sizeof *w);
            float *x = xmalloc((size_t)n * k * sizeof *x);
            lcg_seed(0xB160u + (uint32_t)c);
            for (uint64_t i = 0; i < (uint64_t)m * k; i++) w[i] = f32_to_bf16(lcg_next());
            for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;

            float *want = xmalloc((size_t)n * m * sizeof *want);
            host_gemm_bf16_f64(w, x, want, n, m, k);

            gbuf a = gb_from(x, (uint64_t)n * k);
            gbuf b = gb_from_u16(w, (uint64_t)m * k);
            gbuf o = gb_new((uint64_t)n * m);
            uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
            gb_poison(&o);
            snprintf(lbl, sizeof lbl, "matmul_bf16 vs host f64 (n=%u m=%u k=%u)", n, m, k);
            if (gpu_run("k_matmul_bf16", &a, &b, &o, p)) {
                check_rel_tol(lbl, o.h, want, (uint64_t)n * m, GEMM_TOL);
            }
            gb_free(&a); gb_free(&b); gb_free(&o);
            free(w); free(x); free(want);
        }
        {
            float *w = xmalloc((size_t)m * k * sizeof *w);
            float *x = xmalloc((size_t)n * k * sizeof *x);
            lcg_seed(0xF320u + (uint32_t)c);
            for (uint64_t i = 0; i < (uint64_t)m * k; i++) w[i] = lcg_next();
            for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;

            float *want = xmalloc((size_t)n * m * sizeof *want);
            host_gemm_f32_f64(w, x, want, n, m, k);

            gbuf a = gb_from(x, (uint64_t)n * k);
            gbuf b = gb_from(w, (uint64_t)m * k);
            gbuf o = gb_new((uint64_t)n * m);
            uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
            gb_poison(&o);
            snprintf(lbl, sizeof lbl, "matmul_f32 vs host f64 (n=%u m=%u k=%u)", n, m, k);
            if (gpu_run("k_matmul_f32", &a, &b, &o, p)) {
                check_rel_tol(lbl, o.h, want, (uint64_t)n * m, GEMM_TOL);
            }
            gb_free(&a); gb_free(&b); gb_free(&o);
            free(w); free(x); free(want);
        }
    }
}

/* Gate 2: row n of k_matmul_* == k_matvec_* applied to X[n], same W. Proves
 * the GEMM agrees with the kernel Task 9's own gate already pins down,
 * independent of whether the GEMM's tile geometry is right (a tiling bug
 * that still summed the correct K terms per element would slip past gate 1
 * only by coincidence; comparing every row against the row kernel is the
 * check that would catch a tile/row indexing mistake specifically). */
static void gemm_row_consistency_bf16(uint32_t n, uint32_t m, uint32_t k, uint32_t seed) {
    uint16_t *w = xmalloc((size_t)m * k * sizeof *w);
    float *x = xmalloc((size_t)n * k * sizeof *x);
    lcg_seed(seed);
    for (uint64_t i = 0; i < (uint64_t)m * k; i++) w[i] = f32_to_bf16(lcg_next());
    for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;

    gbuf a = gb_from(x, (uint64_t)n * k);
    gbuf b = gb_from_u16(w, (uint64_t)m * k);
    gbuf o = gb_new((uint64_t)n * m);
    uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_matmul_bf16", &a, &b, &o, p)) {
        gbuf xb = gb_new(k), yb = gb_new(m);
        uint32_t pv[8] = {m, k, 0, 0, 0, 0, 0, 0};
        for (uint32_t r = 0; r < n; r++) {
            memcpy(xb.h, x + (size_t)r * k, k * sizeof(float));
            gb_poison(&yb);
            if (!gpu_run("k_matvec_bf16", &b, &xb, &yb, pv)) break;
            char lbl[96];
            snprintf(lbl, sizeof lbl, "matmul_bf16 row %u == matvec_bf16 (n=%u m=%u k=%u)",
                     r, n, m, k);
            check_rel_tol(lbl, o.h + (size_t)r * m, yb.h, m, GEMM_TOL);
        }
        gb_free(&xb); gb_free(&yb);
    }
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(w); free(x);
}

static void gemm_row_consistency_f32(uint32_t n, uint32_t m, uint32_t k, uint32_t seed) {
    float *w = xmalloc((size_t)m * k * sizeof *w);
    float *x = xmalloc((size_t)n * k * sizeof *x);
    lcg_seed(seed);
    for (uint64_t i = 0; i < (uint64_t)m * k; i++) w[i] = lcg_next();
    for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;

    gbuf a = gb_from(x, (uint64_t)n * k);
    gbuf b = gb_from(w, (uint64_t)m * k);
    gbuf o = gb_new((uint64_t)n * m);
    uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_matmul_f32", &a, &b, &o, p)) {
        gbuf xb = gb_new(k), yb = gb_new(m);
        uint32_t pv[8] = {m, k, 0, 0, 0, 0, 0, 0};
        for (uint32_t r = 0; r < n; r++) {
            memcpy(xb.h, x + (size_t)r * k, k * sizeof(float));
            gb_poison(&yb);
            if (!gpu_run("k_matvec_f32", &b, &xb, &yb, pv)) break;
            char lbl[96];
            snprintf(lbl, sizeof lbl, "matmul_f32 row %u == matvec_f32 (n=%u m=%u k=%u)",
                     r, n, m, k);
            check_rel_tol(lbl, o.h + (size_t)r * m, yb.h, m, GEMM_TOL);
        }
        gb_free(&xb); gb_free(&yb);
    }
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(w); free(x);
}

static void gemm_row_consistency_q8(uint32_t n, uint32_t m, uint32_t k, uint32_t seed) {
    uint64_t wbytes = (uint64_t)m * (k / 32) * 34;
    uint8_t *w = xmalloc((size_t)wbytes);
    float *x = xmalloc((size_t)n * k * sizeof *x);
    lcg_seed(seed);
    build_q8(w, m, k);
    for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;

    gbuf a = gb_from(x, (uint64_t)n * k);
    gbuf b = gb_from_u8(w, wbytes);
    gbuf o = gb_new((uint64_t)n * m);
    uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_matmul_q8", &a, &b, &o, p)) {
        gbuf xb = gb_new(k), yb = gb_new(m);
        uint32_t pv[8] = {m, k, 0, 0, 0, 0, 0, 0};
        for (uint32_t r = 0; r < n; r++) {
            memcpy(xb.h, x + (size_t)r * k, k * sizeof(float));
            gb_poison(&yb);
            if (!gpu_run("k_matvec_q8", &b, &xb, &yb, pv)) break;
            char lbl[96];
            snprintf(lbl, sizeof lbl, "matmul_q8 row %u == matvec_q8 (n=%u m=%u k=%u)",
                     r, n, m, k);
            check_rel_tol(lbl, o.h + (size_t)r * m, yb.h, m, GEMM_Q8_TOL);
        }
        gb_free(&xb); gb_free(&yb);
    }
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(w); free(x);
}

static void metal_matmul_row_consistency(void) {
    /* Non-tile-aligned N and M (SG_GEMM_TM/TN are 16x16), so the tile edge
     * guard (`if (row >= n || col >= m) return`) is exercised, not just the
     * happy path where N and M are exact multiples of the tile shape. */
    gemm_row_consistency_bf16(11, 67, 512, 0xB16Cu);
    gemm_row_consistency_f32(11, 67, 512, 0xF32Cu);
    gemm_row_consistency_q8(11, 67, 512, 0xA8C0u);

    /* And a second, smaller shape entirely inside one tile. */
    gemm_row_consistency_bf16(3, 5, 64, 0xB16Du);
    gemm_row_consistency_f32(3, 5, 64, 0xF32Du);
    gemm_row_consistency_q8(3, 5, 64, 0xA8C1u);
}

/* Gate 3: k_matmul_q8 vs sg_ref_matvec_q8 (Task 7's double-accumulating CPU
 * reference) looped over the N rows, within sg_ref_matvec_q8's own 2e-2
 * tolerance -- the Q8_0 analog of gate 1, which only covers bf16/f32. */
static void metal_matmul_q8_matches_ref_looped(void) {
    const uint32_t n = 6, m = 97, k = 1056;   /* K a multiple of 32; N, M not tile-aligned */
    uint64_t wbytes = (uint64_t)m * (k / 32) * 34;
    uint8_t *w = xmalloc((size_t)wbytes);
    float *x = xmalloc((size_t)n * k * sizeof *x);
    lcg_seed(0xA80Bu);
    build_q8(w, m, k);
    for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;

    float *want = xmalloc((size_t)n * m * sizeof *want);
    for (uint32_t r = 0; r < n; r++) {
        sg_ref_matvec_q8(w, x + (size_t)r * k, want + (size_t)r * m, m, k);
    }

    gbuf a = gb_from(x, (uint64_t)n * k);
    gbuf b = gb_from_u8(w, wbytes);
    gbuf o = gb_new((uint64_t)n * m);
    uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
    gb_poison(&o);
    if (gpu_run("k_matmul_q8", &a, &b, &o, p)) {
        check_rel_tol("matmul_q8 vs sg_ref_matvec_q8 (looped over N rows)",
                      o.h, want, (uint64_t)n * m, GEMM_Q8_TOL);
    }
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(w); free(x); free(want);
}

/* check_params/check_sizes argument checking for the three new kernels,
 * mirroring metal_rejects_bad_arguments's style below for the existing
 * kernels. */
static void metal_matmul_rejects_bad_params(void) {
    gbuf a = gb_new(4096), b = gb_new(4096), o = gb_new(4096);

    uint32_t zero_n[8] = {0, 4, 8, 0, 0, 0, 0, 0};
    sg_err e = sg_gpu_run_op(g_gpu, "k_matmul_f32", a.b, b.b, o.b, zero_n);
    tt_assert(sg_failed(e), "k_matmul_f32 N=0 should be rejected");

    uint32_t zero_m[8] = {4, 0, 8, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matmul_f32", a.b, b.b, o.b, zero_m);
    tt_assert(sg_failed(e), "k_matmul_f32 M=0 should be rejected");

    uint32_t zero_k[8] = {4, 4, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matmul_bf16", a.b, b.b, o.b, zero_k);
    tt_assert(sg_failed(e), "k_matmul_bf16 K=0 should be rejected");

    /* Q8_0 rows are whole 32-element blocks; K not a multiple of 32 must be
     * rejected before check_sizes truncates the block count (the same
     * ordering k_matvec_q8 relies on, tested above for that kernel). */
    uint32_t q8_odd_k[8] = {2, 2, 40, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matmul_q8", a.b, b.b, o.b, q8_odd_k);
    tt_assert(sg_failed(e), "k_matmul_q8 K not a multiple of 32 should be rejected");

    uint32_t zero_k8[8] = {2, 2, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matmul_q8", a.b, b.b, o.b, zero_k8);
    tt_assert(sg_failed(e), "k_matmul_q8 K=0 should be rejected");

    /* Undersized buffers: N, M, K describe a Y larger than o actually is. */
    uint32_t too_big[8] = {100, 100, 8, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matmul_f32", a.b, b.b, o.b, too_big);
    tt_assert(sg_failed(e), "k_matmul_f32 output smaller than N*M should be rejected");

    /* Output overlapping an input: a GEMM tile writing an input another
     * tile has not read yet would be wrong and nondeterministic, same rule
     * as every other kernel. */
    uint32_t alias_p[8] = {4, 4, 4, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matmul_f32", a.b, b.b, a.b, alias_p);
    tt_assert(sg_failed(e), "k_matmul_f32 out aliasing input a should be rejected");

    gb_free(&a); gb_free(&b); gb_free(&o);
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
 * M5.2: fp16 KV cache
 * -------------------------------------------------------------------- */

/* Gate 1: k_kv_store_f16 round-trips exact f16 -- store an f32 value, read
 * back the SAME bits sg_f32_to_f16 (src/kv.c's pure-C reference, already
 * pinned to match Metal's `half` cast bit-for-bit) would produce. n = 777 is
 * not a multiple of SG_TG, so the ragged tail of the elementwise dispatch is
 * exercised too. */
static void metal_kv_store_f16_roundtrips(void) {
    const uint32_t n = 777;
    float *src = xmalloc(n * sizeof *src);
    lcg_seed(0xF16Du);
    for (uint32_t i = 0; i < n; i++) {
        /* A spread of magnitudes -- tiny (near f16 subnormal), huge (near f16
         * overflow) and ordinary -- so round-to-nearest-even edge cases in
         * sg_f32_to_f16 are exercised, not just well-behaved values. */
        float scale = (i % 7 == 0) ? 1e-6f : (i % 5 == 0) ? 6.0e4f : 3.0f;
        src[i] = lcg_next() * scale;
    }

    gbuf a = gb_from(src, n);
    gbuf16 o = gb16_new(n);
    memset(o.h, 0xA5, n * sizeof(uint16_t));
    uint32_t p[8] = {n, 0, 0, 0, 0, 0, 0, 0};

    sg_err e = sg_gpu_run_op(g_gpu, "k_kv_store_f16", a.b, NULL, o.b, p);
    tt_assert(!sg_failed(e), "k_kv_store_f16: %s", e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        uint64_t mismatches = 0;
        uint32_t at = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint16_t want = sg_f32_to_f16(src[i]);
            if (o.h[i] != want) { if (mismatches == 0) at = i; mismatches++; }
        }
        tt_assert(mismatches == 0,
                  "k_kv_store_f16 round-trip: %llu/%u mismatches (first at %u: "
                  "got 0x%04x, want 0x%04x)",
                  (unsigned long long)mismatches, n, at,
                  mismatches ? o.h[at] : 0, mismatches ? sg_f32_to_f16(src[at]) : 0);
        if (mismatches == 0) {
            fprintf(stderr, "   k_kv_store_f16 round-trip: %u/%u values bit-exact\n", n, n);
        }
    }
    gb_free(&a); gb16_free(&o);

    /* Also store at a NONZERO destination offset, into the middle of a
     * larger buffer, poisoning the surrounding halves first: the batched
     * decode path (enc_kv_store) always stores at pos*kv_width, never at
     * offset 0 past the very first position, and *2-vs-*4 byte-offset
     * mistakes are exactly the kind of bug that would only show up once the
     * offset is nonzero. */
    const uint32_t total = 512, off = 200, n2 = 64;
    gbuf a2 = gb_from(src, n2);   /* reuse the first n2 rounded values */

    /* sg_gpu_run_op has no offset concept of its own (it always binds a
     * handle at ITS OWN base); drive the nonzero offset the way enc_kv_store
     * does, by binding a handle whose contents start `off` halves into a
     * larger buffer. sg_gpu_wrap gives that (page-aligned base, arbitrary
     * byte offset from there), which also cross-checks the offset math
     * against a second, independent code path. */
    gbuf16 dst = gb16_new(total);
    memset(dst.h, 0xA5, total * sizeof(uint16_t));
    void *wrapped = NULL;
    sg_err ew = sg_gpu_wrap(g_gpu, dst.h + off, (uint64_t)n2 * sizeof(uint16_t), &wrapped);
    tt_assert(!sg_failed(ew), "sg_gpu_wrap (kv store offset test): %s", ew.msg ? ew.msg : "ok");
    if (!sg_failed(ew)) {
        uint32_t p3[8] = {n2, 0, 0, 0, 0, 0, 0, 0};
        sg_err e3 = sg_gpu_run_op(g_gpu, "k_kv_store_f16", a2.b, NULL, wrapped, p3);
        tt_assert(!sg_failed(e3), "k_kv_store_f16 (nonzero offset): %s", e3.msg ? e3.msg : "ok");
        if (!sg_failed(e3)) {
            uint64_t mism_before = 0, mism_in = 0, mism_after = 0;
            for (uint32_t i = 0; i < off; i++) if (dst.h[i] != (uint16_t)0xA5A5u) mism_before++;
            for (uint32_t i = 0; i < n2; i++) {
                if (dst.h[off + i] != sg_f32_to_f16(src[i])) mism_in++;
            }
            for (uint32_t i = off + n2; i < total; i++) {
                if (dst.h[i] != (uint16_t)0xA5A5u) mism_after++;
            }
            tt_assert(mism_before == 0 && mism_in == 0 && mism_after == 0,
                      "k_kv_store_f16 nonzero-offset store: %llu before / %llu wrong / "
                      "%llu after the target region",
                      (unsigned long long)mism_before, (unsigned long long)mism_in,
                      (unsigned long long)mism_after);
            if (mism_before == 0 && mism_in == 0 && mism_after == 0) {
                fprintf(stderr, "   k_kv_store_f16 nonzero-offset store: exact, "
                                "neighbors untouched\n");
            }
        }
        sg_gpu_buf_free(wrapped);
    }
    gb_free(&a2); gb16_free(&dst);
    free(src);
}

/* Gate 2: k_attn_decode_f16 == k_attn_decode fed the SAME inputs pre-rounded
 * to f16, bit-identical, over 100 reruns. K and V are rounded to their f16
 * value (still stored as f32) before either kernel sees them, so the f32
 * oracle commits no rounding the f16 kernel's half storage did not already
 * commit; widening half -> float back is exact, so the two must match bit
 * for bit. Rerunning the f16 kernel 100 times against one fixed f32-oracle
 * run proves determinism (all 100 runs equal) and equivalence (all 100 equal
 * the oracle) in one pass. */
static void metal_attn_decode_f16_matches_f32(void) {
    const uint32_t n_heads = 8, n_kv = 2, hd = 128, seq = 300, q_stride = 2 * hd;
    lcg_seed(0xA771Fu);

    float *qg = xmalloc((size_t)n_heads * q_stride * sizeof *qg);
    for (uint32_t i = 0; i < n_heads * q_stride; i++) qg[i] = lcg_next();

    uint64_t kvn = (uint64_t)seq * n_kv * hd;
    float *k32 = xmalloc(kvn * sizeof *k32), *v32 = xmalloc(kvn * sizeof *v32);
    uint16_t *k16 = xmalloc(kvn * sizeof *k16), *v16 = xmalloc(kvn * sizeof *v16);
    for (uint64_t i = 0; i < kvn; i++) {
        k16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        v16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        k32[i] = sg_f16_to_f32(k16[i]);   /* the f16 rounding, widened back exactly */
        v32[i] = sg_f16_to_f32(v16[i]);
    }

    float *kv32 = xmalloc(2 * kvn * sizeof *kv32);
    memcpy(kv32, k32, kvn * sizeof *kv32);
    memcpy(kv32 + kvn, v32, kvn * sizeof *kv32);

    uint32_t out_n = n_heads * hd;
    float scale = (float)(1.0 / sqrt((double)hd));

    /* f32 oracle: the EXISTING k_attn_decode kernel, unmodified, over the
     * old combined [K;V] buffer layout, fed the f16-rounded values. */
    gbuf qb = gb_from(qg, (uint64_t)n_heads * q_stride);
    gbuf kvb = gb_from(kv32, 2 * kvn);
    gbuf ref = gb_new(out_n);
    uint32_t pf32[8] = {n_heads, n_kv, hd, seq, q_stride, (uint32_t)kvn, f32_bits(scale), 0};
    gb_poison(&ref);
    bool ok = gpu_run("k_attn_decode", &qb, &kvb, &ref, pf32);
    tt_assert(ok, "k_attn_decode (f32 oracle) failed");

    /* f16 kernel: the SAME q, k, v values, but K/V in separate half buffers. */
    gbuf qb2 = gb_from(qg, (uint64_t)n_heads * q_stride);
    gbuf16 kb16 = gb16_new(kvn), vb16 = gb16_new(kvn);
    memcpy(kb16.h, k16, kvn * sizeof(uint16_t));
    memcpy(vb16.h, v16, kvn * sizeof(uint16_t));
    gbuf o16 = gb_new(out_n);
    uint32_t pf16[8] = {n_heads, n_kv, hd, seq, q_stride, f32_bits(scale), 0, 0};

    float *first_run = xmalloc(out_n * sizeof *first_run);
    uint64_t total_mismatch = 0;
    if (ok) {
        for (int rep = 0; rep < 100; rep++) {
            gb_poison(&o16);
            sg_err e = sg_gpu_run_attn_decode_f16(g_gpu, qb2.b, kb16.b, vb16.b, o16.b, pf16);
            tt_assert(!sg_failed(e), "k_attn_decode_f16 rep %d: %s", rep, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            if (rep == 0) {
                memcpy(first_run, o16.h, out_n * sizeof(float));
                for (uint32_t i = 0; i < out_n; i++) if (o16.h[i] != ref.h[i]) total_mismatch++;
            } else {
                for (uint32_t i = 0; i < out_n; i++) if (o16.h[i] != first_run[i]) total_mismatch++;
            }
        }
    }
    tt_assert(total_mismatch == 0,
              "k_attn_decode_f16: %llu bit-exact mismatches over 100 reruns "
              "(vs the f32 oracle and vs itself)", (unsigned long long)total_mismatch);
    if (total_mismatch == 0) {
        fprintf(stderr, "   k_attn_decode_f16 vs k_attn_decode (f16-rounded inputs): "
                        "bit-identical over 100 reruns\n");
    }

    free(first_run);
    gb_free(&qb); gb_free(&kvb); gb_free(&ref);
    gb_free(&qb2); gb16_free(&kb16); gb16_free(&vb16); gb_free(&o16);
    free(qg); free(k32); free(v32); free(k16); free(v16); free(kv32);
}

/* --------------------------------------------------------------------
 * M5.4: full-attention tiled prefill
 * -------------------------------------------------------------------- */

/* Gate 1: k_rope_chunk applied to a whole chunk of N tokens == k_rope_heads
 * applied once per token at that token's absolute position, BIT-IDENTICAL.
 * Both are GPU kernels reading the same input bytes and the same f32 cos/sin
 * values (the host builds each token's table in double, exactly as the decode
 * path does), and k_rope_chunk's rotation math is k_rope_heads' verbatim, so
 * the two must agree to the byte. Both output buffers are poisoned first, so
 * the untouched non-rotated regions (the [rope_dim, head_dim) tail is copied,
 * but the [head_dim, stride) gate half neither kernel writes) also match:
 * identical 0xA5 on both sides. */
static void rope_chunk_case(uint32_t head_dim, uint32_t rope_dim, uint32_t heads,
                            uint32_t stride, uint32_t n_tok, uint32_t base, float theta) {
    uint64_t slice_floats = (uint64_t)heads * stride;
    uint64_t xn = (uint64_t)n_tok * slice_floats;
    float *x = xmalloc(xn * sizeof *x);
    for (uint64_t i = 0; i < xn; i++) x[i] = lcg_next();

    /* Per-token cos/sin table [n_tok, rope_dim], each token at absolute pos base+t. */
    float *cs = xmalloc((uint64_t)n_tok * rope_dim * sizeof *cs);
    for (uint32_t t = 0; t < n_tok; t++) {
        rope_table(cs + (uint64_t)t * rope_dim, rope_dim, base + t, theta);
    }

    gbuf a = gb_from(x, xn), b = gb_from(cs, (uint64_t)n_tok * rope_dim), o = gb_new(xn);
    uint32_t p[8] = {head_dim, rope_dim, heads, stride, n_tok, 0, 0, 0};
    gb_poison(&o);
    bool ok = gpu_run("k_rope_chunk", &a, &b, &o, p);

    uint64_t mism = 0;
    uint32_t at_tok = 0;
    if (ok) {
        gbuf xr = gb_new(slice_floats), csr = gb_new(rope_dim), ores = gb_new(slice_floats);
        uint32_t pv[8] = {head_dim, rope_dim, heads, stride, 0, 0, 0, 0};
        for (uint32_t t = 0; t < n_tok; t++) {
            memcpy(xr.h, x + (uint64_t)t * slice_floats, (size_t)slice_floats * sizeof(float));
            rope_table(csr.h, rope_dim, base + t, theta);
            gb_poison(&ores);
            if (!gpu_run("k_rope_heads", &xr, &csr, &ores, pv)) { ok = false; break; }
            if (memcmp(ores.h, o.h + (uint64_t)t * slice_floats,
                       (size_t)slice_floats * sizeof(float)) != 0) {
                if (mism == 0) at_tok = t;
                mism++;
            }
        }
        gb_free(&xr); gb_free(&csr); gb_free(&ores);
    }

    char lbl[96];
    snprintf(lbl, sizeof lbl, "k_rope_chunk==k_rope_heads N=%u base=%u heads=%u stride=%u",
             n_tok, base, heads, stride);
    tt_assert(ok && mism == 0, "%s: %llu of %u tokens differed (first at token %u)",
              lbl, (unsigned long long)mism, n_tok, at_tok);
    if (ok && mism == 0) {
        fprintf(stderr, "   %-58s %u/%u tokens bit-identical\n", lbl, n_tok, n_tok);
    }
    gb_free(&a); gb_free(&b); gb_free(&o);
    free(x); free(cs);
}

static void metal_rope_chunk_matches_rope_heads(void) {
    /* The real checkpoint's RoPE parameters: head_dim 256, rope_dim 64 partial,
     * theta 1e7. base 1000 puts every token past the f32-angle danger zone the
     * double-precision host table exists for. */
    const uint32_t hd = 256, rope_dim = 64;
    const float theta = 1e7f;
    const uint32_t ns[4] = {1, 2, 17, 512};
    const uint32_t bases[2] = {0, 1000};
    lcg_seed(0x50BEC0DEu);
    for (int bi = 0; bi < 2; bi++) {
        for (int ni = 0; ni < 4; ni++) {
            /* Q layout: heads slices at stride 2*head_dim (the query interleaved
             * with the untouched attention-gate half). */
            rope_chunk_case(hd, rope_dim, 4, 2 * hd, ns[ni], bases[bi], theta);
        }
    }
    /* K layout: stride == head_dim (no interleaved gate), one representative
     * (N, base) since the kernel treats it as just a different heads/stride. */
    rope_chunk_case(hd, rope_dim, 2, hd, 17, 1000, theta);

    /* Gate 3 (determinism): 100 reruns byte-identical, output poisoned before
     * each run. (Inlined rather than via det_check, which is defined later in
     * this file.) */
    {
        const uint32_t n_tok = 17, heads = 4, stride = 2 * hd, base = 1000;
        uint64_t xn = (uint64_t)n_tok * heads * stride;
        float *x = xmalloc(xn * sizeof *x);
        for (uint64_t i = 0; i < xn; i++) x[i] = lcg_next();
        float *cs = xmalloc((uint64_t)n_tok * rope_dim * sizeof *cs);
        for (uint32_t t = 0; t < n_tok; t++) {
            rope_table(cs + (uint64_t)t * rope_dim, rope_dim, base + t, theta);
        }
        gbuf a = gb_from(x, xn), b = gb_from(cs, (uint64_t)n_tok * rope_dim), o = gb_new(xn);
        uint32_t p[8] = {hd, rope_dim, heads, stride, n_tok, 0, 0, 0};
        /* Byte compare, not float compare: the untouched [head_dim, stride)
         * gate half stays 0xA5 poison (a NaN), and NaN != NaN would be a false
         * mismatch. The poison is re-applied identically each run, so a correct
         * kernel is byte-identical over the whole buffer. */
        uint8_t *first = xmalloc((size_t)xn * sizeof(float));
        uint64_t mism = 0;
        for (int rep = 0; rep < 100; rep++) {
            gb_poison(&o);
            if (!gpu_run("k_rope_chunk", &a, &b, &o, p)) break;
            if (rep == 0) memcpy(first, o.h, (size_t)xn * sizeof(float));
            else if (memcmp(first, o.h, (size_t)xn * sizeof(float)) != 0) mism++;
        }
        tt_assert(mism == 0, "k_rope_chunk: %llu of 99 reruns differed byte-wise",
                  (unsigned long long)mism);
        if (mism == 0) {
            fprintf(stderr, "   k_rope_chunk (17 tok, base 1000): "
                            "byte-identical over 100 reruns\n");
        }
        free(first);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(x); free(cs);
    }
}

/* Gate 2: k_attn_prefill over a chunk == k_attn_decode_f16 run once per token
 * at seq = base+t+1 (the KV built up to and including that token), within the
 * per-op 1e-4 relative bar. The two share one fp16 KV fixture (base prior
 * positions plus the n-token chunk), and k_attn_prefill's per-(token, head)
 * threadgroup mirrors k_attn_decode_f16 statement for statement, so the
 * agreement is in fact bit-exact; the case asserts the 1e-4 gate and also
 * reports the exact-byte token count. Returns the worst relative error. */
static double attn_prefill_case(uint32_t n_heads, uint32_t n_kv, uint32_t hd,
                                uint32_t n_tok, uint32_t base) {
    uint32_t q_stride = 2 * hd;
    float scale = (float)(1.0 / sqrt((double)hd));
    uint64_t seq_max = (uint64_t)base + n_tok;
    uint64_t qn = (uint64_t)n_tok * n_heads * q_stride;
    uint64_t kvn = seq_max * n_kv * hd;
    uint64_t out_n = (uint64_t)n_tok * n_heads * hd;

    float *q = xmalloc(qn * sizeof *q);
    for (uint64_t i = 0; i < qn; i++) q[i] = lcg_next();
    uint16_t *k16 = xmalloc(kvn * sizeof *k16), *v16 = xmalloc(kvn * sizeof *v16);
    for (uint64_t i = 0; i < kvn; i++) {
        k16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        v16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
    }

    gbuf qb = gb_from(q, qn);
    gbuf16 kb = gb16_new(kvn), vb = gb16_new(kvn);
    memcpy(kb.h, k16, kvn * sizeof(uint16_t));
    memcpy(vb.h, v16, kvn * sizeof(uint16_t));
    gbuf ob = gb_new(out_n);
    gb_poison(&ob);
    uint32_t pp[8] = {n_heads, n_kv, hd, base, n_tok, q_stride, f32_bits(scale), 0};
    sg_err e = sg_gpu_run_attn_prefill(g_gpu, qb.b, kb.b, vb.b, ob.b, pp);
    tt_assert(!sg_failed(e), "k_attn_prefill (N=%u base=%u): %s", n_tok, base,
              e.msg ? e.msg : "ok");

    double worst = 0.0;
    uint64_t exact = 0;
    if (!sg_failed(e)) {
        gbuf qtok = gb_new((uint64_t)n_heads * q_stride);
        gbuf dec = gb_new((uint64_t)n_heads * hd);
        bool ok = true;
        for (uint32_t t = 0; t < n_tok && ok; t++) {
            memcpy(qtok.h, q + (uint64_t)t * n_heads * q_stride,
                   (size_t)n_heads * q_stride * sizeof(float));
            uint32_t seq = base + t + 1u;
            uint32_t pd[8] = {n_heads, n_kv, hd, seq, q_stride, f32_bits(scale), 0, 0};
            gb_poison(&dec);
            sg_err ed = sg_gpu_run_attn_decode_f16(g_gpu, qtok.b, kb.b, vb.b, dec.b, pd);
            if (sg_failed(ed)) { tt_assert(0, "decode ref t=%u: %s", t, ed.msg); ok = false; break; }

            const float *got = ob.h + (uint64_t)t * n_heads * hd;
            const float *want = dec.h;
            uint32_t hn = n_heads * hd;
            double sc = 0.0;
            for (uint32_t i = 0; i < hn; i++) { double m = fabs((double)want[i]); if (m > sc) sc = m; }
            if (sc == 0.0) sc = 1.0;
            for (uint32_t i = 0; i < hn; i++) {
                double d = fabs((double)got[i] - (double)want[i]) / sc;
                if (d > worst) worst = d;
            }
            if (memcmp(got, want, (size_t)hn * sizeof(float)) == 0) exact++;
        }
        gb_free(&qtok); gb_free(&dec);
    }

    char lbl[96];
    snprintf(lbl, sizeof lbl, "k_attn_prefill vs decode looped N=%u base=%u", n_tok, base);
    tt_assert(worst < TOL_REL, "%s: worst rel %.3e >= tol %.0e", lbl, worst, TOL_REL);
    fprintf(stderr, "   %-52s rel %.3e  (%llu/%u tokens bit-exact)\n",
            lbl, worst, (unsigned long long)exact, n_tok);
    if (worst > g_worst_rel) {
        g_worst_rel = worst;
        snprintf(g_worst_label, sizeof g_worst_label, "%s", lbl);
    }

    gb_free(&qb); gb16_free(&kb); gb16_free(&vb); gb_free(&ob);
    free(q); free(k16); free(v16);
    return worst;
}

static void metal_attn_prefill_matches_decode(void) {
    /* GQA (8 query heads over 2 kv heads, repeat 4), head_dim 64 for a fast
     * but representative shape. N in {1,2,17,512}, base in {0,1000}. */
    const uint32_t n_heads = 8, n_kv = 2, hd = 64;
    const uint32_t ns[4] = {1, 2, 17, 512};
    const uint32_t bases[2] = {0, 1000};
    lcg_seed(0x9EF11Eu);
    for (int bi = 0; bi < 2; bi++) {
        for (int ni = 0; ni < 4; ni++) {
            attn_prefill_case(n_heads, n_kv, hd, ns[ni], bases[bi]);
        }
    }

    /* Gate 3 (determinism): 100 reruns of k_attn_prefill byte-identical, output
     * poisoned before each run, at one representative (N=17, base=1000). */
    {
        const uint32_t n_tok = 17, base = 1000, q_stride = 2 * hd;
        float scale = (float)(1.0 / sqrt((double)hd));
        uint64_t seq_max = (uint64_t)base + n_tok;
        uint64_t qn = (uint64_t)n_tok * n_heads * q_stride, kvn = seq_max * n_kv * hd;
        uint64_t out_n = (uint64_t)n_tok * n_heads * hd;
        float *q = xmalloc(qn * sizeof *q);
        for (uint64_t i = 0; i < qn; i++) q[i] = lcg_next();
        uint16_t *k16 = xmalloc(kvn * sizeof *k16), *v16 = xmalloc(kvn * sizeof *v16);
        for (uint64_t i = 0; i < kvn; i++) {
            k16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
            v16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        }
        gbuf qb = gb_from(q, qn);
        gbuf16 kb = gb16_new(kvn), vb = gb16_new(kvn);
        memcpy(kb.h, k16, kvn * sizeof(uint16_t));
        memcpy(vb.h, v16, kvn * sizeof(uint16_t));
        gbuf ob = gb_new(out_n);
        uint32_t pp[8] = {n_heads, n_kv, hd, base, n_tok, q_stride, f32_bits(scale), 0};

        float *first = xmalloc(out_n * sizeof *first);
        uint64_t mism = 0;
        for (int rep = 0; rep < 100; rep++) {
            gb_poison(&ob);
            sg_err e = sg_gpu_run_attn_prefill(g_gpu, qb.b, kb.b, vb.b, ob.b, pp);
            tt_assert(!sg_failed(e), "k_attn_prefill det rep %d: %s", rep, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            if (rep == 0) memcpy(first, ob.h, out_n * sizeof *first);
            else for (uint64_t i = 0; i < out_n; i++) if (ob.h[i] != first[i]) mism++;
        }
        tt_assert(mism == 0, "k_attn_prefill: %llu bit mismatches over 100 reruns",
                  (unsigned long long)mism);
        if (mism == 0) {
            fprintf(stderr, "   k_attn_prefill (17 tok, base 1000): "
                            "byte-identical over 100 reruns\n");
        }
        free(first);
        gb_free(&qb); gb16_free(&kb); gb16_free(&vb); gb_free(&ob);
        free(q); free(k16); free(v16);
    }
}

/* --------------------------------------------------------------------
 * M5.5: gated-DeltaNet chunked-scan prefill kernels
 *
 * Each chunk kernel is a sequential replay of a per-token decode op with the
 * recurrent state threaded, so its output is BIT-IDENTICAL to that op looped
 * over the chunk. The oracle is therefore the GPU decode kernel itself (run
 * per token), not the CPU reference: comparing GPU-to-GPU makes the gap exactly
 * zero, which is what these tests assert (check_bit_identical, not check_rel).
 * -------------------------------------------------------------------- */

/* Gate 1 + gate 2 (conv) + gate 5 (conv): k_conv1d_chunk == k_conv1d_step
 * looped with the conv tail threaded, split-chunk == whole-chunk for both output
 * and tail, and 100-rerun byte determinism. conv width 384 (> 256, so the
 * per-channel grid runs past one threadgroup) and ksize 4 (tail 3, the 27B
 * conv_kernel). */
static void metal_conv1d_chunk_matches_step(void) {
    const uint32_t channels = 384, ksize = 4, keep = ksize - 1;
    const uint32_t ns[4] = {1, 2, 7, 64};
    lcg_seed(0x5C0FFEEu);
    float *w = xmalloc((size_t)channels * ksize * sizeof *w);
    for (uint32_t i = 0; i < channels * ksize; i++) w[i] = lcg_next();
    gbuf wb = gb_from(w, (uint64_t)channels * ksize);

    /* Gate 1: chunk == step looped, N in {1,2,7,64}. */
    for (int ni = 0; ni < 4; ni++) {
        uint32_t N = ns[ni];
        float *x = xmalloc((size_t)N * channels * sizeof *x);
        for (uint32_t i = 0; i < N * channels; i++) x[i] = lcg_next() * 2.0f;

        /* Oracle: k_conv1d_step over the N tokens, tail carried in stepbuf's
         * state half (out[channels..]), which gb_new left zeroed. */
        gbuf stepbuf = gb_new((uint64_t)ksize * channels);
        float *ref = xmalloc((size_t)N * channels * sizeof *ref);
        uint32_t ps[8] = {channels, ksize, 0, 0, 0, 0, 0, 0};
        bool ok = true;
        for (uint32_t t = 0; t < N && ok; t++) {
            gbuf xt = gb_from(x + (size_t)t * channels, channels);
            memset(stepbuf.h, 0xA5, channels * sizeof(float));   /* poison out half only */
            ok = gpu_run("k_conv1d_step", &xt, &wb, &stepbuf, ps);
            if (ok) memcpy(ref + (size_t)t * channels, stepbuf.h, channels * sizeof(float));
            gb_free(&xt);
        }

        gbuf xall = gb_from(x, (uint64_t)N * channels);
        gbuf outc = gb_new((uint64_t)N * channels);
        gbuf statec = gb_new((uint64_t)keep * channels);   /* zeroed initial tail */
        gb_poison(&outc);
        uint32_t pc[8] = {channels, ksize, N, 0, 0, 0, 0, 0};
        sg_err e = sg_gpu_run_conv1d_chunk(g_gpu, xall.b, wb.b, outc.b, statec.b, pc);
        tt_assert(!sg_failed(e), "conv1d_chunk N=%u: %s", N, e.msg ? e.msg : "ok");
        if (ok && !sg_failed(e)) {
            char lbl[80];
            snprintf(lbl, sizeof lbl, "k_conv1d_chunk==step looped N=%u", N);
            check_bit_identical(lbl, outc.h, ref, (uint64_t)N * channels);
            snprintf(lbl, sizeof lbl, "k_conv1d_chunk tail N=%u", N);
            check_bit_identical(lbl, statec.h, stepbuf.h + channels, (uint64_t)keep * channels);
        }
        gb_free(&stepbuf); gb_free(&xall); gb_free(&outc); gb_free(&statec);
        free(x); free(ref);
    }

    /* Gate 2 (conv): chunk(N) then chunk(M) sharing one tail == chunk(N+M). */
    {
        const uint32_t N = 7, M = 13, T = N + M;
        float *x = xmalloc((size_t)T * channels * sizeof *x);
        for (uint32_t i = 0; i < T * channels; i++) x[i] = lcg_next() * 2.0f;

        gbuf xw = gb_from(x, (uint64_t)T * channels);
        gbuf outw = gb_new((uint64_t)T * channels), statew = gb_new((uint64_t)keep * channels);
        gb_poison(&outw);
        uint32_t pw[8] = {channels, ksize, T, 0, 0, 0, 0, 0};
        sg_err e = sg_gpu_run_conv1d_chunk(g_gpu, xw.b, wb.b, outw.b, statew.b, pw);

        gbuf x1 = gb_from(x, (uint64_t)N * channels);
        gbuf x2 = gb_from(x + (size_t)N * channels, (uint64_t)M * channels);
        gbuf o1 = gb_new((uint64_t)N * channels), o2 = gb_new((uint64_t)M * channels);
        gbuf st = gb_new((uint64_t)keep * channels);   /* shared, threaded */
        gb_poison(&o1); gb_poison(&o2);
        uint32_t p1[8] = {channels, ksize, N, 0, 0, 0, 0, 0};
        uint32_t p2[8] = {channels, ksize, M, 0, 0, 0, 0, 0};
        sg_err e1 = sg_gpu_run_conv1d_chunk(g_gpu, x1.b, wb.b, o1.b, st.b, p1);
        sg_err e2 = sg_gpu_run_conv1d_chunk(g_gpu, x2.b, wb.b, o2.b, st.b, p2);
        tt_assert(!sg_failed(e) && !sg_failed(e1) && !sg_failed(e2),
                  "conv1d_chunk split/whole dispatch failed");
        if (!sg_failed(e) && !sg_failed(e1) && !sg_failed(e2)) {
            check_bit_identical("k_conv1d_chunk split==whole out[0:N]", o1.h, outw.h,
                                (uint64_t)N * channels);
            check_bit_identical("k_conv1d_chunk split==whole out[N:N+M]", o2.h,
                                outw.h + (size_t)N * channels, (uint64_t)M * channels);
            check_bit_identical("k_conv1d_chunk split==whole tail", st.h, statew.h,
                                (uint64_t)keep * channels);
        }
        gb_free(&xw); gb_free(&outw); gb_free(&statew);
        gb_free(&x1); gb_free(&x2); gb_free(&o1); gb_free(&o2); gb_free(&st);
        free(x);
    }

    /* Gate 5 (conv): 100 reruns byte-identical, output poisoned and the initial
     * tail restored to zero before each. */
    {
        const uint32_t N = 17;
        float *x = xmalloc((size_t)N * channels * sizeof *x);
        for (uint32_t i = 0; i < N * channels; i++) x[i] = lcg_next() * 2.0f;
        gbuf xall = gb_from(x, (uint64_t)N * channels);
        gbuf outc = gb_new((uint64_t)N * channels), statec = gb_new((uint64_t)keep * channels);
        uint32_t pc[8] = {channels, ksize, N, 0, 0, 0, 0, 0};
        float *first = xmalloc((size_t)N * channels * sizeof *first);
        float *stf = xmalloc((size_t)keep * channels * sizeof *stf);
        uint64_t mism = 0;
        for (int rep = 0; rep < 100; rep++) {
            memset(statec.h, 0, (size_t)keep * channels * sizeof(float));
            gb_poison(&outc);
            sg_err e = sg_gpu_run_conv1d_chunk(g_gpu, xall.b, wb.b, outc.b, statec.b, pc);
            tt_assert(!sg_failed(e), "conv1d_chunk det rep %d: %s", rep, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            if (rep == 0) {
                memcpy(first, outc.h, (size_t)N * channels * sizeof(float));
                memcpy(stf, statec.h, (size_t)keep * channels * sizeof(float));
            } else {
                if (memcmp(first, outc.h, (size_t)N * channels * sizeof(float)) != 0) mism++;
                if (memcmp(stf, statec.h, (size_t)keep * channels * sizeof(float)) != 0) mism++;
            }
        }
        tt_assert(mism == 0, "k_conv1d_chunk: %llu byte mismatches over 100 reruns",
                  (unsigned long long)mism);
        if (mism == 0) fprintf(stderr, "   k_conv1d_chunk (17 tok): byte-identical over 100 reruns\n");
        gb_free(&xall); gb_free(&outc); gb_free(&statec);
        free(x); free(first); free(stf);
    }
    gb_free(&wb); free(w);
}

/* k_delta_gates_chunk == k_delta_gates looped, N in {1,2,7,64}, both ssm_a
 * forms (A_log and -exp), plus 100-rerun determinism. */
static void metal_delta_gates_chunk_matches(void) {
    const uint32_t n = 40;
    const uint32_t ns[4] = {1, 2, 7, 64};
    lcg_seed(0xDEADBEEFu);
    float *adt = xmalloc((size_t)2 * n * sizeof *adt);   /* ssm_a[n] then dt_bias[n] */
    for (uint32_t i = 0; i < 2 * n; i++) adt[i] = lcg_next();
    gbuf adtb = gb_from(adt, 2ull * n);

    for (int ne = 0; ne < 2; ne++) {
        uint32_t neg_exp = (uint32_t)ne;
        for (int ni = 0; ni < 4; ni++) {
            uint32_t N = ns[ni];
            float *a = xmalloc((size_t)N * n * sizeof *a), *b = xmalloc((size_t)N * n * sizeof *b);
            for (uint32_t i = 0; i < N * n; i++) { a[i] = lcg_next() * 2.0f; b[i] = lcg_next() * 2.0f; }
            gbuf ab = gb_from(a, (uint64_t)N * n), bb = gb_from(b, (uint64_t)N * n);
            gbuf gc = gb_new(2ull * N * n);
            gb_poison(&gc);
            uint32_t pg[8] = {n, neg_exp, N, 0, 0, 0, 0, 0};
            sg_err e = sg_gpu_run_delta_gates_chunk(g_gpu, ab.b, bb.b, gc.b, adtb.b, pg);
            tt_assert(!sg_failed(e), "delta_gates_chunk N=%u ne=%u: %s", N, neg_exp,
                      e.msg ? e.msg : "ok");

            /* Oracle: k_delta_gates per token over ab_t = [a_t(n), b_t(n)]. */
            float *ref = xmalloc((size_t)2 * N * n * sizeof *ref);
            gbuf abt = gb_new(2ull * n), gt = gb_new(2ull * n);
            bool ok = true;
            for (uint32_t t = 0; t < N && ok; t++) {
                memcpy(abt.h, a + (size_t)t * n, n * sizeof(float));
                memcpy(abt.h + n, b + (size_t)t * n, n * sizeof(float));
                gb_poison(&gt);
                uint32_t pd[8] = {n, neg_exp, 0, 0, 0, 0, 0, 0};
                ok = gpu_run("k_delta_gates", &abt, &adtb, &gt, pd);
                if (ok) memcpy(ref + (size_t)t * 2 * n, gt.h, 2 * n * sizeof(float));
            }
            if (ok && !sg_failed(e)) {
                char lbl[80];
                snprintf(lbl, sizeof lbl, "k_delta_gates_chunk==gates N=%u ne=%u", N, neg_exp);
                check_bit_identical(lbl, gc.h, ref, 2ull * N * n);
            }
            gb_free(&ab); gb_free(&bb); gb_free(&gc); gb_free(&abt); gb_free(&gt);
            free(a); free(b); free(ref);
        }
    }

    /* Gate 5: 100 reruns byte-identical (N=17, -exp form). */
    {
        const uint32_t N = 17, neg_exp = 1;
        float *a = xmalloc((size_t)N * n * sizeof *a), *b = xmalloc((size_t)N * n * sizeof *b);
        for (uint32_t i = 0; i < N * n; i++) { a[i] = lcg_next() * 2.0f; b[i] = lcg_next() * 2.0f; }
        gbuf ab = gb_from(a, (uint64_t)N * n), bb = gb_from(b, (uint64_t)N * n), gc = gb_new(2ull * N * n);
        uint32_t pg[8] = {n, neg_exp, N, 0, 0, 0, 0, 0};
        float *first = xmalloc((size_t)2 * N * n * sizeof *first);
        uint64_t mism = 0;
        for (int rep = 0; rep < 100; rep++) {
            gb_poison(&gc);
            sg_err e = sg_gpu_run_delta_gates_chunk(g_gpu, ab.b, bb.b, gc.b, adtb.b, pg);
            tt_assert(!sg_failed(e), "delta_gates_chunk det rep %d: %s", rep, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            if (rep == 0) memcpy(first, gc.h, (size_t)2 * N * n * sizeof(float));
            else if (memcmp(first, gc.h, (size_t)2 * N * n * sizeof(float)) != 0) mism++;
        }
        tt_assert(mism == 0, "k_delta_gates_chunk: %llu mismatches over 100 reruns",
                  (unsigned long long)mism);
        if (mism == 0) fprintf(stderr, "   k_delta_gates_chunk (17 tok): byte-identical over 100 reruns\n");
        gb_free(&ab); gb_free(&bb); gb_free(&gc); free(a); free(b); free(first);
    }
    gb_free(&adtb); free(adt);
}

/* Gate 3: k_delta_chunk == k_delta_multi looped over the chunk with S threaded,
 * for one head-map (`tiled`) and chunk length N. dk=dv=64, n_k=4, n_v=12 so each
 * k-head maps to 3 v-heads (the 27B's 48/16 ratio), exercising both the tiled
 * (h % n_k) and grouped (h / (n_v/n_k)) maps. */
static void delta_chunk_case(uint32_t dk, uint32_t dv, uint32_t n_v, uint32_t n_k,
                             uint32_t tiled, uint32_t N, uint32_t seed) {
    uint32_t key_dim = n_k * dk, value_dim = n_v * dv, conv_dim = 2 * key_dim + value_dim;
    lcg_seed(seed);
    float *qkv = xmalloc((size_t)N * conv_dim * sizeof *qkv);
    for (uint32_t i = 0; i < N * conv_dim; i++) qkv[i] = lcg_next();
    float *gates = xmalloc((size_t)N * 2 * n_v * sizeof *gates);
    for (uint32_t t = 0; t < N; t++) {
        for (uint32_t h = 0; h < n_v; h++) {
            gates[(size_t)t * 2 * n_v + h]         = 0.5f + 0.4f * lcg_next();   /* beta  */
            gates[(size_t)t * 2 * n_v + n_v + h]   = 0.9f + 0.05f * lcg_next();  /* decay */
        }
    }
    float *S0 = xmalloc((size_t)n_v * dv * dk * sizeof *S0);
    for (uint32_t i = 0; i < n_v * dv * dk; i++) S0[i] = lcg_next() * 0.1f;

    /* Oracle: k_delta_multi per token, S carried in place. */
    gbuf Sm = gb_from(S0, (uint64_t)n_v * dv * dk);
    gbuf qt = gb_new(conv_dim), gt = gb_new(2ull * n_v), ot = gb_new(value_dim);
    float *ref = xmalloc((size_t)N * value_dim * sizeof *ref);
    bool ok = true;
    uint32_t pm[8] = {dk, dv, n_v, n_k, key_dim, tiled, 0, 0};
    for (uint32_t t = 0; t < N && ok; t++) {
        memcpy(qt.h, qkv + (size_t)t * conv_dim, conv_dim * sizeof(float));
        memcpy(gt.h, gates + (size_t)t * 2 * n_v, 2 * n_v * sizeof(float));
        gb_poison(&ot);
        sg_err e = sg_gpu_run_delta_multi(g_gpu, Sm.b, qt.b, ot.b, gt.b, pm);
        if (sg_failed(e)) { tt_assert(0, "delta_multi t=%u: %s", t, e.msg); ok = false; break; }
        memcpy(ref + (size_t)t * value_dim, ot.h, value_dim * sizeof(float));
    }

    gbuf Sc = gb_from(S0, (uint64_t)n_v * dv * dk);
    gbuf qkvb = gb_from(qkv, (uint64_t)N * conv_dim), gatesb = gb_from(gates, (uint64_t)N * 2 * n_v);
    gbuf oc = gb_new((uint64_t)N * value_dim);
    gb_poison(&oc);
    uint32_t pc[8] = {dk, dv, n_v, n_k, key_dim, tiled, N, conv_dim};
    sg_err e = sg_gpu_run_delta_chunk(g_gpu, Sc.b, qkvb.b, oc.b, gatesb.b, pc);
    tt_assert(!sg_failed(e), "delta_chunk N=%u tiled=%u: %s", N, tiled, e.msg ? e.msg : "ok");
    if (ok && !sg_failed(e)) {
        char lbl[96];
        snprintf(lbl, sizeof lbl, "k_delta_chunk==multi looped N=%u tiled=%u", N, tiled);
        check_bit_identical(lbl, oc.h, ref, (uint64_t)N * value_dim);
        snprintf(lbl, sizeof lbl, "k_delta_chunk S N=%u tiled=%u", N, tiled);
        check_bit_identical(lbl, Sc.h, Sm.h, (uint64_t)n_v * dv * dk);
    }
    gb_free(&Sm); gb_free(&qt); gb_free(&gt); gb_free(&ot);
    gb_free(&Sc); gb_free(&qkvb); gb_free(&gatesb); gb_free(&oc);
    free(qkv); free(gates); free(S0); free(ref);
}

static void metal_delta_chunk_matches_multi(void) {
    const uint32_t dk = 64, dv = 64, n_k = 4, n_v = 12;   /* 3 v-heads per k-head */
    const uint32_t key_dim = n_k * dk, value_dim = n_v * dv, conv_dim = 2 * key_dim + value_dim;
    const uint32_t ns[4] = {1, 2, 7, 64};

    /* Gate 3: both head maps, all N. */
    for (int ti = 0; ti < 2; ti++) {
        for (int ni = 0; ni < 4; ni++) {
            delta_chunk_case(dk, dv, n_v, n_k, (uint32_t)ti, ns[ni],
                             0x1234u + (uint32_t)ti * 97u + (uint32_t)ni * 13u);
        }
    }

    /* Gate 2 (S): chunk(N) then chunk(M) sharing one S == chunk(N+M), for both
     * the output and the final S. */
    {
        const uint32_t N = 7, M = 13, T = N + M, tiled = 1;
        lcg_seed(0xB0BAB0Bu);
        float *qkv = xmalloc((size_t)T * conv_dim * sizeof *qkv);
        for (uint32_t i = 0; i < T * conv_dim; i++) qkv[i] = lcg_next();
        float *gates = xmalloc((size_t)T * 2 * n_v * sizeof *gates);
        for (uint32_t t = 0; t < T; t++) for (uint32_t h = 0; h < n_v; h++) {
            gates[(size_t)t * 2 * n_v + h] = 0.5f + 0.4f * lcg_next();
            gates[(size_t)t * 2 * n_v + n_v + h] = 0.9f + 0.05f * lcg_next();
        }
        float *S0 = xmalloc((size_t)n_v * dv * dk * sizeof *S0);
        for (uint32_t i = 0; i < n_v * dv * dk; i++) S0[i] = lcg_next() * 0.1f;

        gbuf Sw = gb_from(S0, (uint64_t)n_v * dv * dk);
        gbuf qw = gb_from(qkv, (uint64_t)T * conv_dim), gw = gb_from(gates, (uint64_t)T * 2 * n_v);
        gbuf ow = gb_new((uint64_t)T * value_dim);
        gb_poison(&ow);
        uint32_t pwh[8] = {dk, dv, n_v, n_k, key_dim, tiled, T, conv_dim};
        sg_err e = sg_gpu_run_delta_chunk(g_gpu, Sw.b, qw.b, ow.b, gw.b, pwh);

        gbuf Ss = gb_from(S0, (uint64_t)n_v * dv * dk);   /* shared, threaded */
        gbuf q1 = gb_from(qkv, (uint64_t)N * conv_dim);
        gbuf q2 = gb_from(qkv + (size_t)N * conv_dim, (uint64_t)M * conv_dim);
        gbuf g1 = gb_from(gates, (uint64_t)N * 2 * n_v);
        gbuf g2 = gb_from(gates + (size_t)N * 2 * n_v, (uint64_t)M * 2 * n_v);
        gbuf o1 = gb_new((uint64_t)N * value_dim), o2 = gb_new((uint64_t)M * value_dim);
        gb_poison(&o1); gb_poison(&o2);
        uint32_t p1[8] = {dk, dv, n_v, n_k, key_dim, tiled, N, conv_dim};
        uint32_t p2[8] = {dk, dv, n_v, n_k, key_dim, tiled, M, conv_dim};
        sg_err e1 = sg_gpu_run_delta_chunk(g_gpu, Ss.b, q1.b, o1.b, g1.b, p1);
        sg_err e2 = sg_gpu_run_delta_chunk(g_gpu, Ss.b, q2.b, o2.b, g2.b, p2);
        tt_assert(!sg_failed(e) && !sg_failed(e1) && !sg_failed(e2),
                  "delta_chunk split/whole dispatch failed");
        if (!sg_failed(e) && !sg_failed(e1) && !sg_failed(e2)) {
            check_bit_identical("k_delta_chunk split==whole out[0:N]", o1.h, ow.h,
                                (uint64_t)N * value_dim);
            check_bit_identical("k_delta_chunk split==whole out[N:N+M]", o2.h,
                                ow.h + (size_t)N * value_dim, (uint64_t)M * value_dim);
            check_bit_identical("k_delta_chunk split==whole S", Ss.h, Sw.h,
                                (uint64_t)n_v * dv * dk);
        }
        gb_free(&Sw); gb_free(&qw); gb_free(&gw); gb_free(&ow);
        gb_free(&Ss); gb_free(&q1); gb_free(&q2); gb_free(&g1); gb_free(&g2);
        gb_free(&o1); gb_free(&o2);
        free(qkv); free(gates); free(S0);
    }

    /* Gate 5 (delta): 100 reruns byte-identical, S restored before each. */
    {
        const uint32_t N = 17, tiled = 1;
        lcg_seed(0xF00D5EEDu);
        float *qkv = xmalloc((size_t)N * conv_dim * sizeof *qkv);
        for (uint32_t i = 0; i < N * conv_dim; i++) qkv[i] = lcg_next();
        float *gates = xmalloc((size_t)N * 2 * n_v * sizeof *gates);
        for (uint32_t t = 0; t < N; t++) for (uint32_t h = 0; h < n_v; h++) {
            gates[(size_t)t * 2 * n_v + h] = 0.5f + 0.4f * lcg_next();
            gates[(size_t)t * 2 * n_v + n_v + h] = 0.9f + 0.05f * lcg_next();
        }
        float *S0 = xmalloc((size_t)n_v * dv * dk * sizeof *S0);
        for (uint32_t i = 0; i < n_v * dv * dk; i++) S0[i] = lcg_next() * 0.1f;

        gbuf Sc = gb_from(S0, (uint64_t)n_v * dv * dk);
        gbuf qkvb = gb_from(qkv, (uint64_t)N * conv_dim), gatesb = gb_from(gates, (uint64_t)N * 2 * n_v);
        gbuf oc = gb_new((uint64_t)N * value_dim);
        uint32_t pc[8] = {dk, dv, n_v, n_k, key_dim, tiled, N, conv_dim};
        float *first = xmalloc((size_t)N * value_dim * sizeof *first);
        float *sfirst = xmalloc((size_t)n_v * dv * dk * sizeof *sfirst);
        uint64_t mism = 0;
        for (int rep = 0; rep < 100; rep++) {
            memcpy(Sc.h, S0, (size_t)n_v * dv * dk * sizeof(float));   /* restore S */
            gb_poison(&oc);
            sg_err e = sg_gpu_run_delta_chunk(g_gpu, Sc.b, qkvb.b, oc.b, gatesb.b, pc);
            tt_assert(!sg_failed(e), "delta_chunk det rep %d: %s", rep, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            if (rep == 0) {
                memcpy(first, oc.h, (size_t)N * value_dim * sizeof(float));
                memcpy(sfirst, Sc.h, (size_t)n_v * dv * dk * sizeof(float));
            } else {
                if (memcmp(first, oc.h, (size_t)N * value_dim * sizeof(float)) != 0) mism++;
                if (memcmp(sfirst, Sc.h, (size_t)n_v * dv * dk * sizeof(float)) != 0) mism++;
            }
        }
        tt_assert(mism == 0, "k_delta_chunk: %llu byte mismatches over 100 reruns",
                  (unsigned long long)mism);
        if (mism == 0) fprintf(stderr, "   k_delta_chunk (17 tok): byte-identical over 100 reruns\n");
        gb_free(&Sc); gb_free(&qkvb); gb_free(&gatesb); gb_free(&oc);
        free(qkv); free(gates); free(S0); free(first); free(sfirst);
    }
}

/* Gate 4: k_rmsnorm_gated_chunk == k_rmsnorm_gated looped, N in {1,2,7,64},
 * plus 100-rerun determinism. dv 512 runs the fold tree past its 256 stride. */
static void metal_rmsnorm_gated_chunk_matches(void) {
    const uint32_t dv = 512, heads = 8, vd = heads * dv;
    const uint32_t ns[4] = {1, 2, 7, 64};
    const float eps = 1e-6f;
    lcg_seed(0xA11CE5u);
    float *w = xmalloc((size_t)dv * sizeof *w);
    for (uint32_t i = 0; i < dv; i++) w[i] = lcg_next();
    gbuf wb = gb_from(w, dv);

    for (int ni = 0; ni < 4; ni++) {
        uint32_t N = ns[ni];
        float *y = xmalloc((size_t)N * vd * sizeof *y), *z = xmalloc((size_t)N * vd * sizeof *z);
        for (uint32_t i = 0; i < N * vd; i++) { y[i] = lcg_next(); z[i] = lcg_next(); }
        gbuf yb = gb_from(y, (uint64_t)N * vd), zb = gb_from(z, (uint64_t)N * vd);
        gbuf ob = gb_new((uint64_t)N * vd);
        gb_poison(&ob);
        uint32_t pc[8] = {dv, heads, f32_bits(eps), N, 0, 0, 0, 0};
        sg_err e = sg_gpu_run_rmsnorm_gated_chunk(g_gpu, yb.b, zb.b, ob.b, wb.b, pc);
        tt_assert(!sg_failed(e), "rmsnorm_gated_chunk N=%u: %s", N, e.msg ? e.msg : "ok");

        /* Oracle: k_rmsnorm_gated per token, zw_t = [z_t(vd), w(dv)]. */
        float *ref = xmalloc((size_t)N * vd * sizeof *ref);
        gbuf yt = gb_new(vd), zwt = gb_new((uint64_t)vd + dv), ot = gb_new(vd);
        bool ok = true;
        uint32_t pd[8] = {dv, heads, f32_bits(eps), 0, 0, 0, 0, 0};
        for (uint32_t t = 0; t < N && ok; t++) {
            memcpy(yt.h, y + (size_t)t * vd, vd * sizeof(float));
            memcpy(zwt.h, z + (size_t)t * vd, vd * sizeof(float));
            memcpy(zwt.h + vd, w, dv * sizeof(float));
            gb_poison(&ot);
            ok = gpu_run("k_rmsnorm_gated", &yt, &zwt, &ot, pd);
            if (ok) memcpy(ref + (size_t)t * vd, ot.h, vd * sizeof(float));
        }
        if (ok && !sg_failed(e)) {
            char lbl[80];
            snprintf(lbl, sizeof lbl, "k_rmsnorm_gated_chunk==gated N=%u", N);
            check_bit_identical(lbl, ob.h, ref, (uint64_t)N * vd);
        }
        gb_free(&yb); gb_free(&zb); gb_free(&ob); gb_free(&yt); gb_free(&zwt); gb_free(&ot);
        free(y); free(z); free(ref);
    }

    /* Gate 5: 100 reruns byte-identical (N=17). */
    {
        const uint32_t N = 17;
        float *y = xmalloc((size_t)N * vd * sizeof *y), *z = xmalloc((size_t)N * vd * sizeof *z);
        for (uint32_t i = 0; i < N * vd; i++) { y[i] = lcg_next(); z[i] = lcg_next(); }
        gbuf yb = gb_from(y, (uint64_t)N * vd), zb = gb_from(z, (uint64_t)N * vd), ob = gb_new((uint64_t)N * vd);
        uint32_t pc[8] = {dv, heads, f32_bits(eps), N, 0, 0, 0, 0};
        float *first = xmalloc((size_t)N * vd * sizeof *first);
        uint64_t mism = 0;
        for (int rep = 0; rep < 100; rep++) {
            gb_poison(&ob);
            sg_err e = sg_gpu_run_rmsnorm_gated_chunk(g_gpu, yb.b, zb.b, ob.b, wb.b, pc);
            tt_assert(!sg_failed(e), "rmsnorm_gated_chunk det rep %d: %s", rep, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            if (rep == 0) memcpy(first, ob.h, (size_t)N * vd * sizeof(float));
            else if (memcmp(first, ob.h, (size_t)N * vd * sizeof(float)) != 0) mism++;
        }
        tt_assert(mism == 0, "k_rmsnorm_gated_chunk: %llu mismatches over 100 reruns",
                  (unsigned long long)mism);
        if (mism == 0) fprintf(stderr, "   k_rmsnorm_gated_chunk (17 tok): byte-identical over 100 reruns\n");
        gb_free(&yb); gb_free(&zb); gb_free(&ob); free(y); free(z); free(first);
    }
    gb_free(&wb); free(w);
}

/* The chunk one-shots' argument guards: an inconsistent or aliasing call must be
 * an error return, not a device-side out-of-bounds read. Buffers are sized so the
 * size checks pass and the specific guard under test (key_dim, or a destructive
 * state carrier overlapping a read-only input) is what fires. */
static void metal_chunk_rejects_bad_arguments(void) {
    const uint32_t dk = 8, dv = 8, n_k = 2, n_v = 4, key_dim = n_k * dk;
    const uint32_t value_dim = n_v * dv, conv_dim = 2 * key_dim + value_dim, N = 2;
    /* S is the largest region (n_v*dv*dk = 256 floats), big enough to double as
     * qkv/out for the aliasing cases below. */
    gbuf S = gb_new((uint64_t)n_v * dv * dk);
    gbuf qkv = gb_new((uint64_t)N * conv_dim), out = gb_new((uint64_t)N * value_dim);
    gbuf gates = gb_new((uint64_t)N * 2 * n_v);

    uint32_t pv[8] = {dk, dv, n_v, n_k, key_dim, 1, N, conv_dim};
    sg_err e = sg_gpu_run_delta_chunk(g_gpu, S.b, qkv.b, out.b, gates.b, pv);
    tt_assert(!sg_failed(e), "delta_chunk valid baseline: %s", e.msg ? e.msg : "ok");

    uint32_t pk[8] = {dk, dv, n_v, n_k, key_dim - 1u, 1, N, conv_dim};
    e = sg_gpu_run_delta_chunk(g_gpu, S.b, qkv.b, out.b, gates.b, pk);
    tt_assert(sg_failed(e), "delta_chunk key_dim < n_k*dk should be rejected");

    e = sg_gpu_run_delta_chunk(g_gpu, S.b, S.b, out.b, gates.b, pv);   /* S aliases qkv */
    tt_assert(sg_failed(e), "delta_chunk S aliasing qkv should be rejected");
    e = sg_gpu_run_delta_chunk(g_gpu, S.b, qkv.b, S.b, gates.b, pv);   /* out aliases S */
    tt_assert(sg_failed(e), "delta_chunk out aliasing S should be rejected");

    uint32_t pz[8] = {dk, dv, 0, n_k, key_dim, 1, N, conv_dim};
    e = sg_gpu_run_delta_chunk(g_gpu, S.b, qkv.b, out.b, gates.b, pz);
    tt_assert(sg_failed(e), "delta_chunk zero n_v should be rejected");
    gb_free(&S); gb_free(&qkv); gb_free(&out); gb_free(&gates);

    const uint32_t ch = 32, ks = 4;
    gbuf w = gb_new((uint64_t)ch * ks), oc = gb_new((uint64_t)N * ch);
    gbuf st = gb_new((uint64_t)(ks - 1) * ch);   /* 96 floats, big enough to double as x */
    uint32_t pc[8] = {ch, ks, N, 0, 0, 0, 0, 0};
    e = sg_gpu_run_conv1d_chunk(g_gpu, st.b, w.b, oc.b, st.b, pc);   /* state aliases x */
    tt_assert(sg_failed(e), "conv1d_chunk state aliasing x should be rejected");
    e = sg_gpu_run_conv1d_chunk(g_gpu, oc.b, w.b, oc.b, st.b, pc);   /* out aliases x */
    tt_assert(sg_failed(e), "conv1d_chunk out aliasing x should be rejected");
    gb_free(&w); gb_free(&oc); gb_free(&st);
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

/* Gate 4 (Task M5.3): 100 reruns of each k_matmul_* kernel, byte-identical,
 * output poisoned before every run (det_check above, already used for the
 * reduction kernels below, does exactly that). Shapes span several tiles in
 * both N and M and are deliberately not tile-aligned. */
static void metal_matmul_deterministic(void) {
    {
        const uint32_t n = 37, m = 91, k = 320;
        uint16_t *w = xmalloc((size_t)m * k * sizeof *w);
        float *x = xmalloc((size_t)n * k * sizeof *x);
        lcg_seed(0xDE70u);
        for (uint64_t i = 0; i < (uint64_t)m * k; i++) w[i] = f32_to_bf16(lcg_next());
        for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 3.0f;
        gbuf a = gb_from(x, (uint64_t)n * k), b = gb_from_u16(w, (uint64_t)m * k);
        gbuf o = gb_new((uint64_t)n * m);
        uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
        det_check("k_matmul_bf16 (37x91x320)", "k_matmul_bf16", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(w); free(x);
    }
    {
        const uint32_t n = 20, m = 48, k = 513;
        float *w = xmalloc((size_t)m * k * sizeof *w);
        float *x = xmalloc((size_t)n * k * sizeof *x);
        lcg_seed(0xF32Du);
        for (uint64_t i = 0; i < (uint64_t)m * k; i++) w[i] = lcg_next();
        for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;
        gbuf a = gb_from(x, (uint64_t)n * k), b = gb_from(w, (uint64_t)m * k);
        gbuf o = gb_new((uint64_t)n * m);
        uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
        det_check("k_matmul_f32 (20x48x513)", "k_matmul_f32", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(w); free(x);
    }
    {
        const uint32_t n = 9, m = 33, k = 1024;
        uint64_t wbytes = (uint64_t)m * (k / 32) * 34;
        uint8_t *w = xmalloc((size_t)wbytes);
        float *x = xmalloc((size_t)n * k * sizeof *x);
        lcg_seed(0xA8C2u);
        build_q8(w, m, k);
        for (uint64_t i = 0; i < (uint64_t)n * k; i++) x[i] = lcg_next() * 2.0f;
        gbuf a = gb_from(x, (uint64_t)n * k), b = gb_from_u8(w, wbytes);
        gbuf o = gb_new((uint64_t)n * m);
        uint32_t p[8] = {n, m, k, 0, 0, 0, 0, 0};
        det_check("k_matmul_q8 (9x33x1024)", "k_matmul_q8", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(w); free(x);
    }
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

    /* matvec_q8, same width. The fused dequant does not change the reduction
     * shape, but it is a distinct kernel and reads a distinct buffer type, so
     * pin its determinism directly. */
    {
        const uint32_t rows = 64, cols = 1024;
        uint64_t wbytes = (uint64_t)rows * (cols / 32) * 34;
        uint8_t *qw = xmalloc((size_t)wbytes);
        float *qx = xmalloc(cols * sizeof *qx);
        build_q8(qw, rows, cols);
        for (uint32_t i = 0; i < cols; i++) qx[i] = lcg_next() * 4.0f;
        gbuf a = gb_from_u8(qw, wbytes), b = gb_from(qx, cols), o = gb_new(rows);
        uint32_t p[8] = {rows, cols, 0, 0, 0, 0, 0, 0};
        det_check("k_matvec_q8 (64x1024)", "k_matvec_q8", &a, &b, &o, p, NULL, 0);
        gb_free(&a); gb_free(&b); gb_free(&o);
        free(qw); free(qx);
    }

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

    /* Q8_0 rows are whole 32-element blocks, so a cols that is not a multiple
     * of 32 has no valid byte layout; it must be rejected before the size
     * rule truncates the block count. */
    uint32_t q8_odd[8] = {8, 40, 0, 0, 0, 0, 0, 0};
    e = sg_gpu_run_op(g_gpu, "k_matvec_q8", a.b, b.b, o.b, q8_odd);
    tt_assert(sg_failed(e), "k_matvec_q8 cols not a multiple of 32 should be rejected");

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

/* --------------------------------------------------------------------
 * P2.2: split-K decode attention (k_attn_decode_splitk_partial +
 * k_attn_decode_splitk_combine) vs the P2.0/P2.1 CPU oracles
 * --------------------------------------------------------------------
 *
 * THESE KERNELS WERE WRITTEN AND COMPILED WITH THE GPU HELD BY A 28-HOUR
 * BENCHMARK, so this subtest had never been executed when it was committed.
 * It is the gate, written to be run the moment the GPU frees; nothing in it
 * had been observed to pass at that point. Everything it asserts is
 * therefore a CLAIM ABOUT INTENT until it runs green.
 *
 * Two oracles, on identical inputs, for every case:
 *   TIGHT   vs sg_ref_attn_decode_splitk at the SAME n_splits. This is the
 *           twin comparison: the Metal pair and that function partition
 *           [0, seq) by the same rule, build the same per-split (m, s, acc)
 *           triple, and fold it with the same log-sum-exp rescaling, so they
 *           must agree partition boundary for partition boundary.
 *   DIRECT  vs sg_ref_attn_decode (one pass over every key, no splitting at
 *           all). This is what proves split-K did not change the ANSWER, not
 *           merely that two split-K implementations agree with each other --
 *           the K-invariance trap the P2.1 review called out.
 *
 * K and V are rounded to f16 before either side sees them, and the CPU
 * oracles get the exactly-widened f32 values, so the only remaining gap is
 * ref.c's double accumulation against the kernel's f32 (the ~1e-7 the rest of
 * this file measures), not a rounding the oracle never committed. */

/* Assert the partial buffers themselves encode the partition the CPU oracle
 * defines: empty splits (which n_splits > seq forces) must carry EXACTLY the
 * documented m = -INFINITY, s = 0, acc = 0, and non-empty splits must carry a
 * finite m with s >= 1.0 (the key achieving that split's own max contributes
 * exp(0) == 1.0, so the sum cannot be smaller). Checked directly rather than
 * only through the combined output, because the combine maps an
 * empty-encoded split and an all-zero-weight split to the same answer and
 * would hide a partition rule that is off by one. */
static void splitk_check_partials(const char *label, const float *m, const float *s,
                                  const float *acc, uint32_t n_heads, uint32_t hd,
                                  uint32_t seq, uint32_t n_splits) {
    uint64_t bad_empty = 0, bad_full = 0, bad_acc = 0;
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < n_splits; i++) {
            uint32_t t0 = (uint32_t)((uint64_t)i * seq / n_splits);
            uint32_t t1 = (uint32_t)((uint64_t)(i + 1) * seq / n_splits);
            size_t at = (size_t)h * n_splits + i;
            if (t0 >= t1) {
                if (!(m[at] == -INFINITY) || s[at] != 0.0f) bad_empty++;
                for (uint32_t d = 0; d < hd; d++) {
                    if (acc[at * hd + d] != 0.0f) bad_acc++;
                }
            } else {
                if (!isfinite(m[at]) || !(s[at] >= 1.0f)) bad_full++;
            }
        }
    }
    tt_assert(bad_empty == 0 && bad_acc == 0 && bad_full == 0,
              "%s: %llu empty splits not encoded as (-inf, 0), %llu with a nonzero acc, "
              "%llu non-empty splits with a non-finite m or s < 1",
              label, (unsigned long long)bad_empty, (unsigned long long)bad_acc,
              (unsigned long long)bad_full);
}

/* One (shape, q_stride) sweep over every n_splits in the gate's list. */
static void splitk_shape(const char *what, uint32_t n_heads, uint32_t n_kv, uint32_t hd,
                         uint32_t seq, uint32_t q_stride, uint32_t seed) {
    const uint32_t splits[] = {1, 2, 3, 7, 64, 257};
    uint32_t out_n = n_heads * hd;
    uint64_t kvn = (uint64_t)seq * n_kv * hd;
    float scale = (float)(1.0 / sqrt((double)hd));

    lcg_seed(seed);
    float *q = xmalloc((size_t)n_heads * q_stride * sizeof *q);
    for (uint32_t h = 0; h < n_heads; h++) {
        float *qh = q + (size_t)h * q_stride;
        for (uint32_t i = 0; i < hd; i++) qh[i] = lcg_next();
        /* The hybrid layout's attention-gate half is NOT query data and must
         * never be read. Poison it with NaN: if the kernel's q_stride
         * handling were wrong by so much as one element, every comparison
         * below would come back NaN rather than merely inaccurate. */
        for (uint32_t i = hd; i < q_stride; i++) qh[i] = (float)NAN;
    }

    uint16_t *k16 = xmalloc((size_t)kvn * sizeof *k16);
    uint16_t *v16 = xmalloc((size_t)kvn * sizeof *v16);
    float *k32 = xmalloc((size_t)kvn * sizeof *k32);
    float *v32 = xmalloc((size_t)kvn * sizeof *v32);
    for (uint64_t i = 0; i < kvn; i++) {
        k16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        v16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        k32[i] = sg_f16_to_f32(k16[i]);   /* the f16 rounding, widened back exactly */
        v32[i] = sg_f16_to_f32(v16[i]);
    }

    /* Oracle 2 (direct, no splitting) is the same for every n_splits. */
    float *want_direct = xmalloc(out_n * sizeof *want_direct);
    sg_ref_attn_decode(q, k32, v32, n_heads, n_kv, hd, seq, q_stride, scale, want_direct);

    gbuf qb = gb_from(q, (uint64_t)n_heads * q_stride);
    gbuf16 kb = gb16_new(kvn), vb = gb16_new(kvn);
    memcpy(kb.h, k16, (size_t)kvn * sizeof(uint16_t));
    memcpy(vb.h, v16, (size_t)kvn * sizeof(uint16_t));

    float *want_split = xmalloc(out_n * sizeof *want_split);
    for (size_t si = 0; si < sizeof splits / sizeof splits[0]; si++) {
        uint32_t ns = splits[si];
        uint32_t p[8] = {n_heads, n_kv, hd, seq, q_stride, f32_bits(scale), ns, 0};
        gbuf mb = gb_new((uint64_t)n_heads * ns);
        gbuf sb = gb_new((uint64_t)n_heads * ns);
        gbuf ab = gb_new((uint64_t)n_heads * ns * hd);
        gbuf ob = gb_new(out_n);
        gb_poison(&mb); gb_poison(&sb); gb_poison(&ab); gb_poison(&ob);

        sg_err e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b,
                                                  mb.b, sb.b, ab.b, p);
        tt_assert(!sg_failed(e), "splitk partial (%s, n_splits %u): %s", what, ns,
                  e.msg ? e.msg : "ok");
        if (!sg_failed(e)) {
            char lbl[96];
            snprintf(lbl, sizeof lbl, "splitk partials (%s, K=%u)", what, ns);
            splitk_check_partials(lbl, mb.h, sb.h, ab.h, n_heads, hd, seq, ns);

            e = sg_gpu_run_attn_splitk_combine(g_gpu, mb.b, sb.b, ab.b, ob.b, p);
            tt_assert(!sg_failed(e), "splitk combine (%s, n_splits %u): %s", what, ns,
                      e.msg ? e.msg : "ok");
            if (!sg_failed(e)) {
                sg_ref_attn_decode_splitk(q, k32, v32, n_heads, n_kv, hd, seq,
                                          q_stride, scale, ns, want_split);
                snprintf(lbl, sizeof lbl, "splitk %s K=%-3u vs ref_splitk", what, ns);
                check_rel(lbl, ob.h, want_split, out_n);
                snprintf(lbl, sizeof lbl, "splitk %s K=%-3u vs ref_decode", what, ns);
                check_rel(lbl, ob.h, want_direct, out_n);
            }
        }
        gb_free(&mb); gb_free(&sb); gb_free(&ab); gb_free(&ob);
    }

    gb_free(&qb); gb16_free(&kb); gb16_free(&vb);
    free(q); free(k16); free(v16); free(k32); free(v32);
    free(want_direct); free(want_split);
}

/* 100 reruns of the same dispatch pair on identical inputs must be
 * BYTE-IDENTICAL, with every output buffer poisoned before each run so a
 * partially-written result cannot pass by inheriting the previous run's
 * bytes. This is the property the fixed-tree folds and the fixed partition
 * rule exist to provide; a stray atomic or a simd_sum would show up here. */
static void splitk_determinism(void) {
    const uint32_t n_heads = 32, n_kv = 8, hd = 128, seq = 1000, q_stride = 2 * hd;
    const uint32_t ns = 7;
    uint32_t out_n = n_heads * hd;
    uint64_t kvn = (uint64_t)seq * n_kv * hd;
    float scale = (float)(1.0 / sqrt((double)hd));

    lcg_seed(0x5D17Fu);
    float *q = xmalloc((size_t)n_heads * q_stride * sizeof *q);
    for (uint32_t i = 0; i < n_heads * q_stride; i++) q[i] = lcg_next();
    uint16_t *k16 = xmalloc((size_t)kvn * sizeof *k16);
    uint16_t *v16 = xmalloc((size_t)kvn * sizeof *v16);
    for (uint64_t i = 0; i < kvn; i++) {
        k16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
        v16[i] = sg_f32_to_f16(lcg_next() * 2.0f);
    }

    gbuf qb = gb_from(q, (uint64_t)n_heads * q_stride);
    gbuf16 kb = gb16_new(kvn), vb = gb16_new(kvn);
    memcpy(kb.h, k16, (size_t)kvn * sizeof(uint16_t));
    memcpy(vb.h, v16, (size_t)kvn * sizeof(uint16_t));
    gbuf mb = gb_new((uint64_t)n_heads * ns);
    gbuf sb = gb_new((uint64_t)n_heads * ns);
    gbuf ab = gb_new((uint64_t)n_heads * ns * hd);
    gbuf ob = gb_new(out_n);
    uint32_t p[8] = {n_heads, n_kv, hd, seq, q_stride, f32_bits(scale), ns, 0};

    float *first = xmalloc(out_n * sizeof *first);
    uint64_t mism = 0;
    for (int rep = 0; rep < 100; rep++) {
        gb_poison(&mb); gb_poison(&sb); gb_poison(&ab); gb_poison(&ob);
        sg_err e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b,
                                                  mb.b, sb.b, ab.b, p);
        if (!sg_failed(e)) e = sg_gpu_run_attn_splitk_combine(g_gpu, mb.b, sb.b, ab.b, ob.b, p);
        tt_assert(!sg_failed(e), "splitk determinism rep %d: %s", rep, e.msg ? e.msg : "ok");
        if (sg_failed(e)) break;
        if (rep == 0) memcpy(first, ob.h, out_n * sizeof(float));
        else for (uint32_t i = 0; i < out_n; i++) if (ob.h[i] != first[i]) mism++;
    }
    tt_assert(mism == 0, "split-K decode attention: %llu bit-exact mismatches over "
                         "100 reruns of the same input", (unsigned long long)mism);
    if (mism == 0) {
        fprintf(stderr, "   split-K decode attention: bit-identical over 100 reruns\n");
    }

    free(first); free(q); free(k16); free(v16);
    gb_free(&qb); gb16_free(&kb); gb16_free(&vb);
    gb_free(&mb); gb_free(&sb); gb_free(&ab); gb_free(&ob);
}

/* The host-side contract: every documented rejection actually rejects. These
 * need no correct kernel at all (check_params and the size rules run before
 * anything is encoded), so they are the one part of this subtest that does
 * not depend on the deferred numeric gates. */
static void splitk_rejects_bad_arguments(void) {
    const uint32_t n_heads = 4, n_kv = 2, hd = 8, seq = 16, ns = 3;
    uint32_t good[8] = {n_heads, n_kv, hd, seq, hd, f32_bits(0.35f), ns, 0};
    gbuf qb = gb_new((uint64_t)n_heads * hd);
    gbuf16 kb = gb16_new((uint64_t)seq * n_kv * hd), vb = gb16_new((uint64_t)seq * n_kv * hd);
    gbuf mb = gb_new((uint64_t)n_heads * ns), sb = gb_new((uint64_t)n_heads * ns);
    gbuf ab = gb_new((uint64_t)n_heads * ns * hd), ob = gb_new((uint64_t)n_heads * hd);
    sg_err e;

    e = sg_gpu_run_attn_splitk_partial(g_gpu, NULL, kb.b, vb.b, mb.b, sb.b, ab.b, good);
    tt_assert(sg_failed(e), "splitk partial should reject a NULL q");
    e = sg_gpu_run_attn_splitk_combine(g_gpu, mb.b, sb.b, ab.b, NULL, good);
    tt_assert(sg_failed(e), "splitk combine should reject a NULL out");

    uint32_t zero_k[8]; memcpy(zero_k, good, sizeof good); zero_k[6] = 0;
    e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b, mb.b, sb.b, ab.b, zero_k);
    tt_assert(sg_failed(e), "splitk partial should reject n_splits == 0");
    e = sg_gpu_run_attn_splitk_combine(g_gpu, mb.b, sb.b, ab.b, ob.b, zero_k);
    tt_assert(sg_failed(e), "splitk combine should reject n_splits == 0");

    uint32_t bad_gqa[8]; memcpy(bad_gqa, good, sizeof good); bad_gqa[1] = 3;
    e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b, mb.b, sb.b, ab.b, bad_gqa);
    tt_assert(sg_failed(e), "splitk partial should reject n_heads not a multiple of n_kv");

    uint32_t bad_stride[8]; memcpy(bad_stride, good, sizeof good); bad_stride[4] = hd - 1;
    e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b, mb.b, sb.b, ab.b, bad_stride);
    tt_assert(sg_failed(e), "splitk partial should reject q_stride < head_dim");

    /* An undersized partial buffer: n_splits raised without growing m/s/acc.
     * This is the mistake that would have the kernel write past the end. */
    uint32_t big_k[8]; memcpy(big_k, good, sizeof good); big_k[6] = ns + 1;
    e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b, mb.b, sb.b, ab.b, big_k);
    tt_assert(sg_failed(e), "splitk partial should reject partial buffers sized for fewer splits");

    /* Aliasing: an output over an input, and two outputs over each other. */
    e = sg_gpu_run_attn_splitk_partial(g_gpu, qb.b, kb.b, vb.b, mb.b, mb.b, ab.b, good);
    tt_assert(sg_failed(e), "splitk partial should reject m and s being the same buffer");
    e = sg_gpu_run_attn_splitk_combine(g_gpu, mb.b, sb.b, ab.b, mb.b, good);
    tt_assert(sg_failed(e), "splitk combine should reject out aliasing an input");

    /* And sg_gpu_run_op must refuse both by name rather than dispatch them
     * with two of their seven bindings missing. */
    e = sg_gpu_run_op(g_gpu, "k_attn_decode_splitk_partial", qb.b, kb.b, ob.b, good);
    tt_assert(sg_failed(e), "sg_gpu_run_op should refuse k_attn_decode_splitk_partial");
    e = sg_gpu_run_op(g_gpu, "k_attn_decode_splitk_combine", mb.b, sb.b, ob.b, good);
    tt_assert(sg_failed(e), "sg_gpu_run_op should refuse k_attn_decode_splitk_combine");

    gb_free(&qb); gb16_free(&kb); gb16_free(&vb);
    gb_free(&mb); gb_free(&sb); gb_free(&ab); gb_free(&ob);
}

static void metal_attn_splitk_matches_ref(void) {
    /* The real decode shape this exists for: Qwen3-4B-Instruct-2507's
     * 32 query heads over 8 kv heads, head_dim 128 (repeat 4). Two sequence
     * lengths on purpose: seq 200 puts n_splits 257 ABOVE seq, so the tail
     * splits are genuinely empty and exercise the -inf/0/0 encoding; seq 1000
     * keeps every split populated and pushes the low n_splits cases past 256
     * keys per threadgroup, where the score loop strides. Both q_stride
     * variants at both lengths: head_dim (dense qwen3, no gate) and
     * 2*head_dim (the hybrid's interleaved attention gate, NaN-poisoned). */
    splitk_shape("32x8x128 seq200 dense", 32, 8, 128, 200, 128, 0x51A7Cu);
    splitk_shape("32x8x128 seq200 gated", 32, 8, 128, 200, 256, 0x51A7Du);
    splitk_shape("32x8x128 seq1000 dense", 32, 8, 128, 1000, 128, 0x51A7Eu);
    splitk_shape("32x8x128 seq1000 gated", 32, 8, 128, 1000, 256, 0x51A7Fu);
    splitk_determinism();
    splitk_rejects_bad_arguments();
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
    tt_run("metal_matvec_q8_matches_ref", metal_matvec_q8_matches_ref);
    tt_run("metal_matmul_matches_host_f64", metal_matmul_matches_host_f64);
    tt_run("metal_matmul_row_consistency", metal_matmul_row_consistency);
    tt_run("metal_matmul_q8_matches_ref_looped", metal_matmul_q8_matches_ref_looped);
    tt_run("metal_matmul_deterministic", metal_matmul_deterministic);
    tt_run("metal_matmul_rejects_bad_params", metal_matmul_rejects_bad_params);
    tt_run("metal_softmax_matches_ref", metal_softmax_matches_ref);
    tt_run("metal_elementwise_match_ref", metal_elementwise_match_ref);
    tt_run("metal_conv1d_step_matches_ref", metal_conv1d_step_matches_ref);
    tt_run("metal_delta_step_matches_ref", metal_delta_step_matches_ref);
    tt_run("metal_rmsnorm_gated_matches_ref", metal_rmsnorm_gated_matches_ref);
    tt_run("metal_attn_decode_matches_ref", metal_attn_decode_matches_ref);
    tt_run("metal_kv_store_f16_roundtrips", metal_kv_store_f16_roundtrips);
    tt_run("metal_attn_decode_f16_matches_f32", metal_attn_decode_f16_matches_f32);
    tt_run("metal_rope_chunk_matches_rope_heads", metal_rope_chunk_matches_rope_heads);
    tt_run("metal_attn_prefill_matches_decode", metal_attn_prefill_matches_decode);
    tt_run("metal_attn_splitk_matches_ref", metal_attn_splitk_matches_ref);
    tt_run("metal_conv1d_chunk_matches_step", metal_conv1d_chunk_matches_step);
    tt_run("metal_delta_gates_chunk_matches", metal_delta_gates_chunk_matches);
    tt_run("metal_delta_chunk_matches_multi", metal_delta_chunk_matches_multi);
    tt_run("metal_rmsnorm_gated_chunk_matches", metal_rmsnorm_gated_chunk_matches);
    tt_run("metal_chunk_rejects_bad_arguments", metal_chunk_rejects_bad_arguments);
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
