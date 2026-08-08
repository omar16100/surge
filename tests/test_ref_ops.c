/* test_ref_ops.c - src/ref.c against fixtures generated from real
 * implementations, not from a prose description of the math.
 *
 * Two fixture files, both produced by tools/make_fixtures.py under
 * /Users/macmini/models/dsv4-venv/bin/python and committed under
 * tests/fixtures/:
 *
 *   ops.bin         numpy reference values for the base scalar ops
 *                   (rmsnorm, full RoPE, matvec bf16/Q8_0, softmax, swiglu).
 *                   Tolerances are the task brief's: 1e-5 for the f32 ops,
 *                   3e-2 for bf16 matvec and 2e-2 for Q8_0 matvec against
 *                   the unquantized answer. The quantized matvecs are ALSO
 *                   checked to 1e-5 against a matvec over the rounded
 *                   weights (the ".out_exact" records), which is what
 *                   actually pins the unpacking rather than just bounding
 *                   the quantization noise.
 *
 *   hybrid_ops.bin  ground truth obtained by CALLING mlx: every record is
 *                   the output of the same mlx primitive or module that
 *                   mlx_lm/models/qwen3_5.py uses. Tolerance 1e-4 max abs
 *                   error, f32 vs f32, per the plan's hybrid revision.
 *
 * The hybrid file has two tiers. Tier 1 is op-by-op (rms_norm with and
 * without a weight, partial RoPE at four positions, silu/sigmoid/softplus,
 * the delta-rule decay gate, the carried-state causal conv, and a chained
 * delta-rule step). Tier 2 is the stronger test the plan allows: two whole
 * mlx submodules -- a real GatedDeltaNet and a real Qwen3NextAttention at
 * small dims -- were each run as a 5-token prefill followed by a 1-token
 * decode against a real mlx cache, and gdn_submodule_matches_mlx /
 * attention_submodule_matches_mlx below rebuild those submodules end to end
 * out of nothing but sg_ref_* calls and compare all six tokens. That
 * catches composition mistakes (wrong q_proj/gate split, wrong head repeat,
 * state carried in the wrong order, RoPE applied before instead of after
 * the qk-norm) that no single-op fixture can.
 *
 * Fixture container format: see the "SURGEOPS" comment block in
 * tools/make_fixtures.py. The fx_* helpers below are its reader.
 */
#include "tinytest.h"
#include "../surge.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------
 * Fixture reader
 * -------------------------------------------------------------------- */

#define FX_F32 0u
#define FX_U16 1u
#define FX_U8  2u
#define FX_DIR_ENTRY 36u

/* Allocation failure in a test is not a test failure to report, it is a
 * reason to stop: soldiering on through a NULL would turn an OOM into a
 * misleading segfault inside an op. */
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "FATAL: out of memory (%zu bytes)\n", n); exit(2); }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "FATAL: out of memory (%zu x %zu)\n", n, sz); exit(2); }
    return p;
}

typedef struct {
    uint8_t *buf;
    uint64_t size;
    uint32_t n_records;
} fixture;

static bool fx_open(fixture *f, const char *path) {
    memset(f, 0, sizeof *f);
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return false; }
    long sz = ftell(fp);
    if (sz < 16) { fclose(fp); return false; }
    rewind(fp);
    uint8_t *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return false; }
    fclose(fp);
    if (memcmp(buf, "SURGEOPS", 8) != 0) { free(buf); return false; }
    uint32_t ver;
    memcpy(&ver, buf + 8, 4);
    if (ver != 1) { free(buf); return false; }
    memcpy(&f->n_records, buf + 12, 4);
    if (16 + (uint64_t)f->n_records * FX_DIR_ENTRY > (uint64_t)sz) { free(buf); return false; }
    f->buf = buf;
    f->size = (uint64_t)sz;
    return true;
}

static void fx_close(fixture *f) { free(f->buf); memset(f, 0, sizeof *f); }

/* Returns the payload pointer for `name`, or NULL. dtype/dims/count are
 * optional outputs; count is the element count (payload bytes / elem size).
 *
 * Every structural property the callers rely on is checked here, so that a
 * truncated or edited fixture is a loud exit rather than an out-of-bounds
 * read inside an op: the name must be NUL-terminated inside the buffer, the
 * payload must lie inside the buffer, the payload length must be exactly
 * prod(dims) * elem_size (so a caller that sizes a read from the record's
 * DIMS can never outrun the record's BYTES), and the payload must be
 * naturally aligned for its element type (the getters hand out float* and
 * uint16_t*; a misaligned one is UB and a UBSan failure). */
static const void *fx_find(const fixture *f, const char *name, uint32_t *dtype,
                           uint64_t dims[4], uint64_t *count) {
    for (uint32_t i = 0; i < f->n_records; i++) {
        const uint8_t *e = f->buf + 16 + (uint64_t)i * FX_DIR_ENTRY;
        uint32_t v[9];
        memcpy(v, e, sizeof v);
        uint32_t name_off = v[0], dt = v[1], n_dims = v[2];
        uint32_t data_off = v[7], nbytes = v[8];

        if (name_off >= f->size) return NULL;
        const char *nm = (const char *)f->buf + name_off;
        if (memchr(nm, '\0', (size_t)(f->size - name_off)) == NULL) return NULL;
        if (strcmp(nm, name) != 0) continue;

        if (dt > FX_U8 || n_dims > 4) return NULL;
        uint32_t esz = (dt == FX_F32) ? 4u : (dt == FX_U16 ? 2u : 1u);
        if ((uint64_t)data_off + nbytes > f->size) return NULL;
        if (data_off % esz != 0) return NULL;

        uint64_t elems = 1;
        for (uint32_t d = 0; d < n_dims; d++) elems *= v[3 + d];
        if (elems > UINT32_MAX / esz || elems * esz != nbytes) return NULL;

        if (dtype) *dtype = dt;
        if (dims) {
            for (uint32_t d = 0; d < 4; d++) dims[d] = (d < n_dims) ? v[3 + d] : 0;
        }
        if (count) *count = elems;
        return f->buf + data_off;
    }
    return NULL;
}

/* Every getter aborts loudly rather than returning garbage: a missing or
 * structurally invalid record means the committed fixture and this test
 * disagree about the fixture's contents, which is never something to
 * soldier on through.
 *
 * `want` is the number of elements the caller is about to read. Pass 0 only
 * when the caller is DISCOVERING the length from this very record (and then
 * asks for `count`); anywhere the length comes from somewhere else -- a
 * scalar record, another array's dims, a computed rows*cols -- state it, so
 * the read is bounded by this record's own payload rather than by another
 * record's idea of how big it should be. */
static const float *fx_f32(const fixture *f, const char *name, uint64_t want,
                           uint64_t *count) {
    uint32_t dt = 0;
    uint64_t n = 0;
    const void *p = fx_find(f, name, &dt, NULL, &n);
    if (!p || dt != FX_F32) {
        fprintf(stderr, "FATAL: fixture record '%s' missing, malformed, or not f32\n", name);
        exit(2);
    }
    if (want != 0 && n < want) {
        fprintf(stderr, "FATAL: fixture record '%s' has %llu f32 elements, "
                        "caller needs %llu\n", name,
                        (unsigned long long)n, (unsigned long long)want);
        exit(2);
    }
    if (count) *count = n;
    return (const float *)p;
}

static const uint16_t *fx_u16(const fixture *f, const char *name, uint64_t want,
                              uint64_t *count) {
    uint32_t dt = 0;
    uint64_t n = 0;
    const void *p = fx_find(f, name, &dt, NULL, &n);
    if (!p || dt != FX_U16) {
        fprintf(stderr, "FATAL: fixture record '%s' missing, malformed, or not u16\n", name);
        exit(2);
    }
    if (want != 0 && n < want) {
        fprintf(stderr, "FATAL: fixture record '%s' has %llu u16 elements, "
                        "caller needs %llu\n", name,
                        (unsigned long long)n, (unsigned long long)want);
        exit(2);
    }
    if (count) *count = n;
    return (const uint16_t *)p;
}

static const void *fx_u8(const fixture *f, const char *name, uint64_t want) {
    uint32_t dt = 0;
    uint64_t n = 0;
    const void *p = fx_find(f, name, &dt, NULL, &n);
    if (!p || dt != FX_U8) {
        fprintf(stderr, "FATAL: fixture record '%s' missing, malformed, or not u8\n", name);
        exit(2);
    }
    if (want != 0 && n < want) {
        fprintf(stderr, "FATAL: fixture record '%s' has %llu bytes, caller needs %llu\n",
                name, (unsigned long long)n, (unsigned long long)want);
        exit(2);
    }
    return p;
}

static uint32_t fx_dim(const fixture *f, const char *name) {
    uint64_t n = 0;
    const float *p = fx_f32(f, name, 1, &n);
    if (n != 1) { fprintf(stderr, "FATAL: '%s' is not a scalar\n", name); exit(2); }
    if (!(p[0] >= 0.0f && p[0] <= 1e9f)) {
        fprintf(stderr, "FATAL: '%s' = %g is not a plausible dimension\n", name, (double)p[0]);
        exit(2);
    }
    return (uint32_t)(p[0] + 0.5f);
}

static float fx_scalar(const fixture *f, const char *name) {
    uint64_t n = 0;
    const float *p = fx_f32(f, name, 1, &n);
    if (n != 1) { fprintf(stderr, "FATAL: '%s' is not a scalar\n", name); exit(2); }
    return p[0];
}

/* --------------------------------------------------------------------
 * Comparison helper
 * -------------------------------------------------------------------- */

static double max_abs_err(const float *got, const float *want, uint64_t n, uint64_t *at) {
    double worst = 0.0;
    if (at) *at = 0;
    for (uint64_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        if (isnan(d)) { if (at) *at = i; return INFINITY; }
        if (d > worst) { worst = d; if (at) *at = i; }
    }
    return worst;
}

static void check_close(const char *label, const float *got, const float *want,
                        uint64_t n, double tol) {
    if (n == 0) {
        /* tt_assert evaluates its varargs whether or not the check fires, so
         * falling through would read got[0]/want[0] out of bounds. */
        tt_assert(0, "%s: nothing to compare (n == 0)", label);
        return;
    }
    uint64_t at = 0;
    double err = max_abs_err(got, want, n, &at);
    tt_assert(err < tol,
              "%s: max abs error %.3e >= %.3e (worst at index %llu: got %.9g, want %.9g)",
              label, err, tol, (unsigned long long)at,
              (double)got[at], (double)want[at]);
    if (err < tol) fprintf(stderr, "   %-38s max |err| %.3e (tol %.0e)\n", label, err, tol);
}

/* --------------------------------------------------------------------
 * Fixture handles, opened once in main().
 * -------------------------------------------------------------------- */

static fixture g_ops, g_hyb;

#define TOL_F32 1e-5
#define TOL_MLX 1e-4

/* --------------------------------------------------------------------
 * Base ops (ops.bin, numpy)
 * -------------------------------------------------------------------- */

static void ref_rmsnorm_matches_numpy(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_ops, "rmsnorm.x", 0, &n);
    const float *w = fx_f32(&g_ops, "rmsnorm.w", n, NULL);
    const float *want = fx_f32(&g_ops, "rmsnorm.out", n, NULL);
    float eps = fx_scalar(&g_ops, "rmsnorm.eps");

    float *got = xmalloc(n * sizeof *got);
    memcpy(got, x, n * sizeof *got);
    sg_ref_rmsnorm(got, w, (uint32_t)n, eps);
    check_close("rmsnorm n=64", got, want, n, TOL_F32);
    free(got);
}

static void ref_rope_matches_numpy(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_ops, "rope.x", 0, &n);
    uint32_t head_dim = fx_dim(&g_ops, "rope.head_dim");
    float theta = fx_scalar(&g_ops, "rope.theta");
    tt_assert(head_dim == n, "rope.head_dim (%u) should equal rope.x length (%llu)",
              head_dim, (unsigned long long)n);

    static const uint32_t positions[] = {0, 1, 4096};
    for (size_t p = 0; p < sizeof positions / sizeof positions[0]; p++) {
        char key[64], label[64];
        snprintf(key, sizeof key, "rope.out_pos%u", positions[p]);
        snprintf(label, sizeof label, "rope head_dim=128 pos=%u", positions[p]);
        const float *want = fx_f32(&g_ops, key, n, NULL);
        float *got = xmalloc(n * sizeof *got);
        memcpy(got, x, n * sizeof *got);
        sg_ref_rope(got, head_dim, positions[p], theta);
        check_close(label, got, want, n, TOL_F32);
        free(got);
    }
}

static void ref_matvec_bf16_matches_numpy(void) {
    uint64_t wn = 0, cols = 0;
    const uint16_t *w = fx_u16(&g_ops, "matvec_bf16.w", 0, &wn);
    const float *x = fx_f32(&g_ops, "matvec_bf16.x", 0, &cols);
    tt_assert(cols == 64, "matvec_bf16.x should be 64 wide, got %llu",
              (unsigned long long)cols);
    uint32_t rows = (uint32_t)(wn / (cols ? cols : 1));
    tt_assert(rows == 8, "matvec_bf16 should be 8x64, got %ux%llu",
              rows, (unsigned long long)cols);
    const float *want = fx_f32(&g_ops, "matvec_bf16.out", rows, NULL);
    const float *want_exact = fx_f32(&g_ops, "matvec_bf16.out_exact", rows, NULL);

    float *y = xmalloc(rows * sizeof *y);
    sg_ref_matvec_bf16(w, x, y, rows, (uint32_t)cols);
    /* Against the rounded weights this must be near-exact; against the
     * original f32 weights only the brief's quantization bound applies. */
    check_close("matvec_bf16 vs rounded weights", y, want_exact, rows, TOL_F32);
    check_close("matvec_bf16 vs f32 weights", y, want, rows, 3e-2);
    free(y);
}

static void ref_matvec_q8_matches_numpy(void) {
    uint64_t cols = 0, rows64 = 0;
    const float *x = fx_f32(&g_ops, "matvec_q8.x", 0, &cols);
    const float *want = fx_f32(&g_ops, "matvec_q8.out", 0, &rows64);
    uint32_t rows = (uint32_t)rows64;
    tt_assert(rows == 8 && cols == 64, "matvec_q8 should be 8x64, got %ux%llu",
              rows, (unsigned long long)cols);
    const float *want_exact = fx_f32(&g_ops, "matvec_q8.out_exact", rows, NULL);
    /* Q8_0: cols/32 blocks per row, 34 bytes each. Stating the exact byte
     * count here is what stops sg_ref_matvec_q8 from reading past a short
     * weight record. */
    const void *w = fx_u8(&g_ops, "matvec_q8.w", (uint64_t)rows * (cols / 32) * 34);

    float *y = xmalloc(rows * sizeof *y);
    sg_ref_matvec_q8(w, x, y, rows, (uint32_t)cols);
    check_close("matvec_q8 vs dequantized weights", y, want_exact, rows, TOL_F32);
    check_close("matvec_q8 vs f32 weights", y, want, rows, 2e-2);
    free(y);
}

static void ref_softmax_matches_numpy(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_ops, "softmax.x", 0, &n);
    const float *want = fx_f32(&g_ops, "softmax.out", n, NULL);

    float *got = xmalloc(n * sizeof *got);
    memcpy(got, x, n * sizeof *got);
    sg_ref_softmax(got, (uint32_t)n);
    check_close("softmax n=32 (with -1e30 mask entries)", got, want, n, TOL_F32);

    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        sum += got[i];
        tt_assert(got[i] >= 0.0f && !isnan(got[i]),
                  "softmax output %llu should be a non-negative number, got %g",
                  (unsigned long long)i, (double)got[i]);
    }
    tt_assert(fabs(sum - 1.0) < 1e-6, "softmax output should sum to 1, got %.9g", sum);
    free(got);
}

static void ref_swiglu_matches_numpy(void) {
    uint64_t n = 0;
    const float *g = fx_f32(&g_ops, "swiglu.gate", 0, &n);
    const float *u = fx_f32(&g_ops, "swiglu.up", n, NULL);
    const float *want = fx_f32(&g_ops, "swiglu.out", n, NULL);

    float *got = xmalloc(n * sizeof *got);
    memcpy(got, g, n * sizeof *got);
    sg_ref_swiglu(got, u, (uint32_t)n);
    check_close("swiglu n=64", got, want, n, TOL_F32);
    free(got);
}

/* --------------------------------------------------------------------
 * Hybrid tier 1: op by op against mlx
 * -------------------------------------------------------------------- */

static void ref_rmsnorm_matches_mlx(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_hyb, "rmsnorm_w.x", 0, &n);
    const float *w = fx_f32(&g_hyb, "rmsnorm_w.w", n, NULL);
    float eps = fx_scalar(&g_hyb, "rmsnorm_w.eps");
    float *got = xmalloc(n * sizeof *got);

    memcpy(got, x, n * sizeof *got);
    sg_ref_rmsnorm(got, w, (uint32_t)n, eps);
    check_close("mx.fast.rms_norm(x, w, eps)", got, fx_f32(&g_hyb, "rmsnorm_w.out", n, NULL),
                n, TOL_MLX);

    /* weight=None is the path qwen3_5.py's DeltaNet takes for q and k. */
    memcpy(got, fx_f32(&g_hyb, "rmsnorm_now.x", n, NULL), n * sizeof *got);
    sg_ref_rmsnorm(got, NULL, (uint32_t)n, 1e-6f);
    check_close("mx.fast.rms_norm(x, None, 1e-6)", got,
                fx_f32(&g_hyb, "rmsnorm_now.out", n, NULL), n, TOL_MLX);
    free(got);
}

static void ref_rope_partial_matches_mlx(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_hyb, "rope_partial.x", 0, &n);
    uint32_t head_dim = fx_dim(&g_hyb, "rope_partial.head_dim");
    uint32_t rope_dim = fx_dim(&g_hyb, "rope_partial.rope_dim");
    float theta = fx_scalar(&g_hyb, "rope_partial.theta");
    tt_assert(rope_dim < head_dim,
              "the partial-RoPE fixture must actually be partial (rope_dim %u < head_dim %u)",
              rope_dim, head_dim);

    uint64_t np = 0;
    const float *positions = fx_f32(&g_hyb, "rope_partial.pos", 0, &np);
    for (uint64_t p = 0; p < np; p++) {
        uint32_t pos = (uint32_t)(positions[p] + 0.5f);
        char key[64], label[80];
        snprintf(key, sizeof key, "rope_partial.out_pos%u", pos);
        snprintf(label, sizeof label, "partial rope %u of %u, pos=%u", rope_dim, head_dim, pos);
        const float *want = fx_f32(&g_hyb, key, n, NULL);

        float *got = xmalloc(n * sizeof *got);
        memcpy(got, x, n * sizeof *got);
        sg_ref_rope_partial(got, head_dim, rope_dim, pos, theta);
        check_close(label, got, want, n, TOL_MLX);
        /* The unrotated tail must be bit-identical to the input, not merely
         * close: "partial" means untouched, not "rotated by ~0". */
        for (uint32_t i = rope_dim; i < head_dim; i++) {
            tt_assert(got[i] == x[i],
                      "partial rope pos=%u: index %u past rope_dim must pass through "
                      "unchanged (got %.9g, want %.9g)", pos, i, (double)got[i], (double)x[i]);
        }
        free(got);
    }
}

/* Full-width RoPE (rope_dim == head_dim) pinned against mlx, not only
 * against ops.bin's numpy restatement of the same formula: sg_ref_rope is a
 * wrapper around sg_ref_rope_partial, and without this its convention would
 * rest entirely on a fixture written in the same convention as the C. */
static void ref_rope_full_matches_mlx(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_hyb, "rope_full.x", 0, &n);
    uint32_t head_dim = fx_dim(&g_hyb, "rope_full.head_dim");
    float theta = fx_scalar(&g_hyb, "rope_full.theta");
    tt_assert(head_dim == n, "rope_full.head_dim (%u) should equal rope_full.x length (%llu)",
              head_dim, (unsigned long long)n);

    uint64_t np = 0;
    const float *positions = fx_f32(&g_hyb, "rope_full.pos", 0, &np);
    for (uint64_t p = 0; p < np; p++) {
        uint32_t pos = (uint32_t)(positions[p] + 0.5f);
        char key[64], label[80];
        snprintf(key, sizeof key, "rope_full.out_pos%u", pos);
        snprintf(label, sizeof label, "full rope %u dims, pos=%u", head_dim, pos);
        const float *want = fx_f32(&g_hyb, key, n, NULL);

        float *got = xmalloc(n * sizeof *got);
        memcpy(got, x, n * sizeof *got);
        sg_ref_rope(got, head_dim, pos, theta);
        check_close(label, got, want, n, TOL_MLX);
        free(got);
    }
}

/* RoPE at the REAL checkpoint's parameters (head_dim 256, rope_dim 64, theta
 * 1e7) out to max_position_embeddings.
 *
 * Two things are asserted, and the split matters. Against the float64
 * reference (".exact_posN") sg_ref_rope_partial must be tight: that is the
 * actual correctness check, and it holds at every position. Against mlx
 * (".mlx_posN") the agreement DEGRADES with position, because mlx rounds the
 * rotation angle to f32 and at pos 262143 that costs it about 8e-3. See the
 * long comment on sg_ref_rope_partial in src/ref.c. Rather than pretend the
 * gap is not there or loosen the whole file's tolerance, the generator
 * measured it per position into "rope_real.mlx_gap" and this test asserts
 * the measurement still holds, so the number is pinned and visible instead
 * of being rediscovered during the M1 gate. */
static void ref_rope_real_dims_vs_mlx_and_exact(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_hyb, "rope_real.x", 0, &n);
    uint32_t head_dim = fx_dim(&g_hyb, "rope_real.head_dim");
    uint32_t rope_dim = fx_dim(&g_hyb, "rope_real.rope_dim");
    float theta = fx_scalar(&g_hyb, "rope_real.theta");
    tt_assert(head_dim == 256 && rope_dim == 64,
              "rope_real should use the real checkpoint dims (256/64), got %u/%u",
              head_dim, rope_dim);

    uint64_t np = 0;
    const float *positions = fx_f32(&g_hyb, "rope_real.pos", 0, &np);
    const float *gaps = fx_f32(&g_hyb, "rope_real.mlx_gap", np, NULL);

    for (uint64_t p = 0; p < np; p++) {
        uint32_t pos = (uint32_t)(positions[p] + 0.5f);
        char key[64], label[96];

        float *got = xmalloc(n * sizeof *got);
        memcpy(got, x, n * sizeof *got);
        sg_ref_rope_partial(got, head_dim, rope_dim, pos, theta);

        snprintf(key, sizeof key, "rope_real.exact_pos%u", pos);
        snprintf(label, sizeof label, "rope 64 of 256 pos=%u vs float64 exact", pos);
        check_close(label, got, fx_f32(&g_hyb, key, n, NULL), n, TOL_F32);

        snprintf(key, sizeof key, "rope_real.mlx_pos%u", pos);
        snprintf(label, sizeof label, "rope 64 of 256 pos=%u vs mlx", pos);
        const float *mlx = fx_f32(&g_hyb, key, n, NULL);
        uint64_t at = 0;
        double gap = max_abs_err(got, mlx, n, &at);
        /* The recorded gap is mlx's own f32 angle-rounding error. Allow a
         * little slack around it, but require it to still be in the same
         * place: a real regression in this function would move it by orders
         * of magnitude, not by 20 percent. */
        double recorded = (double)gaps[p];
        tt_assert(gap <= recorded * 1.2 + 1e-7,
                  "rope pos=%u: gap to mlx is %.3e, recorded %.3e -- this function "
                  "changed relative to mlx", pos, gap, recorded);
        fprintf(stderr, "   %-38s |ref - mlx| %.3e (mlx f32 angle rounding)\n",
                label, gap);
        free(got);
    }
    /* And state the headline consequence once, loudly, so it is in the log. */
    tt_assert((double)gaps[np - 1] > 1e-4,
              "the rope_real fixture is supposed to reach a position where mlx's "
              "own f32 angle rounding exceeds the file's 1e-4 tolerance; recorded "
              "max gap is only %.3e", (double)gaps[np - 1]);
}

static void ref_activations_match_mlx(void) {
    uint64_t n = 0;
    const float *x = fx_f32(&g_hyb, "silu.x", 0, &n);
    float *got = xmalloc(n * sizeof *got);

    memcpy(got, x, n * sizeof *got);
    sg_ref_silu(got, (uint32_t)n);
    check_close("nn.silu", got, fx_f32(&g_hyb, "silu.out", n, NULL), n, TOL_MLX);

    /* sigmoid via the gate op: x[i] = 1 * sigmoid(gate[i]). */
    const float *sx = fx_f32(&g_hyb, "sigmoid.x", n, NULL);
    for (uint64_t i = 0; i < n; i++) got[i] = 1.0f;
    sg_ref_gate_sigmoid(got, sx, (uint32_t)n);
    check_close("mx.sigmoid (via sg_ref_gate_sigmoid)", got,
                fx_f32(&g_hyb, "sigmoid.out", n, NULL), n, TOL_MLX);
    free(got);

    uint64_t ns = 0;
    const float *spx = fx_f32(&g_hyb, "softplus.x", 0, &ns);
    const float *spw = fx_f32(&g_hyb, "softplus.out", ns, NULL);
    float *sp = xmalloc(ns * sizeof *sp);
    for (uint64_t i = 0; i < ns; i++) sp[i] = sg_ref_softplus(spx[i]);
    check_close("nn.softplus (incl. +-30, +88)", sp, spw, ns, TOL_MLX);
    free(sp);
}

static void ref_delta_decay_matches_mlx(void) {
    uint64_t n = 0;
    const float *a_log = fx_f32(&g_hyb, "decay.a_log", 0, &n);
    const float *a = fx_f32(&g_hyb, "decay.a", n, NULL);
    const float *dt = fx_f32(&g_hyb, "decay.dt_bias", n, NULL);
    const float *want = fx_f32(&g_hyb, "decay.out", n, NULL);

    float *got = xmalloc(n * sizeof *got);
    for (uint64_t i = 0; i < n; i++) got[i] = sg_ref_delta_decay(a_log[i], a[i], dt[i]);
    check_close("compute_g = exp(-exp(A_log)*softplus(a+dt_bias))", got, want, n, TOL_MLX);
    free(got);
}

/* The GGUF decay form. blk.N.ssm_a stores -exp(A_log) rather than A_log
 * (convert_hf_to_gguf applies -torch.exp() on the way out, and llama.cpp
 * multiplies the result straight into softplus), so that path must skip the
 * exp and the negation. Reading it as A_log would give
 * exp(-exp(-exp(A_log)*sp)), which is still a plausible-looking number in
 * (0,1) and would never announce itself.
 *
 * Rather than invent a second fixture, this drives the GGUF form with
 * -exp(a_log) taken from the existing mlx decay fixture and requires it to
 * reproduce the mlx answer. So the GGUF path inherits the mlx fixture's
 * validation instead of resting on the reading of a converter script. */
static void ref_delta_decay_neg_a_matches_mlx(void) {
    uint64_t n = 0;
    const float *a_log = fx_f32(&g_hyb, "decay.a_log", 0, &n);
    const float *a = fx_f32(&g_hyb, "decay.a", n, NULL);
    const float *dt = fx_f32(&g_hyb, "decay.dt_bias", n, NULL);
    const float *want = fx_f32(&g_hyb, "decay.out", n, NULL);

    float *got = xmalloc(n * sizeof *got);
    for (uint64_t i = 0; i < n; i++) {
        float neg_a = (float)(-exp((double)a_log[i])); /* what the GGUF stores */
        tt_assert(neg_a < 0.0f, "-exp(A_log) must be negative, got %g at %llu",
                  (double)neg_a, (unsigned long long)i);
        got[i] = sg_ref_delta_decay_neg_a(neg_a, a[i], dt[i]);
        tt_assert(got[i] > 0.0f && got[i] <= 1.0f,
                  "decay must land in (0,1], got %g at %llu",
                  (double)got[i], (unsigned long long)i);
    }
    check_close("GGUF decay form exp(ssm_a*softplus) vs mlx", got, want, n, TOL_MLX);

    /* And the wrong reading must actually differ here, or this test would
     * pass for a build that ignored ssm_a_form entirely. */
    double worst_wrong = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        float wrong = sg_ref_delta_decay((float)(-exp((double)a_log[i])), a[i], dt[i]);
        double d = fabs((double)wrong - (double)want[i]);
        if (d > worst_wrong) worst_wrong = d;
    }
    tt_assert(worst_wrong > 1e-2,
              "feeding the GGUF's ssm_a into the A_log form should be clearly wrong, "
              "but the worst difference is only %.3e -- the two forms are no longer "
              "distinguishable on this fixture", worst_wrong);
    fprintf(stderr, "   %-38s wrong-form max |err| %.3e (must be large)\n",
            "GGUF vs safetensors decay form", worst_wrong);
    free(got);
}

/* The value-head to key-head map differs by source: mlx/safetensors groups
 * (h / repeat), the GGUF converter tiles (h % n_k). Both real 27B layers hit
 * this; the 2B cannot, because n_v == n_k makes every candidate agree. */
static void ssm_head_map_conventions(void) {
    /* 16 key heads feeding 48 value heads, the real 27B shape. */
    const uint32_t nk = 16, nv = 48, repeat = 3;
    bool ever_differ = false;
    for (uint32_t h = 0; h < nv; h++) {
        uint32_t grouped = sg_ssm_k_head(h, nk, nv, false);
        uint32_t tiled = sg_ssm_k_head(h, nk, nv, true);
        tt_assert(grouped == h / repeat, "grouped map for v-head %u should be %u, got %u",
                  h, h / repeat, grouped);
        tt_assert(tiled == h % nk, "tiled map for v-head %u should be %u, got %u",
                  h, h % nk, tiled);
        tt_assert(grouped < nk && tiled < nk, "both maps must stay in range for v-head %u", h);
        if (grouped != tiled) ever_differ = true;
    }
    tt_assert(ever_differ,
              "the two head-map conventions must actually differ at 16k/48v, or the "
              "flag is not testing anything");

    /* And the case that hides the bug: with n_v == n_k they coincide, which
     * is exactly why the 2B cannot expose a wrong choice. */
    for (uint32_t h = 0; h < 16; h++) {
        tt_assert(sg_ssm_k_head(h, 16, 16, false) == sg_ssm_k_head(h, 16, 16, true),
                  "with n_v == n_k the two conventions should agree at v-head %u", h);
    }
    /* Degenerate guard: n_k_heads == 0 must not divide by zero. */
    tt_assert(sg_ssm_k_head(3, 0, 0, false) == 0, "n_k_heads == 0 should return 0");
    tt_assert(sg_ssm_k_head(3, 0, 0, true) == 0, "n_k_heads == 0 should return 0 (tiled)");
}

static void ref_conv1d_causal_matches_mlx(void) {
    uint32_t ch = fx_dim(&g_hyb, "conv1d.channels");
    uint32_t ks = fx_dim(&g_hyb, "conv1d.ksize");
    const float *w = fx_f32(&g_hyb, "conv1d.w", (uint64_t)ch * ks, NULL);
    uint64_t xn = 0;
    const float *x = fx_f32(&g_hyb, "conv1d.x", 0, &xn);
    const float *want = fx_f32(&g_hyb, "conv1d.out", xn, NULL);
    const float *want_state = fx_f32(&g_hyb, "conv1d.state_out", (uint64_t)(ks - 1) * ch, NULL);
    uint32_t steps = (uint32_t)(xn / ch);

    /* mlx ran one Conv1d over concat(zeros[ks-1], x); the C op streams the
     * same thing a token at a time with the state carried between calls,
     * updated in place (state_in == state_out) to exercise the aliasing the
     * header promises is legal. */
    float *state = xcalloc((size_t)(ks - 1) * ch, sizeof *state);
    float *got = xmalloc((size_t)steps * ch * sizeof *got);
    for (uint32_t t = 0; t < steps; t++) {
        float *row = got + (size_t)t * ch;
        memcpy(row, x + (size_t)t * ch, ch * sizeof *row);
        sg_ref_conv1d_causal(row, w, state, state, ch, ks);
    }
    check_close("causal depthwise conv1d, 5 carried steps", got, want, xn, TOL_MLX);
    check_close("conv1d carried state after 5 steps", state, want_state,
                (uint64_t)(ks - 1) * ch, TOL_MLX);

    /* A NULL state_in must behave exactly like an all-zero state. */
    float *first = xmalloc(ch * sizeof *first);
    memcpy(first, x, ch * sizeof *first);
    sg_ref_conv1d_causal(first, w, NULL, NULL, ch, ks);
    check_close("conv1d with NULL state == zero state", first, want, ch, TOL_MLX);

    free(first);
    free(got);
    free(state);
}

static void ref_delta_step_matches_mlx(void) {
    uint32_t dk = fx_dim(&g_hyb, "delta_step.dk");
    uint32_t dv = fx_dim(&g_hyb, "delta_step.dv");
    uint32_t steps = fx_dim(&g_hyb, "delta_step.steps");
    const float *q = fx_f32(&g_hyb, "delta_step.q", (uint64_t)steps * dk, NULL);
    const float *k = fx_f32(&g_hyb, "delta_step.k", (uint64_t)steps * dk, NULL);
    const float *v = fx_f32(&g_hyb, "delta_step.v", (uint64_t)steps * dv, NULL);
    const float *decay = fx_f32(&g_hyb, "delta_step.decay", steps, NULL);
    const float *beta = fx_f32(&g_hyb, "delta_step.beta", steps, NULL);
    const float *s_in = fx_f32(&g_hyb, "delta_step.state_in", (uint64_t)dv * dk, NULL);
    const float *want = fx_f32(&g_hyb, "delta_step.out", (uint64_t)steps * dv, NULL);
    const float *want_state = fx_f32(&g_hyb, "delta_step.state_out", (uint64_t)dv * dk, NULL);

    float *S = xmalloc((size_t)dv * dk * sizeof *S);
    memcpy(S, s_in, (size_t)dv * dk * sizeof *S);
    float *out = xmalloc((size_t)steps * dv * sizeof *out);
    for (uint32_t t = 0; t < steps; t++) {
        sg_ref_delta_step(S, q + (size_t)t * dk, k + (size_t)t * dk, v + (size_t)t * dv,
                          beta[t], decay[t], out + (size_t)t * dv, dk, dv);
    }
    check_close("gated delta rule, 3 chained steps", out, want, (uint64_t)steps * dv, TOL_MLX);
    check_close("gated delta rule state after 3 steps", S, want_state,
                (uint64_t)dv * dk, TOL_MLX);
    free(out);
    free(S);
}

/* --------------------------------------------------------------------
 * Hybrid tier 2: whole submodules rebuilt from sg_ref_* ops
 * -------------------------------------------------------------------- */

/* One token through a gated DeltaNet layer, composed entirely of ref ops.
 * Mirrors qwen3_5.py's GatedDeltaNet.__call__ statement for statement. */
typedef struct {
    uint32_t hidden, hk, hv, dk, dv, ksize, conv_dim, key_dim, value_dim;
    float eps;
    const float *w_conv, *w_qkv, *w_z, *w_b, *w_a, *dt_bias, *a_log, *w_norm, *w_out;
    float *conv_state; /* [ksize-1, conv_dim] */
    float *S;          /* [hv, dv, dk] */
    /* scratch */
    float *qkv, *z, *bv, *av, *y;
} gdn_ctx;

static void gdn_step(gdn_ctx *g, const float *x, float *out) {
    sg_ref_matvec_f32(g->w_qkv, x, g->qkv, g->conv_dim, g->hidden);
    sg_ref_matvec_f32(g->w_z, x, g->z, g->value_dim, g->hidden);
    sg_ref_matvec_f32(g->w_b, x, g->bv, g->hv, g->hidden);
    sg_ref_matvec_f32(g->w_a, x, g->av, g->hv, g->hidden);

    sg_ref_conv1d_causal(g->qkv, g->w_conv, g->conv_state, g->conv_state,
                         g->conv_dim, g->ksize);
    sg_ref_silu(g->qkv, g->conv_dim);

    float *q = g->qkv;
    float *k = g->qkv + g->key_dim;
    float *v = g->qkv + 2 * g->key_dim;

    /* q and k are RMS-normed per key head with NO weight and eps 1e-6
     * (hardcoded in qwen3_5.py, not the config's rms_norm_eps), then
     * scaled: q by 1/head_k_dim, k by 1/sqrt(head_k_dim). */
    double inv = 1.0 / sqrt((double)g->dk);
    for (uint32_t h = 0; h < g->hk; h++) {
        float *qh = q + (size_t)h * g->dk;
        float *kh = k + (size_t)h * g->dk;
        sg_ref_rmsnorm(qh, NULL, g->dk, 1e-6f);
        sg_ref_rmsnorm(kh, NULL, g->dk, 1e-6f);
        for (uint32_t i = 0; i < g->dk; i++) {
            qh[i] = (float)((double)qh[i] * inv * inv);
            kh[i] = (float)((double)kh[i] * inv);
        }
    }

    uint32_t repeat = g->hv / g->hk;
    for (uint32_t h = 0; h < g->hv; h++) {
        uint32_t hk = h / repeat;
        float beta = sg_ref_sigmoid(g->bv[h]);
        float decay = sg_ref_delta_decay(g->a_log[h], g->av[h], g->dt_bias[h]);
        sg_ref_delta_step(g->S + (size_t)h * g->dv * g->dk,
                          q + (size_t)hk * g->dk, k + (size_t)hk * g->dk,
                          v + (size_t)h * g->dv, beta, decay,
                          g->y + (size_t)h * g->dv, g->dk, g->dv);
    }

    /* RMSNormGated: silu(z) * rms_norm(y, w, eps), per value head. */
    for (uint32_t h = 0; h < g->hv; h++) {
        float *yh = g->y + (size_t)h * g->dv;
        float *zh = g->z + (size_t)h * g->dv;
        sg_ref_rmsnorm(yh, g->w_norm, g->dv, g->eps);
        sg_ref_silu(zh, g->dv);
        for (uint32_t i = 0; i < g->dv; i++) yh[i] = (float)((double)yh[i] * (double)zh[i]);
    }

    sg_ref_matvec_f32(g->w_out, g->y, out, g->hidden, g->value_dim);
}

static void gdn_submodule_matches_mlx(void) {
    gdn_ctx g;
    memset(&g, 0, sizeof g);
    g.hidden = fx_dim(&g_hyb, "gdn.hidden");
    g.hk = fx_dim(&g_hyb, "gdn.num_k_heads");
    g.hv = fx_dim(&g_hyb, "gdn.num_v_heads");
    g.dk = fx_dim(&g_hyb, "gdn.head_k_dim");
    g.dv = fx_dim(&g_hyb, "gdn.head_v_dim");
    g.ksize = fx_dim(&g_hyb, "gdn.conv_kernel");
    g.conv_dim = fx_dim(&g_hyb, "gdn.conv_dim");
    g.eps = fx_scalar(&g_hyb, "gdn.rms_eps");
    g.key_dim = g.hk * g.dk;
    g.value_dim = g.hv * g.dv;
    tt_assert(g.conv_dim == 2 * g.key_dim + g.value_dim,
              "gdn conv_dim (%u) should be 2*key_dim + value_dim (%u)",
              g.conv_dim, 2 * g.key_dim + g.value_dim);
    tt_assert(g.hv > g.hk && g.hv % g.hk == 0,
              "the gdn fixture should exercise the value-head repeat (hv %u, hk %u)",
              g.hv, g.hk);

    g.w_conv = fx_f32(&g_hyb, "gdn.conv1d_w", (uint64_t)g.conv_dim * g.ksize, NULL);
    g.w_qkv = fx_f32(&g_hyb, "gdn.in_proj_qkv_w", (uint64_t)g.conv_dim * g.hidden, NULL);
    g.w_z = fx_f32(&g_hyb, "gdn.in_proj_z_w", (uint64_t)g.value_dim * g.hidden, NULL);
    g.w_b = fx_f32(&g_hyb, "gdn.in_proj_b_w", (uint64_t)g.hv * g.hidden, NULL);
    g.w_a = fx_f32(&g_hyb, "gdn.in_proj_a_w", (uint64_t)g.hv * g.hidden, NULL);
    g.dt_bias = fx_f32(&g_hyb, "gdn.dt_bias", g.hv, NULL);
    g.a_log = fx_f32(&g_hyb, "gdn.A_log", g.hv, NULL);
    g.w_norm = fx_f32(&g_hyb, "gdn.norm_w", g.dv, NULL);
    g.w_out = fx_f32(&g_hyb, "gdn.out_proj_w", (uint64_t)g.hidden * g.value_dim, NULL);

    g.conv_state = xcalloc((size_t)(g.ksize - 1) * g.conv_dim, sizeof(float));
    g.S = xcalloc((size_t)g.hv * g.dv * g.dk, sizeof(float));
    g.qkv = xmalloc(g.conv_dim * sizeof(float));
    g.z = xmalloc(g.value_dim * sizeof(float));
    g.bv = xmalloc(g.hv * sizeof(float));
    g.av = xmalloc(g.hv * sizeof(float));
    g.y = xmalloc(g.value_dim * sizeof(float));

    uint32_t pre = fx_dim(&g_hyb, "gdn.prefill_len");
    const float *xp = fx_f32(&g_hyb, "gdn.x_prefill", (uint64_t)pre * g.hidden, NULL);
    const float *xd = fx_f32(&g_hyb, "gdn.x_decode", g.hidden, NULL);
    const float *want_pre = fx_f32(&g_hyb, "gdn.out_prefill", (uint64_t)pre * g.hidden, NULL);
    const float *want_dec = fx_f32(&g_hyb, "gdn.out_decode", g.hidden, NULL);

    float *out = xmalloc((size_t)pre * g.hidden * sizeof *out);
    for (uint32_t t = 0; t < pre; t++) {
        gdn_step(&g, xp + (size_t)t * g.hidden, out + (size_t)t * g.hidden);
    }
    check_close("GatedDeltaNet 5-token prefill vs mlx", out, want_pre,
                (uint64_t)pre * g.hidden, TOL_MLX);

    /* Same C state, now fed the decode token: this is the test that the
     * conv tail and the delta-rule state were carried correctly. */
    float *dec = xmalloc(g.hidden * sizeof *dec);
    gdn_step(&g, xd, dec);
    check_close("GatedDeltaNet decode step vs mlx (carried state)", dec, want_dec,
                g.hidden, TOL_MLX);

    free(dec); free(out);
    free(g.y); free(g.av); free(g.bv); free(g.z); free(g.qkv);
    free(g.S); free(g.conv_state);
}

/* One token through a full-attention layer, composed entirely of ref ops.
 * Mirrors qwen3_next.py's Qwen3NextAttention.__call__. */
typedef struct {
    uint32_t hidden, n_heads, n_kv, head_dim, rope_dim, max_pos;
    float theta, eps;
    const float *w_q, *w_k, *w_v, *w_o, *w_qn, *w_kn;
    float *k_cache, *v_cache; /* [max_pos, n_kv, head_dim] */
    uint32_t used;
    /* scratch */
    float *qg, *kv_k, *kv_v, *scores, *ctx, *gate;
} attn_ctx;

static void attn_step(attn_ctx *a, const float *x, float *out) {
    uint32_t hd = a->head_dim;

    /* q_proj is 2x wide; the reshape is [n_heads, 2*head_dim] and the split
     * is on the LAST axis, so each head's gate sits immediately after that
     * head's queries -- not in one block after all the queries. */
    sg_ref_matvec_f32(a->w_q, x, a->qg, a->n_heads * hd * 2, a->hidden);
    sg_ref_matvec_f32(a->w_k, x, a->kv_k, a->n_kv * hd, a->hidden);
    sg_ref_matvec_f32(a->w_v, x, a->kv_v, a->n_kv * hd, a->hidden);

    uint32_t pos = a->used;
    for (uint32_t h = 0; h < a->n_heads; h++) {
        float *qh = a->qg + (size_t)h * 2 * hd;
        sg_ref_rmsnorm(qh, a->w_qn, hd, a->eps);
        sg_ref_rope_partial(qh, hd, a->rope_dim, pos, a->theta);
        memcpy(a->gate + (size_t)h * hd, a->qg + (size_t)h * 2 * hd + hd, hd * sizeof(float));
    }
    for (uint32_t h = 0; h < a->n_kv; h++) {
        float *kh = a->kv_k + (size_t)h * hd;
        sg_ref_rmsnorm(kh, a->w_kn, hd, a->eps);
        sg_ref_rope_partial(kh, hd, a->rope_dim, pos, a->theta);
    }

    memcpy(a->k_cache + (size_t)pos * a->n_kv * hd, a->kv_k, (size_t)a->n_kv * hd * sizeof(float));
    memcpy(a->v_cache + (size_t)pos * a->n_kv * hd, a->kv_v, (size_t)a->n_kv * hd * sizeof(float));
    a->used++;

    double scale = 1.0 / sqrt((double)hd);
    uint32_t repeat = a->n_heads / a->n_kv;
    for (uint32_t h = 0; h < a->n_heads; h++) {
        uint32_t hk = h / repeat;
        const float *qh = a->qg + (size_t)h * 2 * hd;
        for (uint32_t t = 0; t < a->used; t++) {
            const float *kt = a->k_cache + ((size_t)t * a->n_kv + hk) * hd;
            double dot = 0.0;
            for (uint32_t i = 0; i < hd; i++) dot += (double)qh[i] * (double)kt[i];
            a->scores[t] = (float)(dot * scale);
        }
        sg_ref_softmax(a->scores, a->used);
        float *ch = a->ctx + (size_t)h * hd;
        for (uint32_t i = 0; i < hd; i++) ch[i] = 0.0f;
        for (uint32_t t = 0; t < a->used; t++) {
            const float *vt = a->v_cache + ((size_t)t * a->n_kv + hk) * hd;
            float p = a->scores[t];
            for (uint32_t i = 0; i < hd; i++) ch[i] += p * vt[i];
        }
    }

    sg_ref_gate_sigmoid(a->ctx, a->gate, a->n_heads * hd);
    sg_ref_matvec_f32(a->w_o, a->ctx, out, a->hidden, a->n_heads * hd);
}

static void attention_submodule_matches_mlx(void) {
    attn_ctx a;
    memset(&a, 0, sizeof a);
    a.hidden = fx_dim(&g_hyb, "attn.hidden");
    a.n_heads = fx_dim(&g_hyb, "attn.n_heads");
    a.n_kv = fx_dim(&g_hyb, "attn.n_kv_heads");
    a.head_dim = fx_dim(&g_hyb, "attn.head_dim");
    a.rope_dim = fx_dim(&g_hyb, "attn.rope_dim");
    a.theta = fx_scalar(&g_hyb, "attn.rope_theta");
    a.eps = fx_scalar(&g_hyb, "attn.rms_eps");
    tt_assert(a.rope_dim == a.head_dim / 4,
              "attn fixture should use partial_rotary_factor 0.25 (rope_dim %u, head_dim %u)",
              a.rope_dim, a.head_dim);
    /* n_kv must be at least 2 with a repeat of at least 2. With n_kv == 1
     * every candidate head map (h / repeat, h % n_kv, constant 0) collapses
     * to the same answer and the GQA convention goes completely untested,
     * while both real checkpoints have a repeat > 1 (2B: 8/2, 27B: 24/4). */
    tt_assert(a.n_kv >= 2 && a.n_heads % a.n_kv == 0 && a.n_heads / a.n_kv >= 2,
              "the attn fixture must exercise GQA with n_kv >= 2 and repeat >= 2 "
              "(n_heads %u, n_kv %u) -- otherwise h/repeat and h%%n_kv are "
              "indistinguishable and the head map is not validated",
              a.n_heads, a.n_kv);

    a.w_q = fx_f32(&g_hyb, "attn.q_proj_w", (uint64_t)a.n_heads * a.head_dim * 2 * a.hidden, NULL);
    a.w_k = fx_f32(&g_hyb, "attn.k_proj_w", (uint64_t)a.n_kv * a.head_dim * a.hidden, NULL);
    a.w_v = fx_f32(&g_hyb, "attn.v_proj_w", (uint64_t)a.n_kv * a.head_dim * a.hidden, NULL);
    a.w_o = fx_f32(&g_hyb, "attn.o_proj_w", (uint64_t)a.hidden * a.n_heads * a.head_dim, NULL);
    a.w_qn = fx_f32(&g_hyb, "attn.q_norm_w", a.head_dim, NULL);
    a.w_kn = fx_f32(&g_hyb, "attn.k_norm_w", a.head_dim, NULL);

    uint32_t pre = fx_dim(&g_hyb, "attn.prefill_len");
    a.max_pos = pre + 1;
    uint32_t hd = a.head_dim;
    a.k_cache = xcalloc((size_t)a.max_pos * a.n_kv * hd, sizeof(float));
    a.v_cache = xcalloc((size_t)a.max_pos * a.n_kv * hd, sizeof(float));
    a.qg = xmalloc((size_t)a.n_heads * hd * 2 * sizeof(float));
    a.kv_k = xmalloc((size_t)a.n_kv * hd * sizeof(float));
    a.kv_v = xmalloc((size_t)a.n_kv * hd * sizeof(float));
    a.scores = xmalloc((size_t)a.max_pos * sizeof(float));
    a.ctx = xmalloc((size_t)a.n_heads * hd * sizeof(float));
    a.gate = xmalloc((size_t)a.n_heads * hd * sizeof(float));

    const float *xp = fx_f32(&g_hyb, "attn.x_prefill", (uint64_t)pre * a.hidden, NULL);
    const float *xd = fx_f32(&g_hyb, "attn.x_decode", a.hidden, NULL);
    const float *want_pre = fx_f32(&g_hyb, "attn.out_prefill", (uint64_t)pre * a.hidden, NULL);
    const float *want_dec = fx_f32(&g_hyb, "attn.out_decode", a.hidden, NULL);

    /* mlx ran these five tokens as one causal-masked prefill; the C side
     * streams them one at a time through a growing KV cache. Matching means
     * the causal mask and the incremental cache agree. */
    float *out = xmalloc((size_t)pre * a.hidden * sizeof *out);
    for (uint32_t t = 0; t < pre; t++) {
        attn_step(&a, xp + (size_t)t * a.hidden, out + (size_t)t * a.hidden);
    }
    check_close("Qwen3NextAttention 5-token causal prefill vs mlx", out, want_pre,
                (uint64_t)pre * a.hidden, TOL_MLX);

    float *dec = xmalloc(a.hidden * sizeof *dec);
    attn_step(&a, xd, dec);
    check_close("Qwen3NextAttention decode step vs mlx (KV cache)", dec, want_dec,
                a.hidden, TOL_MLX);

    free(dec); free(out);
    free(a.gate); free(a.ctx); free(a.scores);
    free(a.kv_v); free(a.kv_k); free(a.qg);
    free(a.v_cache); free(a.k_cache);
}

/* --------------------------------------------------------------------
 * Degenerate-input guards (no fixture; these pin the API contract).
 * -------------------------------------------------------------------- */

static void ref_ops_handle_degenerate_inputs(void) {
    /* NULL and zero-length must be no-ops, not crashes. */
    sg_ref_rmsnorm(NULL, NULL, 8, 1e-6f);
    sg_ref_softmax(NULL, 8);
    sg_ref_swiglu(NULL, NULL, 8);
    sg_ref_silu(NULL, 8);
    sg_ref_rope(NULL, 8, 0, 1e4f);
    sg_ref_matvec_f32(NULL, NULL, NULL, 1, 1);
    sg_ref_matvec_bf16(NULL, NULL, NULL, 1, 1);
    sg_ref_matvec_q8(NULL, NULL, NULL, 1, 32);
    sg_ref_conv1d_causal(NULL, NULL, NULL, NULL, 4, 4);
    sg_ref_delta_step(NULL, NULL, NULL, NULL, 0.5f, 0.5f, NULL, 4, 4);

    float one[1] = {3.0f};
    sg_ref_rmsnorm(one, NULL, 0, 1e-6f);
    tt_assert(one[0] == 3.0f, "rmsnorm with n=0 should leave the buffer alone");
    sg_ref_softmax(one, 1);
    tt_assert(one[0] == 1.0f, "softmax of a single element should be exactly 1, got %g",
              (double)one[0]);

    /* An all-mask row (every score the same large negative) must give a
     * uniform distribution, never NaN. */
    float masked[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    sg_ref_softmax(masked, 4);
    for (int i = 0; i < 4; i++) {
        tt_assert(fabs((double)masked[i] - 0.25) < 1e-6,
                  "all-masked softmax entry %d should be 0.25, got %g", i, (double)masked[i]);
    }

    /* mlx masks with finfo(dtype).min, but a surge-side mask built from
     * -INFINITY must not turn the whole row into NaN: -inf minus -inf is
     * NaN, and a NaN row would poison every later layer silently. */
    float neg_inf[3] = {-INFINITY, -INFINITY, -INFINITY};
    sg_ref_softmax(neg_inf, 3);
    for (int i = 0; i < 3; i++) {
        tt_assert(fabs((double)neg_inf[i] - 1.0 / 3.0) < 1e-6,
                  "all -inf softmax entry %d should be 1/3, got %g", i, (double)neg_inf[i]);
    }
    float pos_inf[4] = {1.0f, INFINITY, -2.0f, INFINITY};
    sg_ref_softmax(pos_inf, 4);
    tt_assert(pos_inf[0] == 0.0f && pos_inf[2] == 0.0f,
              "finite entries alongside +inf should get zero mass, got %g and %g",
              (double)pos_inf[0], (double)pos_inf[2]);
    tt_assert(fabs((double)pos_inf[1] - 0.5) < 1e-6 && fabs((double)pos_inf[3] - 0.5) < 1e-6,
              "the two +inf entries should split the mass, got %g and %g",
              (double)pos_inf[1], (double)pos_inf[3]);
    /* A row containing an ordinary large-negative mask alongside real
     * scores must still be finite and sum to 1 (the common case). */
    float mixed[4] = {2.0f, -1e30f, 1.0f, -INFINITY};
    sg_ref_softmax(mixed, 4);
    double msum = 0.0;
    for (int i = 0; i < 4; i++) {
        tt_assert(!isnan(mixed[i]), "mixed-mask softmax entry %d is NaN", i);
        msum += mixed[i];
    }
    tt_assert(fabs(msum - 1.0) < 1e-6, "mixed-mask softmax should sum to 1, got %.9g", msum);

    /* rope_dim > head_dim, and an odd rope_dim, are refused rather than
     * running off the end of the head. */
    float head[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float copy[8];
    memcpy(copy, head, sizeof head);
    sg_ref_rope_partial(head, 8, 16, 3, 1e4f);
    tt_assert(memcmp(head, copy, sizeof head) == 0,
              "rope_partial with rope_dim > head_dim should be a no-op");
    sg_ref_rope_partial(head, 8, 1, 3, 1e4f);
    tt_assert(memcmp(head, copy, sizeof head) == 0,
              "rope_partial with rope_dim < 2 should be a no-op");
    /* mlx raises "[rope] dims must be even" for an odd rotated width, so an
     * odd rope_dim must be refused here too, not half-rotated. */
    sg_ref_rope_partial(head, 8, 7, 3, 1e4f);
    tt_assert(memcmp(head, copy, sizeof head) == 0,
              "rope_partial with an odd rope_dim should be a no-op");
    sg_ref_rope(head, 8, 0, 1e4f);
    tt_assert(memcmp(head, copy, sizeof head) == 0,
              "rope at position 0 should be the identity");

    /* cols not a multiple of 32 is refused by the Q8_0 matvec. */
    float y[2] = {-1.0f, -1.0f};
    float xv[8] = {0};
    uint8_t wq[68] = {0};
    sg_ref_matvec_q8(wq, xv, y, 2, 8);
    tt_assert(y[0] == -1.0f && y[1] == -1.0f,
              "matvec_q8 with cols %% 32 != 0 should leave y untouched");
}

int main(void) {
    if (!fx_open(&g_ops, "tests/fixtures/ops.bin")) {
        fprintf(stderr, "FATAL: cannot open tests/fixtures/ops.bin "
                        "(run tests from the repo root)\n");
        return 2;
    }
    if (!fx_open(&g_hyb, "tests/fixtures/hybrid_ops.bin")) {
        fprintf(stderr, "FATAL: cannot open tests/fixtures/hybrid_ops.bin\n");
        fx_close(&g_ops);
        return 2;
    }

    tt_run("ref_rmsnorm_matches_numpy", ref_rmsnorm_matches_numpy);
    tt_run("ref_rope_matches_numpy", ref_rope_matches_numpy);
    tt_run("ref_matvec_bf16_matches_numpy", ref_matvec_bf16_matches_numpy);
    tt_run("ref_matvec_q8_matches_numpy", ref_matvec_q8_matches_numpy);
    tt_run("ref_softmax_matches_numpy", ref_softmax_matches_numpy);
    tt_run("ref_swiglu_matches_numpy", ref_swiglu_matches_numpy);

    tt_run("ref_rmsnorm_matches_mlx", ref_rmsnorm_matches_mlx);
    tt_run("ref_rope_partial_matches_mlx", ref_rope_partial_matches_mlx);
    tt_run("ref_rope_full_matches_mlx", ref_rope_full_matches_mlx);
    tt_run("ref_rope_real_dims_vs_mlx_and_exact", ref_rope_real_dims_vs_mlx_and_exact);
    tt_run("ref_activations_match_mlx", ref_activations_match_mlx);
    tt_run("ref_delta_decay_matches_mlx", ref_delta_decay_matches_mlx);
    tt_run("ref_delta_decay_neg_a_matches_mlx", ref_delta_decay_neg_a_matches_mlx);
    tt_run("ssm_head_map_conventions", ssm_head_map_conventions);
    tt_run("ref_conv1d_causal_matches_mlx", ref_conv1d_causal_matches_mlx);
    tt_run("ref_delta_step_matches_mlx", ref_delta_step_matches_mlx);

    tt_run("gdn_submodule_matches_mlx", gdn_submodule_matches_mlx);
    tt_run("attention_submodule_matches_mlx", attention_submodule_matches_mlx);

    tt_run("ref_ops_handle_degenerate_inputs", ref_ops_handle_degenerate_inputs);

    fx_close(&g_hyb);
    fx_close(&g_ops);
    return tt_report();
}
