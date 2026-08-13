/* test_kv.c - src/kv.c: pure size math, f16 round-trip, and the sg_kv object
 * over a malloc-backed allocator. Runs under a plain `make check` (no Metal,
 * no GPU) and under `make debug` (ASan/UBSan, SURGE_NO_METAL).
 *
 * The 16 GiB K+V figure is asserted by sg_kv_bytes MATH ONLY: nothing here
 * allocates it. sg_kv_new is exercised at a tiny cap (a few KB) for the
 * object tests; an env-gated path (SURGE_KV_ALLOC=1) allocates the real
 * 27B-shaped cache once on the box, but the default run does not.
 */

#include "surge.h"
#include "tinytest.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* --------------------------------------------------------------------
 * A malloc-backed allocation backend. The "handle" is the host pointer
 * itself (identity host_fn), and alloc zero-fills like sg_gpu_alloc.
 * -------------------------------------------------------------------- */
static uint64_t mock_live = 0;   /* outstanding allocations, to catch leaks */

static sg_err mock_alloc(sg_gpu *g, uint64_t nbytes, void **buf, void **host) {
    (void)g;
    *buf = NULL;
    if (host) *host = NULL;
    if (nbytes == 0) return (sg_err){"mock: zero-length alloc"};
    void *p = calloc(1, (size_t)nbytes);
    if (!p) return (sg_err){"mock: out of memory"};
    *buf = p;
    if (host) *host = p;
    mock_live++;
    return SG_OK;
}
static void mock_free(void *buf) {
    if (buf) { mock_live--; free(buf); }
}
static void *mock_host(void *buf) { return buf; }

/* --------------------------------------------------------------------
 * The verified 27B config (see the M5 plan / model_qwen.c). Only the fields
 * sg_kv reads have to be right; the rest are set for realism.
 * -------------------------------------------------------------------- */
static sg_cfg cfg_27b(void) {
    sg_cfg c;
    memset(&c, 0, sizeof c);
    c.n_layers = 64;
    c.n_heads = 24;
    c.n_kv_heads = 4;
    c.head_dim = 256;
    c.hidden = 5120;
    c.vocab = 151936;
    c.rope_dim = 64;
    c.rope_theta = 1000000.0f;
    c.rms_eps = 1e-6f;
    c.full_attn_interval = 4;
    c.n_k_heads = 16;
    c.head_k_dim = 128;
    c.n_v_heads = 48;
    c.head_v_dim = 128;
    c.conv_kernel = 4;
    return c;
}

/* A small config with both layer kinds, for the object tests. 8 layers,
 * interval 4 -> attn at layers 3 and 7, DeltaNet everywhere else. */
static sg_cfg cfg_small(void) {
    sg_cfg c;
    memset(&c, 0, sizeof c);
    c.n_layers = 8;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 8;         /* kv_width = 16 */
    c.hidden = 32;
    c.vocab = 100;
    c.rope_dim = 8;
    c.rope_theta = 10000.0f;
    c.rms_eps = 1e-6f;
    c.full_attn_interval = 4;
    c.n_k_heads = 2;
    c.head_k_dim = 4;
    c.n_v_heads = 4;
    c.head_v_dim = 4;       /* conv_dim = 2*(2*4)+(4*4) = 32 */
    c.conv_kernel = 3;      /* conv tail rows = 2 */
    return c;
}

/* Task P1: a DENSE config, full_attn_interval == 1 so EVERY layer is
 * full-attention and none is DeltaNet -- the shape a plain qwen3 GGUF (e.g.
 * Qwen3-4B-Instruct-2507) loads into. Deliberately carries NO DeltaNet dims
 * (n_k_heads/n_v_heads/head_k_dim/head_v_dim/conv_kernel all left at their
 * memset 0), matching a real dense model, which has none. This exists to
 * make concrete the P1 brief's "the downstream machinery already tolerates
 * this" claim about kv_is_attn and sg_kv_new's n_gdn==0 guard, rather than
 * leaving it as an unverified reading of kv.c. */
static sg_cfg cfg_dense(void) {
    sg_cfg c;
    memset(&c, 0, sizeof c);
    c.n_layers = 6;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.head_dim = 8;
    c.hidden = 32;
    c.vocab = 100;
    c.rope_dim = 8;          /* full rotary on this fixture's head_dim */
    c.rope_theta = 5000000.0f;
    c.rms_eps = 1e-6f;
    c.full_attn_interval = 1;
    c.attn_output_gate = false;
    return c;
}

/* --------------------------------------------------------------------
 * (1) Pure size math -- the gate numbers, WITHOUT allocating.
 * -------------------------------------------------------------------- */
static void test_sizes(void) {
    sg_cfg c = cfg_27b();

    /* K+V, cap 262144, f16 == exactly 16.0 GiB. */
    tt_assert(sg_kv_bytes(&c, 262144, SG_T_F16) == 17179869184ULL,
              "27B K+V f16 @262144 = %llu, want 17179869184",
              (unsigned long long)sg_kv_bytes(&c, 262144, SG_T_F16));

    /* K+V, cap 131072, f16 == exactly 8.0 GiB. */
    tt_assert(sg_kv_bytes(&c, 131072, SG_T_F16) == 8589934592ULL,
              "27B K+V f16 @131072 = %llu, want 8589934592",
              (unsigned long long)sg_kv_bytes(&c, 131072, SG_T_F16));

    /* f32 K+V is exactly twice the f16 size. */
    tt_assert(sg_kv_bytes(&c, 262144, SG_T_F32) == 2ULL * 17179869184ULL,
              "27B K+V f32 @262144 = %llu, want 34359738368",
              (unsigned long long)sg_kv_bytes(&c, 262144, SG_T_F32));

    /* DeltaNet fixed state == 48*((4-1)*10240 + 48*128*128)*4. */
    uint64_t want_dn = 48ULL * ((4 - 1) * 10240 + 48 * 128 * 128) * 4;
    tt_assert(want_dn == 156893184ULL, "sanity: deltanet formula = %llu",
              (unsigned long long)want_dn);
    tt_assert(sg_kv_state_bytes(&c) == want_dn,
              "27B DeltaNet state = %llu, want %llu",
              (unsigned long long)sg_kv_state_bytes(&c),
              (unsigned long long)want_dn);

    /* Invalid dtype -> 0 (pure math never crashes). */
    tt_assert(sg_kv_bytes(&c, 262144, SG_T_Q8_0) == 0, "q8_0 KV must be 0");

    /* cap policy in the pure math: > max -> 0 (module hard-rejects a larger
     * cap); cap 0 -> 0 (0 positions; the default is sg_kv_new's job). */
    tt_assert(sg_kv_bytes(&c, SG_KV_CAP_MAX + 1, SG_T_F16) == 0, "cap>max -> 0");
    tt_assert(sg_kv_bytes(&c, 0, SG_T_F16) == 0, "cap 0 -> 0");

    /* Overflow / implausible-width guards return 0, never a wrapped size. */
    sg_cfg wide = c;
    wide.head_dim = 0x1000000;      /* kv_width = 4*2^24 > 1<<24 */
    tt_assert(sg_kv_bytes(&wide, 1024, SG_T_F16) == 0, "implausible kv_width -> 0");
    sg_cfg wconv = c;
    wconv.head_k_dim = 0x1000000;   /* key_dim = 16*2^24; 2*key_dim would wrap */
    tt_assert(sg_kv_state_bytes(&wconv) == 0, "implausible conv width -> 0");

    /* Component consistency: total logged by sg_kv_new (K+V + state) is just
     * these two sums; check they add without surprise. */
    tt_assert(sg_kv_bytes(&c, 131072, SG_T_F16) + sg_kv_state_bytes(&c)
                  == 8589934592ULL + 156893184ULL,
              "grand total mismatch");
}

/* --------------------------------------------------------------------
 * (2) f16 round-trip, round-to-nearest-even.
 * -------------------------------------------------------------------- */
static bool is_f16_nan(uint16_t h) {
    return ((h >> 10) & 0x1F) == 0x1F && (h & 0x3FF) != 0;
}

static void test_f16_golden(void) {
    struct { float f; uint16_t h; } g[] = {
        { 0.0f,       0x0000 },
        { -0.0f,      0x8000 },
        { 1.0f,       0x3C00 },
        { -1.0f,      0xBC00 },
        { 2.0f,       0x4000 },
        { 0.5f,       0x3800 },
        { 65504.0f,   0x7BFF },   /* largest finite half */
        { 5.9604645e-8f, 0x0001 }, /* smallest positive subnormal (2^-24) */
        { 6.1035156e-5f, 0x0400 }, /* smallest positive normal (2^-14) */
    };
    for (size_t i = 0; i < sizeof g / sizeof g[0]; i++) {
        uint16_t got = sg_f32_to_f16(g[i].f);
        tt_assert(got == g[i].h, "f32->f16(%g) = 0x%04X, want 0x%04X",
                  (double)g[i].f, got, g[i].h);
        /* and the reverse for the finite ones */
        float back = sg_f16_to_f32(g[i].h);
        tt_assert(back == g[i].f, "f16->f32(0x%04X) = %g, want %g",
                  g[i].h, (double)back, (double)g[i].f);
    }
    /* overflow flushes to Inf; 2^-25 (exact half of the smallest subnormal)
     * ties to even -> 0. */
    tt_assert(sg_f32_to_f16(70000.0f) == 0x7C00, "70000 -> +Inf");
    tt_assert(sg_f32_to_f16(-70000.0f) == 0xFC00, "-70000 -> -Inf");
    tt_assert(sg_f32_to_f16(ldexpf(1.0f, -25)) == 0x0000, "2^-25 ties to 0");
}

/* Exhaustive: every representable half (all 65536 patterns, NaN excluded)
 * survives f16 -> f32 -> f16 unchanged. This pins both directions and every
 * subnormal/zero/Inf edge without a fixture. */
static void test_f16_roundtrip_exhaustive(void) {
    int mismatches = 0;
    for (uint32_t u = 0; u < 65536u; u++) {
        uint16_t h = (uint16_t)u;
        if (is_f16_nan(h)) continue;
        float f = sg_f16_to_f32(h);
        uint16_t h2 = sg_f32_to_f16(f);
        if (h2 != h) { mismatches++; if (mismatches <= 4)
            fprintf(stderr, "  roundtrip 0x%04X -> %g -> 0x%04X\n", h, (double)f, h2); }
    }
    tt_assert(mismatches == 0, "%d f16 round-trip mismatches", mismatches);
}

#if defined(__FLT16_MANT_DIG__)
/* Cross-check against the compiler's hardware _Float16 cast (same IEEE-754
 * binary16, same RNE Metal's `half` uses) on many f32 values, finite only. */
static uint16_t hw_f32_to_f16(float x) {
    _Float16 hf = (_Float16)x;
    uint16_t b;
    memcpy(&b, &hf, 2);
    return b;
}
static float hw_f16_to_f32(uint16_t b) {
    _Float16 hf;
    memcpy(&hf, &b, 2);
    return (float)hf;
}
static void test_f16_vs_hardware(void) {
    uint64_t st = 0x2545F4914F6CDD1DULL;   /* xorshift64 seed */
    int fwd_bad = 0, rev_bad = 0;
    for (int i = 0; i < 400000; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        uint32_t bits = (uint32_t)st;
        float x;
        memcpy(&x, &bits, 4);
        if (x != x) continue;                  /* skip NaN (payload differs) */
        uint16_t mine = sg_f32_to_f16(x), hw = hw_f32_to_f16(x);
        if (mine != hw) { fwd_bad++; if (fwd_bad <= 4)
            fprintf(stderr, "  fwd %g: mine 0x%04X hw 0x%04X\n", (double)x, mine, hw); }
    }
    for (uint32_t u = 0; u < 65536u; u++) {
        uint16_t h = (uint16_t)u;
        if (is_f16_nan(h)) continue;
        float mine = sg_f16_to_f32(h), hw = hw_f16_to_f32(h);
        uint32_t mb, hb;
        memcpy(&mb, &mine, 4); memcpy(&hb, &hw, 4);
        if (mb != hb) { rev_bad++; if (rev_bad <= 4)
            fprintf(stderr, "  rev 0x%04X: mine %g hw %g\n", h, (double)mine, (double)hw); }
    }
    tt_assert(fwd_bad == 0, "%d f32->f16 disagreements with hardware", fwd_bad);
    tt_assert(rev_bad == 0, "%d f16->f32 disagreements with hardware", rev_bad);
}
#endif

/* "store then read back == f32 -> f16 -> f32", actually storing into a K
 * buffer of a real sg_kv (f16 dtype) via the malloc backend. */
static void test_f16_store_readback(void) {
    sg_cfg c = cfg_small();
    sg_kv *kv = NULL;
    sg_err e = sg_kv_new(NULL, &c, 4, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "sg_kv_new f16 small: %s",
              e.msg ? e.msg : "ok");
    if (!kv) return;

    uint16_t *kbuf = (uint16_t *)sg_kv_k(kv, 3);   /* layer 3 is full-attn */
    tt_assert(kbuf != NULL, "attn layer 3 K buffer present");
    if (kbuf) {
        float xs[] = { 0.1f, -3.14159f, 1.0f/3.0f, 123.456f, 1e-3f };
        for (size_t i = 0; i < sizeof xs / sizeof xs[0]; i++) {
            kbuf[i] = sg_f32_to_f16(xs[i]);
            float back = sg_f16_to_f32(kbuf[i]);
            float ref  = sg_f16_to_f32(sg_f32_to_f16(xs[i]));
            uint32_t bb, rb;
            memcpy(&bb, &back, 4); memcpy(&rb, &ref, 4);
            tt_assert(bb == rb, "store/read %g: 0x%08X vs 0x%08X",
                      (double)xs[i], bb, rb);
        }
    }
    sg_kv_free(kv);
}

/* --------------------------------------------------------------------
 * (3) sg_kv object: getters, advance, reset. Tiny cap -> a few KB.
 * -------------------------------------------------------------------- */
static void test_object(void) {
    sg_cfg c = cfg_small();
    sg_kv *kv = NULL;
    sg_err e = sg_kv_new(NULL, &c, 16, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "sg_kv_new small: %s", e.msg ? e.msg : "ok");
    if (!kv) return;

    tt_assert(sg_kv_cap(kv) == 16, "cap = %u", sg_kv_cap(kv));
    tt_assert(sg_kv_used(kv) == 0, "used starts 0");

    /* Layer-kind routing: attn layers 3 and 7 carry K/V and no conv/S;
     * DeltaNet layers carry conv/S and no K/V. */
    for (uint32_t l = 0; l < c.n_layers; l++) {
        bool attn = ((l + 1) % c.full_attn_interval) == 0;
        if (attn) {
            tt_assert(sg_kv_k(kv, l) && sg_kv_v(kv, l),
                      "attn layer %u has K and V", l);
            tt_assert(!sg_kv_conv(kv, l) && !sg_kv_s(kv, l),
                      "attn layer %u has no conv/S", l);
        } else {
            tt_assert(!sg_kv_k(kv, l) && !sg_kv_v(kv, l),
                      "deltanet layer %u has no K/V", l);
            tt_assert(sg_kv_conv(kv, l) && sg_kv_s(kv, l),
                      "deltanet layer %u has conv and S", l);
        }
    }
    /* Out-of-range layer -> NULL, no crash. */
    tt_assert(!sg_kv_k(kv, c.n_layers) && !sg_kv_conv(kv, c.n_layers),
              "out-of-range layer getter is NULL");

    /* advance: in order, monotonic, rejects overflow past cap. */
    tt_assert(!sg_failed(sg_kv_advance(kv, 10)), "advance 10");
    tt_assert(sg_kv_used(kv) == 10, "used = 10");
    tt_assert(!sg_failed(sg_kv_advance(kv, 6)), "advance 6 -> cap");
    tt_assert(sg_kv_used(kv) == 16, "used = 16 (== cap)");
    tt_assert(sg_failed(sg_kv_advance(kv, 1)), "advance past cap rejected");
    tt_assert(sg_kv_used(kv) == 16, "used unchanged after rejected advance");
    /* a large n that would overflow the sum is rejected, not wrapped. */
    sg_kv_reset(kv);
    tt_assert(sg_kv_used(kv) == 0, "reset -> used 0");
    tt_assert(sg_failed(sg_kv_advance(kv, 0xFFFFFFFFu)), "huge advance rejected");
    tt_assert(sg_kv_used(kv) == 0, "used unchanged after huge advance");

    /* reset zeroes conv + S and LEAVES K/V. Dirty everything first. */
    (void)sg_kv_advance(kv, 5);
    uint8_t *conv0 = (uint8_t *)sg_kv_conv(kv, 0);
    uint8_t *s0    = (uint8_t *)sg_kv_s(kv, 0);
    uint16_t *k3   = (uint16_t *)sg_kv_k(kv, 3);
    memset(conv0, 0xAB, 64);    /* conv tail = 2*32 f32 = 256 B; poke 64 */
    memset(s0,    0xCD, 64);
    k3[0] = 0xBEEF; k3[1] = 0x1234;
    sg_kv_reset(kv);
    tt_assert(sg_kv_used(kv) == 0, "reset used 0 (again)");
    int conv_nonzero = 0, s_nonzero = 0;
    for (int i = 0; i < 64; i++) { if (conv0[i]) conv_nonzero++; if (s0[i]) s_nonzero++; }
    tt_assert(conv_nonzero == 0, "reset zeroed conv (%d nonzero)", conv_nonzero);
    tt_assert(s_nonzero == 0, "reset zeroed S (%d nonzero)", s_nonzero);
    tt_assert(k3[0] == 0xBEEF && k3[1] == 0x1234, "reset left K untouched");

    sg_kv_free(kv);
}

/* Task P1: full_attn_interval == 1 (dense, every layer full-attention, zero
 * DeltaNet layers) end to end through sg_kv_new -- kv_is_attn's
 * (layer+1)%1==0 is true for every layer, and sg_kv_new's n_gdn>0 block
 * (which would otherwise demand the DeltaNet dims) never runs since n_gdn is
 * 0. cfg_dense() carries no DeltaNet dims at all, matching a real dense
 * checkpoint. */
static void test_dense_all_attention(void) {
    sg_cfg c = cfg_dense();
    sg_kv *kv = NULL;
    sg_err e = sg_kv_new(NULL, &c, 16, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "sg_kv_new dense (interval=1, no DeltaNet dims): %s",
              e.msg ? e.msg : "ok");
    if (!kv) return;

    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t l = 0; l < c.n_layers; l++) {
        bool attn = sg_kv_k(kv, l) != NULL;
        if (attn) {
            n_attn++;
            tt_assert(sg_kv_v(kv, l) != NULL, "dense layer %u has V too", l);
            tt_assert(!sg_kv_conv(kv, l) && !sg_kv_s(kv, l),
                      "dense layer %u should have no conv/S", l);
        } else {
            n_gdn++;
        }
    }
    tt_assert(n_attn == c.n_layers,
              "all %u dense layers should classify as full-attention, got %u",
              c.n_layers, n_attn);
    tt_assert(n_gdn == 0, "dense ssm/DeltaNet census should be 0, got %u", n_gdn);

    /* The pure size math agrees: nonzero K+V, exactly zero DeltaNet state. */
    tt_assert(sg_kv_state_bytes(&c) == 0,
              "a dense (all-attention) config's DeltaNet state size must be 0");
    tt_assert(sg_kv_bytes(&c, 16, SG_T_F16) > 0,
              "a dense config's K+V size must be nonzero");

    sg_kv_free(kv);
}

/* --------------------------------------------------------------------
 * (4) Rejections: cap > max, bad dtype, missing backend.
 * -------------------------------------------------------------------- */
static void test_rejections(void) {
    sg_cfg c = cfg_small();
    sg_kv *kv = NULL;

    /* cap > SG_KV_CAP_MAX hard-rejected with a clear message. */
    sg_err e = sg_kv_new(NULL, &c, SG_KV_CAP_MAX + 1, SG_T_F16, &kv);
    tt_assert(sg_failed(e) && !kv, "cap 262145 rejected");
    tt_assert(e.msg && strstr(e.msg, "cap"), "cap error mentions cap: %s",
              e.msg ? e.msg : "(null)");

    /* exactly at the max is allowed (uses the small width, ~1 MB/buf). */
    kv = NULL;
    e = sg_kv_new(NULL, &c, SG_KV_CAP_MAX, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "cap == max allowed: %s", e.msg ? e.msg : "ok");
    if (kv) { tt_assert(sg_kv_cap(kv) == SG_KV_CAP_MAX, "cap set to max"); sg_kv_free(kv); }

    /* unsupported KV dtype. */
    kv = NULL;
    e = sg_kv_new(NULL, &c, 8, SG_T_Q8_0, &kv);
    tt_assert(sg_failed(e) && !kv, "q8_0 KV dtype rejected");

    /* implausible DeltaNet width rejected before any allocation. */
    sg_cfg wconv = cfg_small();
    wconv.head_k_dim = 0x1000000;   /* key_dim = 2*2^24 > 1<<24 */
    kv = NULL;
    e = sg_kv_new(NULL, &wconv, 8, SG_T_F16, &kv);
    tt_assert(sg_failed(e) && !kv, "implausible conv_dim rejected by sg_kv_new");

    /* cap == 0 -> default. */
    kv = NULL;
    e = sg_kv_new(NULL, &c, 0, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "cap 0 -> default: %s", e.msg ? e.msg : "ok");
    if (kv) { tt_assert(sg_kv_cap(kv) == SG_KV_CAP_DEFAULT, "cap 0 -> %u",
                        sg_kv_cap(kv)); sg_kv_free(kv); }

    /* no backend registered -> clear error, no allocation. */
    sg_kv_set_backend(NULL, NULL, NULL);
    kv = NULL;
    e = sg_kv_new(NULL, &c, 8, SG_T_F16, &kv);
    tt_assert(sg_failed(e) && !kv, "missing backend rejected");
    sg_kv_set_backend(mock_alloc, mock_free, mock_host);   /* restore */
}

/* --------------------------------------------------------------------
 * (5) Env-gated real-size allocation. NOT run by default make check.
 * -------------------------------------------------------------------- */
static void test_alloc_gated(void) {
    const char *env = getenv("SURGE_KV_ALLOC");
    if (!env || env[0] == '\0' || env[0] == '0') {
        fprintf(stderr, "  (skipped; set SURGE_KV_ALLOC=1 to run)\n");
        return;
    }
    uint32_t cap = 131072;   /* 8 GiB f16 with the real 27B shape */
    long v = atol(env);
    if (v > 1) cap = (v <= SG_KV_CAP_MAX) ? (uint32_t)v : SG_KV_CAP_MAX;

    sg_cfg c = cfg_27b();
    sg_kv *kv = NULL;
    sg_err e = sg_kv_new(NULL, &c, cap, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "real-size sg_kv_new: %s", e.msg ? e.msg : "ok");
    if (!kv) return;
    tt_assert(sg_kv_cap(kv) == cap, "cap = %u", sg_kv_cap(kv));
    tt_assert(!sg_failed(sg_kv_advance(kv, cap)), "advance to cap");
    tt_assert(sg_kv_used(kv) == cap, "used == cap");
    tt_assert(sg_failed(sg_kv_advance(kv, 1)), "advance past cap rejected");
    sg_kv_free(kv);
}

int main(void) {
    sg_kv_set_backend(mock_alloc, mock_free, mock_host);

    tt_run("sizes (16 GiB math, no alloc)", test_sizes);
    tt_run("f16 golden vectors", test_f16_golden);
    tt_run("f16 round-trip exhaustive", test_f16_roundtrip_exhaustive);
#if defined(__FLT16_MANT_DIG__)
    tt_run("f16 vs hardware _Float16", test_f16_vs_hardware);
#endif
    tt_run("f16 store/read-back in K buffer", test_f16_store_readback);
    tt_run("sg_kv object (getters/advance/reset)", test_object);
    tt_run("dense config (full_attn_interval=1, all-attention)", test_dense_all_attention);
    tt_run("rejections (cap/dtype/backend)", test_rejections);
    tt_run("env-gated real-size alloc", test_alloc_gated);

    tt_assert(mock_live == 0, "leaked %llu buffers", (unsigned long long)mock_live);
    return tt_report();
}
