/* test_attn_decode.c - src/ref.c's sg_ref_attn_decode and
 * sg_ref_attn_decode_splitk (Task P2.1): the direct and split-K CPU oracles
 * for the coming Metal split-K decode-attention kernel. Pure C, no Metal, no
 * GPU; runs under a plain `make check` and under `make debug` (ASan/UBSan,
 * SURGE_NO_METAL).
 *
 * Gates covered (task-P2.1-brief.md):
 *   1. split-K equals direct: max ABSOLUTE error < 1e-6 for n_parts in
 *      {1,2,3,7,64,257} against ragged (non-dividing) seq, including
 *      n_parts > seq (forces empty partitions), across three (n_heads,
 *      n_kv_heads, head_dim) shapes: a small heavy-GQA shape, a 1:1 shape,
 *      and the real Qwen3-4B-Instruct-2507 shape (32/8/128).
 *   2. n_parts == 1 is bit-identical to the direct function.
 *   3. GQA is actually exercised: the real 32/8 (repeat 4) shape and a
 *      repeat-1 (n_heads==n_kv_heads) shape, both asserting the GQA head
 *      MAPPING itself (not just that the numbers come out close), via the
 *      identical-query trick: two heads sharing a kv head, given the same
 *      query, must produce bit-identical output; two heads on DIFFERENT kv
 *      heads, given the same query, must produce different output (the
 *      vacuity guard -- otherwise a hk-collapsed-to-0 bug would pass the
 *      first check trivially).
 *   4. q_stride both ways (head_dim dense, 2*head_dim hybrid), proving the
 *      gate half is never read as query data: a NaN-poisoned gate half must
 *      produce the exact same, finite output as the no-gate-at-all layout.
 *   5. 100x determinism: byte-identical repeated calls, both functions.
 *   6. Bonus (not a numbered gate, but cheap and matches this project's
 *      house style): seq==0's documented all-zero output, and NULL/
 *      zero-dimension inputs' documented no-op / no-crash behavior.
 *
 * Design note on gate 2: sg_ref_attn_decode is implemented in src/ref.c as a
 * thin wrapper that calls the exact same static core sg_ref_attn_decode_
 * splitk uses, with n_parts hardcoded to 1 -- so bit-exactness at n_parts==1
 * is structural, not coincidental (same reasoning P2.0's test file used for
 * its own shared range_summary() primitive, taken one step further into the
 * production code itself). This test still exercises and pins that identity
 * directly (test_n_parts_1_bit_exact below) rather than taking the
 * implementation's word for it.
 */
#include "surge.h"
#include "tinytest.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* --------------------------------------------------------------------
 * Deterministic xorshift64 RNG (same construction as tests/test_attn_combine.c
 * and tests/test_bench.c), for reproducible synthetic q/K/V.
 * -------------------------------------------------------------------- */
static uint64_t rng_state = 0x5EED5EEDCAFEBABEULL;
static double rng_unit(void) {   /* uniform in [0, 1) */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
}
static float rng_range(float lo, float hi) {
    return (float)((double)lo + rng_unit() * (double)(hi - lo));
}
static void fill_random(float *buf, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) buf[i] = rng_range(lo, hi);
}

static float max_abs_err(const float *a, const float *b, uint32_t n) {
    float worst = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static int bit_equal(const float *a, const float *b, uint32_t n) {
    uint32_t ba, bb;
    for (uint32_t i = 0; i < n; i++) {
        memcpy(&ba, &a[i], 4);
        memcpy(&bb, &b[i], 4);
        if (ba != bb) return 0;
    }
    return 1;
}

typedef struct { uint32_t n_heads, n_kv_heads, head_dim; } shape_cfg;

/* --------------------------------------------------------------------
 * Gate 1: split-K equals direct, n_parts in {1,2,3,7,64,257}, ragged seq,
 * n_parts > seq forcing empty partitions, across three shapes including the
 * real Qwen3-4B-Instruct-2507 one (32/8/128, repeat 4).
 * -------------------------------------------------------------------- */
static float g_worst_abs_err = 0.0f;

static void check_equiv(uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                        uint32_t seq, uint32_t n_parts) {
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_n = (size_t)seq * n_kv_heads * head_dim;
    float *kc = malloc(kv_n * sizeof(float));
    float *vc = malloc(kv_n * sizeof(float));
    fill_random(kc, kv_n, -1.0f, 1.0f);
    fill_random(vc, kv_n, -3.0f, 3.0f);

    float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
    fill_random(q, (size_t)n_heads * head_dim, -1.0f, 1.0f);

    float *direct_out = malloc((size_t)n_heads * head_dim * sizeof(float));
    float *splitk_out = malloc((size_t)n_heads * head_dim * sizeof(float));
    sg_ref_attn_decode(q, kc, vc, n_heads, n_kv_heads, head_dim, seq, head_dim,
                       scale, direct_out);
    sg_ref_attn_decode_splitk(q, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                              head_dim, scale, n_parts, splitk_out);

    uint32_t out_n = n_heads * head_dim;
    float err = max_abs_err(direct_out, splitk_out, out_n);
    if (err > g_worst_abs_err) g_worst_abs_err = err;
    tt_assert(err < 1e-6f,
              "n_heads=%u n_kv_heads=%u head_dim=%u seq=%u n_parts=%u: "
              "worst abs err=%.3e (want <1e-6)",
              n_heads, n_kv_heads, head_dim, seq, n_parts, (double)err);
    for (uint32_t i = 0; i < out_n; i++) {
        tt_assert(isfinite(splitk_out[i]),
                  "n_heads=%u n_kv_heads=%u head_dim=%u seq=%u n_parts=%u: "
                  "splitk out[%u]=%g not finite",
                  n_heads, n_kv_heads, head_dim, seq, n_parts, i, (double)splitk_out[i]);
    }

    free(kc); free(vc); free(q); free(direct_out); free(splitk_out);
}

static void test_splitk_equals_direct(void) {
    static const shape_cfg shapes[] = {
        { 4, 1, 8 },     /* small, heavy GQA (repeat=4) */
        { 8, 8, 16 },    /* 1:1, no GQA */
        { 32, 8, 128 },  /* real Qwen3-4B-Instruct-2507 shape (repeat=4) */
    };
    /* Ragged against every n_parts below except (500, 2) (500/2==250 exactly),
     * same set P2.0's test file used for its own combine equivalence gate.
     * 37 < every n_parts in {64,257} and 101 < 257, so n_parts > seq (empty
     * partitions) is exercised repeatedly, not just once. */
    static const uint32_t seqs[] = { 37, 101, 500 };
    static const uint32_t n_parts_list[] = { 1, 2, 3, 7, 64, 257 };

    for (size_t sh = 0; sh < sizeof shapes / sizeof *shapes; sh++) {
        for (size_t si = 0; si < sizeof seqs / sizeof *seqs; si++) {
            for (size_t ni = 0; ni < sizeof n_parts_list / sizeof *n_parts_list; ni++) {
                check_equiv(shapes[sh].n_heads, shapes[sh].n_kv_heads,
                           shapes[sh].head_dim, seqs[si], n_parts_list[ni]);
            }
        }
    }
}

/* --------------------------------------------------------------------
 * Gate 2: n_parts == 1 is bit-exact vs direct.
 * -------------------------------------------------------------------- */
static void test_n_parts_1_bit_exact(void) {
    static const shape_cfg shapes[] = {
        { 4, 1, 8 }, { 8, 8, 16 }, { 32, 8, 128 }, { 3, 1, 5 },
    };
    static const uint32_t seqs[] = { 1, 5, 37, 100 };

    for (size_t sh = 0; sh < sizeof shapes / sizeof *shapes; sh++) {
        for (size_t si = 0; si < sizeof seqs / sizeof *seqs; si++) {
            uint32_t n_heads = shapes[sh].n_heads, n_kv_heads = shapes[sh].n_kv_heads,
                     head_dim = shapes[sh].head_dim, seq = seqs[si];
            float scale = 1.0f / sqrtf((float)head_dim);
            size_t kv_n = (size_t)seq * n_kv_heads * head_dim;
            float *kc = malloc(kv_n * sizeof(float));
            float *vc = malloc(kv_n * sizeof(float));
            fill_random(kc, kv_n, -1.0f, 1.0f);
            fill_random(vc, kv_n, -3.0f, 3.0f);
            float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
            fill_random(q, (size_t)n_heads * head_dim, -1.0f, 1.0f);

            float *direct_out = malloc((size_t)n_heads * head_dim * sizeof(float));
            float *splitk_out = malloc((size_t)n_heads * head_dim * sizeof(float));
            sg_ref_attn_decode(q, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                               head_dim, scale, direct_out);
            sg_ref_attn_decode_splitk(q, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                                      head_dim, scale, 1, splitk_out);

            tt_assert(bit_equal(direct_out, splitk_out, n_heads * head_dim),
                      "n_heads=%u n_kv_heads=%u head_dim=%u seq=%u: "
                      "n_parts==1 not bit-exact vs direct",
                      n_heads, n_kv_heads, head_dim, seq);

            free(kc); free(vc); free(q); free(direct_out); free(splitk_out);
        }
    }
}

/* --------------------------------------------------------------------
 * Gate 3: GQA actually exercised, both shapes, mapping proven (not just
 * numerically close).
 * -------------------------------------------------------------------- */
static void test_gqa_real_shape(void) {
    uint32_t n_heads = 32, n_kv_heads = 8, head_dim = 128, seq = 80;
    uint32_t repeat = n_heads / n_kv_heads; /* 4 */
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Distinctly-offset random data per kv head, so a wrong (or collapsed)
     * kv-head mapping changes the result. */
    float *kc = malloc((size_t)seq * n_kv_heads * head_dim * sizeof(float));
    float *vc = malloc((size_t)seq * n_kv_heads * head_dim * sizeof(float));
    for (uint32_t hk = 0; hk < n_kv_heads; hk++) {
        for (uint32_t t = 0; t < seq; t++) {
            float *kt = kc + ((size_t)t * n_kv_heads + hk) * head_dim;
            float *vt = vc + ((size_t)t * n_kv_heads + hk) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                kt[d] = rng_range(-1.0f, 1.0f) + (float)hk;
                vt[d] = rng_range(-2.0f, 2.0f) + (float)(hk * 10);
            }
        }
    }

    float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
    fill_random(q, (size_t)n_heads * head_dim, -1.0f, 1.0f);
    /* Heads 0,1,2,3 all map to kv head 0 (repeat==4): force them to share
     * head 0's exact query row. Head 4 maps to kv head 1 (a DIFFERENT kv
     * head): also force it to share head 0's query row, as the vacuity
     * guard below. */
    for (uint32_t h = 1; h < repeat; h++) {
        memcpy(q + (size_t)h * head_dim, q + 0 * head_dim, head_dim * sizeof(float));
    }
    memcpy(q + (size_t)repeat * head_dim, q + 0 * head_dim, head_dim * sizeof(float));

    float *out = malloc((size_t)n_heads * head_dim * sizeof(float));
    sg_ref_attn_decode(q, kc, vc, n_heads, n_kv_heads, head_dim, seq, head_dim,
                       scale, out);

    for (uint32_t h = 1; h < repeat; h++) {
        tt_assert(bit_equal(out + 0 * head_dim, out + (size_t)h * head_dim, head_dim),
                  "GQA real shape: heads 0 and %u share kv head 0 (repeat=%u) with an "
                  "identical query but produced different output", h, repeat);
    }

    int differs = 0;
    for (uint32_t d = 0; d < head_dim; d++) {
        if (out[0 * head_dim + d] != out[(size_t)repeat * head_dim + d]) { differs = 1; break; }
    }
    tt_assert(differs,
              "GQA real shape: head 0 (kv 0) and head %u (kv 1) got an identical query and "
              "STILL produced identical output -- hk looks hardcoded/collapsed", repeat);

    for (uint32_t i = 0; i < n_heads * head_dim; i++) {
        tt_assert(isfinite(out[i]), "GQA real shape: out[%u]=%g not finite", i, (double)out[i]);
    }

    free(kc); free(vc); free(q); free(out);
}

static void test_gqa_repeat1(void) {
    uint32_t n_heads = 6, n_kv_heads = 6, head_dim = 16, seq = 40;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Distinctly-offset data per kv head, same trick as the real-shape test
     * above, but here every head has its OWN kv head (repeat==1): give
     * every head the SAME query row and assert every pair of the n_heads
     * outputs is DIFFERENT -- proving hk==h for every h (no collapsing, no
     * off-by-one, no reversed mapping), since any such bug would make at
     * least one pair coincide. */
    float *kc = malloc((size_t)seq * n_kv_heads * head_dim * sizeof(float));
    float *vc = malloc((size_t)seq * n_kv_heads * head_dim * sizeof(float));
    for (uint32_t hk = 0; hk < n_kv_heads; hk++) {
        for (uint32_t t = 0; t < seq; t++) {
            float *kt = kc + ((size_t)t * n_kv_heads + hk) * head_dim;
            float *vt = vc + ((size_t)t * n_kv_heads + hk) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                kt[d] = rng_range(-1.0f, 1.0f) + (float)hk;
                vt[d] = rng_range(-2.0f, 2.0f) + (float)(hk * 10);
            }
        }
    }

    float *q_row = malloc((size_t)head_dim * sizeof(float));
    fill_random(q_row, head_dim, -1.0f, 1.0f);
    float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
    for (uint32_t h = 0; h < n_heads; h++) {
        memcpy(q + (size_t)h * head_dim, q_row, head_dim * sizeof(float));
    }

    float *out = malloc((size_t)n_heads * head_dim * sizeof(float));
    sg_ref_attn_decode(q, kc, vc, n_heads, n_kv_heads, head_dim, seq, head_dim,
                       scale, out);

    for (uint32_t a = 0; a < n_heads; a++) {
        for (uint32_t b = a + 1; b < n_heads; b++) {
            int differs = 0;
            for (uint32_t d = 0; d < head_dim; d++) {
                if (out[(size_t)a * head_dim + d] != out[(size_t)b * head_dim + d]) { differs = 1; break; }
            }
            tt_assert(differs,
                      "GQA repeat=1: heads %u and %u (distinct kv heads, identical query) "
                      "produced identical output -- hk!=h mapping looks broken", a, b);
        }
    }
    for (uint32_t i = 0; i < n_heads * head_dim; i++) {
        tt_assert(isfinite(out[i]), "GQA repeat=1: out[%u]=%g not finite", i, (double)out[i]);
    }

    free(kc); free(vc); free(q_row); free(q); free(out);
}

/* --------------------------------------------------------------------
 * Gate 4: q_stride both ways, gate half never read as query data.
 * -------------------------------------------------------------------- */
static void test_q_stride_gate_never_read(void) {
    uint32_t n_heads = 6, n_kv_heads = 2, head_dim = 24, seq = 90;
    float scale = 1.0f / sqrtf((float)head_dim);

    float *kc = malloc((size_t)seq * n_kv_heads * head_dim * sizeof(float));
    float *vc = malloc((size_t)seq * n_kv_heads * head_dim * sizeof(float));
    fill_random(kc, (size_t)seq * n_kv_heads * head_dim, -1.0f, 1.0f);
    fill_random(vc, (size_t)seq * n_kv_heads * head_dim, -2.0f, 2.0f);

    /* Dense q_stride==head_dim: n_heads rows of head_dim real query values. */
    float *q_dense = malloc((size_t)n_heads * head_dim * sizeof(float));
    fill_random(q_dense, (size_t)n_heads * head_dim, -1.0f, 1.0f);

    /* Hybrid q_stride==2*head_dim: identical query values in [0,head_dim)
     * per head; [head_dim, 2*head_dim) -- the gate half -- is POISONED with
     * NaN, so any accidental read of it propagates into the output and
     * fails the finiteness / bit-equality checks below. */
    uint32_t q_stride2 = 2 * head_dim;
    float *q_hybrid = malloc((size_t)n_heads * q_stride2 * sizeof(float));
    for (uint32_t h = 0; h < n_heads; h++) {
        memcpy(q_hybrid + (size_t)h * q_stride2, q_dense + (size_t)h * head_dim,
              head_dim * sizeof(float));
        for (uint32_t d = 0; d < head_dim; d++) {
            q_hybrid[(size_t)h * q_stride2 + head_dim + d] = NAN;
        }
    }

    uint32_t out_n = n_heads * head_dim;
    float *out_dense = malloc((size_t)out_n * sizeof(float));
    float *out_hybrid = malloc((size_t)out_n * sizeof(float));
    sg_ref_attn_decode(q_dense, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                       head_dim, scale, out_dense);
    sg_ref_attn_decode(q_hybrid, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                       q_stride2, scale, out_hybrid);

    for (uint32_t i = 0; i < out_n; i++) {
        tt_assert(isfinite(out_hybrid[i]),
                  "q_stride=2*head_dim direct: out[%u]=%g not finite (NaN gate half leaked in?)",
                  i, (double)out_hybrid[i]);
    }
    tt_assert(bit_equal(out_dense, out_hybrid, out_n),
              "q_stride=head_dim vs q_stride=2*head_dim (NaN-poisoned gate half) direct "
              "outputs differ -- the gate half was read as query data");

    /* Same property through split-K, several n_parts: proves the q_stride
     * offset arithmetic (h*q_stride) is correct there too, not just in the
     * n_parts==1 direct path. */
    uint32_t nps[] = { 2, 7, 23 };
    for (size_t k = 0; k < sizeof nps / sizeof *nps; k++) {
        float *out_sk = malloc((size_t)out_n * sizeof(float));
        sg_ref_attn_decode_splitk(q_hybrid, kc, vc, n_heads, n_kv_heads, head_dim,
                                  seq, q_stride2, scale, nps[k], out_sk);
        for (uint32_t i = 0; i < out_n; i++) {
            tt_assert(isfinite(out_sk[i]),
                      "q_stride=2*head_dim split-K n_parts=%u: out[%u]=%g not finite",
                      nps[k], i, (double)out_sk[i]);
        }
        float err = max_abs_err(out_dense, out_sk, out_n);
        tt_assert(err < 1e-6f,
                  "q_stride=2*head_dim split-K n_parts=%u vs dense direct: worst abs err=%.3e",
                  nps[k], (double)err);
        free(out_sk);
    }

    free(kc); free(vc); free(q_dense); free(q_hybrid); free(out_dense); free(out_hybrid);
}

/* --------------------------------------------------------------------
 * Gate 5: 100x determinism, byte-identical, both functions.
 * -------------------------------------------------------------------- */
static void test_determinism_100x(void) {
    uint32_t n_heads = 12, n_kv_heads = 3, head_dim = 20, seq = 130, n_parts = 11;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_n = (size_t)seq * n_kv_heads * head_dim;
    float *kc = malloc(kv_n * sizeof(float));
    float *vc = malloc(kv_n * sizeof(float));
    fill_random(kc, kv_n, -1.0f, 1.0f);
    fill_random(vc, kv_n, -3.0f, 3.0f);
    float *q = malloc((size_t)n_heads * head_dim * sizeof(float));
    fill_random(q, (size_t)n_heads * head_dim, -1.0f, 1.0f);

    size_t out_n = (size_t)n_heads * head_dim;
    float *first_direct = malloc(out_n * sizeof(float));
    float *first_splitk = malloc(out_n * sizeof(float));
    sg_ref_attn_decode(q, kc, vc, n_heads, n_kv_heads, head_dim, seq, head_dim,
                       scale, first_direct);
    sg_ref_attn_decode_splitk(q, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                              head_dim, scale, n_parts, first_splitk);

    float *run = malloc(out_n * sizeof(float));
    int direct_mismatches = 0, splitk_mismatches = 0;
    for (int iter = 0; iter < 100; iter++) {
        sg_ref_attn_decode(q, kc, vc, n_heads, n_kv_heads, head_dim, seq, head_dim,
                           scale, run);
        if (!bit_equal(first_direct, run, (uint32_t)out_n)) direct_mismatches++;

        sg_ref_attn_decode_splitk(q, kc, vc, n_heads, n_kv_heads, head_dim, seq,
                                  head_dim, scale, n_parts, run);
        if (!bit_equal(first_splitk, run, (uint32_t)out_n)) splitk_mismatches++;
    }
    tt_assert(direct_mismatches == 0,
              "%d/100 sg_ref_attn_decode runs diverged from the first call", direct_mismatches);
    tt_assert(splitk_mismatches == 0,
              "%d/100 sg_ref_attn_decode_splitk runs diverged from the first call", splitk_mismatches);

    free(kc); free(vc); free(q); free(first_direct); free(first_splitk); free(run);
}

/* --------------------------------------------------------------------
 * Bonus: seq==0 documented zero output, NULL/zero-dim no-op / no-crash.
 * -------------------------------------------------------------------- */
static void test_seq_zero_is_defined_zero(void) {
    uint32_t n_heads = 5, n_kv_heads = 1, head_dim = 12, seq = 0;
    float scale = 1.0f / sqrtf((float)head_dim);
    float q[5 * 12];
    fill_random(q, sizeof q / sizeof *q, -1.0f, 1.0f);
    /* kc/vc are allocated (non-NULL) but never dereferenced with seq==0:
     * every partition's t0>=t1 empty-range guard fires before any K/V
     * read, so their contents (and size beyond "non-NULL") do not matter. */
    float kc_dummy[12], vc_dummy[12];
    float out[5 * 12];
    memset(out, 0xAA, sizeof out);
    sg_ref_attn_decode(q, kc_dummy, vc_dummy, n_heads, n_kv_heads, head_dim, seq,
                       head_dim, scale, out);
    for (size_t i = 0; i < sizeof out / sizeof *out; i++) {
        tt_assert(out[i] == 0.0f,
                  "seq==0: out[%zu]=%g, want 0.0 (documented all-empty convention)",
                  i, (double)out[i]);
    }
}

static void test_null_and_zero_dim_no_crash(void) {
    float dummy_q[8], dummy_kv[8];
    for (int i = 0; i < 8; i++) { dummy_q[i] = 1.0f; dummy_kv[i] = 1.0f; }

    /* out == NULL: no crash, nothing to inspect (ASan/UBSan back this). */
    sg_ref_attn_decode(dummy_q, dummy_kv, dummy_kv, 2, 1, 4, 2, 4, 1.0f, NULL);
    sg_ref_attn_decode_splitk(dummy_q, dummy_kv, dummy_kv, 2, 1, 4, 2, 4, 1.0f, 3, NULL);

    float sentinel[4] = { 1.0f, 2.0f, 3.0f, 4.0f }, check[4];

    /* head_dim == 0: out left untouched. */
    memcpy(check, sentinel, sizeof sentinel);
    sg_ref_attn_decode(dummy_q, dummy_kv, dummy_kv, 2, 1, 0, 2, 0, 1.0f, check);
    tt_assert(memcmp(check, sentinel, sizeof sentinel) == 0, "head_dim==0: out must be left untouched");

    /* n_heads == 0: out left untouched. */
    memcpy(check, sentinel, sizeof sentinel);
    sg_ref_attn_decode(dummy_q, dummy_kv, dummy_kv, 0, 1, 4, 2, 4, 1.0f, check);
    tt_assert(memcmp(check, sentinel, sizeof sentinel) == 0, "n_heads==0: out must be left untouched");

    /* n_kv_heads == 0: out left untouched (mirrors k_attn_decode_f16's own
     * dispatch guard). */
    memcpy(check, sentinel, sizeof sentinel);
    sg_ref_attn_decode(dummy_q, dummy_kv, dummy_kv, 2, 0, 4, 2, 4, 1.0f, check);
    tt_assert(memcmp(check, sentinel, sizeof sentinel) == 0, "n_kv_heads==0: out must be left untouched");

    /* NULL q with n_heads>0: contract violation, out left untouched. */
    memcpy(check, sentinel, sizeof sentinel);
    sg_ref_attn_decode(NULL, dummy_kv, dummy_kv, 2, 1, 4, 2, 4, 1.0f, check);
    tt_assert(memcmp(check, sentinel, sizeof sentinel) == 0, "NULL q: out must be left untouched");

    tt_assert(1, "NULL/zero-dim calls completed without crashing");
}

int main(void) {
    tt_run("split-K equals direct (n_parts in {1,2,3,7,64,257}, ragged seq, incl real GQA shape)",
           test_splitk_equals_direct);
    tt_run("n_parts==1 bit-exact identity", test_n_parts_1_bit_exact);
    tt_run("GQA mapping: real 32/8 shape (repeat=4)", test_gqa_real_shape);
    tt_run("GQA mapping: n_heads==n_kv_heads (repeat=1)", test_gqa_repeat1);
    tt_run("q_stride both ways: gate half never read as query data", test_q_stride_gate_never_read);
    tt_run("100x determinism (direct and split-K)", test_determinism_100x);
    tt_run("seq==0 is the documented all-empty zero output", test_seq_zero_is_defined_zero);
    tt_run("NULL/zero-dim inputs: no crash, documented no-op", test_null_and_zero_dim_no_crash);

    fprintf(stderr, "gate 1 worst observed: max absolute error=%.3e (want <1e-6)\n",
            (double)g_worst_abs_err);
    return tt_report();
}
