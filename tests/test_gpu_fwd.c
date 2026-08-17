/* test_gpu_fwd.c - the Task 10 Metal decode path against the Task 8 CPU
 * reference forward, whole-model, on the mini hybrid fixture.
 *
 * This is the ungated, milliseconds-long version of the M2 gate. The oracle
 * is sg_ref_forward, not the stored mlx logits: test_ref_fwd.c already pins
 * ref.c to mlx, so asking "does the GPU compute what ref.c computes" puts any
 * divergence on exactly one side.
 *
 * tests/fixtures/mini_fwd/ ships the SAME 4-layer hybrid model twice, and
 * running both matters here for the same reason it mattered in Task 8:
 *
 *   model.safetensors  bf16 matmuls, RESIDUAL norms (+1.0 at load),
 *                      ssm_a = A_log, GROUPED value heads
 *   model.gguf         f32 matmuls, ABSOLUTE norms, ssm_a = -exp(A_log),
 *                      TILED value heads (n_v_heads 4 != n_k_heads 2)
 *
 * so between them they exercise both matvec kernels, both decay forms, both
 * value-head maps and both norm conventions on the GPU. The real 2B has
 * n_v_heads == n_k_heads and cannot expose the head map at all.
 *
 * Three properties are checked per model: per-position logit parity against
 * ref (relative, 1e-4, the same bar as the per-op tests), IDENTICAL ARGMAX at
 * every position (that is what the M2 gate actually compares), and
 * byte-identical reruns after sg_gpu_state_reset (the determinism the gate
 * rests on).
 */
#ifdef SURGE_NO_METAL

#include <stdio.h>
int main(void) {
    fprintf(stderr, "SKIP test_gpu_fwd: built with -DSURGE_NO_METAL "
                    "(Metal and the ASan/UBSan run do not mix)\n");
    return 0;
}

#else

#include "tinytest.h"
#include "../surge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MINI_DIR "tests/fixtures/mini_fwd"

static sg_gpu *g_gpu;
static double g_worst_rel = 0.0;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

static int32_t *read_ids_file(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char buf[8192];
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = '\0';
    size_t cap = 64, n = 0;
    int32_t *ids = xmalloc(cap * sizeof *ids);
    const char *p = buf;
    while (*p) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) { p++; continue; }
        if (n == cap) { cap *= 2; ids = realloc(ids, cap * sizeof *ids); }
        ids[n++] = (int32_t)v;
        p = end;
    }
    *n_out = n;
    return ids;
}

static uint32_t argmax_f32(const float *v, uint32_t n) {
    uint32_t arg = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[arg]) arg = i;
    return arg;
}

/* One model, both paths, every position. `label` names the checkpoint format
 * so a failure says which of the two conventions broke. */
static void compare_paths(const char *label, sg_model *m, const int32_t *ids,
                          size_t n_ids) {
    sg_ref_state *rs = NULL;
    sg_err e = sg_ref_state_new(m, (uint32_t)n_ids, &rs);
    tt_assert(!sg_failed(e), "%s: sg_ref_state_new: %s", label, e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    e = sg_gpu_load_model(g_gpu, m);
    tt_assert(!sg_failed(e), "%s: sg_gpu_load_model: %s", label, e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_ref_state_free(rs); return; }
    e = sg_gpu_state_new(g_gpu, m, (uint32_t)n_ids);
    tt_assert(!sg_failed(e), "%s: sg_gpu_state_new: %s", label, e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_ref_state_free(rs); return; }

    uint32_t vocab = m->cfg.vocab;
    float *gpu_all = xmalloc(n_ids * vocab * sizeof *gpu_all);
    double worst = 0.0;
    uint32_t argmax_agree = 0;

    for (uint32_t t = 0; t < n_ids; t++) {
        const float *rl = NULL, *gl = NULL;
        e = sg_ref_forward(rs, m, ids[t], t, &rl);
        tt_assert(!sg_failed(e), "%s: sg_ref_forward %u: %s", label, t, e.msg ? e.msg : "ok");
        if (sg_failed(e)) break;
        e = sg_gpu_forward(g_gpu, m, ids[t], t, &gl);
        tt_assert(!sg_failed(e), "%s: sg_gpu_forward %u: %s", label, t, e.msg ? e.msg : "ok");
        if (sg_failed(e)) break;
        memcpy(gpu_all + (size_t)t * vocab, gl, vocab * sizeof *gpu_all);

        double scale = 0.0, err = 0.0;
        for (uint32_t i = 0; i < vocab; i++) {
            double a = fabs((double)rl[i]);
            double d = fabs((double)gl[i] - (double)rl[i]);
            if (a > scale) scale = a;
            if (d > err) err = d;
        }
        double rel = (scale > 0.0) ? err / scale : err;
        if (rel > worst) worst = rel;
        if (argmax_f32(gl, vocab) == argmax_f32(rl, vocab)) argmax_agree++;
    }
    if (worst > g_worst_rel) g_worst_rel = worst;

    tt_assert(worst < 1e-4, "%s: worst relative logit gap vs ref is %.3e (bar 1e-4)",
              label, worst);
    tt_assert(argmax_agree == n_ids, "%s: argmax agrees at %u/%zu positions",
              label, argmax_agree, n_ids);

    /* Determinism: the same sequence again, through a reset state, must
     * produce the same BYTES. This is the property that makes a byte-exact
     * token gate meaningful at all -- without it a divergence could be
     * scheduling rather than arithmetic. */
    sg_gpu_state_reset(g_gpu);
    uint64_t identical = 0;
    for (uint32_t t = 0; t < n_ids; t++) {
        const float *gl = NULL;
        if (sg_failed(sg_gpu_forward(g_gpu, m, ids[t], t, &gl))) break;
        for (uint32_t i = 0; i < vocab; i++) {
            if (gl[i] == gpu_all[(size_t)t * vocab + i]) identical++;
        }
    }
    tt_assert(identical == (uint64_t)n_ids * vocab,
              "%s: rerun after sg_gpu_state_reset matched %llu/%llu logits bit-exactly",
              label, (unsigned long long)identical,
              (unsigned long long)((uint64_t)n_ids * vocab));

    free(gpu_all);
    sg_ref_state_free(rs);
}

static void mini_st_matches_ref(void) {
    size_t n_ids = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    tt_assert(ids && n_ids > 1, "read %s/ids.txt", MINI_DIR);
    if (!ids || n_ids < 2) { free(ids); return; }

    sg_st *s = NULL;
    sg_err e = sg_st_open(MINI_DIR, &s);
    tt_assert(!sg_failed(e), "sg_st_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); return; }

    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(!sg_failed(e), "sg_model_from_st: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_st_close(s); free(ids); return; }

    /* The two source conventions this checkpoint carries, asserted rather
     * than assumed, so the test cannot quietly stop covering them. */
    tt_assert(m.wtype == SG_T_BF16, "safetensors mini should have bf16 matmuls");
    tt_assert(m.norms_are_residual, "safetensors mini should have residual norms");
    tt_assert(m.ssm_a_form == SG_SSM_A_LOG, "safetensors mini should store A_log");
    tt_assert(!m.v_heads_tiled, "safetensors mini should use grouped value heads");

    compare_paths("mini/safetensors", &m, ids, n_ids);

    sg_model_free(&m);
    sg_st_close(s);
    free(ids);
}

static void mini_gguf_matches_ref(void) {
    size_t n_ids = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    tt_assert(ids && n_ids > 1, "read %s/ids.txt", MINI_DIR);
    if (!ids || n_ids < 2) { free(ids); return; }

    sg_gguf *gg = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); return; }

    sg_model m;
    e = sg_model_from_gguf(gg, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(gg); free(ids); return; }

    tt_assert(m.wtype == SG_T_F32, "gguf mini should have f32 matmuls");
    tt_assert(!m.norms_are_residual, "gguf mini should have absolute norms");
    tt_assert(m.ssm_a_form == SG_SSM_A_NEG_EXP, "gguf mini should store -exp(A_log)");
    tt_assert(m.v_heads_tiled, "gguf mini should use tiled value heads");
    tt_assert(m.cfg.n_v_heads != m.cfg.n_k_heads,
              "the gguf mini must have n_v_heads != n_k_heads or the tiled map "
              "is untested (%u vs %u)", m.cfg.n_v_heads, m.cfg.n_k_heads);

    compare_paths("mini/gguf", &m, ids, n_ids);

    sg_model_free(&m);
    sg_gguf_close(gg);
    free(ids);
}

/* M5.2: the DEFAULT KV dtype (f16, set explicitly in main() before this runs)
 * over the same mini hybrid fixture as mini_gguf_matches_ref. This is gate
 * 4's bar -- "a short greedy decode with f16 KV runs, output finite/
 * in-range/coherent" -- not mini_gguf_matches_ref's 1e-4 logit-parity bar:
 * the task brief's correctness philosophy explicitly expects fp16 KV to add
 * ~1e-3 logit noise, absorbed by argmax at the token level, so this checks
 * finiteness, in-range argmax, and logs (without hard-failing on) argmax
 * agreement against the f32 ref instead of re-litigating logit parity. */
static void mini_f16_kv_decode_coherent(void) {
    size_t n_ids = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    tt_assert(ids && n_ids > 1, "read %s/ids.txt", MINI_DIR);
    if (!ids || n_ids < 2) { free(ids); return; }

    sg_gguf *gg = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); return; }

    sg_model m;
    e = sg_model_from_gguf(gg, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(gg); free(ids); return; }

    sg_ref_state *rs = NULL;
    sg_err re = sg_ref_state_new(&m, (uint32_t)n_ids, &rs);
    tt_assert(!sg_failed(re), "sg_ref_state_new: %s", re.msg ? re.msg : "ok");

    e = sg_gpu_load_model(g_gpu, &m);
    tt_assert(!sg_failed(e), "sg_gpu_load_model: %s", e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        e = sg_gpu_state_new(g_gpu, &m, (uint32_t)n_ids);
        tt_assert(!sg_failed(e), "sg_gpu_state_new: %s", e.msg ? e.msg : "ok");
    }

    if (!sg_failed(e) && !sg_failed(re)) {
        uint32_t vocab = m.cfg.vocab;
        bool all_finite = true, all_in_range = true;
        uint32_t argmax_agree = 0;

        for (uint32_t t = 0; t < n_ids; t++) {
            const float *gl = NULL;
            e = sg_gpu_forward(g_gpu, &m, ids[t], t, &gl);
            tt_assert(!sg_failed(e), "sg_gpu_forward %u: %s", t, e.msg ? e.msg : "ok");
            if (sg_failed(e)) break;
            for (uint32_t i = 0; i < vocab; i++) if (!isfinite(gl[i])) all_finite = false;
            uint32_t arg = argmax_f32(gl, vocab);
            if (arg >= vocab) all_in_range = false;

            const float *rl = NULL;
            re = sg_ref_forward(rs, &m, ids[t], t, &rl);
            if (!sg_failed(re) && argmax_f32(rl, vocab) == arg) argmax_agree++;
        }
        tt_assert(all_finite, "f16-KV decode logits are all finite");
        tt_assert(all_in_range, "f16-KV decode argmaxes are in vocab range");
        fprintf(stderr, "   f16-KV decode: argmax agreed with the f32 ref at %u/%zu positions\n",
                argmax_agree, n_ids);
    }

    if (rs) sg_ref_state_free(rs);
    sg_model_free(&m);
    sg_gguf_close(gg);
    free(ids);
}

/* Task P2.3: the DECODE PATH's split-K wiring, on the same mini hybrid
 * fixture, run past the SG_TG*4 == 1024 threshold at which enc_attn switches
 * from k_attn_decode_f16 to the k_attn_decode_splitk_partial + _combine pair.
 *
 * The per-op gate for those two kernels lives in test_metal_ops.c and is
 * untouched; what is new and untested there is the WIRING: the hand-rolled 2D
 * dispatch inside sg_gpu_forward's open command buffer, the m/s/acc buffers
 * shared by every full-attention layer and every step, the dedicated
 * splitk_scratch, and the threshold itself.
 *
 * THE THRESHOLD IS WHAT MAKES THIS NON-VACUOUS. Both modes run the SAME kernel
 * below the threshold, so for every position with seq = pos+1 < 1024 the two
 * runs must be BYTE-IDENTICAL; at and above it they run different kernels over
 * a different partition of the same sum, so they must AGREE NUMERICALLY while
 * DIFFERING in at least one bit somewhere. Asserting both directions pins the
 * switch to exactly the documented sequence length: a wiring that never
 * dispatched split-K would fail the "differs somewhere above" half, and one
 * that dispatched it too early would fail the byte-identical half.
 *
 * Three passes over the same synthetic id sequence: incumbent (pinned with
 * SURGE_ATTN_SPLITK=0), split-K (the default), and split-K again through a
 * reset state for a byte-identical determinism rerun. */
/* > 1024 so the threshold is crossed, and chosen so that
 * n_splits = clamp(N/SG_TG, 4, 1024) == 6 is DIFFERENT from this fixture's
 * n_heads == 4. That matters: the partial is dispatched on a 2D grid
 * (x = split, y = head), and a swapped-axis bug produces plausible-looking
 * output whenever n_splits happens to equal n_heads (P2.2's own deferred-gate
 * list says so). At 1088 the two would both be 4 and this gate would be blind
 * to exactly the bug the hand-rolled dispatch is most likely to have. */
#define SPLITK_GATE_N 1600u
#define SPLITK_GATE_THRESHOLD 1024u  /* SG_TG * 4, mirrored from metal.m */

static void splitk_gate_run(sg_model *m, const int32_t *ids, uint32_t n,
                            const char *mode, float *logits_out, uint32_t *arg_out) {
    uint32_t vocab = m->cfg.vocab;
    setenv("SURGE_ATTN_SPLITK", mode, 1);
    sg_err e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (splitk=%s): %s", mode,
              e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    for (uint32_t t = 0; t < n; t++) {
        const float *gl = NULL;
        e = sg_gpu_forward(g_gpu, m, ids[t], t, &gl);
        if (sg_failed(e)) {
            tt_assert(false, "sg_gpu_forward %u (splitk=%s): %s", t, mode,
                      e.msg ? e.msg : "ok");
            return;
        }
        memcpy(logits_out + (size_t)t * vocab, gl, (size_t)vocab * sizeof *gl);
        arg_out[t] = argmax_f32(gl, vocab);
    }
}

static void mini_f16_splitk_decode_matches_incumbent(void) {
    size_t n_seed = 0;
    int32_t *seed = read_ids_file(MINI_DIR "/ids.txt", &n_seed);
    tt_assert(seed && n_seed > 1, "read %s/ids.txt", MINI_DIR);
    if (!seed || n_seed < 2) { free(seed); return; }

    sg_gguf *gg = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(seed); return; }

    sg_model m;
    e = sg_model_from_gguf(gg, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(gg); free(seed); return; }

    e = sg_gpu_load_model(g_gpu, &m);
    tt_assert(!sg_failed(e), "sg_gpu_load_model: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); free(seed); return; }

    const uint32_t n = SPLITK_GATE_N;
    uint32_t vocab = m.cfg.vocab;
    /* The fixture ships 12 ids; cycle them to reach past the threshold. The
     * content does not matter here (this compares two kernels against each
     * other, not against an oracle), only that it is in-vocab and identical
     * across the three passes. */
    int32_t *ids = xmalloc((size_t)n * sizeof *ids);
    for (uint32_t t = 0; t < n; t++) ids[t] = seed[t % n_seed];

    size_t lbytes = (size_t)n * vocab * sizeof(float);
    float *l_inc = xmalloc(lbytes), *l_sk = xmalloc(lbytes), *l_sk2 = xmalloc(lbytes);
    uint32_t *a_inc = xmalloc((size_t)n * sizeof *a_inc);
    uint32_t *a_sk = xmalloc((size_t)n * sizeof *a_sk);
    uint32_t *a_sk2 = xmalloc((size_t)n * sizeof *a_sk2);
    memset(l_inc, 0, lbytes);
    memset(l_sk, 0, lbytes);
    memset(l_sk2, 0, lbytes);

    /* How many times to REPEAT the split-K pass for the determinism check
     * below. One extra pass keeps `make check` in the seconds it lives in;
     * SURGE_SPLITK_DET_REPS=100 turns this into the 100x byte-identical rerun
     * the task's gate list asks for, at roughly a second per pass. */
    uint32_t det_reps = 1;
    const char *reps_env = getenv("SURGE_SPLITK_DET_REPS");
    if (reps_env) {
        long v = strtol(reps_env, NULL, 10);
        if (v > 0 && v < 100000) det_reps = (uint32_t)v;
    }

    splitk_gate_run(&m, ids, n, "0", l_inc, a_inc);
    splitk_gate_run(&m, ids, n, "1", l_sk, a_sk);

    /* 1. Below the threshold both modes dispatch k_attn_decode_f16, so every
     *    bit must match. A single difference here means split-K engaged
     *    earlier than documented. */
    uint32_t below_diff = 0;
    for (uint32_t t = 0; t + 1 < SPLITK_GATE_THRESHOLD && t < n; t++) {
        if (memcmp(l_inc + (size_t)t * vocab, l_sk + (size_t)t * vocab,
                   (size_t)vocab * sizeof(float)) != 0) below_diff++;
    }
    tt_assert(below_diff == 0,
              "below seq %u the two modes must be byte-identical, %u positions differ",
              SPLITK_GATE_THRESHOLD, below_diff);

    /* 2. At and above it they must DIFFER somewhere, or split-K was never
     *    dispatched and everything below this is vacuous. */
    uint32_t above_diff = 0, above_total = 0;
    for (uint32_t t = SPLITK_GATE_THRESHOLD - 1; t < n; t++) {
        above_total++;
        if (memcmp(l_inc + (size_t)t * vocab, l_sk + (size_t)t * vocab,
                   (size_t)vocab * sizeof(float)) != 0) above_diff++;
    }
    tt_assert(above_diff > 0,
              "at seq >= %u split-K must actually change the bits (0 of %u "
              "positions differ, so the split-K path never ran)",
              SPLITK_GATE_THRESHOLD, above_total);

    /* 3. The two paths must still AGREE. The metric is DELIBERATELY NOT this
     *    file's bare per-element relative ratio: task P2.0 established (and
     *    its reviewer independently confirmed against an all-double gold
     *    reference) that such a ratio explodes on a logit that lands near zero
     *    by cancellation, reporting a huge number for an error of one float
     *    ULP. Two summation orders over the same keys is exactly the case that
     *    triggers it. So this asserts what actually matters for a decode:
     *
     *      - identical argmax at every position (the token-level property),
     *      - the worst ABSOLUTE logit delta is smaller than the SMALLEST
     *        top1-top2 margin any position had, which is the M3.4 gate's
     *        robustness argument: if the largest disagreement anywhere is
     *        under the closest call anywhere, no position could have flipped,
     *        so the argmax agreement above is not luck,
     *      - the worst delta relative to the logit SCALE at its own position
     *        (max |logit| there), reported and bounded, which is the
     *        cancellation-proof version of the relative bar. */
    uint32_t arg_mismatch = 0;
    double worst_abs = 0.0, worst_scaled = 0.0, min_margin = INFINITY;
    for (uint32_t t = 0; t < n; t++) {
        if (a_inc[t] != a_sk[t]) arg_mismatch++;
        const float *a = l_inc + (size_t)t * vocab, *b = l_sk + (size_t)t * vocab;
        double scale = 0.0, top1 = -INFINITY, top2 = -INFINITY;
        for (uint32_t i = 0; i < vocab; i++) {
            double av = fabs((double)a[i]);
            if (av > scale) scale = av;
            if ((double)a[i] > top1) { top2 = top1; top1 = (double)a[i]; }
            else if ((double)a[i] > top2) { top2 = (double)a[i]; }
        }
        if (scale < 1.0) scale = 1.0;
        if (top1 - top2 < min_margin) min_margin = top1 - top2;
        for (uint32_t i = 0; i < vocab; i++) {
            double d = fabs((double)a[i] - (double)b[i]);
            if (d > worst_abs) worst_abs = d;
            if (d / scale > worst_scaled) worst_scaled = d / scale;
        }
    }
    tt_assert(arg_mismatch == 0,
              "split-K vs incumbent argmax differs at %u of %u positions",
              arg_mismatch, n);
    tt_assert(worst_abs < min_margin,
              "worst |logit delta| %.3e must stay under the smallest top1-top2 "
              "margin %.3e, or an argmax could flip", worst_abs, min_margin);
    tt_assert(worst_scaled < 1e-4,
              "split-K vs incumbent worst logit delta relative to the position's "
              "logit scale is %.3e", worst_scaled);

    /* 4. Determinism: every split-K rerun through a fresh state must be BIT
     *    identical to the first, including the m/s/acc buffers being reused by
     *    every layer and every step (a stale-buffer bug would show up as a
     *    run-to-run difference, not as a wrong-looking number). memcmp rather
     *    than float ==, per the P2.2 fix round: == calls a stable NaN a
     *    mismatch and misses a +0.0/-0.0 flip, which are the two signals that
     *    matter here. */
    uint32_t det_diff = 0;
    for (uint32_t r = 0; r < det_reps; r++) {
        splitk_gate_run(&m, ids, n, "1", l_sk2, a_sk2);
        for (uint32_t t = 0; t < n; t++) {
            if (memcmp(l_sk + (size_t)t * vocab, l_sk2 + (size_t)t * vocab,
                       (size_t)vocab * sizeof(float)) != 0) det_diff++;
        }
    }
    tt_assert(det_diff == 0, "split-K decode reruns differ at %u of %u "
              "(position, rerun) pairs over %u reruns", det_diff, n * det_reps,
              det_reps);

    fprintf(stderr, "   split-K decode: %u/%u positions differ from the incumbent "
                    "above seq %u, 0 below; worst |delta| %.3e (scaled %.3e) vs "
                    "min top1-top2 margin %.3e; argmax %u/%u agree; %u rerun(s) "
                    "byte-identical at %u/%u positions\n",
            above_diff, above_total, SPLITK_GATE_THRESHOLD, worst_abs, worst_scaled,
            min_margin, n - arg_mismatch, n, det_reps, n * det_reps - det_diff,
            n * det_reps);

    unsetenv("SURGE_ATTN_SPLITK");
    free(a_sk2); free(a_sk); free(a_inc);
    free(l_sk2); free(l_sk); free(l_inc);
    free(ids);
    sg_model_free(&m);
    sg_gguf_close(gg);
    free(seed);
}

/* Task P2.4 (written in fix round 1, review finding I1): the GQA-shared partial
 * IN THE DECODE PATH, on the same mini hybrid fixture, past the same seq 1024
 * threshold.
 *
 * WHY A BYTE-IDENTITY COMPARISON ALONE WOULD BE WORTHLESS HERE. Unlike P2.3's
 * subtest, the two kernels this one switches between are contracted to produce
 * the SAME BYTES, so "the logits match" is ALSO exactly what happens when the
 * GQA kernel is never dispatched: someone narrows the group band in
 * splitk_gqa_use, or attn_splitk_gqa stops being set, or the pipeline selection
 * regresses, and every logit still matches because the per-head kernel ran both
 * times. The gate would be green and the 4x traffic saving would be gone.
 *
 * So this asserts THREE things, and the first is the load-bearing one:
 *   1. WHICH KERNEL RAN, from sg_gpu_splitk_dispatch_counts: with the switch on,
 *      every split-K dispatch this state encoded must be a GQA dispatch and the
 *      per-head count must be exactly 0; with it off, the reverse. A selection
 *      that silently declined shows up as gqa == 0 and FAILS here.
 *   2. The POLICY, from sg_gpu_splitk_gqa_selected, which calls the same
 *      internal predicate the encoder consults: the documented group band (the
 *      real 4B 32/8 and 27B 24/4 shapes are in), repeat == 1 is out, repeat == 9
 *      is out, a non-multiple is out, n_kv_heads == 0 is out, and with the switch
 *      off EVERYTHING is out. Those five rules previously existed only as a
 *      comment.
 *   3. Only then, byte-identity: with the counters proving the two runs really
 *      did dispatch different kernels, every logit bit matching is the actual
 *      end-to-end statement of the task's contract, and it is a STRONGER bar
 *      than P2.3's A/B could use (that one changes the partition of the sum, so
 *      it changes rounding; this one must not change anything).
 *
 * Both passes run the same ids through a fresh state, with SURGE_ATTN_SPLITK
 * pinned on so the split-K path is live in both. */
static void splitk_gqa_gate_run(sg_model *m, const int32_t *ids, uint32_t n,
                                const char *gqa_mode, float *logits_out,
                                uint64_t *per_head_out, uint64_t *gqa_out,
                                bool *policy_ok) {
    uint32_t vocab = m->cfg.vocab;
    setenv("SURGE_ATTN_SPLITK", "1", 1);
    setenv("SURGE_ATTN_SPLITK_GQA", gqa_mode, 1);
    sg_err e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (gqa=%s): %s", gqa_mode,
              e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    /* The policy rules, read through the same predicate the encoder uses. `on`
     * is what every in-band shape must answer while the switch is on. */
    bool on = strcmp(gqa_mode, "1") == 0;
    *policy_ok =
        sg_gpu_splitk_gqa_selected(g_gpu, 32, 8) == on &&   /* 4B dense, repeat 4 */
        sg_gpu_splitk_gqa_selected(g_gpu, 24, 4) == on &&   /* 27B, repeat 6 */
        sg_gpu_splitk_gqa_selected(g_gpu, 4, 2) == on &&    /* this fixture, repeat 2 */
        sg_gpu_splitk_gqa_selected(g_gpu, 16, 2) == on &&   /* repeat 8, the boundary */
        !sg_gpu_splitk_gqa_selected(g_gpu, 8, 8) &&         /* repeat 1: nothing to share */
        !sg_gpu_splitk_gqa_selected(g_gpu, 18, 2) &&        /* repeat 9: past the bound */
        !sg_gpu_splitk_gqa_selected(g_gpu, 32, 5) &&        /* not a multiple */
        !sg_gpu_splitk_gqa_selected(g_gpu, 32, 0);          /* no kv heads */

    for (uint32_t t = 0; t < n; t++) {
        const float *gl = NULL;
        e = sg_gpu_forward(g_gpu, m, ids[t], t, &gl);
        if (sg_failed(e)) {
            tt_assert(false, "sg_gpu_forward %u (gqa=%s): %s", t, gqa_mode,
                      e.msg ? e.msg : "ok");
            return;
        }
        memcpy(logits_out + (size_t)t * vocab, gl, (size_t)vocab * sizeof *gl);
    }
    sg_gpu_splitk_dispatch_counts(g_gpu, per_head_out, gqa_out);
}

static void mini_f16_splitk_gqa_dispatches_and_matches(void) {
    size_t n_seed = 0;
    int32_t *seed = read_ids_file(MINI_DIR "/ids.txt", &n_seed);
    tt_assert(seed && n_seed > 1, "read %s/ids.txt", MINI_DIR);
    if (!seed || n_seed < 2) { free(seed); return; }

    sg_gguf *gg = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(seed); return; }

    sg_model m;
    e = sg_model_from_gguf(gg, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(gg); free(seed); return; }

    e = sg_gpu_load_model(g_gpu, &m);
    tt_assert(!sg_failed(e), "sg_gpu_load_model: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); free(seed); return; }

    /* This fixture is 4 query heads over 2 kv heads, i.e. repeat 2, which is
     * INSIDE the documented [2, 8] band: the GQA kernel is genuinely selectable
     * here, which is what makes assertion 1 below meaningful. */
    tt_assert(m.cfg.n_heads == 4 && m.cfg.n_kv_heads == 2,
              "the mini fixture must stay a GQA shape for this gate (got %u heads "
              "over %u kv heads); repeat 1 would make it decline by policy and the "
              "dispatch assertions below would be testing nothing",
              m.cfg.n_heads, m.cfg.n_kv_heads);

    const uint32_t n = SPLITK_GATE_N;
    uint32_t vocab = m.cfg.vocab;
    int32_t *ids = xmalloc((size_t)n * sizeof *ids);
    for (uint32_t t = 0; t < n; t++) ids[t] = seed[t % n_seed];

    size_t lbytes = (size_t)n * vocab * sizeof(float);
    float *l_off = xmalloc(lbytes), *l_on = xmalloc(lbytes);
    memset(l_off, 0, lbytes);
    memset(l_on, 0, lbytes);
    uint64_t ph_off = 0, gq_off = 0, ph_on = 0, gq_on = 0;
    bool pol_off = false, pol_on = false;

    splitk_gqa_gate_run(&m, ids, n, "0", l_off, &ph_off, &gq_off, &pol_off);
    splitk_gqa_gate_run(&m, ids, n, "1", l_on, &ph_on, &gq_on, &pol_on);

    /* 1. The positive control. Every step at or above the threshold encodes one
     *    split-K partial per full-attention layer, so both runs must have
     *    encoded the same NONZERO number of them, and each must have used
     *    exactly one kernel. */
    tt_assert(gq_on > 0,
              "SURGE_ATTN_SPLITK_GQA=1 encoded 0 GQA partial dispatches over %u "
              "positions, so the GQA kernel never ran and every comparison below "
              "is vacuous", n);
    tt_assert(ph_on == 0,
              "SURGE_ATTN_SPLITK_GQA=1 still encoded %llu per-head partial "
              "dispatches", (unsigned long long)ph_on);
    tt_assert(ph_off > 0 && gq_off == 0,
              "SURGE_ATTN_SPLITK_GQA=0 must encode only per-head partials "
              "(per_head %llu, gqa %llu)",
              (unsigned long long)ph_off, (unsigned long long)gq_off);
    tt_assert(gq_on == ph_off,
              "the two modes must dispatch the same NUMBER of split-K partials "
              "(gqa %llu vs per-head %llu); a difference means the switch changed "
              "more than which kernel runs",
              (unsigned long long)gq_on, (unsigned long long)ph_off);

    /* 2. The policy rules, including the band edges that were previously only a
     *    comment. */
    tt_assert(pol_on, "with SURGE_ATTN_SPLITK_GQA=1 the group-size policy is wrong: "
                      "an in-band shape declined or an out-of-band one was accepted");
    tt_assert(pol_off, "with SURGE_ATTN_SPLITK_GQA=0 some shape still selected the "
                       "GQA kernel");

    /* 3. Now that the counters prove different kernels ran, byte-identity is the
     *    contract itself. memcmp, not float ==, for the P2.2 reason. */
    uint32_t diff = 0;
    for (uint32_t t = 0; t < n; t++) {
        if (memcmp(l_off + (size_t)t * vocab, l_on + (size_t)t * vocab,
                   (size_t)vocab * sizeof(float)) != 0) diff++;
    }
    tt_assert(diff == 0,
              "the GQA partial must be BYTE-IDENTICAL end to end: %u of %u "
              "positions have different logit bits", diff, n);

    fprintf(stderr, "   split-K GQA decode: %llu GQA dispatches with the switch on "
                    "(0 per-head), %llu per-head with it off, logits byte-identical "
                    "at %u/%u positions\n",
            (unsigned long long)gq_on, (unsigned long long)ph_off, n - diff, n);

    unsetenv("SURGE_ATTN_SPLITK_GQA");
    unsetenv("SURGE_ATTN_SPLITK");
    free(l_on); free(l_off);
    free(ids);
    sg_model_free(&m);
    sg_gguf_close(gg);
    free(seed);
}

/* Q8_0 now loads on the Metal path (M3.2+M3.3). This env-guarded check wants
 * a real Q8_0 gguf, loads it, and greedily decodes a few tokens, asserting the
 * outputs are valid (in-vocab, finite logits) and NOT degenerate (not the same
 * token every step). The rigorous Q8_0-vs-ref numeric gate is M3.4; here the
 * bar is "loads + coherent", matching the M3.3 gate text. */
static void q8_loads_and_decodes(void) {
    const char *path = getenv("SURGE_GGUF");
    if (!path) {
        fprintf(stderr, "   SKIP: set SURGE_GGUF to a Q8_0 gguf to check the "
                        "Metal Q8_0 load + decode\n");
        return;
    }
    sg_gguf *gg = NULL;
    if (sg_failed(sg_gguf_open(path, &gg))) return;
    sg_model m;
    if (sg_failed(sg_model_from_gguf(gg, &m))) { sg_gguf_close(gg); return; }
    if (m.wtype != SG_T_Q8_0) {
        fprintf(stderr, "   SKIP: SURGE_GGUF is not a Q8_0 model (wtype %d)\n",
                (int)m.wtype);
        sg_model_free(&m);
        sg_gguf_close(gg);
        return;
    }

    sg_err e = sg_gpu_load_model(g_gpu, &m);
    tt_assert(!sg_failed(e), "sg_gpu_load_model (Q8_0): %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); return; }

    const uint32_t n_gen = 8;
    const int32_t prompt = 1;   /* any in-vocab id; we only check coherence */
    e = sg_gpu_state_new(g_gpu, &m, 1 + n_gen);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (Q8_0): %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); return; }

    const float *lg = NULL;
    e = sg_gpu_forward(g_gpu, &m, prompt, 0, &lg);
    tt_assert(!sg_failed(e), "sg_gpu_forward[0] (Q8_0): %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); return; }

    uint32_t vocab = m.cfg.vocab;
    int32_t gen[8];
    bool all_finite = true, all_in_range = true;
    for (uint32_t i = 0; i < n_gen; i++) {
        for (uint32_t k = 0; k < vocab; k++) if (!isfinite(lg[k])) all_finite = false;
        uint32_t arg = argmax_f32(lg, vocab);
        if (arg >= vocab) all_in_range = false;
        gen[i] = (int32_t)arg;
        if (i + 1 == n_gen) break;
        e = sg_gpu_forward(g_gpu, &m, (int32_t)arg, 1 + i, &lg);
        tt_assert(!sg_failed(e), "sg_gpu_forward[%u] (Q8_0): %s", i + 1, e.msg ? e.msg : "ok");
        if (sg_failed(e)) break;
    }
    tt_assert(all_finite, "Q8_0 decode logits are all finite");
    tt_assert(all_in_range, "Q8_0 decode argmaxes are in vocab range");
    /* Degenerate-output guard: a broken dequant/dispatch tends to collapse to
     * one repeated token. Require at least two distinct tokens over 8 steps. */
    uint32_t distinct = 1;
    for (uint32_t i = 1; i < n_gen; i++) {
        bool seen = false;
        for (uint32_t j = 0; j < i; j++) if (gen[j] == gen[i]) { seen = true; break; }
        if (!seen) distinct++;
    }
    tt_assert(distinct >= 2, "Q8_0 decode produced %u distinct tokens over %u steps "
              "(a single repeated token signals a broken Q8_0 path)", distinct, n_gen);
    fprintf(stderr, "   Q8_0 decode ids:");
    for (uint32_t i = 0; i < n_gen; i++) fprintf(stderr, " %d", gen[i]);
    fprintf(stderr, "\n");

    sg_model_free(&m);
    sg_gguf_close(gg);
}

int main(void) {
    sg_err e = sg_gpu_init(&g_gpu);
    if (sg_failed(e)) {
        fprintf(stderr, "SKIP test_gpu_fwd: %s\n", e.msg);
        return 0;
    }

    /* M5.2 changed sg_gpu_state_new's DEFAULT KV dtype to f16. Pin the
     * pre-existing M2 gate to f32 explicitly so it keeps comparing against
     * the exact same combined-buffer k_attn_decode path it always has,
     * regardless of the new default; see gate 3 in the M5.2 task brief. */
    setenv("SURGE_KV_DTYPE", "f32", 1);
    tt_run("mini_st_matches_ref", mini_st_matches_ref);
    tt_run("mini_gguf_matches_ref", mini_gguf_matches_ref);
    tt_run("q8_loads_and_decodes", q8_loads_and_decodes);

    /* M5.2's new default path: no override, but set explicitly for clarity
     * and so this test does not silently start testing f32 again if the
     * default ever changes back. */
    setenv("SURGE_KV_DTYPE", "f16", 1);
    tt_run("mini_f16_kv_decode_coherent", mini_f16_kv_decode_coherent);
    /* P2.3: the split-K decode wiring, which only exists on the f16 path. */
    tt_run("mini_f16_splitk_decode_matches_incumbent",
           mini_f16_splitk_decode_matches_incumbent);
    /* P2.4: WHICH split-K partial the decode path selected, plus the group-size
     * policy and the end-to-end byte-identity that only means something once the
     * first two hold. */
    tt_run("mini_f16_splitk_gqa_dispatches_and_matches",
           mini_f16_splitk_gqa_dispatches_and_matches);
    unsetenv("SURGE_KV_DTYPE");

    fprintf(stderr, "worst relative logit gap vs ref: %.3e\n", g_worst_rel);
    sg_gpu_free(g_gpu);
    return tt_report();
}

#endif /* SURGE_NO_METAL */
