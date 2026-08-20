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
#define SPLITK_GATE_THRESHOLD 1024u  /* SG_TG * 4, mirrored from metal_internal.h */
#define SPLITK_CAP_GATE_TG 256u      /* SG_TG itself, mirrored (P2.6 and P2.7) */

/* ---- Task P2.7: what the GQA gates below cost, and why they cannot be short.
 *
 * P2.7 gave splitk_gqa_use a MEASURED THREADGROUP FLOOR: the GQA partial is
 * selected only when the grid it would dispatch,
 * splitk_gqa_n_splits(seq, cap) * n_kv_heads, is at least
 * SG_SPLITK_GQA_MIN_TG == 128 threadgroups, because below that the collapsed
 * grid starves this machine's 80 GPU cores and the kernel measurably LOSES.
 *
 * This fixture has 2 kv heads, the worst possible multiplier, so it needs 64
 * splits to reach 128 threadgroups, i.e. seq >= 64 * SG_TG == 16384. There is
 * no shortcut: decode is sequential, so a gate that wants a GQA dispatch has to
 * walk there. SPLITK_GATE_N == 1600 (which both GQA gates used before this task)
 * now yields 6 splits * 2 kv = 12 threadgroups and would select the PER-HEAD
 * kernel in both arms, turning the positive control and the cap-override gate
 * into two runs of the same kernel: green, and testing nothing. Raising the run
 * length is what keeps them non-vacuous, at about 15 s per pass on this fixture.
 *
 * The P2.3 subtest above deliberately keeps SPLITK_GATE_N == 1600: its threshold
 * is the split-K one (1024), which this task did not touch. */
#define MINI_KV_HEADS 2u             /* the fixture's n_kv_heads, asserted below */
#define SPLITK_GQA_MIN_TG 128u       /* SG_SPLITK_GQA_MIN_TG, mirrored from metal.m */
#define SPLITK_GQA_FLOOR_NS (SPLITK_GQA_MIN_TG / MINI_KV_HEADS)          /* 64 */
#define SPLITK_GQA_FLOOR_SEQ (SPLITK_CAP_GATE_TG * SPLITK_GQA_FLOOR_NS)  /* 16384 */
/* 66 * SG_TG: past the floor (16384) and past the cap-override divergence point
 * SG_TG * (64 + 1) == 16640, with 257 positions above it, so both GQA gates get
 * a window rather than a single position. */
#define SPLITK_GQA_GATE_N 16896u

/* A seq deep enough that the P2.7 floor is satisfied for every shape the
 * group-band cases below name (the narrowest is 4 heads over 2 kv: 256 splits *
 * 2 == 512 threadgroups), so those cases test the GROUP BAND and nothing else. */
#define SPLITK_GQA_DEEP_SEQ 65536u

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
 * All three passes run the same ids through a fresh state, with SURGE_ATTN_SPLITK
 * pinned on so the split-K path is live in each.
 *
 * TASK P4.0 ADDED THE THIRD ARM: `gqa_mode == NULL` UNSETS SURGE_ATTN_SPLITK_GQA
 * and therefore measures THE DEFAULT. That arm is the whole gate for the default
 * flip: the "0" and "1" arms pin the switch explicitly, so they would keep
 * passing unchanged whichever way the default pointed, and a flip that silently
 * failed to take effect (or a later flip back) would show up nowhere. With the
 * default ON, this arm must reproduce the "1" arm's dispatch counts EXACTLY. */
static void splitk_gqa_gate_run(sg_model *m, const int32_t *ids, uint32_t n,
                                const char *gqa_mode, float *logits_out,
                                uint64_t *per_head_out, uint64_t *gqa_out,
                                bool *policy_ok) {
    uint32_t vocab = m->cfg.vocab;
    const char *label = gqa_mode ? gqa_mode : "<unset, the default>";
    setenv("SURGE_ATTN_SPLITK", "1", 1);
    if (gqa_mode) setenv("SURGE_ATTN_SPLITK_GQA", gqa_mode, 1);
    else unsetenv("SURGE_ATTN_SPLITK_GQA");
    sg_err e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (gqa=%s): %s", label,
              e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    /* The policy rules, read through the same predicate the encoder uses. `on`
     * is what every in-band shape must answer while the switch is on. Every case
     * is asked at SPLITK_GQA_DEEP_SEQ so that P2.7's threadgroup floor is
     * satisfied for all of them and these cases stay about the GROUP BAND; the
     * floor itself is gated separately, in splitk_gqa_floor_policy below. */
    /* P4.0: unset means the DEFAULT, and the default is now ON, so the unset arm
     * must satisfy exactly the same policy table as the "1" arm. */
    bool on = gqa_mode == NULL || strcmp(gqa_mode, "1") == 0;
    const uint32_t ds = SPLITK_GQA_DEEP_SEQ;
    *policy_ok =
        sg_gpu_splitk_gqa_selected(g_gpu, 32, 8, ds) == on &&   /* 4B dense, repeat 4 */
        sg_gpu_splitk_gqa_selected(g_gpu, 24, 4, ds) == on &&   /* 27B, repeat 6 */
        sg_gpu_splitk_gqa_selected(g_gpu, 4, 2, ds) == on &&    /* this fixture, repeat 2 */
        sg_gpu_splitk_gqa_selected(g_gpu, 16, 2, ds) == on &&   /* repeat 8, the boundary */
        !sg_gpu_splitk_gqa_selected(g_gpu, 8, 8, ds) &&         /* repeat 1: nothing to share */
        !sg_gpu_splitk_gqa_selected(g_gpu, 18, 2, ds) &&        /* repeat 9: past the bound */
        !sg_gpu_splitk_gqa_selected(g_gpu, 32, 5, ds) &&        /* not a multiple */
        !sg_gpu_splitk_gqa_selected(g_gpu, 32, 0, ds);          /* no kv heads */

    for (uint32_t t = 0; t < n; t++) {
        const float *gl = NULL;
        e = sg_gpu_forward(g_gpu, m, ids[t], t, &gl);
        if (sg_failed(e)) {
            tt_assert(false, "sg_gpu_forward %u (gqa=%s): %s", t, label,
                      e.msg ? e.msg : "ok");
            return;
        }
        memcpy(logits_out + (size_t)t * vocab, gl, (size_t)vocab * sizeof *gl);
    }
    sg_gpu_splitk_dispatch_counts(g_gpu, per_head_out, gqa_out);
}

/* Task P2.7: THE OCCUPANCY FLOOR, against the table that was measured for it.
 *
 * The floor exists because the GQA kernel is SLOWER than the per-head one when
 * its collapsed grid is too small to fill the machine, so the property to gate is
 * two-sided: the predicate must be FALSE at every shape/seq where the sweep
 * measured a loss and TRUE where it measured a win. A one-sided "it is true
 * somewhere" test would pass for a floor of 0, which is the bug this task exists
 * to fix.
 *
 * Every `want` below is the sign of a measured speedup (per-head time / GQA time,
 * `./tests/bench_splitk.bin --reps 20` both arms, 3 alternating rounds,
 * 2026-08-17; a second study of 6 rounds at --reps 50 is quoted where it exists).
 * The numbers are in the note column so a future retune has to argue with the
 * measurement rather than with a constant. Read through
 * sg_gpu_splitk_gqa_selected, which calls the same predicate the encoder
 * consults.
 *
 * Requires the caller's state to have the GQA switch ON and the shipped cap, both
 * asserted here rather than assumed: at a lowered cap the whole table shifts,
 * since the cap lowers the split count and therefore the grid. */
static void splitk_gqa_floor_policy(void) {
    struct { uint32_t n_heads, n_kv, seq; bool want; const char *note; } cases[] = {
        /* 27B decode shape, 24 heads over 4 kv (repeat 6): threadgroups = 4*n_splits */
        { 24, 4,   2048, false, "32 tg, measured 0.938x" },
        { 24, 4,   2560, false, "40 tg, measured 0.867x" },
        { 24, 4,   4096, false, "64 tg, measured 0.990x / 0.967x" },
        { 24, 4,   5120, false, "80 tg, measured 0.988x / 0.956x" },
        { 24, 4,   6144, false, "96 tg, 1.025x here but the 4B shape loses at 96 tg" },
        { 24, 4,   8192, true,  "128 tg, measured 1.148x" },
        { 24, 4,  16384, true,  "256 tg, measured 1.248x" },
        { 24, 4, 262144, true,  "cap 256 binds: 1024 tg" },
        /* 4B dense shape, 32 heads over 8 kv (repeat 4): threadgroups = 8*n_splits */
        { 32, 8,   2048, false, "64 tg, measured 0.975x" },
        { 32, 8,   2560, false, "80 tg, measured 0.925x" },
        { 32, 8,   3072, false, "96 tg, measured 0.975x / 0.974x" },
        { 32, 8,   3584, false, "112 tg, measured 1.008x / 0.991x, a wash" },
        { 32, 8,   4096, true,  "128 tg, measured 1.004x twice" },
        { 32, 8,   8192, true,  "256 tg, measured 1.056x" },
        { 32, 8,  16384, true,  "512 tg, measured 1.182x" },
        /* This fixture's own shape, pinning the boundary from BOTH sides at the
         * exact seq where 2 kv heads reach 128 threadgroups. These two are what
         * make the dispatch-count assertions below predictable. */
        { 4, 2, SPLITK_GQA_FLOOR_SEQ - 1u, false, "63 splits * 2 kv = 126 tg" },
        { 4, 2, SPLITK_GQA_FLOOR_SEQ,      true,  "64 splits * 2 kv = 128 tg" },
    };
    tt_assert(sg_gpu_splitk_gqa_selected(g_gpu, 4, 2, SPLITK_GQA_FLOOR_SEQ),
              "this table needs a state with SURGE_ATTN_SPLITK_GQA=1; the fixture "
              "shape declines even above the floor, so every `false` below would "
              "pass for the wrong reason");
    tt_assert(sg_gpu_splitk_gqa_cap(g_gpu) == 256u,
              "this table is the SHIPPED cap's table; the state resolved cap %u, "
              "which moves every threadgroup count in it",
              sg_gpu_splitk_gqa_cap(g_gpu));
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        bool got = sg_gpu_splitk_gqa_selected(g_gpu, cases[i].n_heads,
                                              cases[i].n_kv, cases[i].seq);
        tt_assert(got == cases[i].want,
                  "P2.7 floor: %u heads / %u kv at seq %u must be %s (%s), got %s",
                  cases[i].n_heads, cases[i].n_kv, cases[i].seq,
                  cases[i].want ? "SELECTED" : "declined", cases[i].note,
                  got ? "SELECTED" : "declined");
    }
    /* The floor must be MONOTONE in seq for a fixed shape, or "turns on at depth"
     * is not what it does. n_splits is nondecreasing in seq and the kv count is
     * fixed, so once selected it must stay selected. */
    bool seen_true = false, went_back = false;
    for (uint32_t ns = 1; ns <= 300; ns++) {
        bool sel = sg_gpu_splitk_gqa_selected(g_gpu, 24, 4, ns * SPLITK_CAP_GATE_TG);
        if (sel) seen_true = true;
        else if (seen_true) went_back = true;
    }
    tt_assert(seen_true && !went_back,
              "the floor must be monotone in seq for a fixed shape "
              "(selected somewhere: %d, then declined again: %d)",
              (int)seen_true, (int)went_back);
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
    tt_assert(m.cfg.n_heads == 4 && m.cfg.n_kv_heads == MINI_KV_HEADS,
              "the mini fixture must stay a GQA shape for this gate (got %u heads "
              "over %u kv heads); repeat 1 would make it decline by policy and the "
              "dispatch assertions below would be testing nothing",
              m.cfg.n_heads, m.cfg.n_kv_heads);

    /* P2.7: SPLITK_GQA_GATE_N, not SPLITK_GATE_N. The floor needs 128
     * threadgroups and this fixture has 2 kv heads, so the GQA kernel is not
     * selected at all below seq 16384; at 1600 this gate would run the per-head
     * kernel in both arms and assert byte-identity between a kernel and itself. */
    const uint32_t n = SPLITK_GQA_GATE_N;
    tt_assert(n > SPLITK_GQA_FLOOR_SEQ,
              "the run (%u positions) must reach past the P2.7 floor seq %u or the "
              "GQA kernel is never selected and this gate is vacuous",
              n, SPLITK_GQA_FLOOR_SEQ);
    uint32_t vocab = m.cfg.vocab;
    int32_t *ids = xmalloc((size_t)n * sizeof *ids);
    for (uint32_t t = 0; t < n; t++) ids[t] = seed[t % n_seed];

    size_t lbytes = (size_t)n * vocab * sizeof(float);
    float *l_off = xmalloc(lbytes), *l_on = xmalloc(lbytes), *l_def = xmalloc(lbytes);
    memset(l_off, 0, lbytes);
    memset(l_on, 0, lbytes);
    memset(l_def, 0, lbytes);
    uint64_t ph_off = 0, gq_off = 0, ph_on = 0, gq_on = 0, ph_def = 0, gq_def = 0;
    bool pol_off = false, pol_on = false, pol_def = false;

    splitk_gqa_gate_run(&m, ids, n, "0", l_off, &ph_off, &gq_off, &pol_off);
    splitk_gqa_gate_run(&m, ids, n, "1", l_on, &ph_on, &gq_on, &pol_on);
    /* P4.0: the arm with NO env var set, i.e. what a user of this engine gets.
     * Runs last of the three so the state it leaves behind is a DEFAULT one. */
    splitk_gqa_gate_run(&m, ids, n, NULL, l_def, &ph_def, &gq_def, &pol_def);
    /* The P2.7 floor table, asserted against the live state the run above left
     * behind (the switch must be on there, which the table requires; since P4.0
     * that state is the DEFAULT one, so this is also a check that the default
     * reaches the same predicate the "1" arm does). */
    splitk_gqa_floor_policy();

    /* 1. THE POSITIVE CONTROL, and after P2.7 it is a two-sided one. Every step
     *    at or above the split-K threshold encodes one split-K partial per
     *    full-attention layer, so the TOTAL is the same in both arms; what the
     *    guard decides is how that total SPLITS between the two kernels with the
     *    switch on. Below the floor seq the per-head kernel runs even with the
     *    switch on, at and above it the GQA one does, so both counts are
     *    predictable to the dispatch and are asserted exactly:
     *
     *      total  = (n - SPLITK_GATE_THRESHOLD + 1) * full-attention layers
     *      gqa    = (n - SPLITK_GQA_FLOOR_SEQ + 1) * full-attention layers
     *      per-head with the switch on = total - gqa
     *
     *    The layer count is DERIVED from the off arm rather than hardcoded, so
     *    this does not silently encode "the mini fixture has one full-attention
     *    layer"; exact equality then pins WHERE the floor is, which a
     *    `gqa > 0` test alone would not (a floor of 0 passes that). */
    uint64_t sk_positions = (uint64_t)n - SPLITK_GATE_THRESHOLD + 1u;
    uint64_t gqa_positions = (uint64_t)n - SPLITK_GQA_FLOOR_SEQ + 1u;
    tt_assert(ph_off > 0 && gq_off == 0,
              "SURGE_ATTN_SPLITK_GQA=0 must encode only per-head partials "
              "(per_head %llu, gqa %llu)",
              (unsigned long long)ph_off, (unsigned long long)gq_off);
    tt_assert(ph_off % sk_positions == 0,
              "the off arm encoded %llu per-head partials over %llu split-K "
              "positions, which is not a whole number of full-attention layers; "
              "the counts below cannot be derived",
              (unsigned long long)ph_off, (unsigned long long)sk_positions);
    uint64_t layers = ph_off / sk_positions;
    tt_assert(gq_on > 0,
              "SURGE_ATTN_SPLITK_GQA=1 encoded 0 GQA partial dispatches over %u "
              "positions, so the GQA kernel never ran and every comparison below "
              "is vacuous", n);
    tt_assert(gq_on == gqa_positions * layers,
              "with the switch on the GQA kernel must run at EXACTLY the positions "
              "at or above the P2.7 floor seq %u: expected %llu dispatches "
              "(%llu positions x %llu full-attn layers), got %llu",
              SPLITK_GQA_FLOOR_SEQ, (unsigned long long)(gqa_positions * layers),
              (unsigned long long)gqa_positions, (unsigned long long)layers,
              (unsigned long long)gq_on);
    tt_assert(ph_on == (sk_positions - gqa_positions) * layers,
              "with the switch on the PER-HEAD kernel must run at exactly the "
              "split-K positions BELOW the floor seq %u: expected %llu, got %llu; "
              "0 would mean the floor is not being applied at all",
              SPLITK_GQA_FLOOR_SEQ,
              (unsigned long long)((sk_positions - gqa_positions) * layers),
              (unsigned long long)ph_on);
    tt_assert(gq_on + ph_on == ph_off,
              "the two modes must dispatch the same NUMBER of split-K partials "
              "(gqa %llu + per-head %llu vs per-head %llu); a difference means the "
              "switch changed more than which kernel runs",
              (unsigned long long)gq_on, (unsigned long long)ph_on,
              (unsigned long long)ph_off);

    /* 1b. TASK P4.0: WHICH KERNEL THE DEFAULT PICKS. The two arms above pin the
     *     switch explicitly, so they say nothing about the default and would pass
     *     unchanged whichever way it pointed. This is the assertion that fails if
     *     the default flips back, or if a flip forward never reached the encoder.
     *
     *     Stated as EXACT EQUALITY with the "1" arm rather than as `gq_def > 0`,
     *     for the reason assertion 1 gives: an existence test also passes for a
     *     default that reached the switch but lost the P2.7 floor, and the split
     *     between the two counters is where that floor is.
     *
     *     Note what the `ph_off > 0 && gq_off == 0` assertion above now ALSO
     *     proves, which it did not before this task: that SURGE_ATTN_SPLITK_GQA=0
     *     still turns the GQA path off now that off is the non-default direction.
     *     The override is gated in BOTH directions here, on counters. */
    tt_assert(gq_def == gq_on && ph_def == ph_on,
              "with SURGE_ATTN_SPLITK_GQA UNSET the decode path must dispatch "
              "exactly what SURGE_ATTN_SPLITK_GQA=1 dispatches (gqa %llu vs %llu, "
              "per-head %llu vs %llu): since task P4.0 the GQA partial is the "
              "SHIPPED DEFAULT, so any difference means the default moved",
              (unsigned long long)gq_def, (unsigned long long)gq_on,
              (unsigned long long)ph_def, (unsigned long long)ph_on);

    /* 2. The policy rules, including the band edges that were previously only a
     *    comment. */
    tt_assert(pol_on, "with SURGE_ATTN_SPLITK_GQA=1 the group-size policy is wrong: "
                      "an in-band shape declined or an out-of-band one was accepted");
    tt_assert(pol_off, "with SURGE_ATTN_SPLITK_GQA=0 some shape still selected the "
                       "GQA kernel");
    tt_assert(pol_def, "with SURGE_ATTN_SPLITK_GQA UNSET the group-size policy must "
                       "answer exactly as it does with =1; it did not, so the "
                       "default and the explicit opt-in are not the same state");

    /* 3. Now that the counters prove different kernels ran, byte-identity is the
     *    contract itself. memcmp, not float ==, for the P2.2 reason. */
    uint32_t diff = 0, diff_def = 0;
    for (uint32_t t = 0; t < n; t++) {
        if (memcmp(l_off + (size_t)t * vocab, l_on + (size_t)t * vocab,
                   (size_t)vocab * sizeof(float)) != 0) diff++;
        /* P4.0: the default arm against the PINNED-OFF arm, which is the
         * statement a user cares about: turning the new default off changes no
         * bit of any logit. Compared against l_off and not against l_on so this
         * is an independent comparison rather than a transitive one. */
        if (memcmp(l_off + (size_t)t * vocab, l_def + (size_t)t * vocab,
                   (size_t)vocab * sizeof(float)) != 0) diff_def++;
    }
    tt_assert(diff == 0,
              "the GQA partial must be BYTE-IDENTICAL end to end: %u of %u "
              "positions have different logit bits", diff, n);
    tt_assert(diff_def == 0,
              "the DEFAULT decode path must be byte-identical to the one pinned "
              "with SURGE_ATTN_SPLITK_GQA=0: %u of %u positions differ", diff_def, n);

    fprintf(stderr, "   split-K GQA decode: %llu GQA dispatches with the switch on "
                    "(seq >= %u, the P2.7 floor) + %llu per-head below it, %llu "
                    "per-head with it off, logits byte-identical at %u/%u positions\n",
            (unsigned long long)gq_on, SPLITK_GQA_FLOOR_SEQ,
            (unsigned long long)ph_on, (unsigned long long)ph_off, n - diff, n);
    fprintf(stderr, "   split-K GQA default (P4.0, env UNSET): %llu GQA + %llu "
                    "per-head dispatches, identical to the =1 arm; logits "
                    "byte-identical to the =0 arm at %u/%u positions\n",
            (unsigned long long)gq_def, (unsigned long long)ph_def,
            n - diff_def, n);

    unsetenv("SURGE_ATTN_SPLITK_GQA");
    unsetenv("SURGE_ATTN_SPLITK");
    free(l_def); free(l_on); free(l_off);
    free(ids);
    sg_model_free(&m);
    sg_gguf_close(gg);
    free(seed);
}

/* Task P2.6: THE GREEDY-TOKEN GATE FOR THE REGIME P2.5 INTRODUCED.
 *
 * P2.5 gave the GQA arm its own split policy, clamp(min(seq/SG_TG, 256), 4,
 * 1024). From seq 65792 == SG_TG * (256 + 1) on, that returns FEWER splits than
 * the per-head splitk_n_splits, so the two arms partition the same keys
 * differently and their logits agree only to float rounding. surge's
 * correctness standard is byte-exact greedy TOKENS, and nothing tested them
 * there: the P2.4 positive control above runs below the natural divergence seq,
 * where both policies return the same count, and test_metal_ops.c's subtest pins
 * n_splits identically on both sides, bypassing the policy entirely. The
 * argument that the margin covers it (P2.4 measured a worst logit delta of
 * 9.537e-07 against a smallest top1-top2 margin of 1.385e-03) was an argument,
 * not a gate.
 *
 * WHY THIS RUNS IN SECONDS INSTEAD OF HOURS. The property under test is only
 * "the two arms pick DIFFERENT n_splits, do greedy tokens still match". That
 * needs divergence, not depth, and the divergence point is SG_TG * (cap + 1)
 * for whatever cap the state resolved. A lowered SURGE_SPLITK_GQA_CAP therefore
 * reproduces the EXACT mechanism of the natural 257-vs-256 case at a seq a test
 * can reach. P2.6 used cap 4 (divergence at seq 1280, per-head 6 splits against
 * GQA 4 at seq 1600); P2.7's threadgroup floor makes cap 4 unselectable on this
 * fixture, so the cap is now 64 and the divergence sits at seq 16640 -- see
 * SPLITK_CAP_GATE_CAP below for why that is forced and what it costs.
 *
 * FIVE ASSERTIONS, AND THE FIRST THREE EXIST TO STOP THIS PASSING VACUOUSLY:
 *   1. THE OVERRIDE WAS PARSED. sg_gpu_splitk_gqa_cap reports the cap, read off
 *      the live state. If SURGE_SPLITK_GQA_CAP were ignored, this fails here
 *      instead of silently making 4 and 5 below.
 *   2. THE POLICIES REALLY DIVERGE, by specific value, read through the same
 *      functions the encoder dispatches with: per-head cap+1 vs GQA cap at the
 *      divergence seq, per-head n/SG_TG vs GQA cap at the run's own seq, and
 *      EQUAL (cap and cap) one position earlier. That last case pins the boundary
 *      from below, so a cap that took effect at the wrong seq fails rather than
 *      passing on a wider window. It also asserts that AT THE DEFAULT cap the two
 *      would be equal at the run's seq, which is the explicit statement that this
 *      subtest has content only because of the override.
 *   3. WHICH KERNEL RAN, from sg_gpu_splitk_dispatch_counts, exactly as the
 *      P2.4 control does: a declined selection would otherwise show up as two
 *      identical runs and a green gate. Since P2.7 the "on" arm's dispatches
 *      split at the floor seq, and BOTH counts are asserted exactly.
 *   4. THE BITS ACTUALLY CHANGED at and above the divergence seq, and did NOT
 *      change below it. Both halves are load-bearing. If the cap silently did
 *      nothing, the two arms would use the same n_splits and be byte-identical
 *      everywhere, failing the "differs above" half; if the cap bound too early
 *      the "byte-identical below" half fails. This is P2.3's mutation-proof
 *      shape applied to the policy instead of the threshold.
 *   5. THE TOKENS. Identical argmax at every position, plus the M3.4 robustness
 *      bound: the worst absolute logit delta anywhere must be under the
 *      smallest top1-top2 margin anywhere, so the agreement is not luck.
 *
 * WHAT BREAKS IT (mutations, all of them checked by hand and reported in
 * docs/17082026_splitk_gqa_threadgroups.md): making sg_gpu_state_new ignore
 * SURGE_SPLITK_GQA_CAP fails assertion 1 and then 4; having enc_attn_splitk
 * pass SG_SPLITK_GQA_N_SPLITS_CAP instead of the state's cap fails 4 with 0
 * positions differing; a real numeric bug in the GQA partial fails 5. */
/* P2.7 FORCED THIS CAP UP FROM 4 TO 64, and the arithmetic leaves no choice.
 * The cap is a CEILING on the GQA arm's split count, so it also caps the GQA
 * grid at cap * n_kv_heads threadgroups: at cap 4 on this 2-kv fixture that is 8
 * threadgroups, far under P2.7's measured floor of 128, so the GQA kernel would
 * never be selected at any seq and this whole subtest would compare the per-head
 * kernel with itself. The smallest cap that can reach the floor is
 * SPLITK_GQA_FLOOR_NS == 128 / 2 == 64, which puts the divergence point at
 * SG_TG * 65 == 16640 and is why SPLITK_GQA_GATE_N has to be past that.
 *
 * WHAT THAT COSTS, stated rather than hidden: the divergence is now 65 splits
 * against 64 (a 1.016x split-count ratio) instead of P2.6's 6 against 4 (1.5x),
 * so the rounding perturbation under test is SMALLER. It is also much closer to
 * the natural 257-against-256 case at the shipped cap that no test can reach,
 * and it is the same 1.016x ratio as the real-model M0/M1 A/B P2.6 ran at cap 63
 * (see docs/17082026_splitk_gqa_threadgroups.md). The two-sided bit assertions
 * below are what prove the smaller perturbation is still detectable. */
#define SPLITK_CAP_GATE_CAP SPLITK_GQA_FLOOR_NS
/* The first seq at which the capped GQA policy returns less than the per-head
 * one: seq / SG_TG first reaches cap + 1 there. Same closed form as the natural
 * 65792 at the shipped cap of 256. */
#define SPLITK_CAP_GATE_DIVERGE (SPLITK_CAP_GATE_TG * (SPLITK_CAP_GATE_CAP + 1u))

/* What one arm observed about the policy, read off the LIVE state so that a
 * mis-parsed or ignored override shows up as a failed assertion rather than as
 * two arms that quietly agree. */
struct splitk_cap_obs {
    uint32_t cap;              /* sg_gpu_splitk_gqa_cap: was the override parsed */
    uint32_t ns_ph_at_n;       /* per-head n_splits at SPLITK_GATE_N */
    uint32_t ns_gqa_at_n;      /* GQA n_splits at SPLITK_GATE_N, this state's cap */
    uint32_t ns_ph_div, ns_gqa_div;      /* the same pair at the divergence point */
    uint32_t ns_ph_pre, ns_gqa_pre;      /* and one seq BELOW it, where they must agree */
    uint32_t ns_gqa_default_at_n;        /* the shipped cap's answer, for the vacuity note */
    uint64_t per_head, gqa;    /* dispatch counters for this state */
};

static void splitk_cap_gate_run(sg_model *m, const int32_t *ids, uint32_t n,
                                const char *gqa_mode, float *logits_out,
                                uint32_t *arg_out, struct splitk_cap_obs *obs) {
    uint32_t vocab = m->cfg.vocab;
    char cap_str[16];
    snprintf(cap_str, sizeof cap_str, "%u", SPLITK_CAP_GATE_CAP);
    setenv("SURGE_ATTN_SPLITK", "1", 1);
    setenv("SURGE_ATTN_SPLITK_GQA", gqa_mode, 1);
    /* Set in BOTH arms on purpose: the cap must not perturb the per-head arm,
     * and the byte-comparison below would hide it if only one arm carried it. */
    setenv("SURGE_SPLITK_GQA_CAP", cap_str, 1);
    sg_err e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (gqa=%s, cap=%s): %s", gqa_mode,
              cap_str, e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    obs->cap = sg_gpu_splitk_gqa_cap(g_gpu);
    obs->ns_ph_at_n = sg_gpu_splitk_n_splits(n);
    obs->ns_gqa_at_n = sg_gpu_splitk_gqa_n_splits_at(g_gpu, n);
    obs->ns_ph_div = sg_gpu_splitk_n_splits(SPLITK_CAP_GATE_DIVERGE);
    obs->ns_gqa_div = sg_gpu_splitk_gqa_n_splits_at(g_gpu, SPLITK_CAP_GATE_DIVERGE);
    obs->ns_ph_pre = sg_gpu_splitk_n_splits(SPLITK_CAP_GATE_DIVERGE - 1u);
    obs->ns_gqa_pre = sg_gpu_splitk_gqa_n_splits_at(g_gpu, SPLITK_CAP_GATE_DIVERGE - 1u);
    obs->ns_gqa_default_at_n = sg_gpu_splitk_gqa_n_splits(n);

    for (uint32_t t = 0; t < n; t++) {
        const float *gl = NULL;
        e = sg_gpu_forward(g_gpu, m, ids[t], t, &gl);
        if (sg_failed(e)) {
            tt_assert(false, "sg_gpu_forward %u (gqa=%s, cap=%s): %s", t, gqa_mode,
                      cap_str, e.msg ? e.msg : "ok");
            return;
        }
        memcpy(logits_out + (size_t)t * vocab, gl, (size_t)vocab * sizeof *gl);
        arg_out[t] = argmax_f32(gl, vocab);
    }
    sg_gpu_splitk_dispatch_counts(g_gpu, &obs->per_head, &obs->gqa);
}

/* An unusable SURGE_SPLITK_GQA_CAP must FAIL sg_gpu_state_new, not warn and
 * fall back the way the other three split-K env vars do. A silently ignored
 * value is precisely how the gate above would go vacuous, so the rejection is
 * part of the gate, not hygiene. */
static void splitk_cap_rejects_bad_values(sg_model *m, uint32_t n) {
    const char *bad[] = {
        "0",      /* would clamp every seq to SG_SPLITK_MIN */
        "3",      /* below SG_SPLITK_MIN: the floor clamp eats it */
        "1025",   /* above SG_SPLITK_MAX: the ceiling clamp eats it */
        "-4",     /* negative */
        "4x",     /* trailing garbage: strtol alone would accept this as 4 */
        "x",      /* no digits at all */
        " ",      /* whitespace only */
        " 4",     /* leading whitespace: strtol alone would accept this as 4 */
        "+4",     /* leading sign: strtol alone would accept this as 4 */
        "99999999999999999999",  /* strtol range error */
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        setenv("SURGE_SPLITK_GQA_CAP", bad[i], 1);
        sg_err e = sg_gpu_state_new(g_gpu, m, n);
        tt_assert(sg_failed(e),
                  "SURGE_SPLITK_GQA_CAP='%s' must be rejected, not ignored "
                  "(an ignored value makes the cap-override gate vacuous)", bad[i]);
    }
    /* And the in-band edges must be ACCEPTED, or the rejection above is just a
     * blanket refusal and the override is unusable. */
    const char *good[] = { "4", "256", "1024" };
    for (size_t i = 0; i < sizeof good / sizeof good[0]; i++) {
        setenv("SURGE_SPLITK_GQA_CAP", good[i], 1);
        sg_err e = sg_gpu_state_new(g_gpu, m, n);
        tt_assert(!sg_failed(e), "SURGE_SPLITK_GQA_CAP='%s' must be accepted: %s",
                  good[i], e.msg ? e.msg : "ok");
        if (!sg_failed(e)) {
            uint32_t want = (uint32_t)strtoul(good[i], NULL, 10);
            tt_assert(sg_gpu_splitk_gqa_cap(g_gpu) == want,
                      "SURGE_SPLITK_GQA_CAP='%s' parsed as %u", good[i],
                      sg_gpu_splitk_gqa_cap(g_gpu));
        }
    }
    /* Unset must resolve to the shipped default, which is what every real run
     * gets. Asserted through the same accessor, so "unset" and "256" are proven
     * to be the same state rather than assumed to be. */
    unsetenv("SURGE_SPLITK_GQA_CAP");
    sg_err e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new with no cap override: %s",
              e.msg ? e.msg : "ok");
    tt_assert(sg_gpu_splitk_gqa_cap(g_gpu) == 256u,
              "with SURGE_SPLITK_GQA_CAP unset the resolved cap must be the "
              "measured 256, got %u", sg_gpu_splitk_gqa_cap(g_gpu));
}

/* Task P2.7, and it closes a mutation the two runs above CANNOT see: the guard
 * must measure the grid the dispatch will really have, i.e. the CAPPED split
 * count, not the uncapped splitk_n_splits(seq). At the cap this gate runs with
 * (64 on a 2-kv fixture) the two agree at every seq that matters, so a guard
 * reading the uncapped value passes both runs; at cap 4 they differ by 16x and
 * one call settles it. No forwards, so this costs nothing.
 *
 * cap 4 on the 4B shape is 4 splits * 8 kv == 32 threadgroups, which is under the
 * floor at EVERY seq including 262144, where an uncapped guard would compute
 * 1024 * 8 == 8192 and select the GQA kernel into a 32-threadgroup dispatch. The
 * same two shapes with no override must be SELECTED at that seq, or the first
 * half proved nothing about the cap. */
static void splitk_gqa_floor_uses_state_cap(sg_model *m, uint32_t n) {
    const uint32_t deep = 262144u;
    setenv("SURGE_SPLITK_GQA_CAP", "4", 1);
    sg_err e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (cap 4): %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;
    tt_assert(sg_gpu_splitk_gqa_cap(g_gpu) == 4u,
              "this check needs cap 4 to be live, got %u", sg_gpu_splitk_gqa_cap(g_gpu));
    tt_assert(!sg_gpu_splitk_gqa_selected(g_gpu, 32, 8, deep),
              "at cap 4 the 4B shape's GQA grid is 4 splits x 8 kv = 32 "
              "threadgroups at EVERY seq, so it must be declined even at seq %u; "
              "selecting it means the floor is reading the UNCAPPED split count "
              "(%u) instead of the one the dispatch uses (%u)",
              deep, sg_gpu_splitk_n_splits(deep),
              sg_gpu_splitk_gqa_n_splits_at(g_gpu, deep));
    tt_assert(!sg_gpu_splitk_gqa_selected(g_gpu, 24, 4, deep),
              "same for the 27B shape at cap 4: 4 splits x 4 kv = 16 threadgroups");

    unsetenv("SURGE_SPLITK_GQA_CAP");
    e = sg_gpu_state_new(g_gpu, m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (no cap): %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;
    tt_assert(sg_gpu_splitk_gqa_selected(g_gpu, 32, 8, deep)
                  && sg_gpu_splitk_gqa_selected(g_gpu, 24, 4, deep),
              "with the cap at its default both real shapes must be SELECTED at seq "
              "%u (256 splits x 8 kv and x 4 kv); if they are not, the declines "
              "above were not caused by the cap", deep);
}

static void mini_f16_splitk_gqa_cap_override_greedy_matches(void) {
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

    /* The mirrored SG_TG must stay in step with the threshold this file already
     * mirrors (SG_TG * SG_SPLITK_MIN == SG_TG * 4), or SPLITK_CAP_GATE_DIVERGE
     * is computed from a stale constant and every seq boundary below is wrong. */
    tt_assert(SPLITK_CAP_GATE_TG * 4u == SPLITK_GATE_THRESHOLD,
              "mirrored SG_TG (%u) disagrees with the mirrored split-K threshold "
              "(%u); one of the two is stale",
              SPLITK_CAP_GATE_TG, SPLITK_GATE_THRESHOLD);
    /* And the run must be long enough to reach past the divergence point, or
     * assertion 4 has no positions to look at. */
    tt_assert(SPLITK_GQA_GATE_N > SPLITK_CAP_GATE_DIVERGE,
              "the gate run (%u positions) must reach past the divergence seq %u",
              SPLITK_GQA_GATE_N, SPLITK_CAP_GATE_DIVERGE);
    /* P2.7: and past the threadgroup floor, which is a SEPARATE and lower
     * boundary (16384 against 16640 here). Between the two the GQA kernel runs at
     * the same n_splits as the per-head one, which is where assertion 4's
     * "identical below" half gets its teeth: those positions run the GQA KERNEL
     * and must still match bit for bit. */
    tt_assert(SPLITK_CAP_GATE_DIVERGE > SPLITK_GQA_FLOOR_SEQ,
              "the divergence seq %u must sit above the P2.7 floor seq %u, or the "
              "GQA kernel is not even selected where the policies diverge",
              SPLITK_CAP_GATE_DIVERGE, SPLITK_GQA_FLOOR_SEQ);
    tt_assert(m.cfg.n_heads == 4 && m.cfg.n_kv_heads == MINI_KV_HEADS,
              "the mini fixture must stay a GQA shape (got %u heads over %u kv)",
              m.cfg.n_heads, m.cfg.n_kv_heads);

    const uint32_t n = SPLITK_GQA_GATE_N;
    uint32_t vocab = m.cfg.vocab;
    int32_t *ids = xmalloc((size_t)n * sizeof *ids);
    for (uint32_t t = 0; t < n; t++) ids[t] = seed[t % n_seed];

    size_t lbytes = (size_t)n * vocab * sizeof(float);
    float *l_off = xmalloc(lbytes), *l_on = xmalloc(lbytes);
    uint32_t *a_off = xmalloc((size_t)n * sizeof *a_off);
    uint32_t *a_on = xmalloc((size_t)n * sizeof *a_on);
    memset(l_off, 0, lbytes);
    memset(l_on, 0, lbytes);
    struct splitk_cap_obs off = {0}, on = {0};

    splitk_cap_gate_run(&m, ids, n, "0", l_off, a_off, &off);
    splitk_cap_gate_run(&m, ids, n, "1", l_on, a_on, &on);

    /* 1. The override was PARSED, on both arms' states. */
    tt_assert(off.cap == SPLITK_CAP_GATE_CAP && on.cap == SPLITK_CAP_GATE_CAP,
              "SURGE_SPLITK_GQA_CAP=%u was not applied (resolved cap %u with the "
              "GQA switch off, %u with it on); without it both arms pick the same "
              "n_splits and everything below is vacuous",
              SPLITK_CAP_GATE_CAP, off.cap, on.cap);

    /* 2. The two policies diverge, BY SPECIFIC VALUE, at and only at the
     *    documented seq, read through the functions the encoder dispatches
     *    with. Cap 64 (P2.7): per-head 64 vs GQA 64 at seq 16639, 65 vs 64 at
     *    16640, 66 vs 64 at 16896. */
    tt_assert(on.ns_ph_pre == SPLITK_CAP_GATE_CAP && on.ns_gqa_pre == SPLITK_CAP_GATE_CAP,
              "one seq below the divergence point (%u) the two policies must still "
              "agree at %u splits, got per-head %u and GQA %u",
              SPLITK_CAP_GATE_DIVERGE - 1u, SPLITK_CAP_GATE_CAP,
              on.ns_ph_pre, on.ns_gqa_pre);
    tt_assert(on.ns_ph_div == SPLITK_CAP_GATE_CAP + 1u
                  && on.ns_gqa_div == SPLITK_CAP_GATE_CAP,
              "at the divergence seq %u the policies must split %u vs %u, got "
              "per-head %u and GQA %u",
              SPLITK_CAP_GATE_DIVERGE, SPLITK_CAP_GATE_CAP + 1u, SPLITK_CAP_GATE_CAP,
              on.ns_ph_div, on.ns_gqa_div);
    tt_assert(on.ns_ph_at_n == n / SPLITK_CAP_GATE_TG
                  && on.ns_gqa_at_n == SPLITK_CAP_GATE_CAP,
              "at the run's own seq %u the policies must split %u vs %u, got "
              "per-head %u and GQA %u",
              n, n / SPLITK_CAP_GATE_TG, SPLITK_CAP_GATE_CAP,
              on.ns_ph_at_n, on.ns_gqa_at_n);
    tt_assert(on.ns_gqa_at_n < on.ns_ph_at_n,
              "the GQA arm must ask for FEWER splits than the per-head arm here "
              "(GQA %u, per-head %u); equal counts mean the same partition and no "
              "rounding difference to test", on.ns_gqa_at_n, on.ns_ph_at_n);
    /* The vacuity statement, asserted rather than commented: at the SHIPPED cap
     * this run's seq is far below the divergence point, so the two policies
     * agree and this whole subtest would be a duplicate of the P2.4 control. */
    tt_assert(on.ns_gqa_default_at_n == on.ns_ph_at_n,
              "at the shipped cap the two policies must still agree at seq %u "
              "(got GQA %u vs per-head %u); if they already diverged there, this "
              "subtest is not testing the override it claims to",
              n, on.ns_gqa_default_at_n, on.ns_ph_at_n);
    /* The cap must not touch the per-head policy at all. */
    tt_assert(off.ns_ph_at_n == on.ns_ph_at_n,
              "the cap changed the PER-HEAD split count (%u vs %u); it must only "
              "affect the GQA arm", off.ns_ph_at_n, on.ns_ph_at_n);

    /* 3. WHICH KERNEL RAN, the P2.4 positive control repeated under the
     *    override: without this a declined GQA selection looks like a pass. P2.7
     *    splits the "on" arm at the floor seq, exactly as it does in the P2.4
     *    control, so the two counts are asserted separately and must still sum to
     *    the off arm's total. */
    uint64_t cap_sk_positions = (uint64_t)n - SPLITK_GATE_THRESHOLD + 1u;
    uint64_t cap_gqa_positions = (uint64_t)n - SPLITK_GQA_FLOOR_SEQ + 1u;
    tt_assert(off.per_head > 0 && off.gqa == 0,
              "with the GQA switch off every split-K dispatch must be a per-head "
              "one (per-head %llu, gqa %llu)",
              (unsigned long long)off.per_head, (unsigned long long)off.gqa);
    tt_assert(off.per_head % cap_sk_positions == 0,
              "the off arm's %llu per-head dispatches are not a whole number of "
              "full-attention layers over %llu split-K positions",
              (unsigned long long)off.per_head, (unsigned long long)cap_sk_positions);
    uint64_t cap_layers = off.per_head / cap_sk_positions;
    tt_assert(on.gqa == cap_gqa_positions * cap_layers,
              "with the switch on the GQA kernel must run at exactly the positions "
              "at or above the P2.7 floor seq %u: expected %llu, got %llu",
              SPLITK_GQA_FLOOR_SEQ,
              (unsigned long long)(cap_gqa_positions * cap_layers),
              (unsigned long long)on.gqa);
    tt_assert(on.per_head == (cap_sk_positions - cap_gqa_positions) * cap_layers,
              "with the switch on the per-head kernel must run at exactly the "
              "split-K positions below the floor seq %u: expected %llu, got %llu",
              SPLITK_GQA_FLOOR_SEQ,
              (unsigned long long)((cap_sk_positions - cap_gqa_positions) * cap_layers),
              (unsigned long long)on.per_head);
    tt_assert(on.gqa + on.per_head == off.per_head,
              "the two arms must dispatch the same NUMBER of split-K partials "
              "(%llu + %llu vs %llu)",
              (unsigned long long)on.gqa, (unsigned long long)on.per_head,
              (unsigned long long)off.per_head);

    /* 4. The bits must be IDENTICAL below the divergence seq and DIFFERENT at or
     *    above it. Position t carries seq = t + 1. */
    uint32_t below_diff = 0, above_diff = 0, above_total = 0;
    bool boundary_differs = false;
    for (uint32_t t = 0; t < n; t++) {
        bool differs = memcmp(l_off + (size_t)t * vocab, l_on + (size_t)t * vocab,
                              (size_t)vocab * sizeof(float)) != 0;
        if (t + 1u < SPLITK_CAP_GATE_DIVERGE) {
            if (differs) below_diff++;
        } else {
            above_total++;
            if (differs) above_diff++;
            if (t + 1u == SPLITK_CAP_GATE_DIVERGE) boundary_differs = differs;
        }
    }
    /* Two reasons every position below the divergence seq must match, and P2.7
     * added the second: below the FLOOR seq the same per-head kernel runs in both
     * arms, and between the floor and the divergence the GQA kernel runs at the
     * SAME n_splits as the per-head one, where the two kernels are contracted to
     * be byte-identical. So this half now also gates P2.4's byte-identity claim
     * inside the real decode path at a grid the guard actually admits. */
    tt_assert(below_diff == 0,
              "below seq %u the two policies pick the SAME n_splits and the two "
              "kernels are byte-identical, so every logit bit must match; %u "
              "positions differ", SPLITK_CAP_GATE_DIVERGE, below_diff);
    /* BOTH of the next two, and the second one is why. `above_diff > 0` alone is a
     * ONE-SIDED existence test over the window above, so a mutation that
     * shifts the ACTUAL divergence LATER (dispatching with cap+1, say, while the
     * accessors still report cap) would still find some differing position up here
     * and pass. Requiring the position at EXACTLY the divergence seq to differ, and
     * requiring EVERY position at or above it to differ, pins the boundary with the
     * BITS instead of only through the accessor assertions above. Both are safe to
     * state as equalities rather than inequalities because each mode is
     * deterministic (P2.3's own subtest reruns split-K 100x byte-identically), so
     * this is a fixed count for this fixture and cap, not a sampled one. */
    tt_assert(boundary_differs,
              "the position at exactly the divergence seq %u must differ: that is "
              "the first seq where the two policies split %u vs %u, and if its bits "
              "match, the cap the DISPATCH used is not the cap the accessors report",
              SPLITK_CAP_GATE_DIVERGE, SPLITK_CAP_GATE_CAP + 1u, SPLITK_CAP_GATE_CAP);
    tt_assert(above_diff == above_total && above_total > 0,
              "at seq >= %u the capped GQA policy must change the bits at EVERY "
              "position (%u of %u differ); 0 means the override never reached the "
              "dispatch, and anything in between means the divergence starts "
              "somewhere other than where the policy says it does",
              SPLITK_CAP_GATE_DIVERGE, above_diff, above_total);

    /* 5. THE TOKENS, which is what this task exists for, plus the M3.4
     *    robustness bound. Same metric choice as the P2.3 subtest above: no bare
     *    per-element relative ratio, because two summation orders over the same
     *    keys is exactly the case that explodes it near a cancelled logit. */
    uint32_t arg_mismatch = 0;
    double worst_abs = 0.0, worst_scaled = 0.0, min_margin = INFINITY;
    for (uint32_t t = 0; t < n; t++) {
        if (a_off[t] != a_on[t]) arg_mismatch++;
        const float *a = l_off + (size_t)t * vocab, *b = l_on + (size_t)t * vocab;
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
              "GREEDY TOKENS MUST MATCH with the two policies splitting %u vs %u: "
              "argmax differs at %u of %u positions",
              on.ns_ph_at_n, on.ns_gqa_at_n, arg_mismatch, n);
    tt_assert(worst_abs < min_margin,
              "worst |logit delta| %.3e must stay under the smallest top1-top2 "
              "margin %.3e, or an argmax could flip", worst_abs, min_margin);
    tt_assert(worst_scaled < 1e-4,
              "worst logit delta relative to the position's logit scale is %.3e",
              worst_scaled);

    fprintf(stderr, "   split-K GQA cap override: cap %u so the policies split %u "
                    "(per-head) vs %u (GQA) at seq %u and diverge from seq %u; "
                    "%llu GQA dispatches with the switch on (+%llu per-head below "
                    "the P2.7 floor seq %u), %llu per-head with it off; %u/%u "
                    "positions differ above the divergence seq, 0 of %u below; "
                    "worst |delta| %.3e (scaled %.3e) vs min top1-top2 margin "
                    "%.3e; greedy argmax %u/%u agree\n",
            on.cap, on.ns_ph_at_n, on.ns_gqa_at_n, n, SPLITK_CAP_GATE_DIVERGE,
            (unsigned long long)on.gqa, (unsigned long long)on.per_head,
            SPLITK_GQA_FLOOR_SEQ, (unsigned long long)off.per_head,
            above_diff, above_total, n - above_total, worst_abs, worst_scaled,
            min_margin, n - arg_mismatch, n);
    fprintf(stderr, "   (mutation that breaks it: dropping the SURGE_SPLITK_GQA_CAP "
                    "parse fails the resolved-cap assertion and then the "
                    "\"%u of %u positions differ\" one; dispatching with the "
                    "compiled cap instead of the state's fails the same one with 0 "
                    "of %u)\n", above_diff, above_total, above_total);

    /* The rejection half of the override's contract, on a state that is about to
     * be replaced anyway. Runs last so a failure here cannot leave the arms
     * above sharing a broken state. */
    splitk_gqa_floor_uses_state_cap(&m, n);
    splitk_cap_rejects_bad_values(&m, n);

    unsetenv("SURGE_SPLITK_GQA_CAP");
    unsetenv("SURGE_ATTN_SPLITK_GQA");
    unsetenv("SURGE_ATTN_SPLITK");
    free(a_on); free(a_off);
    free(l_on); free(l_off);
    free(ids);
    sg_model_free(&m);
    sg_gguf_close(gg);
    free(seed);
}

/* --------------------------------------------------------------------
 * Task P2.8: the ONLINE-softmax split-K arm's env var and selection policy
 * --------------------------------------------------------------------
 *
 * WHAT THIS GATES AND WHAT IT DELIBERATELY DOES NOT. It costs no forward passes:
 * every assertion is about sg_gpu_state_new's parse of SURGE_ATTN_SPLITK_ONLINE
 * and about sg_gpu_splitk_online_selected, which calls the same predicate the
 * decode encoder consults. That covers the three things that would otherwise
 * exist only as comments: the value is REJECTED rather than ignored, the
 * head_dim <= SG_TG bound is real, and the online arm inherits the GQA group
 * band and P2.7's threadgroup floor instead of quietly having its own.
 *
 * IT DOES NOT ASSERT TOKEN OR LOGIT AGREEMENT, and that is on purpose rather
 * than for cost. The online kernel is NOT byte-identical to the four-pass ones
 * (streaming reorders the exponential sums), so the end-to-end statement is
 * "same greedy tokens, logits agreeing to float rounding", and the honest place
 * for a token divergence is a report with the argmax margin at the divergence,
 * not a red `make check`. That comparison is the deferred real-model A/B in
 * docs/17082026_splitk_gqa_threadgroups.md, on the 4B Q8_0 checkpoint at a depth
 * where the floor admits the kernel; this fixture has 2 kv heads, so it would
 * need 16384 positions before the online kernel is even selected.
 *
 * The state's max_ctx is small on purpose: no forward runs, and the policy is a
 * pure function of its arguments rather than of max_ctx. */
static void mini_f16_splitk_online_policy(void) {
    sg_gguf *gg = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    sg_model m;
    e = sg_model_from_gguf(gg, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(gg); return; }

    e = sg_gpu_load_model(g_gpu, &m);
    tt_assert(!sg_failed(e), "sg_gpu_load_model: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); return; }

    /* sg_gpu_splitk_gqa_selected has no head_dim argument, so it reads the
     * STATE's head_dim to decide whether the online arm has taken the dispatch
     * (see splitk_gqa_use). The mutual-exclusion assertion below therefore needs
     * this fixture's own head_dim to be inside the online bound. */
    tt_assert(m.cfg.head_dim > 0 && m.cfg.head_dim <= SPLITK_CAP_GATE_TG,
              "this gate needs the fixture's head_dim (%u) to be in [1, %u], the "
              "online kernel's register bound", m.cfg.head_dim, SPLITK_CAP_GATE_TG);

    const uint32_t n = 1024u;              /* no forwards; keep the state cheap */
    const uint32_t deep = SPLITK_GQA_DEEP_SEQ;   /* 65536: past the P2.7 floor */
    setenv("SURGE_ATTN_SPLITK", "1", 1);
    setenv("SURGE_ATTN_SPLITK_GQA", "0", 1);

    /* 1. REJECTED, NOT IGNORED. Every one of these would otherwise silently mean
     *    "off" and make an A/B whose on-arm was never on pass perfectly. */
    const char *bad[] = { "on", "true", "yes", "2", "-1", "01", "1 ", " 1", "00", "x" };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        setenv("SURGE_ATTN_SPLITK_ONLINE", bad[i], 1);
        e = sg_gpu_state_new(g_gpu, &m, n);
        tt_assert(sg_failed(e),
                  "SURGE_ATTN_SPLITK_ONLINE='%s' must be rejected, not ignored "
                  "(an ignored value makes the online A/B vacuous)", bad[i]);
    }

    /* 2. "0" is accepted and selects nothing, which is also the shipped default. */
    setenv("SURGE_ATTN_SPLITK_ONLINE", "0", 1);
    e = sg_gpu_state_new(g_gpu, &m, n);
    tt_assert(!sg_failed(e), "SURGE_ATTN_SPLITK_ONLINE=0 must be accepted: %s",
              e.msg ? e.msg : "ok");
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 24, 4, 256, deep)
                  && !sg_gpu_splitk_online_selected(g_gpu, 32, 8, 128, deep),
              "with SURGE_ATTN_SPLITK_ONLINE=0 no shape may select the online kernel");

    /* 3. "1" is accepted, and the two REAL decode shapes are admitted at depth.
     *    27B head_dim 256 is exactly SG_TG, the boundary that matters: it is the
     *    widest head_dim whose acc fits one register per head per thread, and it
     *    is the shape this whole line of work is measured on. */
    setenv("SURGE_ATTN_SPLITK_ONLINE", "1", 1);
    e = sg_gpu_state_new(g_gpu, &m, n);
    tt_assert(!sg_failed(e), "SURGE_ATTN_SPLITK_ONLINE=1 must be accepted: %s",
              e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_gguf_close(gg); return; }
    tt_assert(sg_gpu_splitk_online_selected(g_gpu, 24, 4, 256, deep),
              "the 27B shape (24h/4kv/256d, head_dim == SG_TG) must be admitted at "
              "seq %u", deep);
    tt_assert(sg_gpu_splitk_online_selected(g_gpu, 32, 8, 128, deep),
              "the 4B dense shape (32h/8kv/128d) must be admitted at seq %u", deep);

    /* 4. The head_dim bound, both sides of it. 257 and 512 are past the point
     *    where a thread owns one output dim, so the kernel would re-stream the
     *    split per band and lose to the four-pass kernel it replaces. */
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 24, 4, SPLITK_CAP_GATE_TG + 1, deep),
              "head_dim %u (one past SG_TG) must be declined: that is where the "
              "running accumulator stops fitting in registers",
              SPLITK_CAP_GATE_TG + 1);
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 24, 4, 512, deep),
              "head_dim 512 must be declined for the same reason");
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 24, 4, 0, deep),
              "head_dim 0 must be declined");

    /* 5. The GQA group band, inherited rather than re-stated. */
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 8, 8, 128, deep),
              "repeat 1 must be declined: there is nothing to share");
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 18, 2, 128, deep),
              "repeat 9 must be declined: past SG_SPLITK_GQA_MAX");
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 32, 5, 128, deep),
              "a n_heads that is not a multiple of n_kv_heads must be declined");
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 32, 0, 128, deep),
              "n_kv_heads 0 must be declined");

    /* 6. P2.7's threadgroup floor, inherited too: the same shape is admitted at
     *    depth and declined at short context. This fixture's 2 kv heads need 64
     *    splits, i.e. seq 16384, so seq 2048 gives 8 * 2 == 16 threadgroups. */
    tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 4, MINI_KV_HEADS,
                                             m.cfg.head_dim, 2048u),
              "the online arm must obey the P2.7 floor: 4h/2kv at seq 2048 is "
              "%u splits x %u kv threadgroups, under the floor of %u",
              sg_gpu_splitk_gqa_n_splits_at(g_gpu, 2048u), MINI_KV_HEADS,
              SPLITK_GQA_MIN_TG);
    tt_assert(sg_gpu_splitk_online_selected(g_gpu, 4, MINI_KV_HEADS,
                                            m.cfg.head_dim, deep),
              "the same shape must be admitted at seq %u, or the decline above is "
              "not about the floor", deep);

    /* 7. MUTUAL EXCLUSION. With both kernel switches on, the online arm takes the
     *    dispatch and the four-pass GQA predicate must answer false, or the
     *    encoder's two counters would both claim one dispatch. */
    setenv("SURGE_ATTN_SPLITK_GQA", "1", 1);
    e = sg_gpu_state_new(g_gpu, &m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (both switches on): %s",
              e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        tt_assert(sg_gpu_splitk_online_selected(g_gpu, 4, MINI_KV_HEADS,
                                                m.cfg.head_dim, deep),
                  "with both switches on the online arm must be selected");
        tt_assert(!sg_gpu_splitk_gqa_selected(g_gpu, 4, MINI_KV_HEADS, deep),
                  "with both switches on the FOUR-PASS GQA predicate must answer "
                  "false for the state's own shape, or one dispatch would be "
                  "counted twice");
        /* And with only the GQA switch on, the four-pass arm is selected again,
         * so the line above is about precedence and not a blanket false. */
        setenv("SURGE_ATTN_SPLITK_ONLINE", "0", 1);
        e = sg_gpu_state_new(g_gpu, &m, n);
        tt_assert(!sg_failed(e), "sg_gpu_state_new (gqa only): %s",
                  e.msg ? e.msg : "ok");
        if (!sg_failed(e)) {
            tt_assert(sg_gpu_splitk_gqa_selected(g_gpu, 4, MINI_KV_HEADS, deep),
                      "with only SURGE_ATTN_SPLITK_GQA=1 the four-pass GQA arm must "
                      "be selected at seq %u", deep);
        }
    }

    /* 8. Split-K off means online off: the online kernel reads half-typed
     *    separate K and V, and the incumbent path has no split-K at all. */
    setenv("SURGE_ATTN_SPLITK", "0", 1);
    setenv("SURGE_ATTN_SPLITK_ONLINE", "1", 1);
    e = sg_gpu_state_new(g_gpu, &m, n);
    tt_assert(!sg_failed(e), "sg_gpu_state_new (splitk off, online on): %s",
              e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        tt_assert(!sg_gpu_splitk_online_selected(g_gpu, 24, 4, 256, deep),
                  "with SURGE_ATTN_SPLITK=0 the online arm must be off too");
    }

    /* 9. No forward ran on any of these states, so the counter must be 0. This is
     *    the anti-vacuity anchor for the deferred end-to-end gate: that gate's
     *    claim is that the counter becomes NONZERO. */
    tt_assert(sg_gpu_splitk_online_dispatches(g_gpu) == 0,
              "no decode step ran, so the online dispatch count must be 0, got %llu",
              (unsigned long long)sg_gpu_splitk_online_dispatches(g_gpu));

    unsetenv("SURGE_ATTN_SPLITK_ONLINE");
    unsetenv("SURGE_ATTN_SPLITK_GQA");
    unsetenv("SURGE_ATTN_SPLITK");
    sg_model_free(&m);
    sg_gguf_close(gg);
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
    /* P2.6: the same decode path with the GQA split cap overridden, so the two
     * arms genuinely pick DIFFERENT n_splits and the greedy tokens are gated in
     * the regime P2.5 introduced. Placed after the P2.4 control rather than
     * before it as belt-and-braces, NOT because the order is required: each
     * subtest sets its own env and clears SURGE_SPLITK_GQA_CAP when it finishes,
     * and tt_assert counts rather than aborts, so the cleanup runs even on
     * failure. The order only matters if this subtest dies between its setenv and
     * its unsetenv, which would leak the override into the P2.4 control's
     * byte-identity assertion. */
    tt_run("mini_f16_splitk_gqa_cap_override_greedy_matches",
           mini_f16_splitk_gqa_cap_override_greedy_matches);
    /* P2.8: the online-softmax arm's env parse and selection policy. No forwards,
     * so it costs nothing and runs last; the end-to-end token comparison for that
     * arm is a deferred real-model gate, for the reason stated at the subtest. */
    tt_run("mini_f16_splitk_online_policy", mini_f16_splitk_online_policy);
    unsetenv("SURGE_KV_DTYPE");

    fprintf(stderr, "worst relative logit gap vs ref: %.3e\n", g_worst_rel);
    sg_gpu_free(g_gpu);
    return tt_report();
}

#endif /* SURGE_NO_METAL */
