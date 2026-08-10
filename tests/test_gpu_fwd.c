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
    tt_run("mini_st_matches_ref", mini_st_matches_ref);
    tt_run("mini_gguf_matches_ref", mini_gguf_matches_ref);
    tt_run("q8_loads_and_decodes", q8_loads_and_decodes);
    fprintf(stderr, "worst relative logit gap vs ref: %.3e\n", g_worst_rel);
    sg_gpu_free(g_gpu);
    return tt_report();
}

#endif /* SURGE_NO_METAL */
