/* test_ref_fwd.c - the reference forward pass against mlx-lm's own logits.
 *
 * Two tiers, mirroring how every other part of this project is validated:
 *
 * TIER 1 (ungated, milliseconds): tests/fixtures/mini_fwd/ is a complete
 * 4-layer hybrid qwen3_5 checkpoint -- three gated-DeltaNet layers and one
 * full-attention layer -- with random but architecturally real weights, plus
 * the logits a real mlx_lm.models.qwen3_5.Model produced for a fixed
 * 12-token sequence, teacher-forced. sg_ref_forward has to reproduce all 12
 * positions. That pins the whole composition: layer-type dispatch, the
 * per-layer union state (KV cache vs conv tail + delta S-matrix), the GQA
 * head map, the DeltaNet value-head map, the partial-RoPE width, the
 * attention output gate, RMSNormGated, and -- the reason the fixture stores
 * an unsanitized conv1d and residual norm weights -- the +1.0 norm shift.
 *
 * TIER 2 (SURGE_ST-gated, minutes): the same comparison on the real
 * Qwen3.5-2B is the M1 gate, and it lives outside `make check` because it
 * needs a 4.5 GB checkpoint and mlx-lm. It is run by tools/tf_compare.py
 * against src/cli_ref.c's --logits dump. What runs here under SURGE_ST is
 * the cheap half of it: load the real model, assert the config and the
 * norm-shift decision, and run ONE position, comparing against the frozen
 * first-position logits of the first M1 fixture when it is present.
 *
 * The fixture dims are chosen so no head map is degenerate; see MINI_FWD in
 * tools/make_fixtures.py for the reasoning behind each one.
 */
#include "surge.h"
#include "tinytest.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MINI_DIR "tests/fixtures/mini_fwd"

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "FATAL: out of memory\n"); exit(2); }
    return p;
}

/* Reads a whole file of raw little-endian f32 (the same format cli_ref's
 * --logits writes, so the mini fixture and the M1 dumps read identically). */
static float *read_f32_file(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || sz % 4 != 0) { fclose(f); return NULL; }
    rewind(f);
    float *v = xmalloc((size_t)sz ? (size_t)sz : 4);
    if (sz > 0 && fread(v, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(v); return NULL;
    }
    fclose(f);
    *n_out = (size_t)sz / 4;
    return v;
}

/* Whole file into a NUL-terminated buffer; no fixed-size read, because a
 * silently truncated id list would compare a different sequence than the one
 * the fixture was frozen from. */
static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = xmalloc((size_t)sz + 1);
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return NULL;
    }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* Parses one comma-separated line of ids. Advances *cursor past the line. */
static int32_t *parse_id_line(const char **cursor, size_t *n_out) {
    const char *p = *cursor;
    while (*p == '\n' || *p == '\r') p++;
    if (!*p) { *cursor = p; *n_out = 0; return NULL; }

    size_t cap = 64, n = 0;
    int32_t *ids = xmalloc(cap * sizeof *ids);
    while (*p && *p != '\n' && *p != '\r') {
        while (*p == ',' || *p == ' ') p++;
        if (!*p || *p == '\n' || *p == '\r') break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (n == cap) {
            cap *= 2;
            int32_t *g = realloc(ids, cap * sizeof *ids);
            if (!g) { free(ids); fprintf(stderr, "FATAL: out of memory\n"); exit(2); }
            ids = g;
        }
        ids[n++] = (int32_t)v;
        p = end;
    }
    while (*p == '\n' || *p == '\r') p++;
    *cursor = p;
    *n_out = n;
    return ids;
}

static int32_t *read_ids_file(const char *path, size_t *n_out) {
    char *buf = read_text_file(path);
    if (!buf) return NULL;
    const char *cursor = buf;
    int32_t *ids = parse_id_line(&cursor, n_out);
    free(buf);
    return ids;
}

/* Reports max |delta| and the top-1 argmax agreement, which are exactly the
 * two numbers the M1 gate is stated in. */
static void check_logits(const char *what, const float *got, const float *want,
                         uint32_t positions, uint32_t vocab, float tol) {
    double worst = 0.0;
    uint32_t top1_ok = 0;
    uint32_t worst_pos = 0;
    for (uint32_t t = 0; t < positions; t++) {
        const float *g = got + (size_t)t * vocab, *w = want + (size_t)t * vocab;
        uint32_t ag = 0, aw = 0;
        for (uint32_t i = 1; i < vocab; i++) {
            if (g[i] > g[ag]) ag = i;
            if (w[i] > w[aw]) aw = i;
        }
        if (ag == aw) top1_ok++;
        for (uint32_t i = 0; i < vocab; i++) {
            double d = fabs((double)g[i] - (double)w[i]);
            if (d > worst) { worst = d; worst_pos = t; }
        }
    }
    fprintf(stderr, "   %-46s top-1 %u/%u, max |delta| %.3e (tol %g, worst at pos %u)\n",
            what, top1_ok, positions, worst, (double)tol, worst_pos);
    tt_assert(top1_ok == positions, "%s: top-1 agreement %u/%u", what, top1_ok, positions);
    tt_assert(worst <= (double)tol, "%s: max |delta| %.6e exceeds %g", what, worst,
              (double)tol);
}

/* --------------------------------------------------------------------
 * Tier 1: the whole model against mlx, ungated
 * -------------------------------------------------------------------- */

static void mini_fwd_config_is_as_written(const sg_model *m) {
    const sg_cfg *c = &m->cfg;
    tt_assert(c->n_layers == 4, "n_layers %u", c->n_layers);
    tt_assert(c->hidden == 32 && c->ffn_hidden == 24 && c->vocab == 40,
              "hidden %u ffn %u vocab %u", c->hidden, c->ffn_hidden, c->vocab);
    tt_assert(c->n_heads == 4 && c->n_kv_heads == 2 && c->head_dim == 16,
              "heads %u kv %u head_dim %u", c->n_heads, c->n_kv_heads, c->head_dim);
    /* int(16 * 0.25); the truncation and the "< head_dim" are both load-bearing. */
    tt_assert(c->rope_dim == 4, "rope_dim %u (want int(head_dim*0.25))", c->rope_dim);
    tt_assert(c->full_attn_interval == 4, "full_attn_interval %u", c->full_attn_interval);
    tt_assert(c->n_k_heads == 2 && c->n_v_heads == 4, "nk %u nv %u",
              c->n_k_heads, c->n_v_heads);
    tt_assert(c->head_k_dim == 32 && c->head_v_dim == 32, "dk %u dv %u",
              c->head_k_dim, c->head_v_dim);
    tt_assert(c->conv_kernel == 4, "conv_kernel %u", c->conv_kernel);
    tt_assert(c->tied_embeddings && m->lm_head == m->tok_emb,
              "the fixture ties its embeddings, as the real 2B does");
    /* The fixture's eps must NOT be the 1e-6 that qwen3_5.py hardcodes for
     * the DeltaNet q/k normalization, or the two eps values coincide and
     * confusing them in ref.c becomes bit-identical here (and on both real
     * checkpoints, which also use 1e-6). See MINI_FWD in
     * tools/make_fixtures.py; a regeneration must not weaken this. */
    tt_assert(c->rms_eps > 1e-5f,
              "mini_fwd's rms_norm_eps (%g) must differ from the DeltaNet q/k "
              "hardcoded 1e-6, or the two eps paths are indistinguishable",
              (double)c->rms_eps);

    /* The fixture stores conv1d.weight unsanitized ([C,1,K]) precisely so
     * this decision has to be made correctly; getting it wrong shifts every
     * norm in the model by 1.0 and is otherwise silent. */
    tt_assert(m->norms_are_residual,
              "mini_fwd stores residual norm weights (unsanitized conv1d), "
              "so norms_are_residual must be true");
    tt_assert(m->dense_type == SG_T_BF16, "dense_type %d", (int)m->dense_type);
    tt_assert(m->ssm_a_type == SG_T_F32, "ssm_a_type %d", (int)m->ssm_a_type);
    tt_assert(m->ssm_norm_type == SG_T_F32, "ssm_norm_type %d", (int)m->ssm_norm_type);
    tt_assert(m->ssm_a_form == SG_SSM_A_LOG, "safetensors ssm_a is A_log");
    tt_assert(!m->v_heads_tiled, "safetensors value heads are grouped, not tiled");

    /* Layer kinds, the thing the whole hybrid rests on. */
    for (uint32_t i = 0; i < c->n_layers; i++) {
        bool attn = m->layers[i].q_proj != NULL;
        tt_assert(attn == (i == 3), "layer %u should be %s", i,
                  (i == 3) ? "full-attention" : "gated-DeltaNet");
    }
}

static void mini_fwd_matches_mlx(void) {
    sg_st *s = NULL;
    sg_err e = sg_st_open(MINI_DIR, &s);
    tt_assert(!sg_failed(e), "sg_st_open(%s): %s", MINI_DIR, e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(!sg_failed(e), "sg_model_from_st: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_st_close(s); return; }

    mini_fwd_config_is_as_written(&m);

    size_t n_ids = 0, n_ref = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    float *ref = read_f32_file(MINI_DIR "/logits.f32", &n_ref);
    tt_assert(ids != NULL && n_ids > 1, "read %s/ids.txt", MINI_DIR);
    tt_assert(ref != NULL && n_ids > 0 && n_ref == n_ids * m.cfg.vocab,
              "%s/logits.f32 has %zu floats, want %zu x %u", MINI_DIR, n_ref,
              n_ids, m.cfg.vocab);
    if (!ids || !ref || n_ref != n_ids * m.cfg.vocab) {
        free(ids); free(ref); sg_model_free(&m); sg_st_close(s); return;
    }

    sg_ref_state *st = NULL;
    e = sg_ref_state_new(&m, (uint32_t)n_ids, &st);
    tt_assert(!sg_failed(e), "sg_ref_state_new: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); free(ref); sg_model_free(&m); sg_st_close(s); return; }

    float *got = xmalloc(n_ids * m.cfg.vocab * sizeof *got);
    for (uint32_t t = 0; t < n_ids; t++) {
        const float *lg = NULL;
        e = sg_ref_forward(st, &m, ids[t], t, &lg);
        tt_assert(!sg_failed(e), "sg_ref_forward pos %u: %s", t, e.msg ? e.msg : "ok");
        if (sg_failed(e)) break;
        memcpy(got + (size_t)t * m.cfg.vocab, lg, m.cfg.vocab * sizeof *got);
    }
    /* f32 mlx vs f32-with-double-accumulators C over 4 layers: the gap is
     * arithmetic order only, and it is three orders of magnitude tighter
     * than the M1 gate's 1e-2. */
    check_logits("mini hybrid model, 12 positions vs mlx", got, ref,
                 (uint32_t)n_ids, m.cfg.vocab, 1e-4f);

    /* The state must be reusable: rewinding and re-running the same
     * sequence has to give bit-identical logits, or the M1 dumps are not
     * reproducible and Task 10's byte-exact gate has no floor. */
    sg_ref_state_reset(st);
    uint64_t identical = 0;
    for (uint32_t t = 0; t < n_ids; t++) {
        const float *lg = NULL;
        if (sg_failed(sg_ref_forward(st, &m, ids[t], t, &lg))) break;
        for (uint32_t i = 0; i < m.cfg.vocab; i++) {
            if (lg[i] == got[(size_t)t * m.cfg.vocab + i]) identical++;
        }
    }
    tt_assert(identical == (uint64_t)n_ids * m.cfg.vocab,
              "after sg_ref_state_reset the rerun matched %llu/%llu logits bit-exactly",
              (unsigned long long)identical,
              (unsigned long long)((uint64_t)n_ids * m.cfg.vocab));

    free(got);
    free(ids);
    free(ref);
    sg_ref_state_free(st);
    sg_model_free(&m);
    sg_st_close(s);
}

/* The same model, the same mlx logits, through the GGUF path.
 *
 * tests/fixtures/mini_fwd/model.gguf is the mini checkpoint after every
 * transform the GGUF converter applies: norms with the +1.0 already baked
 * in, ssm_a holding -exp(A_log), the value heads in TILED order, dims
 * reversed, and every tensor F32. Four code paths exist only here and are
 * reachable by nothing else in the suite:
 *
 *   sg_model.norms_are_residual == false   (no shift at use)
 *   sg_model.ssm_a_form == SG_SSM_A_NEG_EXP
 *   sg_model.v_heads_tiled == true         (sg_ssm_k_head's other branch)
 *   sg_model.dense_type == SG_T_F32
 *
 * The 2B cannot exercise the head map at all (n_v_heads == n_k_heads), and
 * the real 27B has no mlx oracle. This fixture has 4 value heads over 2 key
 * heads, so the tiled and grouped maps genuinely differ.
 *
 * Scope limit, stated so nobody over-reads this: the fixture is BUILT from
 * the same rule surge implements, so it proves the forward USES the rule,
 * not that the rule is right. The rule's correctness is pinned separately by
 * tests/test_model.c's SURGE_GGUF_TWIN cross-check against an HF copy of the
 * real 27B. */
static void mini_fwd_gguf_matches_mlx(void) {
    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &g);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    sg_model m;
    e = sg_model_from_gguf(g, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(g); return; }

    tt_assert(m.ssm_a_form == SG_SSM_A_NEG_EXP, "gguf ssm_a is -exp(A_log)");
    tt_assert(m.v_heads_tiled, "gguf value heads are tiled");
    tt_assert(!m.norms_are_residual, "gguf norms are absolute");
    tt_assert(m.dense_type == SG_T_F32, "gguf small tensors are F32");
    tt_assert(m.wtype == SG_T_F32, "this fixture stores its matmuls as F32");
    tt_assert(m.cfg.n_v_heads == 4 && m.cfg.n_k_heads == 2,
              "the gguf fixture must have n_v_heads != n_k_heads (%u, %u), or the "
              "tiled and grouped head maps are indistinguishable",
              m.cfg.n_v_heads, m.cfg.n_k_heads);
    tt_assert(sg_ssm_k_head(1, m.cfg.n_k_heads, m.cfg.n_v_heads, true)
                  != sg_ssm_k_head(1, m.cfg.n_k_heads, m.cfg.n_v_heads, false),
              "the two head maps must actually disagree on this shape");
    tt_assert(m.cfg.rope_dim == 4 && m.cfg.full_attn_interval == 4,
              "gguf rope_dim %u interval %u", m.cfg.rope_dim, m.cfg.full_attn_interval);

    size_t n_ids = 0, n_ref = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    float *ref = read_f32_file(MINI_DIR "/logits.f32", &n_ref);
    if (!ids || !ref || n_ids == 0 || n_ref != n_ids * m.cfg.vocab) {
        tt_assert(false, "mini_fwd ids/logits fixtures are unreadable");
        free(ids); free(ref); sg_model_free(&m); sg_gguf_close(g); return;
    }

    sg_ref_state *st = NULL;
    e = sg_ref_state_new(&m, (uint32_t)n_ids, &st);
    tt_assert(!sg_failed(e), "sg_ref_state_new (gguf): %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); free(ref); sg_model_free(&m); sg_gguf_close(g); return; }

    float *got = xmalloc(n_ids * m.cfg.vocab * sizeof *got);
    for (uint32_t t = 0; t < n_ids; t++) {
        const float *lg = NULL;
        if (sg_failed(sg_ref_forward(st, &m, ids[t], t, &lg))) break;
        memcpy(got + (size_t)t * m.cfg.vocab, lg, m.cfg.vocab * sizeof *got);
    }
    check_logits("same model via GGUF, 12 positions vs mlx", got, ref,
                 (uint32_t)n_ids, m.cfg.vocab, 1e-4f);

    free(got); free(ids); free(ref);
    sg_ref_state_free(st);
    sg_model_free(&m);
    sg_gguf_close(g);
}

/* --------------------------------------------------------------------
 * API contract
 * -------------------------------------------------------------------- */

static void ref_forward_rejects_bad_input(void) {
    sg_st *s = NULL;
    if (sg_failed(sg_st_open(MINI_DIR, &s))) return;
    sg_model m;
    if (sg_failed(sg_model_from_st(s, &m))) { sg_st_close(s); return; }

    sg_ref_state *st = NULL;
    tt_assert(sg_failed(sg_ref_state_new(&m, 0, &st)), "max_ctx 0 must be rejected");
    tt_assert(sg_failed(sg_ref_state_new(NULL, 4, &st)), "a NULL model must be rejected");

    sg_err e = sg_ref_state_new(&m, 4, &st);
    tt_assert(!sg_failed(e) && st != NULL, "sg_ref_state_new: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_st_close(s); return; }

    const float *lg = NULL;
    tt_assert(sg_failed(sg_ref_forward(st, &m, -1, 0, &lg)),
              "a negative token id must be rejected");
    tt_assert(sg_failed(sg_ref_forward(st, &m, (int32_t)m.cfg.vocab, 0, &lg)),
              "a token id >= vocab must be rejected");
    tt_assert(sg_failed(sg_ref_forward(st, &m, 0, 1, &lg)),
              "starting at position 1 must be rejected (caches are append-only)");
    tt_assert(!sg_failed(sg_ref_forward(st, &m, 0, 0, &lg)) && lg != NULL,
              "position 0 must run");
    tt_assert(sg_failed(sg_ref_forward(st, &m, 0, 0, &lg)),
              "replaying position 0 must be rejected");
    tt_assert(!sg_failed(sg_ref_forward(st, &m, 1, 1, &lg)), "position 1 must run");
    tt_assert(!sg_failed(sg_ref_forward(st, &m, 1, 2, &lg)), "position 2 must run");
    tt_assert(!sg_failed(sg_ref_forward(st, &m, 1, 3, &lg)), "position 3 must run");
    tt_assert(sg_failed(sg_ref_forward(st, &m, 1, 4, &lg)),
              "position 4 must be rejected at max_ctx 4");

    /* Every buffer in the state is sized from the model it was built for, so
     * being handed a different one is a buffer overrun, not a mix-up. */
    sg_model other;
    if (!sg_failed(sg_model_from_st(s, &other))) {
        sg_ref_state_reset(st);
        tt_assert(sg_failed(sg_ref_forward(st, &other, 0, 0, &lg)),
                  "a state must refuse a different sg_model");
        sg_model_free(&other);
    }

    sg_ref_state_free(st);
    sg_ref_state_free(NULL);        /* must be a no-op, not a crash */
    sg_ref_state_reset(NULL);
    sg_model_free(&m);
    sg_st_close(s);
}

/* A model whose tensor layout contradicts full_attention_interval must be
 * refused rather than run with one of the two beliefs picked arbitrarily.
 * mini_model_st is a 2-layer model with interval 2 (layer 0 DeltaNet, layer
 * 1 attention); flipping the interval to 4 makes both layers "should be
 * DeltaNet" while layer 1 carries attention tensors. */
static void ref_state_rejects_layer_kind_mismatch(void) {
    sg_st *s = NULL;
    if (sg_failed(sg_st_open("tests/fixtures/mini_model_st", &s))) return;
    sg_model m;
    if (sg_failed(sg_model_from_st(s, &m))) { sg_st_close(s); return; }

    sg_ref_state *st = NULL;
    m.cfg.full_attn_interval = 4;
    sg_err e = sg_ref_state_new(&m, 2, &st);
    tt_assert(sg_failed(e), "a layer kind that contradicts the interval must be rejected");
    if (!sg_failed(e)) sg_ref_state_free(st);

    sg_model_free(&m);
    sg_st_close(s);
}

/* --------------------------------------------------------------------
 * Tier 2: the real 2B, gated on SURGE_ST
 * -------------------------------------------------------------------- */

static void real_model_forward_smoke(void) {
    const char *dir = getenv("SURGE_ST");
    if (!dir) {
        fprintf(stderr, "   SKIP: set SURGE_ST=/Users/macmini/models/qwen35-2b to run "
                        "one real 2B position (~2 s); the full M1 gate is "
                        "tools/tf_compare.py\n");
        return;
    }
    sg_st *s = NULL;
    sg_err e = sg_st_open(dir, &s);
    tt_assert(!sg_failed(e), "sg_st_open(%s): %s", dir, e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;
    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(!sg_failed(e), "sg_model_from_st: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_st_close(s); return; }

    tt_assert(m.cfg.rope_dim == 64 && m.cfg.head_dim == 256,
              "2B rope_dim %u of head_dim %u (want 64 of 256)", m.cfg.rope_dim,
              m.cfg.head_dim);
    tt_assert(m.cfg.full_attn_interval == 4, "2B full_attn_interval %u",
              m.cfg.full_attn_interval);
    tt_assert(m.cfg.n_k_heads == 16 && m.cfg.n_v_heads == 16
                  && m.cfg.head_k_dim == 128 && m.cfg.head_v_dim == 128
                  && m.cfg.conv_kernel == 4,
              "2B DeltaNet dims %u/%u/%u/%u/%u", m.cfg.n_k_heads, m.cfg.n_v_heads,
              m.cfg.head_k_dim, m.cfg.head_v_dim, m.cfg.conv_kernel);
    tt_assert(m.norms_are_residual,
              "the real 2B has both mtp.* tensors and an unsanitized conv1d, so "
              "its norm weights are residual");

    sg_ref_state *st = NULL;
    e = sg_ref_state_new(&m, 2, &st);
    tt_assert(!sg_failed(e), "sg_ref_state_new on the 2B: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_model_free(&m); sg_st_close(s); return; }

    const float *lg = NULL;
    e = sg_ref_forward(st, &m, 9419 /* "Hello" */, 0, &lg);
    tt_assert(!sg_failed(e), "sg_ref_forward on the 2B: %s", e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        uint32_t arg = 0;
        double sum = 0.0;
        bool finite = true;
        for (uint32_t i = 0; i < m.cfg.vocab; i++) {
            if (!isfinite(lg[i])) finite = false;
            if (lg[i] > lg[arg]) arg = i;
            sum += (double)lg[i];
        }
        fprintf(stderr, "   2B position 0: argmax %u, max logit %.4f, mean %.4f\n",
                arg, (double)lg[arg], sum / (double)m.cfg.vocab);
        tt_assert(finite, "every 2B logit must be finite");
        tt_assert(lg[arg] > 5.0f && lg[arg] < 60.0f,
                  "a trained model's top logit should be O(10), got %.4f",
                  (double)lg[arg]);
    }

    sg_ref_state_free(st);
    sg_model_free(&m);
    sg_st_close(s);
}

/* The frozen M1 regression fixture.
 *
 * tools/tf_compare.py's full dumps are 1017 MB, so what is committed is
 * tests/fixtures/m1/pNN.f32: per position, [argmax, max, mean, rms] followed
 * by DIGEST_SAMPLES logits sampled at a fixed stride. Every entry depends on
 * the whole forward, and the mean and RMS cannot be preserved by a localized
 * error, so this catches a ref.c regression without a gigabyte of fixtures.
 *
 * ALL sixteen prompts are checked, not just the first: a digest nothing reads
 * is not a fixture. Only the leading SURGE_M1_POSITIONS positions of each are
 * recomputed (default 2, about 1.3 s each on the 2B at -O2), since the point
 * is to detect a changed forward, not to re-run the gate -- tools/tf_compare.py
 * is the gate.
 *
 * The comparison is ref.c against numbers ref.c produced, so the expected
 * delta is zero; the tolerance only allows for floating-point reassociation
 * between build configurations (the -O0 sanitizer build runs this too) and
 * for the f32 storage of the mean and RMS. Measured: 2.2e-07. */
#define DIGEST_SAMPLES 64
#define DIGEST_HEADER 4
#define DIGEST_PROMPTS 16
#define DIGEST_TOL 1e-5

static void m1_frozen_digest_still_reproduces(void) {
    const char *dir = getenv("SURGE_ST");
    if (!dir) {
        fprintf(stderr, "   SKIP: SURGE_ST unset, so the 16 frozen M1 digests are "
                        "not rechecked (tests/fixtures/m1/pNN.f32)\n");
        return;
    }
    char *ids_text = read_text_file("tests/fixtures/m1/ids.txt");
    if (!ids_text) {
        fprintf(stderr, "   SKIP: tests/fixtures/m1/ids.txt absent (run "
                        "tools/tf_compare.py to create it)\n");
        return;
    }

    uint32_t per_prompt = 2;
    const char *env = getenv("SURGE_M1_POSITIONS");
    if (env) {
        unsigned long v = strtoul(env, NULL, 10);
        if (v > 0 && v < 1000) per_prompt = (uint32_t)v;
    }

    sg_st *s = NULL;
    if (sg_failed(sg_st_open(dir, &s))) { free(ids_text); return; }
    sg_model m;
    if (sg_failed(sg_model_from_st(s, &m))) { sg_st_close(s); free(ids_text); return; }

    const uint32_t row = DIGEST_HEADER + DIGEST_SAMPLES;
    const uint32_t stride = m.cfg.vocab / DIGEST_SAMPLES;
    const char *cursor = ids_text;
    double worst = 0.0;
    uint64_t checked = 0, argmax_ok = 0;
    uint32_t prompts_seen = 0;

    for (uint32_t p = 0; p < DIGEST_PROMPTS; p++) {
        size_t n_ids = 0;
        int32_t *ids = parse_id_line(&cursor, &n_ids);
        if (!ids || n_ids == 0) { free(ids); break; }

        char path[64];
        int r = snprintf(path, sizeof path, "tests/fixtures/m1/p%02u.f32", p);
        size_t n_dig = 0;
        float *dig = (r > 0 && (size_t)r < sizeof path) ? read_f32_file(path, &n_dig) : NULL;
        tt_assert(dig != NULL, "%s is missing", path);
        /* tt_assert is NON-FATAL, so every bound below needs its own bail-out;
         * without them a short or truncated fixture is a heap overread. */
        if (!dig) { free(ids); continue; }
        if (n_dig == 0 || n_dig % row != 0) {
            tt_assert(false, "%s has %zu floats, not a positive multiple of %u",
                      path, n_dig, row);
            free(dig); free(ids);
            continue;
        }
        uint64_t rows = n_dig / row;
        tt_assert(rows == n_ids, "%s has %llu positions but ids.txt line %u has %zu ids",
                  path, (unsigned long long)rows, p, n_ids);

        uint32_t want = per_prompt;
        if (want > rows) want = (uint32_t)rows;
        if (want > n_ids) want = (uint32_t)n_ids;

        sg_ref_state *st = NULL;
        sg_err e = sg_ref_state_new(&m, want, &st);
        tt_assert(!sg_failed(e), "sg_ref_state_new: %s", e.msg ? e.msg : "ok");
        if (sg_failed(e)) { free(dig); free(ids); continue; }

        for (uint32_t t = 0; t < want; t++) {
            const float *lg = NULL;
            if (sg_failed(sg_ref_forward(st, &m, ids[t], t, &lg))) break;
            const float *d = dig + (size_t)t * row;

            uint32_t arg = 0;
            double sum = 0.0, sumsq = 0.0;
            for (uint32_t i = 0; i < m.cfg.vocab; i++) {
                if (lg[i] > lg[arg]) arg = i;
                sum += (double)lg[i];
                sumsq += (double)lg[i] * (double)lg[i];
            }
            checked++;
            if ((float)arg == d[0]) argmax_ok++;
            double now[DIGEST_HEADER];
            now[0] = (double)arg;
            now[1] = (double)lg[arg];
            now[2] = sum / (double)m.cfg.vocab;
            now[3] = sqrt(sumsq / (double)m.cfg.vocab);
            for (uint32_t i = 1; i < DIGEST_HEADER; i++) {
                double dd = fabs(now[i] - (double)d[i]);
                if (dd > worst) worst = dd;
            }
            for (uint32_t i = 0; i < DIGEST_SAMPLES; i++) {
                double dd = fabs((double)lg[(size_t)i * stride]
                                 - (double)d[DIGEST_HEADER + i]);
                if (dd > worst) worst = dd;
            }
        }
        prompts_seen++;
        sg_ref_state_free(st);
        free(dig);
        free(ids);
    }

    fprintf(stderr, "   frozen M1 digests: %u prompts, %llu positions, argmax "
                    "%llu/%llu, max |delta| %.3e (tol %g)\n",
            prompts_seen, (unsigned long long)checked,
            (unsigned long long)argmax_ok, (unsigned long long)checked, worst,
            (double)DIGEST_TOL);
    tt_assert(prompts_seen == DIGEST_PROMPTS, "checked %u of %u frozen prompts",
              prompts_seen, DIGEST_PROMPTS);
    tt_assert(checked > 0, "no frozen M1 position was actually recomputed");
    tt_assert(argmax_ok == checked, "frozen M1 argmax matched %llu/%llu",
              (unsigned long long)argmax_ok, (unsigned long long)checked);
    tt_assert(worst < DIGEST_TOL, "frozen M1 digest max |delta| %.6e", worst);

    sg_model_free(&m);
    sg_st_close(s);
    free(ids_text);
}

int main(void) {
    tt_run("mini_fwd_matches_mlx", mini_fwd_matches_mlx);
    tt_run("mini_fwd_gguf_matches_mlx", mini_fwd_gguf_matches_mlx);
    tt_run("ref_forward_rejects_bad_input", ref_forward_rejects_bad_input);
    tt_run("ref_state_rejects_layer_kind_mismatch", ref_state_rejects_layer_kind_mismatch);
    tt_run("real_model_forward_smoke", real_model_forward_smoke);
    tt_run("m1_frozen_digest_still_reproduces", m1_frozen_digest_still_reproduces);
    return tt_report();
}
