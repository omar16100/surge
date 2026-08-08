/* test_model.c - tests for model_qwen.c's config extraction + weight-name
 * mapping (sg_model_from_gguf, sg_model_from_st).
 *
 * Fully env-gated on SURGE_GGUF and SURGE_ST (auto-skip with a notice when
 * unset, matching test_tok.c's precedent for tests that need a real,
 * multi-GB model on disk rather than a small synthetic fixture -- there is
 * no fixture-based test here because Task 6 is a pure name-mapping layer
 * over gguf.c/st.c, which already have their own fixture coverage).
 *
 * Real-file correction pinned by these tests, discovered during
 * reconnaissance and NOT anticipated by the original task brief: both real
 * checkpoints (Qwen3.6-27B-Q8_0.gguf and the Qwen3.5-2B safetensors dir) are
 * a HYBRID of full-attention and linear-attention (gated Delta-Net / SSM)
 * layers, not a uniform dense stack. Layer 0 is always a linear-attention
 * layer in both checkpoints (full_attention_interval=4, first full-attention
 * layer is index 3); layer n_layers-1 happens to be full-attention in both
 * (64 and 24 are both multiples of 4). So instead of the brief's literal
 * "layer 0 and n_layers-1 have every pointer non-NULL", these tests assert:
 *   - layer 3 and layer n_layers-1 (both known full-attention): every
 *     sg_layer_w pointer non-NULL.
 *   - layer 0 (known linear-attention): the shared MLP/norm pointers
 *     (gate_proj, up_proj, down_proj, ln1, ln2) non-NULL, but the
 *     attention-only pointers (q_proj, k_proj, v_proj, o_proj, q_norm,
 *     k_norm) NULL -- pinning the hybrid-architecture discovery as a
 *     regression check rather than silently working around it.
 *
 * Second correction: full-attention layers gate their q_proj 2x wide
 * (attn_output_gate in the real config.json / Qwen3NextAttention's q_proj
 * producing num_attention_heads*head_dim*2, confirmed against real tensor
 * shapes below), so "n_heads * head_dim == q_proj row count" from the brief
 * does not hold; the real relation is n_heads*head_dim*2 == q_proj rows,
 * k/v stay unscaled (n_kv_heads*head_dim exactly, no gate).
 */
#include "tinytest.h"
#include "../surge.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool nearly_eq(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static void assert_cfg_sane(const sg_cfg *c, const char *label) {
    tt_assert(c->n_layers > 0, "%s: n_layers should be nonzero", label);
    tt_assert(c->n_heads > 0, "%s: n_heads should be nonzero", label);
    tt_assert(c->n_kv_heads > 0, "%s: n_kv_heads should be nonzero", label);
    tt_assert(c->head_dim > 0, "%s: head_dim should be nonzero", label);
    tt_assert(c->hidden > 0, "%s: hidden should be nonzero", label);
    tt_assert(c->ffn_hidden > 0, "%s: ffn_hidden should be nonzero", label);
    tt_assert(c->vocab > 0, "%s: vocab should be nonzero", label);
    tt_assert(c->rope_theta > 0.0f, "%s: rope_theta should be nonzero", label);
    tt_assert(c->rms_eps > 0.0f, "%s: rms_eps should be nonzero", label);
    tt_assert(c->n_heads % c->n_kv_heads == 0,
              "%s: n_kv_heads (%u) should divide n_heads (%u)",
              label, c->n_kv_heads, c->n_heads);
}

static void assert_layer_full_attn(const sg_layer_w *lw, uint32_t idx, const char *label) {
    tt_assert(lw->q_proj != NULL, "%s: layer %u q_proj should be non-NULL", label, idx);
    tt_assert(lw->k_proj != NULL, "%s: layer %u k_proj should be non-NULL", label, idx);
    tt_assert(lw->v_proj != NULL, "%s: layer %u v_proj should be non-NULL", label, idx);
    tt_assert(lw->o_proj != NULL, "%s: layer %u o_proj should be non-NULL", label, idx);
    tt_assert(lw->q_norm != NULL, "%s: layer %u q_norm should be non-NULL", label, idx);
    tt_assert(lw->k_norm != NULL, "%s: layer %u k_norm should be non-NULL", label, idx);
    tt_assert(lw->gate_proj != NULL, "%s: layer %u gate_proj should be non-NULL", label, idx);
    tt_assert(lw->up_proj != NULL, "%s: layer %u up_proj should be non-NULL", label, idx);
    tt_assert(lw->down_proj != NULL, "%s: layer %u down_proj should be non-NULL", label, idx);
    tt_assert(lw->ln1 != NULL, "%s: layer %u ln1 should be non-NULL", label, idx);
    tt_assert(lw->ln2 != NULL, "%s: layer %u ln2 should be non-NULL", label, idx);
}

static void assert_layer_linear_attn(const sg_layer_w *lw, uint32_t idx, const char *label) {
    tt_assert(lw->q_proj == NULL, "%s: layer %u (linear-attn) q_proj should be NULL", label, idx);
    tt_assert(lw->k_proj == NULL, "%s: layer %u (linear-attn) k_proj should be NULL", label, idx);
    tt_assert(lw->v_proj == NULL, "%s: layer %u (linear-attn) v_proj should be NULL", label, idx);
    tt_assert(lw->o_proj == NULL, "%s: layer %u (linear-attn) o_proj should be NULL", label, idx);
    tt_assert(lw->q_norm == NULL, "%s: layer %u (linear-attn) q_norm should be NULL", label, idx);
    tt_assert(lw->k_norm == NULL, "%s: layer %u (linear-attn) k_norm should be NULL", label, idx);
    tt_assert(lw->gate_proj != NULL, "%s: layer %u gate_proj should still be non-NULL", label, idx);
    tt_assert(lw->up_proj != NULL, "%s: layer %u up_proj should still be non-NULL", label, idx);
    tt_assert(lw->down_proj != NULL, "%s: layer %u down_proj should still be non-NULL", label, idx);
    tt_assert(lw->ln1 != NULL, "%s: layer %u ln1 should still be non-NULL", label, idx);
    tt_assert(lw->ln2 != NULL, "%s: layer %u ln2 should still be non-NULL", label, idx);
}

static float g_gguf_rope_theta = 0.0f, g_gguf_rms_eps = 0.0f;
static uint32_t g_gguf_vocab = 0;
static bool g_gguf_ran = false;

static float g_st_rope_theta = 0.0f, g_st_rms_eps = 0.0f;
static uint32_t g_st_vocab = 0;
static bool g_st_ran = false;

static void model_from_gguf_real(void) {
    const char *path = getenv("SURGE_GGUF");
    if (!path || !*path) {
        fprintf(stderr, "SKIP: SURGE_GGUF not set; skipping real-model gguf model test "
                        "(set it to a qwen35 gguf, e.g. "
                        "/Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf)\n");
        return;
    }

    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open(path, &g);
    tt_assert(!sg_failed(e), "open %s should succeed: %s", path, e.msg ? e.msg : "");
    if (!g) return;

    sg_model m;
    e = sg_model_from_gguf(g, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf should succeed: %s", e.msg ? e.msg : "");
    if (sg_failed(e)) { sg_gguf_close(g); return; }

    assert_cfg_sane(&m.cfg, "gguf");

    /* Real Qwen3.6-27B-Q8_0.gguf dims (recon'd against the actual file). */
    tt_assert(m.cfg.n_layers == 64, "gguf n_layers should be 64, got %u", m.cfg.n_layers);
    tt_assert(m.cfg.n_heads == 24, "gguf n_heads should be 24, got %u", m.cfg.n_heads);
    tt_assert(m.cfg.n_kv_heads == 4, "gguf n_kv_heads should be 4, got %u", m.cfg.n_kv_heads);
    tt_assert(m.cfg.head_dim == 256, "gguf head_dim should be 256, got %u", m.cfg.head_dim);
    tt_assert(m.cfg.hidden == 5120, "gguf hidden should be 5120, got %u", m.cfg.hidden);
    tt_assert(m.cfg.vocab == 248320, "gguf vocab should be 248320, got %u", m.cfg.vocab);
    tt_assert(nearly_eq(m.cfg.rope_theta, 1e7f, 1.0f),
              "gguf rope_theta should be 1e7, got %f", (double)m.cfg.rope_theta);
    tt_assert(!m.cfg.tied_embeddings, "gguf output.weight is present, embeddings should not be tied");
    tt_assert(m.wtype == SG_T_Q8_0, "gguf wtype should be SG_T_Q8_0, got %d", (int)m.wtype);

    tt_assert(m.tok_emb != NULL, "gguf tok_emb should be non-NULL");
    tt_assert(m.out_norm != NULL, "gguf out_norm should be non-NULL");
    tt_assert(m.lm_head != NULL, "gguf lm_head should be non-NULL");
    tt_assert(m.lm_head != m.tok_emb, "gguf lm_head should differ from tok_emb when untied");

    /* head_dim must come from the explicit key_length metadata key, not
     * hidden/heads: 5120/24 isn't an integer. */
    tt_assert(m.cfg.hidden % m.cfg.n_heads != 0,
              "sanity: hidden (%u) should NOT divide evenly by n_heads (%u) in this "
              "checkpoint, or the head_dim-from-metadata requirement isn't exercised",
              m.cfg.hidden, m.cfg.n_heads);

    assert_layer_full_attn(&m.layers[3], 3, "gguf");
    assert_layer_full_attn(&m.layers[m.cfg.n_layers - 1], m.cfg.n_layers - 1, "gguf");
    assert_layer_linear_attn(&m.layers[0], 0, "gguf");

    /* Cross-check the mapping's wiring (not just non-NULL) and the real q
     * gate factor directly against the tensor directory. */
    const sg_tensor *q3 = sg_gguf_tensor(g, "blk.3.attn_q.weight");
    const sg_tensor *k3 = sg_gguf_tensor(g, "blk.3.attn_k.weight");
    tt_assert(q3 != NULL && k3 != NULL, "blk.3.attn_q/k.weight should be found directly");
    if (q3 && k3) {
        tt_assert(m.layers[3].q_proj == q3->data,
                  "gguf layer 3 q_proj should point at blk.3.attn_q.weight's data");
        tt_assert(m.layers[3].k_proj == k3->data,
                  "gguf layer 3 k_proj should point at blk.3.attn_k.weight's data");
        uint64_t q_out = q3->dims[q3->n_dims - 1];
        uint64_t k_out = k3->dims[k3->n_dims - 1];
        tt_assert(q_out == 2ull * m.cfg.n_heads * m.cfg.head_dim,
                  "gguf q_proj out rows (%llu) should be 2*n_heads*head_dim (%llu) -- "
                  "full-attention layers gate q_proj 2x wide in this checkpoint",
                  (unsigned long long)q_out,
                  (unsigned long long)(2ull * m.cfg.n_heads * m.cfg.head_dim));
        tt_assert(k_out == (uint64_t)m.cfg.n_kv_heads * m.cfg.head_dim,
                  "gguf k_proj out rows (%llu) should be n_kv_heads*head_dim (%llu), no gate",
                  (unsigned long long)k_out,
                  (unsigned long long)((uint64_t)m.cfg.n_kv_heads * m.cfg.head_dim));
    }

    g_gguf_rope_theta = m.cfg.rope_theta;
    g_gguf_rms_eps = m.cfg.rms_eps;
    g_gguf_vocab = m.cfg.vocab;
    g_gguf_ran = true;

    sg_model_free(&m);
    sg_gguf_close(g);
}

static void model_from_st_real(void) {
    const char *dir = getenv("SURGE_ST");
    if (!dir || !*dir) {
        fprintf(stderr, "SKIP: SURGE_ST not set; skipping real-model st model test "
                        "(set it to a safetensors model dir, e.g. "
                        "/Users/macmini/models/qwen35-2b)\n");
        return;
    }

    sg_st *s = NULL;
    sg_err e = sg_st_open(dir, &s);
    tt_assert(!sg_failed(e), "open %s should succeed: %s", dir, e.msg ? e.msg : "");
    if (!s) return;

    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(!sg_failed(e), "sg_model_from_st should succeed: %s", e.msg ? e.msg : "");
    if (sg_failed(e)) { sg_st_close(s); return; }

    assert_cfg_sane(&m.cfg, "st");

    /* Real Qwen3.5-2B config.json dims (recon'd against the actual file). */
    tt_assert(m.cfg.n_layers == 24, "st n_layers should be 24, got %u", m.cfg.n_layers);
    tt_assert(m.cfg.n_heads == 8, "st n_heads should be 8, got %u", m.cfg.n_heads);
    tt_assert(m.cfg.n_kv_heads == 2, "st n_kv_heads should be 2, got %u", m.cfg.n_kv_heads);
    tt_assert(m.cfg.head_dim == 256, "st head_dim should be 256, got %u", m.cfg.head_dim);
    tt_assert(m.cfg.hidden == 2048, "st hidden should be 2048, got %u", m.cfg.hidden);
    tt_assert(m.cfg.vocab == 248320, "st vocab should be 248320, got %u", m.cfg.vocab);
    tt_assert(nearly_eq(m.cfg.rope_theta, 1e7f, 1.0f),
              "st rope_theta should be 1e7, got %f", (double)m.cfg.rope_theta);
    tt_assert(m.cfg.tied_embeddings, "st checkpoint has tie_word_embeddings=true, lm_head.weight absent");
    tt_assert(m.wtype == SG_T_BF16, "st wtype should be SG_T_BF16, got %d", (int)m.wtype);

    tt_assert(m.tok_emb != NULL, "st tok_emb should be non-NULL");
    tt_assert(m.out_norm != NULL, "st out_norm should be non-NULL");
    tt_assert(m.lm_head != NULL, "st lm_head should be non-NULL");
    tt_assert(m.lm_head == m.tok_emb, "st lm_head should equal tok_emb when tied");

    assert_layer_full_attn(&m.layers[3], 3, "st");
    assert_layer_full_attn(&m.layers[m.cfg.n_layers - 1], m.cfg.n_layers - 1, "st");
    assert_layer_linear_attn(&m.layers[0], 0, "st");

    const uint16_t *q3 = NULL, *k3 = NULL;
    uint64_t q_dims[4] = {0}, k_dims[4] = {0};
    uint32_t q_nd = 0, k_nd = 0;
    bool found_q = sg_st_tensor(s, "model.language_model.layers.3.self_attn.q_proj.weight",
                                 &q3, q_dims, &q_nd);
    bool found_k = sg_st_tensor(s, "model.language_model.layers.3.self_attn.k_proj.weight",
                                 &k3, k_dims, &k_nd);
    tt_assert(found_q && found_k, "layer 3 self_attn q/k_proj should be found directly");
    if (found_q && found_k) {
        tt_assert(m.layers[3].q_proj == q3, "st layer 3 q_proj should point at the same data");
        tt_assert(m.layers[3].k_proj == k3, "st layer 3 k_proj should point at the same data");
        /* safetensors shape is [out_features, in_features]. */
        tt_assert(q_dims[0] == 2ull * m.cfg.n_heads * m.cfg.head_dim,
                  "st q_proj out rows (%llu) should be 2*n_heads*head_dim (%llu) -- "
                  "attn_output_gate doubles q_proj width",
                  (unsigned long long)q_dims[0],
                  (unsigned long long)(2ull * m.cfg.n_heads * m.cfg.head_dim));
        tt_assert(k_dims[0] == (uint64_t)m.cfg.n_kv_heads * m.cfg.head_dim,
                  "st k_proj out rows (%llu) should be n_kv_heads*head_dim (%llu), no gate",
                  (unsigned long long)k_dims[0],
                  (unsigned long long)((uint64_t)m.cfg.n_kv_heads * m.cfg.head_dim));
    }

    g_st_rope_theta = m.cfg.rope_theta;
    g_st_rms_eps = m.cfg.rms_eps;
    g_st_vocab = m.cfg.vocab;
    g_st_ran = true;

    sg_model_free(&m);
    sg_st_close(s);
}

/* "The two loaders agree on cfg for their respective models where
 * comparable" (brief, Step 1): the 27B gguf and 2B safetensors checkpoints
 * are different model sizes (different n_layers/hidden/heads), so those
 * fields can't match -- but they share the same tokenizer vocab and the
 * same rope_theta/rms_eps hyperparameter family, which should agree. */
static void model_loaders_agree_where_comparable(void) {
    if (!g_gguf_ran || !g_st_ran) {
        fprintf(stderr, "SKIP: cross-loader agreement check needs both "
                        "SURGE_GGUF and SURGE_ST\n");
        return;
    }
    tt_assert(g_gguf_vocab == g_st_vocab,
              "gguf vocab (%u) and st vocab (%u) should agree (same tokenizer)",
              g_gguf_vocab, g_st_vocab);
    tt_assert(nearly_eq(g_gguf_rope_theta, g_st_rope_theta, 1.0f),
              "gguf rope_theta (%f) and st rope_theta (%f) should agree",
              (double)g_gguf_rope_theta, (double)g_st_rope_theta);
    tt_assert(nearly_eq(g_gguf_rms_eps, g_st_rms_eps, 1e-9f),
              "gguf rms_eps (%f) and st rms_eps (%f) should agree",
              (double)g_gguf_rms_eps, (double)g_st_rms_eps);
}

static void model_from_gguf_null_args(void) {
    sg_model m;
    sg_err e = sg_model_from_gguf(NULL, &m);
    tt_assert(sg_failed(e), "sg_model_from_gguf(NULL, ...) should fail");
}

static void model_from_st_null_args(void) {
    sg_model m;
    sg_err e = sg_model_from_st(NULL, &m);
    tt_assert(sg_failed(e), "sg_model_from_st(NULL, ...) should fail");
}

int main(void) {
    tt_run("model_from_gguf_null_args", model_from_gguf_null_args);
    tt_run("model_from_st_null_args", model_from_st_null_args);
    tt_run("model_from_gguf_real", model_from_gguf_real);
    tt_run("model_from_st_real", model_from_st_real);
    tt_run("model_loaders_agree_where_comparable", model_loaders_agree_where_comparable);
    return tt_report();
}
