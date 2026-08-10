/* test_model.c - tests for model_qwen.c's config extraction + weight-name
 * mapping (sg_model_from_gguf, sg_model_from_st).
 *
 * Coverage comes in two tiers. The mini-model fixture tests
 * (model_from_st_mini_fixture, model_from_st_rejects_partial_group) run
 * UNCONDITIONALLY against tests/fixtures/mini_model_st{,_partial}/ and cover
 * the whole safetensors mapping including the linear-attention path; Task 6
 * originally had no fixture tier at all, which meant a plain `make check`
 * ran two null-argument assertions and nothing else.
 *
 * The real-model tests stay env-gated on SURGE_GGUF and SURGE_ST (there is
 * no synthetic substitute for a multi-GB checkpoint), and SURGE_GGUF_TWIN
 * gates the GGUF-vs-HF cross-check. Skipped gates are recorded and reported
 * as a loud banner at the end of the run rather than a line scrolling past,
 * because "0 failures" from a run where the gates never executed is exactly
 * the kind of green that hides a problem.
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

/* The nine DeltaNet pointers, as one group. Task 7 added these; the two
 * assert_layer_* helpers below check that the ssm group and the attention
 * group are exact complements on every layer, which is what makes
 * "layer kind == which tensors are present" a safe rule for the forward
 * pass to dispatch on. */
#define SSM_FIELDS(X) \
    X(ssm_in_qkv) X(ssm_in_z) X(ssm_in_b) X(ssm_in_a) X(ssm_a) \
    X(ssm_dt_bias) X(ssm_conv1d) X(ssm_norm) X(ssm_out)

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
#define X(f) tt_assert(lw->f == NULL, \
        "%s: layer %u (full-attn) " #f " should be NULL", label, idx);
    SSM_FIELDS(X)
#undef X
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
#define X(f) tt_assert(lw->f != NULL, \
        "%s: layer %u (linear-attn) " #f " should be non-NULL", label, idx);
    SSM_FIELDS(X)
#undef X
}

/* Mean of an F32 norm-weight vector, used to pin the +1.0 scale difference
 * between the two sources (see the long note in surge.h's sg_layer_w
 * comment): mlx's TextModel.sanitize adds 1.0 to the input_layernorm /
 * post_attention_layernorm / model.norm / q_norm / k_norm weights when the
 * checkpoint carries mtp.* tensors or an unsanitized conv1d weight. The real
 * 2B safetensors checkpoint carries both and is therefore stored
 * "residual" (centred near 0); the GGUF converter baked the shift in, so the
 * GGUF's are absolute (centred near 1). A forward pass that feeds either one
 * straight into sg_ref_rmsnorm without knowing which it has is wrong by a
 * factor of ~10 on the safetensors path, so this is asserted, not assumed. */
static double f32_mean(const void *p, uint64_t n) {
    const float *v = (const float *)p;
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) s += (double)v[i];
    return n ? s / (double)n : 0.0;
}

static double bf16_mean(const void *p, uint64_t n) {
    const uint16_t *v = (const uint16_t *)p;
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)v[i] << 16;
        float f;
        memcpy(&f, &bits, sizeof f);
        s += (double)f;
    }
    return n ? s / (double)n : 0.0;
}

/* Skipped-gate registry, reported as a banner by main(). */
#define MAX_SKIPS 8
static struct { const char *env; const char *covers; const char *hint; } g_skips[MAX_SKIPS];
static int g_n_skips = 0;

static void note_skip(const char *env, const char *covers, const char *hint) {
    if (g_n_skips < MAX_SKIPS) {
        g_skips[g_n_skips].env = env;
        g_skips[g_n_skips].covers = covers;
        g_skips[g_n_skips].hint = hint;
        g_n_skips++;
    }
    fprintf(stderr, "SKIP (%s unset): %s\n", env, covers);
}

static void report_skips(void) {
    if (g_n_skips == 0) {
        fprintf(stderr, "all real-model gates RAN (no env vars unset)\n");
        return;
    }
    fprintf(stderr,
            "\n!! %d real-model gate(s) DID NOT RUN. The check count above excludes\n"
            "!! them, so a clean result here is NOT evidence about the real files.\n",
            g_n_skips);
    for (int i = 0; i < g_n_skips; i++) {
        fprintf(stderr, "!!   %-16s %s\n", g_skips[i].env, g_skips[i].covers);
        fprintf(stderr, "!!   %-16s   set e.g. %s\n", "", g_skips[i].hint);
    }
    fputc('\n', stderr);
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
        note_skip("SURGE_GGUF",
                  "real-GGUF config, hybrid layer mapping, DeltaNet tensor shapes, "
                  "norm-weight scale (~110 checks)",
                  "SURGE_GGUF=/Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf");
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

    /* Linear-attention layer wiring, checked against the file directly and
     * against the DeltaNet shape relations from qwen3_5.py (Task 7). The
     * qwen35.ssm.* metadata keys name the DeltaNet dims: group_count is the
     * key-head count, state_size the key head dim, time_step_rank the
     * value-head count, inner_size the total value width, conv_kernel the
     * short conv's kernel. Layer 0 is a linear-attention layer. */
    uint32_t k_heads = 0, k_head_dim = 0, v_heads = 0, v_dim = 0, conv_k = 0;
    bool ssm_meta = sg_gguf_get_u32(g, "qwen35.ssm.group_count", &k_heads)
                 && sg_gguf_get_u32(g, "qwen35.ssm.state_size", &k_head_dim)
                 && sg_gguf_get_u32(g, "qwen35.ssm.time_step_rank", &v_heads)
                 && sg_gguf_get_u32(g, "qwen35.ssm.inner_size", &v_dim)
                 && sg_gguf_get_u32(g, "qwen35.ssm.conv_kernel", &conv_k);
    tt_assert(ssm_meta, "gguf should carry the qwen35.ssm.* DeltaNet metadata keys");
    if (ssm_meta) {
        uint32_t key_dim = k_heads * k_head_dim;
        uint32_t conv_dim = 2 * key_dim + v_dim;
        const sg_tensor *qkv = sg_gguf_tensor(g, "blk.0.attn_qkv.weight");
        const sg_tensor *z = sg_gguf_tensor(g, "blk.0.attn_gate.weight");
        const sg_tensor *cv = sg_gguf_tensor(g, "blk.0.ssm_conv1d.weight");
        const sg_tensor *al = sg_gguf_tensor(g, "blk.0.ssm_a");
        const sg_tensor *dt = sg_gguf_tensor(g, "blk.0.ssm_dt.bias");
        const sg_tensor *nw = sg_gguf_tensor(g, "blk.0.ssm_norm.weight");
        const sg_tensor *ow = sg_gguf_tensor(g, "blk.0.ssm_out.weight");
        const sg_tensor *ba = sg_gguf_tensor(g, "blk.0.ssm_beta.weight");
        const sg_tensor *aa = sg_gguf_tensor(g, "blk.0.ssm_alpha.weight");
        tt_assert(qkv && z && cv && al && dt && nw && ow && ba && aa,
                  "all nine blk.0 DeltaNet tensors should be found directly");
        if (qkv && z && cv && al && dt && nw && ow && ba && aa) {
            tt_assert(m.layers[0].ssm_in_qkv == qkv->data,
                      "gguf layer 0 ssm_in_qkv should point at blk.0.attn_qkv.weight");
            tt_assert(m.layers[0].ssm_in_z == z->data,
                      "gguf layer 0 ssm_in_z should point at blk.0.attn_gate.weight");
            tt_assert(m.layers[0].ssm_in_b == ba->data,
                      "gguf layer 0 ssm_in_b should point at blk.0.ssm_beta.weight");
            tt_assert(m.layers[0].ssm_in_a == aa->data,
                      "gguf layer 0 ssm_in_a should point at blk.0.ssm_alpha.weight");
            tt_assert(m.layers[0].ssm_a == al->data,
                      "gguf layer 0 ssm_a should point at blk.0.ssm_a");
            tt_assert(m.layers[0].ssm_conv1d == cv->data,
                      "gguf layer 0 ssm_conv1d should point at blk.0.ssm_conv1d.weight");

            /* GGUF stores dims fastest-axis-first, so a [in, out] linear is
             * dims[0]=in, dims[1]=out. */
            tt_assert(qkv->dims[1] == conv_dim,
                      "blk.0.attn_qkv out width (%llu) should be 2*key_dim+value_dim (%u)",
                      (unsigned long long)qkv->dims[1], conv_dim);
            tt_assert(qkv->dims[0] == m.cfg.hidden,
                      "blk.0.attn_qkv in width (%llu) should be hidden (%u)",
                      (unsigned long long)qkv->dims[0], m.cfg.hidden);
            tt_assert(z->dims[1] == v_dim,
                      "blk.0.attn_gate out width (%llu) should be value_dim (%u) -- it is "
                      "the DeltaNet's in_proj_z, despite the attn_ name prefix",
                      (unsigned long long)z->dims[1], v_dim);
            tt_assert(cv->dims[0] == conv_k && cv->dims[1] == conv_dim,
                      "blk.0.ssm_conv1d.weight should be [conv_kernel %u, conv_dim %u], "
                      "got [%llu, %llu]", conv_k, conv_dim,
                      (unsigned long long)cv->dims[0], (unsigned long long)cv->dims[1]);
            tt_assert(al->dims[0] == v_heads && dt->dims[0] == v_heads,
                      "blk.0.ssm_a and ssm_dt.bias should both be [num_v_heads %u], "
                      "got [%llu] and [%llu]", v_heads,
                      (unsigned long long)al->dims[0], (unsigned long long)dt->dims[0]);
            tt_assert(v_heads != 0 && nw->dims[0] == v_dim / v_heads,
                      "blk.0.ssm_norm.weight should be [head_v_dim %u], got [%llu]",
                      v_heads ? v_dim / v_heads : 0, (unsigned long long)nw->dims[0]);
            tt_assert(ow->dims[0] == v_dim && ow->dims[1] == m.cfg.hidden,
                      "blk.0.ssm_out.weight should be [value_dim %u, hidden %u], got [%llu, %llu]",
                      v_dim, m.cfg.hidden,
                      (unsigned long long)ow->dims[0], (unsigned long long)ow->dims[1]);
            tt_assert(ba->dims[1] == v_heads && aa->dims[1] == v_heads,
                      "blk.0.ssm_beta/alpha out widths should both be num_v_heads (%u), "
                      "got %llu and %llu", v_heads,
                      (unsigned long long)ba->dims[1], (unsigned long long)aa->dims[1]);
        }
        /* A full-attention layer must carry none of them. */
        tt_assert(sg_gguf_tensor(g, "blk.3.ssm_conv1d.weight") == NULL,
                  "blk.3 (full-attention) should have no ssm_conv1d.weight");
        tt_assert(sg_gguf_tensor(g, "blk.3.attn_qkv.weight") == NULL,
                  "blk.3 (full-attention) should have no attn_qkv.weight");
    }

    /* Norm-weight scale: the GGUF's are absolute (mlx's +1.0 sanitize shift
     * is already baked in by the converter), so their means sit near 1. All
     * of these are F32 in the GGUF even though the matmul weights are Q8_0. */
    tt_assert(m.layers[0].ln1 != NULL && m.layers[3].q_norm != NULL,
              "gguf norm pointers needed for the scale check");
    if (m.layers[0].ln1 && m.layers[3].q_norm) {
        double ln1_mean = f32_mean(m.layers[0].ln1, m.cfg.hidden);
        double qn_mean = f32_mean(m.layers[3].q_norm, m.cfg.head_dim);
        tt_assert(ln1_mean > 0.5,
                  "gguf blk.0.attn_norm.weight mean is %.4f; the GGUF's norm weights "
                  "should be ABSOLUTE (near 1), i.e. mlx's +1.0 sanitize shift already "
                  "applied by the converter", ln1_mean);
        tt_assert(qn_mean > 0.5,
                  "gguf blk.3.attn_q_norm.weight mean is %.4f; expected absolute (near 1)",
                  qn_mean);
        fprintf(stderr, "   gguf norm means: attn_norm %.4f, attn_q_norm %.4f "
                        "(absolute scale)\n", ln1_mean, qn_mean);
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
        note_skip("SURGE_ST",
                  "real-safetensors config, hybrid layer mapping, F32/BF16 DeltaNet "
                  "dtypes, norm-weight residual scale (~55 checks)",
                  "SURGE_ST=/Users/macmini/models/qwen35-2b");
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

    /* Linear-attention layer wiring (Task 7). config.json's text_config
     * names the DeltaNet dims directly; st.c's config lookup already
     * descends into text_config. Layer 0 is a linear-attention layer. */
    uint32_t k_heads = 0, k_head_dim = 0, v_heads = 0, v_head_dim = 0, conv_k = 0;
    bool ssm_cfg = sg_st_config_u32(s, "linear_num_key_heads", &k_heads)
                && sg_st_config_u32(s, "linear_key_head_dim", &k_head_dim)
                && sg_st_config_u32(s, "linear_num_value_heads", &v_heads)
                && sg_st_config_u32(s, "linear_value_head_dim", &v_head_dim)
                && sg_st_config_u32(s, "linear_conv_kernel_dim", &conv_k);
    tt_assert(ssm_cfg, "config.json should carry the linear_* DeltaNet dims");
    if (ssm_cfg) {
        uint32_t key_dim = k_heads * k_head_dim;
        uint32_t v_dim = v_heads * v_head_dim;
        uint32_t conv_dim = 2 * key_dim + v_dim;

        const uint16_t *qkv = NULL, *z = NULL, *cv = NULL, *dt = NULL,
                       *ow = NULL, *bw = NULL, *aw = NULL;
        const float *al = NULL, *nw = NULL;
        uint64_t d_qkv[4] = {0}, d_z[4] = {0}, d_cv[4] = {0}, d_al[4] = {0},
                 d_dt[4] = {0}, d_nw[4] = {0}, d_ow[4] = {0}, d_bw[4] = {0}, d_aw[4] = {0};
        uint32_t nd_cv = 0;
#define ST_L0(nm, var, dims, nd) \
        sg_st_tensor(s, "model.language_model.layers.0.linear_attn." nm, &var, dims, nd)
        bool found = ST_L0("in_proj_qkv.weight", qkv, d_qkv, NULL)
                  && ST_L0("in_proj_z.weight", z, d_z, NULL)
                  && ST_L0("conv1d.weight", cv, d_cv, &nd_cv)
                  && ST_L0("dt_bias", dt, d_dt, NULL)
                  && ST_L0("out_proj.weight", ow, d_ow, NULL)
                  && ST_L0("in_proj_b.weight", bw, d_bw, NULL)
                  && ST_L0("in_proj_a.weight", aw, d_aw, NULL);
#undef ST_L0
        /* A_log and norm.weight are F32 in this checkpoint while the other
         * seven tensors of the same layer are BF16 -- sg_st_tensor cannot
         * see them at all, which is why sg_st_tensor_f32 exists. */
        found = found
             && sg_st_tensor_f32(s, "model.language_model.layers.0.linear_attn.A_log",
                                 &al, d_al, NULL)
             && sg_st_tensor_f32(s, "model.language_model.layers.0.linear_attn.norm.weight",
                                 &nw, d_nw, NULL);
        tt_assert(found, "all nine layer-0 linear_attn tensors should be found directly");

        tt_assert(!sg_st_tensor(s, "model.language_model.layers.0.linear_attn.A_log",
                                NULL, NULL, NULL),
                  "linear_attn.A_log is F32, so the bf16 accessor must NOT return it");

        if (found) {
            tt_assert(m.layers[0].ssm_in_qkv == qkv, "st layer 0 ssm_in_qkv wiring");
            tt_assert(m.layers[0].ssm_in_z == z, "st layer 0 ssm_in_z wiring");
            tt_assert(m.layers[0].ssm_in_b == bw, "st layer 0 ssm_in_b wiring");
            tt_assert(m.layers[0].ssm_in_a == aw, "st layer 0 ssm_in_a wiring");
            tt_assert(m.layers[0].ssm_a == al, "st layer 0 ssm_a wiring");
            tt_assert(m.layers[0].ssm_dt_bias == dt, "st layer 0 ssm_dt_bias wiring");
            tt_assert(m.layers[0].ssm_conv1d == cv, "st layer 0 ssm_conv1d wiring");
            tt_assert(m.layers[0].ssm_norm == nw, "st layer 0 ssm_norm wiring");
            tt_assert(m.layers[0].ssm_out == ow, "st layer 0 ssm_out wiring");

            /* safetensors shape is [out_features, in_features]. */
            tt_assert(d_qkv[0] == conv_dim && d_qkv[1] == m.cfg.hidden,
                      "in_proj_qkv should be [2*key_dim+value_dim %u, hidden %u], got [%llu, %llu]",
                      conv_dim, m.cfg.hidden,
                      (unsigned long long)d_qkv[0], (unsigned long long)d_qkv[1]);
            tt_assert(d_z[0] == v_dim, "in_proj_z out width (%llu) should be value_dim (%u)",
                      (unsigned long long)d_z[0], v_dim);
            /* Unsanitized HF layout: [conv_dim, 1, kernel]. mlx's sanitize
             * moves the axes to [conv_dim, kernel, 1]; surge reads the raw
             * checkpoint, so the kernel is the LAST axis here. */
            tt_assert(nd_cv == 3 && d_cv[0] == conv_dim && d_cv[1] == 1 && d_cv[2] == conv_k,
                      "conv1d.weight should be [conv_dim %u, 1, kernel %u], got rank %u "
                      "[%llu, %llu, %llu]", conv_dim, conv_k, nd_cv,
                      (unsigned long long)d_cv[0], (unsigned long long)d_cv[1],
                      (unsigned long long)d_cv[2]);
            tt_assert(d_al[0] == v_heads && d_dt[0] == v_heads,
                      "A_log and dt_bias should both be [num_v_heads %u], got [%llu] and [%llu]",
                      v_heads, (unsigned long long)d_al[0], (unsigned long long)d_dt[0]);
            tt_assert(d_nw[0] == v_head_dim,
                      "linear_attn.norm.weight should be [value_head_dim %u], got [%llu]",
                      v_head_dim, (unsigned long long)d_nw[0]);
            tt_assert(d_ow[0] == m.cfg.hidden && d_ow[1] == v_dim,
                      "out_proj should be [hidden %u, value_dim %u], got [%llu, %llu]",
                      m.cfg.hidden, v_dim,
                      (unsigned long long)d_ow[0], (unsigned long long)d_ow[1]);
            tt_assert(d_bw[0] == v_heads && d_aw[0] == v_heads,
                      "in_proj_b/in_proj_a out widths should both be num_v_heads (%u), "
                      "got %llu and %llu", v_heads,
                      (unsigned long long)d_bw[0], (unsigned long long)d_aw[0]);
        }

        tt_assert(!sg_st_tensor(s, "model.language_model.layers.3.linear_attn.in_proj_qkv.weight",
                                NULL, NULL, NULL),
                  "layer 3 (full-attention) should have no linear_attn.in_proj_qkv.weight");
    }

    /* The mirror of the GGUF check above, and the one that matters: this
     * checkpoint carries mtp.* tensors AND an unsanitized conv1d weight
     * ([6144, 1, 4], last axis != 1), so mlx's TextModel.sanitize adds 1.0
     * to these five norm families at load time. The file therefore stores
     * them centred near 0, and a forward pass must add the 1.0 back. The
     * assertion is deliberately two-sided so that a future checkpoint which
     * flips the convention fails loudly here instead of producing quietly
     * wrong logits. ssm_norm is NOT in mlx's shift list and is checked to
     * confirm it stays on the absolute scale. */
    tt_assert(m.layers[0].ln1 != NULL && m.layers[3].q_norm != NULL
              && m.layers[0].ssm_norm != NULL,
              "st norm pointers needed for the scale check");
    if (m.layers[0].ln1 && m.layers[3].q_norm && m.layers[0].ssm_norm) {
        double ln1_mean = bf16_mean(m.layers[0].ln1, m.cfg.hidden);
        double qn_mean = bf16_mean(m.layers[3].q_norm, m.cfg.head_dim);
        double ssm_mean = f32_mean(m.layers[0].ssm_norm, 128);
        tt_assert(ln1_mean < 0.5,
                  "st layers.0.input_layernorm.weight mean is %.4f; this checkpoint "
                  "stores norm weights RESIDUAL (near 0) and mlx's sanitize adds 1.0 "
                  "at load time -- the forward pass must do the same", ln1_mean);
        tt_assert(qn_mean < 0.5,
                  "st layers.3.self_attn.q_norm.weight mean is %.4f; expected residual "
                  "(near 0), same +1.0 shift as input_layernorm", qn_mean);
        tt_assert(ssm_mean > 0.5,
                  "st layers.0.linear_attn.norm.weight mean is %.4f; ssm_norm is NOT in "
                  "mlx's sanitize shift list and should already be absolute (near 1)",
                  ssm_mean);
        fprintf(stderr, "   st norm means: input_layernorm %.4f, q_norm %.4f "
                        "(residual, need +1.0), linear_attn.norm %.4f (absolute)\n",
                ln1_mean, qn_mean, ssm_mean);
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
                        "SURGE_GGUF and SURGE_ST (already reported above)\n");
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

/* ---------------------------------------------------------------------
 * Ungated coverage of the safetensors DeltaNet mapping.
 *
 * tests/fixtures/mini_model_st/ is a 2-layer hybrid model (layer 0
 * linear-attention, layer 1 full-attention) small enough to commit. Before
 * it existed, every line of model_qwen.c's linear-attention mapping was
 * reachable only under the SURGE_ST-gated real-model test, so a plain
 * `make check` ran exactly two assertions in this file and none of the
 * mapping. The fixture also deliberately stores linear_attn.A_log as BF16
 * and linear_attn.norm.weight as F32, because the two real checkpoints
 * disagree about those dtypes and both probe branches need exercising.
 * --------------------------------------------------------------------- */

static void model_from_st_mini_fixture(void) {
    sg_st *s = NULL;
    sg_err e = sg_st_open("tests/fixtures/mini_model_st", &s);
    tt_assert(!sg_failed(e), "open mini_model_st should succeed: %s", e.msg ? e.msg : "");
    if (!s) return;

    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(!sg_failed(e), "sg_model_from_st on the mini model should succeed: %s",
              e.msg ? e.msg : "");
    if (sg_failed(e)) { sg_st_close(s); return; }

    tt_assert(m.cfg.n_layers == 2, "mini n_layers should be 2, got %u", m.cfg.n_layers);
    tt_assert(m.cfg.hidden == 8, "mini hidden should be 8, got %u", m.cfg.hidden);
    tt_assert(m.cfg.head_dim == 4, "mini head_dim should be 4, got %u", m.cfg.head_dim);
    tt_assert(m.cfg.tied_embeddings, "mini model has no lm_head.weight, should be tied");

    /* The whole point: both layer kinds map correctly with no real model. */
    assert_layer_linear_attn(&m.layers[0], 0, "mini_st");
    assert_layer_full_attn(&m.layers[1], 1, "mini_st");

    /* Per-source DeltaNet semantics for a safetensors model. */
    tt_assert(m.ssm_a_form == SG_SSM_A_LOG,
              "safetensors ssm_a is mlx's A_log verbatim, form should be SG_SSM_A_LOG");
    tt_assert(!m.v_heads_tiled,
              "safetensors uses mlx's grouped value-head order, v_heads_tiled should be false");
    /* Both dtype-probe branches: BF16 A_log, F32 norm weight. */
    tt_assert(m.ssm_a_type == SG_T_BF16,
              "mini A_log is BF16, ssm_a_type should be SG_T_BF16, got %d", (int)m.ssm_a_type);
    tt_assert(m.ssm_norm_type == SG_T_F32,
              "mini linear_attn.norm.weight is F32, ssm_norm_type should be SG_T_F32, got %d",
              (int)m.ssm_norm_type);

    /* And that the pointers really are the tensors they claim to be. */
    const uint16_t *qkv = NULL;
    const float *nw = NULL;
    uint64_t d[4] = {0};
    bool ok = sg_st_tensor(s, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                           &qkv, d, NULL);
    tt_assert(ok, "mini layer 0 in_proj_qkv should be directly findable");
    if (ok) {
        tt_assert(m.layers[0].ssm_in_qkv == qkv, "mini ssm_in_qkv wiring");
        /* conv_dim = 2*key_dim + value_dim = 2*(2*4) + (4*4) = 32. */
        tt_assert(d[0] == 32 && d[1] == 8,
                  "mini in_proj_qkv should be [32, 8], got [%llu, %llu]",
                  (unsigned long long)d[0], (unsigned long long)d[1]);
    }
    ok = sg_st_tensor_f32(s, "model.language_model.layers.0.linear_attn.norm.weight",
                          &nw, NULL, NULL);
    tt_assert(ok, "mini layer 0 linear_attn.norm.weight should be findable as F32");
    if (ok) tt_assert(m.layers[0].ssm_norm == nw, "mini ssm_norm wiring");

    sg_model_free(&m);
    sg_st_close(s);
}

/* The group invariant has to reject a partial group, not just accept
 * complete ones: the partial fixture is the same model with
 * linear_attn.dt_bias removed, which before this check loaded "fine" with a
 * NULL sitting in the middle of an otherwise-populated DeltaNet layer. */
static void model_from_st_rejects_partial_group(void) {
    sg_st *s = NULL;
    sg_err e = sg_st_open("tests/fixtures/mini_model_st_partial", &s);
    tt_assert(!sg_failed(e), "open mini_model_st_partial should succeed: %s",
              e.msg ? e.msg : "");
    if (!s) return;

    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(sg_failed(e),
              "a layer with 8 of 9 DeltaNet tensors should be rejected, not loaded");
    if (!sg_failed(e)) sg_model_free(&m);
    else fprintf(stderr, "   (rejected as expected: %s)\n", e.msg);
    sg_st_close(s);
}

/* ---------------------------------------------------------------------
 * The GGUF verification that Task 7's first report wrongly deferred to the
 * M1 gate.
 *
 * That deferral was structurally impossible: M1 runs the ref forward against
 * mlx-lm on the 2B SAFETENSORS checkpoint, so it never touches a GGUF at
 * all; there is no mlx oracle for a GGUF; and the 2B has n_v_heads ==
 * n_k_heads (16 and 16), so the value-head ordering question cannot even
 * manifest there. Nothing about the GGUF's DeltaNet contract would have been
 * tested by it.
 *
 * What actually settles it is a checkpoint pair: the GGUF and an HF
 * safetensors copy of THE SAME MODEL. Then ssm_a can be compared elementwise
 * against A_log, which pins two things at once:
 *
 *   1. WHAT ssm_a HOLDS. If the GGUF stored A_log verbatim, ssm_a would
 *      equal A_log. It does not; it equals -exp(A_log). The decay gate must
 *      therefore skip the exp() and the negation on this path
 *      (sg_ref_delta_decay_neg_a, selected by sg_model.ssm_a_form).
 *   2. THE VALUE-HEAD ORDER. The equality only holds after un-tiling: GGUF
 *      value head p carries HF value head g where p = (g % repeat) * n_k +
 *      (g / repeat), which is exactly the statement that the GGUF's key-head
 *      map is p %% n_k_heads while HF's is g / repeat. Under the identity
 *      ordering the two disagree by ~50, so this is not a near-miss.
 *
 * Gated on SURGE_GGUF_TWIN, an HF safetensors dir of the same model as
 * SURGE_GGUF (e.g. /Users/macmini/models/qwen36-27b-8bit, an mlx 8-bit
 * repack of Qwen3.6-27B; its A_log and dt_bias stay unquantized, which is
 * all this needs). The quantized in_proj_* tensors are not compared: the
 * reorder permutation is shared across every reordered tensor, so pinning it
 * on A_log pins it everywhere, and dequantizing mlx's affine format here
 * would add a second thing that could be wrong.
 * --------------------------------------------------------------------- */

/* Reads through sg_st_read_f32, not a pointer accessor: the twin checkpoint
 * is written by mlx, which does not align its data sections, so most of its
 * tensors (A_log included) sit at odd byte offsets and the pointer
 * accessors correctly refuse them. */
static float twin_read1(const sg_st *s, const char *name, uint32_t i, bool *ok) {
    float v = 0.0f;
    *ok = sg_st_read_f32(s, name, i, 1, &v);
    return v;
}

static void gguf_ssm_a_and_head_order_vs_hf_twin(void) {
    const char *gpath = getenv("SURGE_GGUF");
    const char *tdir = getenv("SURGE_GGUF_TWIN");
    if (!gpath || !*gpath || !tdir || !*tdir) {
        note_skip("SURGE_GGUF_TWIN",
                  "GGUF ssm_a semantics (-exp(A_log)) and tiled value-head order, "
                  "cross-checked against an HF copy of the same model",
                  "SURGE_GGUF_TWIN=/Users/macmini/models/qwen36-27b-8bit "
                  "(needs SURGE_GGUF too)");
        return;
    }

    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open(gpath, &g);
    tt_assert(!sg_failed(e), "open %s should succeed: %s", gpath, e.msg ? e.msg : "");
    if (!g) return;
    sg_st *s = NULL;
    e = sg_st_open(tdir, &s);
    tt_assert(!sg_failed(e), "open twin %s should succeed: %s", tdir, e.msg ? e.msg : "");
    if (!s) { sg_gguf_close(g); return; }

    uint32_t n_k = 0, n_v = 0;
    bool have = sg_gguf_get_u32(g, "qwen35.ssm.group_count", &n_k)
             && sg_gguf_get_u32(g, "qwen35.ssm.time_step_rank", &n_v);
    tt_assert(have, "gguf should carry ssm.group_count and ssm.time_step_rank");
    if (!have || n_k == 0 || n_v % n_k != 0) { sg_st_close(s); sg_gguf_close(g); return; }
    tt_assert(n_v > n_k,
              "the twin cross-check is only meaningful when n_v_heads (%u) > n_k_heads "
              "(%u); with them equal the two head orders coincide", n_v, n_k);
    uint32_t repeat = n_v / n_k;

    /* Layers 0, 1, 2 are linear-attention (full_attention_interval 4). */
    static const uint32_t layers[] = {0, 1, 2};
    uint32_t checked = 0;
    double worst_tiled = 0.0, worst_identity = 0.0;
    int neg_total = 0, neg_count = 0;

    for (size_t li = 0; li < sizeof layers / sizeof layers[0]; li++) {
        uint32_t L = layers[li];
        char gname[96], tname[192];
        snprintf(gname, sizeof gname, "blk.%u.ssm_a", L);
        snprintf(tname, sizeof tname,
                 "language_model.model.layers.%u.linear_attn.A_log", L);

        const sg_tensor *ga = sg_gguf_tensor(g, gname);
        if (!ga) continue;
        tt_assert(ga->type == SG_T_F32, "%s should be F32, got %d", gname, (int)ga->type);
        tt_assert(ga->dims[0] == n_v, "%s should be [n_v_heads %u], got [%llu]",
                  gname, n_v, (unsigned long long)ga->dims[0]);
        if (ga->type != SG_T_F32 || ga->dims[0] != n_v) continue;
        const float *gv = (const float *)ga->data;

        bool ok = false;
        (void)twin_read1(s, tname, 0, &ok);
        if (!ok) continue;

        for (uint32_t gi = 0; gi < n_v; gi++) {
            float a_log = twin_read1(s, tname, gi, &ok);
            if (!ok) break;
            double want = -exp((double)a_log);
            /* HF head gi sits at GGUF slot p under the converter's reorder. */
            uint32_t p = (gi % repeat) * n_k + (gi / repeat);
            double d_tiled = fabs((double)gv[p] - want);
            double d_ident = fabs((double)gv[gi] - want);
            if (d_tiled > worst_tiled) worst_tiled = d_tiled;
            if (d_ident > worst_identity) worst_identity = d_ident;
            /* And the key-head map that reorder implies. */
            tt_assert(sg_ssm_k_head(p, n_k, n_v, true) == gi / repeat,
                      "layer %u: GGUF slot %u should tile to the same key head (%u) that "
                      "HF head %u groups to (%u)", L, p,
                      sg_ssm_k_head(p, n_k, n_v, true), gi, gi / repeat);
            neg_total++;
            if (gv[gi] < 0.0f) neg_count++;
        }
        checked++;
    }

    tt_assert(checked > 0, "no linear-attention layer could be cross-checked; is "
                           "SURGE_GGUF_TWIN really a copy of the same model?");
    if (checked == 0) { sg_st_close(s); sg_gguf_close(g); return; }

    /* bf16 A_log round-tripped through exp() lands around 1e-6 relative;
     * the values reach ~140 in magnitude, so allow a little absolute room. */
    tt_assert(worst_tiled < 1e-3,
              "GGUF ssm_a should equal -exp(HF A_log) under the TILED value-head "
              "reindex, but the worst mismatch is %.3e", worst_tiled);
    /* And the wrong reading must be unmistakably wrong, not marginally so:
     * if this ever got small, the test would have stopped proving anything. */
    tt_assert(worst_identity > 1.0,
              "the identity (non-reordered) value-head reading should be grossly "
              "wrong, but its worst mismatch is only %.3e -- the tiled-vs-grouped "
              "distinction is no longer being tested", worst_identity);
    tt_assert(neg_count == neg_total,
              "all GGUF ssm_a values should be strictly negative (they are "
              "-exp(A_log)), got %d of %d", neg_count, neg_total);

    fprintf(stderr,
            "   twin cross-check over %u linear-attn layers, %u v-heads each:\n"
            "     ssm_a vs -exp(A_log), tiled reindex   max |err| %.3e  (PASS)\n"
            "     ssm_a vs -exp(A_log), identity order  max |err| %.3e  (must be large)\n"
            "     ssm_a strictly negative               %d/%d\n",
            checked, n_v, worst_tiled, worst_identity, neg_count, neg_total);

    sg_st_close(s);
    sg_gguf_close(g);
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
    tt_run("model_from_st_mini_fixture", model_from_st_mini_fixture);
    tt_run("model_from_st_rejects_partial_group", model_from_st_rejects_partial_group);
    tt_run("model_from_gguf_real", model_from_gguf_real);
    tt_run("model_from_st_real", model_from_st_real);
    tt_run("model_loaders_agree_where_comparable", model_loaders_agree_where_comparable);
    tt_run("gguf_ssm_a_and_head_order_vs_hf_twin", gguf_ssm_a_and_head_order_vs_hf_twin);
    report_skips();
    return tt_report();
}
