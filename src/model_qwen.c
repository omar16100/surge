/* model_qwen.c - config extraction + weight-name mapping for the qwen3_5 /
 * qwen35 architecture family, from both GGUF and safetensors sources.
 *
 * This is a pure name-mapping layer: it does not copy or interpret tensor
 * data, only looks up tensors/config keys by the checkpoint's own naming
 * convention and stores the resulting pointers in sg_model. Every tensor
 * lookup here is a direct pass-through to sg_gguf_tensor()/sg_st_tensor(),
 * which already bounds-check and validate on open; this file adds no new
 * unchecked memory access.
 *
 * Real-file corrections found during reconnaissance, before writing any of
 * this (see the design spec and task-6-brief.md for the original,
 * pre-recon assumptions this corrects):
 *
 * 1. The real GGUF's general.architecture string is "qwen35", not "qwen3_5"
 *    as the design spec assumed. All qwen35.* metadata keys use that prefix.
 *    This loader reads general.architecture first and uses it verbatim as
 *    the key prefix, accepting either "qwen35" or "qwen3_5".
 *
 * 2. Both real checkpoints (Qwen3.6-27B-Q8_0.gguf and the Qwen3.5-2B
 *    safetensors dir) are NOT a uniform dense transformer, despite the task
 *    being framed as "the qwen3_5 dense architecture". They are a hybrid of
 *    full (softmax) attention layers and linear-attention (gated Delta-Net /
 *    SSM) layers, interleaved every full_attention_interval layers (4 in
 *    both checkpoints: qwen35.full_attention_interval in GGUF,
 *    text_config.full_attention_interval in config.json). Confirmed by
 *    tensor/key dumps: GGUF blk.N carries blk.N.ssm_{a,alpha,beta,conv1d,
 *    dt.bias,norm,out}* and blk.N.attn_{gate,qkv}.weight for linear-attention
 *    layers (3 out of every 4), and blk.N.attn_{q,k,v,q_norm,k_norm,output}
 *    only for the remaining full-attention layer; the safetensors dir has
 *    the analogous split via config.json's text_config.layer_types array and
 *    per-layer tensor names (linear_attn.* vs self_attn.*). Every layer,
 *    either kind, shares the same MLP (gate/up/down_proj) and pre/post norms
 *    (ln1/ln2). This loader maps names layer-by-layer regardless: the six
 *    attention-only sg_layer_w pointers (q_proj, k_proj, v_proj, o_proj,
 *    q_norm, k_norm) come back NULL, not an error, on a linear-attention
 *    layer, since the named tensor genuinely does not exist there. Callers
 *    (the eventual ref forward pass) must treat a NULL q_proj as "this layer
 *    is not a full-attention layer, do not attempt dense attention on it" --
 *    this loader does not attempt to model or map the SSM tensors.
 *
 * 3. The GGUF norm name for the pre-MLP norm is blk.N.post_attention_norm.
 *    weight, not blk.N.ffn_norm.weight as the original brief assumed --
 *    confirmed absent (ffn_norm.weight) vs present (post_attention_norm.
 *    weight) in the real tensor directory.
 *
 * 4. head_dim cannot be derived as hidden/n_heads: in the real 27B GGUF,
 *    hidden=5120 and n_heads=24 do not divide evenly. head_dim always comes
 *    from an explicit metadata key (qwen35.attention.key_length in GGUF,
 *    head_dim in config.json / text_config).
 *
 * 5. On full-attention layers, q_proj's output width is 2*n_heads*head_dim,
 *    not n_heads*head_dim: both real checkpoints set attn_output_gate=true,
 *    which folds a same-width gate into q_proj (confirmed against real
 *    tensor shapes: GGUF blk.3.attn_q.weight is [5120,12288] where
 *    12288 = 2*24*256; safetensors layer 3's self_attn.q_proj.weight is
 *    [4096,2048] where 4096 = 2*8*256). k_proj/v_proj stay unscaled
 *    (n_kv_heads*head_dim exactly). This loader still just hands back the
 *    raw q_proj pointer; splitting query from gate is the forward pass's
 *    job, not this mapping layer's.
 *
 * 6. The safetensors checkpoint is multimodal-wrapped: every text-model
 *    tensor is nested one level deeper than the plain HF convention, under
 *    "language_model" (e.g. model.language_model.embed_tokens.weight, not
 *    model.embed_tokens.weight), and config.json nests most hyperparameters
 *    (including rope_theta, two levels deep under
 *    text_config.rope_parameters.rope_theta) under "text_config". Every
 *    tensor lookup here tries the plain HF name first (for a future
 *    pure-text checkpoint), then falls back to the language_model-nested
 *    name. The config.json text_config and rope_parameters fallbacks are
 *    handled by st.c's sg_st_config_u32/f32 (st_config_lookup), extended
 *    there for the rope_parameters nesting.
 *
 * 7. lm_head.weight (safetensors) / output.weight (GGUF) tied-embeddings
 *    detection is done by tensor presence, not by reading the
 *    tie_word_embeddings config bool: the GGUF side has no such key at all,
 *    and using presence keeps both loaders' tied-embeddings logic
 *    symmetric and directly tied to what this loader can actually act on.
 *    Independently verified consistent with the real checkpoint's
 *    tie_word_embeddings=true / lm_head.weight-absent.
 */
#include "surge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool gguf_arch_recognized(const char *a) {
    return a && (strcmp(a, "qwen35") == 0 || strcmp(a, "qwen3_5") == 0);
}

sg_err sg_model_from_gguf(const sg_gguf *g, sg_model *m) {
    if (!m) return (sg_err){"model: invalid arguments"};
    memset(m, 0, sizeof(*m));
    if (!g) return (sg_err){"model: invalid arguments"};

    const char *arch = NULL;
    if (!sg_gguf_get_str(g, "general.architecture", &arch) || !gguf_arch_recognized(arch)) {
        return (sg_err){"model: unrecognized gguf architecture (expected qwen35 or qwen3_5)"};
    }

    sg_cfg cfg = {0};
    char key[160];

#define GGUF_U32(suffix, dst, errmsg) do { \
        int _r = snprintf(key, sizeof key, "%s.%s", arch, suffix); \
        if (_r <= 0 || (size_t)_r >= sizeof key) return (sg_err){"model: gguf metadata key too long"}; \
        if (!sg_gguf_get_u32(g, key, &(dst))) return (sg_err){errmsg}; \
    } while (0)
#define GGUF_F32(suffix, dst, errmsg) do { \
        int _r = snprintf(key, sizeof key, "%s.%s", arch, suffix); \
        if (_r <= 0 || (size_t)_r >= sizeof key) return (sg_err){"model: gguf metadata key too long"}; \
        if (!sg_gguf_get_f32(g, key, &(dst))) return (sg_err){errmsg}; \
    } while (0)

    GGUF_U32("block_count", cfg.n_layers, "model: gguf missing <arch>.block_count");
    GGUF_U32("attention.head_count", cfg.n_heads, "model: gguf missing <arch>.attention.head_count");
    GGUF_U32("attention.head_count_kv", cfg.n_kv_heads,
             "model: gguf missing <arch>.attention.head_count_kv");
    /* From the explicit key_length, NOT hidden/heads (see file header note 4). */
    GGUF_U32("attention.key_length", cfg.head_dim, "model: gguf missing <arch>.attention.key_length");
    GGUF_U32("embedding_length", cfg.hidden, "model: gguf missing <arch>.embedding_length");
    GGUF_U32("feed_forward_length", cfg.ffn_hidden, "model: gguf missing <arch>.feed_forward_length");
    GGUF_F32("rope.freq_base", cfg.rope_theta, "model: gguf missing <arch>.rope.freq_base");
    GGUF_F32("attention.layer_norm_rms_epsilon", cfg.rms_eps,
             "model: gguf missing <arch>.attention.layer_norm_rms_epsilon");

#undef GGUF_U32
#undef GGUF_F32

    sg_gguf_kv_type elem_type;
    const void *arr_data = NULL;
    uint64_t vocab_count = 0;
    if (!sg_gguf_get_arr(g, "tokenizer.ggml.tokens", &elem_type, &arr_data, &vocab_count)
        || elem_type != SG_GGUF_STR) {
        return (sg_err){"model: gguf missing tokenizer.ggml.tokens"};
    }
    if (vocab_count == 0 || vocab_count > UINT32_MAX) {
        return (sg_err){"model: gguf tokenizer.ggml.tokens has an implausible count"};
    }
    cfg.vocab = (uint32_t)vocab_count;

    if (cfg.n_layers == 0) return (sg_err){"model: gguf block_count is zero"};
    /* No real model has anywhere near this many layers; a corrupt/malicious
     * block_count near UINT32_MAX would otherwise size a huge (if
     * overflow-safe) calloc below purely to fail on the first missing
     * per-layer tensor. Mirrors gguf.c's own implausible-count guard. */
    if (cfg.n_layers > 100000) return (sg_err){"model: gguf block_count is implausibly large"};

    const sg_tensor *tok_emb_t = sg_gguf_tensor(g, "token_embd.weight");
    if (!tok_emb_t) return (sg_err){"model: gguf missing token_embd.weight"};
    const sg_tensor *out_norm_t = sg_gguf_tensor(g, "output_norm.weight");
    if (!out_norm_t) return (sg_err){"model: gguf missing output_norm.weight"};
    const sg_tensor *output_t = sg_gguf_tensor(g, "output.weight"); /* absent => tied */
    cfg.tied_embeddings = (output_t == NULL);

    sg_layer_w *layers = calloc(cfg.n_layers, sizeof(*layers));
    if (!layers) return (sg_err){"model: out of memory"};

    char name[160];
#define REQ(fmt, field, errmsg) do { \
        int _r = snprintf(name, sizeof name, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof name) { free(layers); return (sg_err){"model: gguf tensor name too long"}; } \
        const sg_tensor *_t = sg_gguf_tensor(g, name); \
        if (!_t) { free(layers); return (sg_err){errmsg}; } \
        lw->field = _t->data; \
    } while (0)
#define OPT(fmt, field) do { \
        int _r = snprintf(name, sizeof name, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof name) { free(layers); return (sg_err){"model: gguf tensor name too long"}; } \
        const sg_tensor *_t = sg_gguf_tensor(g, name); \
        lw->field = _t ? _t->data : NULL; \
    } while (0)

    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        sg_layer_w *lw = &layers[i];

        REQ("blk.%u.attn_norm.weight", ln1, "model: gguf missing blk.N.attn_norm.weight");
        /* post_attention_norm.weight, not ffn_norm.weight (see file header note 3). */
        REQ("blk.%u.post_attention_norm.weight", ln2,
            "model: gguf missing blk.N.post_attention_norm.weight");
        REQ("blk.%u.ffn_gate.weight", gate_proj, "model: gguf missing blk.N.ffn_gate.weight");
        REQ("blk.%u.ffn_up.weight", up_proj, "model: gguf missing blk.N.ffn_up.weight");
        REQ("blk.%u.ffn_down.weight", down_proj, "model: gguf missing blk.N.ffn_down.weight");

        /* Optional: NULL on a linear-attention layer (see file header note 2). */
        OPT("blk.%u.attn_q.weight", q_proj);
        OPT("blk.%u.attn_k.weight", k_proj);
        OPT("blk.%u.attn_v.weight", v_proj);
        OPT("blk.%u.attn_output.weight", o_proj);
        OPT("blk.%u.attn_q_norm.weight", q_norm);
        OPT("blk.%u.attn_k_norm.weight", k_norm);
    }
#undef REQ
#undef OPT

    m->cfg = cfg;
    m->tok_emb = tok_emb_t->data;
    m->out_norm = out_norm_t->data;
    m->lm_head = output_t ? output_t->data : tok_emb_t->data;
    m->layers = layers;
    m->wtype = tok_emb_t->type;
    return SG_OK;
}

/* Tries "model.<suffix>" first (a future pure-text qwen3_5 checkpoint), then
 * falls back to "model.language_model.<suffix>" (the real Qwen3.5-2B
 * checkpoint on disk is multimodal-wrapped; see file header note 6). */
static const void *st_dense_tensor(const sg_st *s, const char *suffix) {
    char plain[192], nested[192];
    int r1 = snprintf(plain, sizeof plain, "model.%s", suffix);
    int r2 = snprintf(nested, sizeof nested, "model.language_model.%s", suffix);
    if (r1 <= 0 || (size_t)r1 >= sizeof plain) return NULL;
    if (r2 <= 0 || (size_t)r2 >= sizeof nested) return NULL;

    const uint16_t *data = NULL;
    if (sg_st_tensor(s, plain, &data, NULL, NULL)) return data;
    if (sg_st_tensor(s, nested, &data, NULL, NULL)) return data;
    return NULL;
}

sg_err sg_model_from_st(const sg_st *s, sg_model *m) {
    if (!m) return (sg_err){"model: invalid arguments"};
    memset(m, 0, sizeof(*m));
    if (!s) return (sg_err){"model: invalid arguments"};

    sg_cfg cfg = {0};
    if (!sg_st_config_u32(s, "num_hidden_layers", &cfg.n_layers)) {
        return (sg_err){"model: config.json missing num_hidden_layers"};
    }
    if (!sg_st_config_u32(s, "num_attention_heads", &cfg.n_heads)) {
        return (sg_err){"model: config.json missing num_attention_heads"};
    }
    if (!sg_st_config_u32(s, "num_key_value_heads", &cfg.n_kv_heads)) {
        return (sg_err){"model: config.json missing num_key_value_heads"};
    }
    /* From the explicit head_dim key, NOT hidden/heads (see file header note 4). */
    if (!sg_st_config_u32(s, "head_dim", &cfg.head_dim)) {
        return (sg_err){"model: config.json missing head_dim"};
    }
    if (!sg_st_config_u32(s, "hidden_size", &cfg.hidden)) {
        return (sg_err){"model: config.json missing hidden_size"};
    }
    if (!sg_st_config_u32(s, "intermediate_size", &cfg.ffn_hidden)) {
        return (sg_err){"model: config.json missing intermediate_size"};
    }
    if (!sg_st_config_u32(s, "vocab_size", &cfg.vocab)) {
        return (sg_err){"model: config.json missing vocab_size"};
    }
    /* Nested under text_config.rope_parameters in the real checkpoint;
     * st_config_lookup (st.c) checks that nesting (see file header note 6). */
    if (!sg_st_config_f32(s, "rope_theta", &cfg.rope_theta)) {
        return (sg_err){"model: config.json missing rope_theta"};
    }
    if (!sg_st_config_f32(s, "rms_norm_eps", &cfg.rms_eps)) {
        return (sg_err){"model: config.json missing rms_norm_eps"};
    }

    if (cfg.n_layers == 0) return (sg_err){"model: config.json num_hidden_layers is zero"};
    /* Same implausible-count guard as sg_model_from_gguf. */
    if (cfg.n_layers > 100000) {
        return (sg_err){"model: config.json num_hidden_layers is implausibly large"};
    }

    const void *tok_emb = st_dense_tensor(s, "embed_tokens.weight");
    if (!tok_emb) return (sg_err){"model: safetensors missing embed_tokens.weight"};
    const void *out_norm = st_dense_tensor(s, "norm.weight");
    if (!out_norm) return (sg_err){"model: safetensors missing norm.weight"};

    /* lm_head.weight lives at the checkpoint's top level, never nested under
     * "model." or "model.language_model." (see file header note 7). */
    const uint16_t *lm_head_data = NULL;
    bool has_lm_head = sg_st_tensor(s, "lm_head.weight", &lm_head_data, NULL, NULL);
    cfg.tied_embeddings = !has_lm_head;

    sg_layer_w *layers = calloc(cfg.n_layers, sizeof(*layers));
    if (!layers) return (sg_err){"model: out of memory"};

    char suffix[160];
#define REQ(fmt, field, errmsg) do { \
        int _r = snprintf(suffix, sizeof suffix, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof suffix) { free(layers); return (sg_err){"model: safetensors tensor name too long"}; } \
        const void *_p = st_dense_tensor(s, suffix); \
        if (!_p) { free(layers); return (sg_err){errmsg}; } \
        lw->field = _p; \
    } while (0)
#define OPT(fmt, field) do { \
        int _r = snprintf(suffix, sizeof suffix, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof suffix) { free(layers); return (sg_err){"model: safetensors tensor name too long"}; } \
        lw->field = st_dense_tensor(s, suffix); \
    } while (0)

    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        sg_layer_w *lw = &layers[i];

        REQ("layers.%u.input_layernorm.weight", ln1,
            "model: safetensors missing layers.N.input_layernorm.weight");
        REQ("layers.%u.post_attention_layernorm.weight", ln2,
            "model: safetensors missing layers.N.post_attention_layernorm.weight");
        REQ("layers.%u.mlp.gate_proj.weight", gate_proj,
            "model: safetensors missing layers.N.mlp.gate_proj.weight");
        REQ("layers.%u.mlp.up_proj.weight", up_proj,
            "model: safetensors missing layers.N.mlp.up_proj.weight");
        REQ("layers.%u.mlp.down_proj.weight", down_proj,
            "model: safetensors missing layers.N.mlp.down_proj.weight");

        /* Optional: NULL on a linear-attention layer (see file header note 2). */
        OPT("layers.%u.self_attn.q_proj.weight", q_proj);
        OPT("layers.%u.self_attn.k_proj.weight", k_proj);
        OPT("layers.%u.self_attn.v_proj.weight", v_proj);
        OPT("layers.%u.self_attn.o_proj.weight", o_proj);
        OPT("layers.%u.self_attn.q_norm.weight", q_norm);
        OPT("layers.%u.self_attn.k_norm.weight", k_norm);
    }
#undef REQ
#undef OPT

    m->cfg = cfg;
    m->tok_emb = tok_emb;
    m->out_norm = out_norm;
    m->lm_head = has_lm_head ? (const void *)lm_head_data : tok_emb;
    m->layers = layers;
    m->wtype = SG_T_BF16; /* sg_st_tensor only ever retrieves BF16 tensors */
    return SG_OK;
}

void sg_model_free(sg_model *m) {
    if (!m) return;
    free(m->layers);
    m->layers = NULL;
}
