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
 *
 * 8. (Task 7) The linear-attention layers' own nine tensors are now mapped
 *    too, into sg_layer_w's ssm_* fields, by the same tensor-presence rule
 *    as note 2's attention group: whichever group is absent on a layer
 *    comes back NULL, and no config layer-type list is consulted. Names
 *    were taken verbatim from the real files' tensor directories, which
 *    corrected two guesses: the GGUF's a/b projections are
 *    blk.N.ssm_alpha.weight and blk.N.ssm_beta.weight (not bare ssm_alpha /
 *    ssm_beta), while blk.N.ssm_a really is bare with no .weight suffix.
 *    The two "attn_"-prefixed names on a linear layer are not a mistake in
 *    the converter: blk.N.attn_qkv.weight is the DeltaNet's in_proj_qkv and
 *    blk.N.attn_gate.weight is its in_proj_z, confirmed by their shapes
 *    ([5120, 10240] == hidden -> 2*key_dim+value_dim, and [5120, 6144] ==
 *    hidden -> value_dim, with key_dim/value_dim read off qwen35.ssm.*).
 *    The alpha<->in_proj_a / beta<->in_proj_b pairing was flagged in Task 7
 *    as "by name only" (both tensors are [5120, 48], so shape cannot
 *    disambiguate them). VERIFIED NUMERICALLY IN TASK 8, by dequantizing
 *    both sides of the twin pair (GGUF Q8_0 and the mlx 8-bit affine repack
 *    of the same model) and comparing under the tiled value-head reindex:
 *
 *      layer  alpha<->a   beta<->b  ||  SWAPPED alpha<->b   beta<->a
 *        0    1.217e-03  5.054e-04  ||       1.265e-01    1.264e-01
 *        1    1.638e-03  2.348e-03  ||       3.340e-01    3.339e-01
 *        2    1.823e-03  2.047e-03  ||       2.479e-01    2.479e-01
 *        4    1.962e-03  1.694e-03  ||       3.491e-01    3.491e-01
 *
 *    The named pairing agrees to the two quantizers' combined noise; the
 *    swap is two orders of magnitude worse. Corroborated end to end: the
 *    27B GGUF greedily completes "The capital of France is" with " Paris.",
 *    which a swapped beta/decay pairing could not produce.
 *
 * 9. (Task 7) The safetensors checkpoint is dtype-mixed WITHIN a linear-
 *    attention layer: linear_attn.A_log and linear_attn.norm.weight are
 *    F32 while the other seven tensors of the same layer are BF16 (36 F32
 *    tensors in the 2B file = 18 linear layers x 2). mlx's own
 *    cast_predicate explicitly exempts A_log from casting, so this is
 *    intentional upstream, not a packaging artifact. st.c's index was
 *    extended in Task 7 to carry F32 tensors as well, reachable through
 *    sg_st_tensor_f32; those two fields are looked up with it here.
 *
 * 10. (Task 8) Both loaders now also fill the forward pass's shape and
 *    semantics fields: sg_cfg.rope_dim / full_attn_interval and the five
 *    DeltaNet dims, plus sg_model.dense_type and norms_are_residual. The
 *    DeltaNet dims are OPTIONAL on both sides, because a checkpoint with no
 *    linear-attention layer legitimately has none of them; sg_ref_state_new
 *    is where their absence becomes an error, and only if a DeltaNet layer
 *    actually exists. GGUF spells them qwen35.ssm.group_count (key heads),
 *    .state_size (key head dim), .time_step_rank (value heads), .inner_size
 *    (value heads * value head dim) and .conv_kernel; config.json spells them
 *    linear_num_key_heads / linear_key_head_dim / linear_num_value_heads /
 *    linear_value_head_dim / linear_conv_kernel_dim, i.e. the GGUF states the
 *    value width as a product and config.json states it as a factor.
 *
 * 11. (Task 8) The GGUF loader now VERIFIES each tensor's dtype as it maps
 *    it, instead of leaving the forward pass to trust surge.h's dtype table:
 *    matmul weights must equal token_embd's type and the small tensors
 *    (norms, conv1d, dt bias, ssm_a, ssm_norm) must be F32. That check is
 *    what lets sg_model.dense_type be a single value rather than a guess.
 *    Measured on Qwen3.6-27B-Q8_0: every matmul tensor is Q8_0 and every
 *    small tensor is F32, no exceptions.
 *
 * 12. (Task 8) sg_model.norms_are_residual reproduces mlx's own trigger
 *    rather than hardcoding "safetensors means residual". TextModel.sanitize
 *    shifts the five norm weights by +1.0 when the checkpoint has mtp.*
 *    tensors OR an unsanitized conv1d.weight (last dim != 1); an already
 *    sanitized, mtp-free safetensors checkpoint must NOT be shifted. st.c
 *    exposes no tensor enumeration, so the conv1d test is the load-bearing
 *    one here (it fires on the real 2B, whose conv1d.weight is [6144,1,4]),
 *    with a probe for the two plausible spellings of the mtp head as a
 *    secondary trigger.
 *
 * 13. (Task P1) A second, PLAIN DENSE architecture is now accepted from GGUF:
 *    general.architecture == "qwen3" (no "5"/"_5"), e.g.
 *    Qwen3-4B-Instruct-2507-Q8_0.gguf. It is a uniform 36-layer full-attention
 *    transformer, not a hybrid: zero gated-DeltaNet layers, and every
 *    full-attention layer has a SINGLE-width q_proj with no folded output
 *    gate (cfg.attn_output_gate, surge.h). Three real-file corrections found
 *    while wiring this up, none anticipated by the task brief that scoped it:
 *
 *      a. general.architecture == "qwen3" implies dense: it carries no
 *         qwen3.full_attention_interval key at all (that key is specific to
 *         the qwen35/qwen3_5 hybrid converter), so a dense checkpoint sets
 *         cfg.full_attn_interval = 1 outright rather than reading it -- every
 *         layer is then a full-attention layer by the same
 *         (L+1)%full_attn_interval==0 rule the hybrid uses.
 *
 *      b. It ALSO carries no qwen3.rope.dimension_count key. This is the
 *         dangerous one: Qwen3 uses FULL rotary (rope_dim == head_dim, 128 of
 *         128), and rope_dim is even and <= head_dim for every value from 2
 *         to 128, so a wrong default here passes every downstream validity
 *         check silently. In particular the safetensors loader's existing
 *         default (int(head_dim * partial_rotary_factor), 0.25 when the key
 *         is absent -- correct for the qwen3_5 hybrid, which really is a
 *         partial-rotary model) would be the WRONG default to copy for dense
 *         qwen3 and would silently compute rope_dim 32 instead of 128,
 *         loading and running with wrong output and no error. The dense
 *         branch below therefore defaults cfg.rope_dim to cfg.head_dim (full
 *         rotary) and only lets an explicit qwen3.rope.dimension_count
 *         override it. tests/test_model.c's dense config-extraction test
 *         asserts rope_dim == 128 specifically to catch a regression to 32.
 *
 *      c. The pre-MLP norm tensor is named blk.N.ffn_norm.weight on this
 *         file, not blk.N.post_attention_norm.weight like the hybrid GGUF
 *         (note 3 above) -- llama.cpp's standard norm name, confirmed absent/
 *         present the same way note 3 was originally confirmed. The layer
 *         loop tries the hybrid name first, then falls back to this one, by
 *         tensor presence rather than by branching on architecture (same
 *         convention as notes 2 and 8), so neither path's lookup order
 *         changes based on which architecture is loaded.
 */
#include "surge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool gguf_arch_recognized(const char *a) {
    return a && (strcmp(a, "qwen35") == 0 || strcmp(a, "qwen3_5") == 0
                 || strcmp(a, "qwen3") == 0);
}

/* "qwen3" (bare) is the DENSE Qwen3 family (e.g. Qwen3-4B-Instruct-2507):
 * full softmax attention on every layer, no gated-DeltaNet/SSM layers, and a
 * SINGLE-width q_proj with no folded output gate -- confirmed against the
 * real file's tensor directory (blk.N.attn_q.weight is [hidden, n_heads*
 * head_dim], not [hidden, 2*n_heads*head_dim], and there are zero blk.N.ssm_*
 * or blk.N.attn_gate.weight tensors anywhere). "qwen35"/"qwen3_5" stay the
 * hybrid family the rest of this file's notes describe. */
static bool gguf_arch_is_dense_qwen3(const char *a) {
    return a && strcmp(a, "qwen3") == 0;
}

/* The derived widths every tensor's size is checked against. Computed in
 * u64 from the config so the check itself cannot wrap. */
typedef struct {
    uint64_t hidden, ffn, vocab, head_dim;
    uint64_t attn_w;     /* n_heads    * head_dim  (o_proj's input width) */
    uint64_t kv_w;       /* n_kv_heads * head_dim  (k_proj/v_proj rows)   */
    uint64_t key_dim, value_dim, conv_dim, n_v, head_v_dim, conv_k;
    bool have_ssm_dims;
} shape_t;

static shape_t shapes_of(const sg_cfg *c) {
    shape_t s;
    s.hidden = c->hidden;
    s.ffn = c->ffn_hidden;
    s.vocab = c->vocab;
    s.head_dim = c->head_dim;
    s.attn_w = (uint64_t)c->n_heads * c->head_dim;
    s.kv_w = (uint64_t)c->n_kv_heads * c->head_dim;
    s.key_dim = (uint64_t)c->n_k_heads * c->head_k_dim;
    s.value_dim = (uint64_t)c->n_v_heads * c->head_v_dim;
    s.conv_dim = 2 * s.key_dim + s.value_dim;
    s.n_v = c->n_v_heads;
    s.head_v_dim = c->head_v_dim;
    s.conv_k = c->conv_kernel;
    s.have_ssm_dims = c->n_k_heads && c->n_v_heads && c->head_k_dim
                      && c->head_v_dim && c->conv_kernel;
    return s;
}

/* THE SHAPE CHECK, and why it is not optional.
 *
 * Every width the forward pass uses -- every matvec's rows and cols, the
 * embedding row stride, the conv kernel span -- comes from metadata, while
 * the bytes come from the file. Nothing before this compared the two. A
 * config that overstates any dimension (a vocab_size larger than
 * token_embd's row count, a doubled hidden_size, a doubled
 * linear_value_head_dim) therefore made the forward read past the end of a
 * tensor: inside a large mmap that silently returns the neighbouring
 * tensor's bytes and produces a wrong answer with no diagnostic, and at the
 * end of the mapping it is a SIGBUS.
 *
 * Comparing ELEMENT COUNTS rather than per-axis extents is deliberate: it is
 * dimension-order agnostic (GGUF stores dims fastest-axis-first and
 * safetensors slowest-axis-first, and conv1d is 3-D on one side and 2-D on
 * the other), and it is exactly the property that bounds every access this
 * code performs. A transposed-but-same-size tensor is not caught here; that
 * is what the mlx logit comparison is for. */
static uint64_t tensor_elems(const uint64_t dims[4], uint32_t n_dims) {
    if (n_dims == 0 || n_dims > 4) return 0;
    uint64_t n = 1;
    /* A zero extent must produce 0, NOT be skipped. Both readers compute
     * byte lengths by multiplying straight through, so a tensor declared
     * [2048, 0] occupies zero bytes and both readers accept it; treating the
     * 0 as a 1 here would report 2048 elements, pass this check against
     * hidden, and hand the forward a pointer to 0 bytes. That is exactly the
     * overread this function exists to prevent. */
    for (uint32_t i = 0; i < n_dims; i++) n *= dims[i];
    return n;
}

static sg_err check_elems(const char *name, uint64_t got, uint64_t want) {
    if (got == want && got != 0) return SG_OK;
    fprintf(stderr,
            "model: tensor %s holds %llu elements but the config implies %llu\n",
            name, (unsigned long long)got, (unsigned long long)want);
    return (sg_err){"model: a tensor's element count contradicts the config"};
}

/* Every layer must be cleanly one kind or the other: all six attention
 * tensors and no ssm tensors, or all nine ssm tensors and no attention
 * tensors. Both loaders detect layer kind purely by tensor presence, so a
 * PARTIAL group is the one thing that quietly breaks that rule -- the
 * forward pass would see a non-NULL q_proj (or ssm_in_qkv), decide the layer
 * kind from it, and then dereference a NULL sibling several ops later, far
 * from the actual cause. A renamed or dropped tensor in a future converter
 * is exactly how that would arrive, so it is rejected at load time with the
 * layer index and the counts, not left as a landmine.
 *
 * The MLP/norm tensors are not checked here; they are REQ'd on every layer
 * already and a missing one fails earlier with its own message. */
static sg_err check_layer_groups(const sg_layer_w *lw, uint32_t layer) {
    const void *attn[] = { lw->q_proj, lw->k_proj, lw->v_proj,
                           lw->o_proj, lw->q_norm, lw->k_norm };
    const void *ssm[] = { lw->ssm_in_qkv, lw->ssm_in_z, lw->ssm_in_b,
                          lw->ssm_in_a, lw->ssm_a, lw->ssm_dt_bias,
                          lw->ssm_conv1d, lw->ssm_norm, lw->ssm_out };
    unsigned n_attn = 0, n_ssm = 0;
    for (size_t i = 0; i < sizeof attn / sizeof *attn; i++) if (attn[i]) n_attn++;
    for (size_t i = 0; i < sizeof ssm / sizeof *ssm; i++) if (ssm[i]) n_ssm++;

    /* Static messages only: sg_err carries a const char *, so it cannot own
     * a formatted string. The counts go to stderr, which is where anyone
     * debugging a rejected checkpoint will be looking anyway. */
    if (n_attn == 6 && n_ssm == 0) return SG_OK;
    if (n_attn == 0 && n_ssm == 9) return SG_OK;

    fprintf(stderr,
            "model: layer %u has an incomplete tensor group: %u/6 full-attention "
            "tensors and %u/9 gated-DeltaNet tensors present (a layer must be "
            "entirely one kind or the other)\n", layer, n_attn, n_ssm);
    if (n_attn == 0 && n_ssm == 0) {
        return (sg_err){"model: layer has neither a full-attention nor a "
                        "gated-DeltaNet tensor group"};
    }
    return (sg_err){"model: layer has an incomplete full-attention or "
                    "gated-DeltaNet tensor group"};
}

sg_err sg_model_from_gguf(const sg_gguf *g, sg_model *m) {
    if (!m) return (sg_err){"model: invalid arguments"};
    memset(m, 0, sizeof(*m));
    if (!g) return (sg_err){"model: invalid arguments"};

    const char *arch = NULL;
    if (!sg_gguf_get_str(g, "general.architecture", &arch) || !gguf_arch_recognized(arch)) {
        return (sg_err){"model: unrecognized gguf architecture "
                        "(expected qwen35, qwen3_5 or qwen3)"};
    }
    bool dense = gguf_arch_is_dense_qwen3(arch);

    sg_cfg cfg = {0};
    /* Hybrid: q_proj folds a sigmoid output gate (file header note 5). Dense
     * qwen3: no gate at all. Set before shapes_of() runs below so every
     * width derived from it (the q_proj element check here, g->q_width in
     * metal.m, st->q_width in ref.c) agrees. */
    cfg.attn_output_gate = !dense;
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
/* Absent leaves dst untouched: the DeltaNet dims genuinely do not exist on a
 * checkpoint with no linear-attention layer (see file header note 10). */
#define GGUF_U32_OPT(suffix, dst) do { \
        int _r = snprintf(key, sizeof key, "%s.%s", arch, suffix); \
        if (_r > 0 && (size_t)_r < sizeof key) (void)sg_gguf_get_u32(g, key, &(dst)); \
    } while (0)

    GGUF_U32("block_count", cfg.n_layers, "model: gguf missing <arch>.block_count");
    GGUF_U32("attention.head_count", cfg.n_heads, "model: gguf missing <arch>.attention.head_count");
    GGUF_U32("attention.head_count_kv", cfg.n_kv_heads,
             "model: gguf missing <arch>.attention.head_count_kv");
    /* From the explicit key_length, NOT hidden/heads (see file header note 4). */
    GGUF_U32("attention.key_length", cfg.head_dim, "model: gguf missing <arch>.attention.key_length");
    /* key_length is used as THE head dim: for v_proj's rows, the V-cache
     * stride and o_proj's input width. Both real files set value_length to
     * the same 256, and nothing downstream could tell the difference if they
     * ever diverged (the shape check derives both sides from key_length), so
     * refuse a file where they differ rather than read V as garbage. */
    {
        uint32_t vlen = 0;
        int _r = snprintf(key, sizeof key, "%s.attention.value_length", arch);
        if (_r > 0 && (size_t)_r < sizeof key && sg_gguf_get_u32(g, key, &vlen)
            && vlen != cfg.head_dim) {
            fprintf(stderr, "model: gguf attention.value_length %u differs from "
                            "attention.key_length %u; surge assumes one head dim\n",
                    vlen, cfg.head_dim);
            return (sg_err){"model: gguf value_length differs from key_length"};
        }
    }
    GGUF_U32("embedding_length", cfg.hidden, "model: gguf missing <arch>.embedding_length");
    GGUF_U32("feed_forward_length", cfg.ffn_hidden, "model: gguf missing <arch>.feed_forward_length");
    GGUF_F32("rope.freq_base", cfg.rope_theta, "model: gguf missing <arch>.rope.freq_base");
    GGUF_F32("attention.layer_norm_rms_epsilon", cfg.rms_eps,
             "model: gguf missing <arch>.attention.layer_norm_rms_epsilon");
    if (dense) {
        /* THE TRAP (Task P1, most important fix in this file). The hybrid
         * branch below requires <arch>.rope.dimension_count outright, which
         * is correct there because both real hybrid checkpoints state it
         * (64 of 256). The real dense Qwen3-4B-Instruct-2507 GGUF carries NO
         * qwen3.rope.dimension_count key at all (verified against the actual
         * file), so naively reusing sg_model_from_st's
         * partial_rotary_factor-0.25-of-head_dim default here would compute
         * rope_dim = 32 of 128 -- Qwen3 has no partial_rotary_factor concept
         * and actually uses FULL rotary. rope_dim 32 is even and <= head_dim,
         * so it passes every validity check in ref.c and metal.m and would
         * load and run, producing WRONG logits with no error anywhere. The
         * only safe default for a dense qwen3 file is full rotary
         * (rope_dim == head_dim); the optional read only OVERRIDES that if a
         * future dense checkpoint ever states the key explicitly. Covered by
         * tests/test_model.c's anti-trap regression (asserts 128, not 32). */
        cfg.rope_dim = cfg.head_dim;
        GGUF_U32_OPT("rope.dimension_count", cfg.rope_dim);

        /* Real dense qwen3 GGUFs carry no <arch>.full_attention_interval key
         * either (that key is specific to the qwen35/qwen3_5 hybrid
         * converter): dense means every layer is a full-attention layer, and
         * interval 1 makes (L+1) % 1 == 0 true for every L, which is exactly
         * what kv_is_attn / check_layer_groups / the ref.c and metal.m
         * layer-kind cross-checks already expect (file header note 2's rule
         * still decides layer kind by tensor presence; this just makes the
         * config side of that cross-check agree for every layer). */
        cfg.full_attn_interval = 1;
    } else {
        /* The GGUF states the rotated width outright, so there is no
         * partial_rotary_factor multiplication to round here. Required, byte-
         * identical to before Task P1: both real hybrid checkpoints state it. */
        GGUF_U32("rope.dimension_count", cfg.rope_dim,
                 "model: gguf missing <arch>.rope.dimension_count");
        GGUF_U32("full_attention_interval", cfg.full_attn_interval,
                 "model: gguf missing <arch>.full_attention_interval");
    }

    /* DeltaNet dims: optional (see file header note 10). inner_size is the
     * whole value width, so head_v_dim is a division rather than a key. */
    uint32_t ssm_inner = 0;
    GGUF_U32_OPT("ssm.group_count", cfg.n_k_heads);
    GGUF_U32_OPT("ssm.state_size", cfg.head_k_dim);
    GGUF_U32_OPT("ssm.time_step_rank", cfg.n_v_heads);
    GGUF_U32_OPT("ssm.inner_size", ssm_inner);
    GGUF_U32_OPT("ssm.conv_kernel", cfg.conv_kernel);
    if (cfg.n_v_heads != 0 && ssm_inner != 0) {
        if (ssm_inner % cfg.n_v_heads != 0) {
            return (sg_err){"model: gguf <arch>.ssm.inner_size is not a multiple of "
                            "<arch>.ssm.time_step_rank"};
        }
        cfg.head_v_dim = ssm_inner / cfg.n_v_heads;
    }

#undef GGUF_U32
#undef GGUF_F32
#undef GGUF_U32_OPT

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

    /* mat = whatever token_embd is (Q8_0 in the real file); dense = F32.
     * Both are checked per tensor below rather than assumed, so that
     * sg_model.dense_type is a fact about this file (see note 11). */
    sg_tensor_type mat_type = tok_emb_t->type;
    if (out_norm_t->type != SG_T_F32) {
        return (sg_err){"model: gguf output_norm.weight is not F32"};
    }
    if (output_t && output_t->type != mat_type) {
        return (sg_err){"model: gguf output.weight dtype differs from token_embd.weight"};
    }

    shape_t sh = shapes_of(&cfg);
    sg_err se = check_elems("token_embd.weight",
                            tensor_elems(tok_emb_t->dims, tok_emb_t->n_dims),
                            sh.vocab * sh.hidden);
    if (sg_failed(se)) return se;
    se = check_elems("output_norm.weight",
                     tensor_elems(out_norm_t->dims, out_norm_t->n_dims), sh.hidden);
    if (sg_failed(se)) return se;
    if (output_t) {
        se = check_elems("output.weight",
                         tensor_elems(output_t->dims, output_t->n_dims),
                         sh.vocab * sh.hidden);
        if (sg_failed(se)) return se;
    }

    sg_layer_w *layers = calloc(cfg.n_layers, sizeof(*layers));
    if (!layers) return (sg_err){"model: out of memory"};

    char name[160];
    /* want_type and want_elems are checked, not assumed: a converter that
     * changes a norm to F16, or a config that overstates a dimension, would
     * otherwise be read at the wrong width and produce plausible garbage
     * several hundred ops later (or read off the end of the mapping). */
#define TCHK(_t, _want_type, _want_elems) do { \
        if ((_t)->type != (_want_type)) { \
            fprintf(stderr, "model: gguf tensor %s has type %d, expected %d\n", \
                    (_t)->name, (int)(_t)->type, (int)(_want_type)); \
            free(layers); \
            return (sg_err){"model: gguf tensor has an unexpected dtype"}; \
        } \
        sg_err _e = check_elems((_t)->name, tensor_elems((_t)->dims, (_t)->n_dims), \
                                (_want_elems)); \
        if (sg_failed(_e)) { free(layers); return _e; } \
    } while (0)
#define REQ(fmt, field, want, elems, errmsg) do { \
        int _r = snprintf(name, sizeof name, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof name) { free(layers); return (sg_err){"model: gguf tensor name too long"}; } \
        const sg_tensor *_t = sg_gguf_tensor(g, name); \
        if (!_t) { free(layers); return (sg_err){errmsg}; } \
        TCHK(_t, (want), (elems)); \
        lw->field = _t->data; \
    } while (0)
#define OPT(fmt, field, want, elems) do { \
        int _r = snprintf(name, sizeof name, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof name) { free(layers); return (sg_err){"model: gguf tensor name too long"}; } \
        const sg_tensor *_t = sg_gguf_tensor(g, name); \
        if (_t) TCHK(_t, (want), (elems)); \
        lw->field = _t ? _t->data : NULL; \
    } while (0)

    /* q_proj's element count: DOUBLE width (queries + folded output gate) on
     * the hybrid, single width on dense qwen3 (Task P1; see cfg.attn_output_gate
     * above and file header note 5). Computed once, outside the loop: it does
     * not depend on i. */
    uint64_t q_mult = cfg.attn_output_gate ? 2 : 1;

    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        sg_layer_w *lw = &layers[i];

        REQ("blk.%u.attn_norm.weight", ln1, SG_T_F32, sh.hidden,
            "model: gguf missing blk.N.attn_norm.weight");
        /* The hybrid GGUF spells the pre-MLP norm post_attention_norm.weight
         * (file header note 3). The real DENSE qwen3 GGUF
         * (Qwen3-4B-Instruct-2507, Task P1) spells the SAME tensor
         * ffn_norm.weight instead -- llama.cpp's standard norm name for this
         * position, confirmed against the real file's tensor directory (it
         * has no post_attention_norm.weight tensor at all). Try the hybrid
         * name first (so the hybrid path's lookup order, and therefore its
         * behavior, is unchanged), then fall back to the dense name, rather
         * than branching on architecture -- consistent with this file's
         * tensor-presence-decides convention (notes 2 and 8). */
        OPT("blk.%u.post_attention_norm.weight", ln2, SG_T_F32, sh.hidden);
        if (!lw->ln2) {
            REQ("blk.%u.ffn_norm.weight", ln2, SG_T_F32, sh.hidden,
                "model: gguf missing blk.N.post_attention_norm.weight or "
                "blk.N.ffn_norm.weight");
        }
        REQ("blk.%u.ffn_gate.weight", gate_proj, mat_type, sh.ffn * sh.hidden,
            "model: gguf missing blk.N.ffn_gate.weight");
        REQ("blk.%u.ffn_up.weight", up_proj, mat_type, sh.ffn * sh.hidden,
            "model: gguf missing blk.N.ffn_up.weight");
        REQ("blk.%u.ffn_down.weight", down_proj, mat_type, sh.hidden * sh.ffn,
            "model: gguf missing blk.N.ffn_down.weight");

        /* Optional: NULL on a linear-attention layer (see file header note 2). */
        OPT("blk.%u.attn_q.weight", q_proj, mat_type, q_mult * sh.attn_w * sh.hidden);
        OPT("blk.%u.attn_k.weight", k_proj, mat_type, sh.kv_w * sh.hidden);
        OPT("blk.%u.attn_v.weight", v_proj, mat_type, sh.kv_w * sh.hidden);
        OPT("blk.%u.attn_output.weight", o_proj, mat_type, sh.hidden * sh.attn_w);
        OPT("blk.%u.attn_q_norm.weight", q_norm, SG_T_F32, sh.head_dim);
        OPT("blk.%u.attn_k_norm.weight", k_norm, SG_T_F32, sh.head_dim);

        /* The mirror-image group: present only on a linear-attention layer,
         * NULL on a full-attention one (see file header note 8). Names are
         * verbatim from the real file's tensor directory, including the two
         * that do not follow the ".weight" convention (ssm_a and the dt bias,
         * spelled ssm_dt.bias). ssm_a does NOT hold A_log on this GGUF path:
         * the converter has already applied -exp() to it, so it stores
         * -exp(A_log) and is consumed via sg_ref_delta_decay_neg_a. The
         * safetensors path stores A_log verbatim and is consumed via
         * sg_ref_delta_decay. See file header note 3. */
        if (!sh.have_ssm_dims && sg_gguf_tensor(g, (snprintf(name, sizeof name,
                "blk.%u.ssm_a", i), name))) {
            free(layers);
            return (sg_err){"model: gguf has linear-attention layers but no "
                            "<arch>.ssm.* dimensions"};
        }
        OPT("blk.%u.attn_qkv.weight", ssm_in_qkv, mat_type, sh.conv_dim * sh.hidden);
        OPT("blk.%u.attn_gate.weight", ssm_in_z, mat_type, sh.value_dim * sh.hidden);
        OPT("blk.%u.ssm_beta.weight", ssm_in_b, mat_type, sh.n_v * sh.hidden);
        OPT("blk.%u.ssm_alpha.weight", ssm_in_a, mat_type, sh.n_v * sh.hidden);
        OPT("blk.%u.ssm_a", ssm_a, SG_T_F32, sh.n_v);
        OPT("blk.%u.ssm_dt.bias", ssm_dt_bias, SG_T_F32, sh.n_v);
        OPT("blk.%u.ssm_conv1d.weight", ssm_conv1d, SG_T_F32, sh.conv_dim * sh.conv_k);
        OPT("blk.%u.ssm_norm.weight", ssm_norm, SG_T_F32, sh.head_v_dim);
        OPT("blk.%u.ssm_out.weight", ssm_out, mat_type, sh.hidden * sh.value_dim);

        sg_err ge = check_layer_groups(lw, i);
        if (sg_failed(ge)) { free(layers); return ge; }
    }
#undef REQ
#undef OPT
#undef TCHK

    m->cfg = cfg;
    m->tok_emb = tok_emb_t->data;
    m->out_norm = out_norm_t->data;
    m->lm_head = output_t ? output_t->data : tok_emb_t->data;
    m->layers = layers;
    m->wtype = tok_emb_t->type;
    /* Per-source DeltaNet semantics (see notes 3 and 4 in surge.h). Both are
     * properties of the GGUF converter, not of this particular file. */
    m->ssm_a_form = SG_SSM_A_NEG_EXP;
    m->v_heads_tiled = true;
    m->ssm_a_type = SG_T_F32;    /* blk.N.ssm_a is F32, checked above */
    m->ssm_norm_type = SG_T_F32; /* blk.N.ssm_norm.weight likewise */
    m->dense_type = SG_T_F32;    /* every norm/conv/bias, checked above */
    /* The converter baked mlx's +1.0 norm shift in already (see note 2 in
     * surge.h and note 12 above): GGUF norms are absolute. */
    m->norms_are_residual = false;
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

/* Same two-name probe, for the two DeltaNet tensors the real checkpoint
 * stores as F32 rather than BF16 (see file header note 9). Kept separate
 * from st_dense_tensor rather than folded into one "any dtype" helper so
 * that a tensor turning up in the wrong dtype is a loud NULL here instead
 * of a silently misinterpreted pointer. */
static const void *st_dense_tensor_f32(const sg_st *s, const char *suffix) {
    char plain[192], nested[192];
    int r1 = snprintf(plain, sizeof plain, "model.%s", suffix);
    int r2 = snprintf(nested, sizeof nested, "model.language_model.%s", suffix);
    if (r1 <= 0 || (size_t)r1 >= sizeof plain) return NULL;
    if (r2 <= 0 || (size_t)r2 >= sizeof nested) return NULL;

    const float *data = NULL;
    if (sg_st_tensor_f32(s, plain, &data, NULL, NULL)) return data;
    if (sg_st_tensor_f32(s, nested, &data, NULL, NULL)) return data;
    return NULL;
}

/* Same probe, but for the two DeltaNet tensors whose dtype is NOT stable
 * across safetensors checkpoints: linear_attn.A_log and linear_attn.norm.
 * weight are F32 in the Qwen3.5-2B bf16 checkpoint (mlx's cast_predicate
 * exempts A_log from casting) and BF16 in the mlx 8-bit repack of
 * Qwen3.6-27B. A F32-only lookup silently returns NULL on the latter, so
 * this tries F32 first, then BF16, and reports which it found through
 * *found_type. Callers record that on sg_model so the forward pass can read
 * the right width instead of guessing from sg_model.wtype. */
static const void *st_dense_tensor_any(const sg_st *s, const char *suffix,
                                       sg_tensor_type *found_type) {
    const void *p = st_dense_tensor_f32(s, suffix);
    if (p) { if (found_type) *found_type = SG_T_F32; return p; }
    p = st_dense_tensor(s, suffix);
    if (p) { if (found_type) *found_type = SG_T_BF16; return p; }
    return NULL;
}

/* Same two-name probe, but reporting the tensor's shape rather than its
 * data. Used only to ask whether conv1d.weight is in HF's unsanitized
 * [conv_dim, 1, kernel] layout, which is one of the two conditions under
 * which mlx shifts the norm weights by +1.0 (see file header note 12). */
static bool st_dense_dims(const sg_st *s, const char *suffix,
                          uint64_t dims[4], uint32_t *n_dims) {
    char plain[192], nested[192];
    int r1 = snprintf(plain, sizeof plain, "model.%s", suffix);
    int r2 = snprintf(nested, sizeof nested, "model.language_model.%s", suffix);
    if (r1 <= 0 || (size_t)r1 >= sizeof plain) return false;
    if (r2 <= 0 || (size_t)r2 >= sizeof nested) return false;

    const uint16_t *d = NULL;
    if (sg_st_tensor(s, plain, &d, dims, n_dims)) return true;
    if (sg_st_tensor(s, nested, &d, dims, n_dims)) return true;
    const float *f = NULL;
    if (sg_st_tensor_f32(s, plain, &f, dims, n_dims)) return true;
    if (sg_st_tensor_f32(s, nested, &f, dims, n_dims)) return true;
    return false;
}

/* mlx's second trigger, `any("mtp." in k for k in weights)`. st.c has no
 * tensor-enumeration API, so this probes the two names the real Qwen3.5
 * multi-token-prediction head actually uses instead. It is a SECONDARY
 * trigger: the conv1d test above is the one that fires on every checkpoint
 * this project has seen, and both conditions produce the same answer on the
 * real 2B (which has 15 mtp.* tensors AND an unsanitized conv1d). */
static bool st_has_mtp_head(const sg_st *s) {
    static const char *const names[] = {
        "mtp.fc.weight", "model.mtp.fc.weight", "model.language_model.mtp.fc.weight",
    };
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) {
        const uint16_t *d = NULL;
        if (sg_st_tensor(s, names[i], &d, NULL, NULL)) return true;
        const float *f = NULL;
        if (sg_st_tensor_f32(s, names[i], &f, NULL, NULL)) return true;
    }
    return false;
}

sg_err sg_model_from_st(const sg_st *s, sg_model *m) {
    if (!m) return (sg_err){"model: invalid arguments"};
    memset(m, 0, sizeof(*m));
    if (!s) return (sg_err){"model: invalid arguments"};

    sg_cfg cfg = {0};
    /* This loader only ever maps the qwen3_5/qwen35 hybrid safetensors
     * checkpoint (see file header note 2): every full-attention layer's
     * q_proj is double width with a folded sigmoid output gate. Task P1's
     * dense qwen3 is GGUF-only so far (sg_model_from_gguf derives this from
     * general.architecture); this stays unconditionally true here so the
     * hybrid st path is unaffected -- leaving it at {0}'s default false would
     * silently halve q_proj's expected width and break every full-attention
     * layer on this path. */
    cfg.attn_output_gate = true;
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

    /* rope_dim is mlx's int(head_dim * partial_rotary_factor), with mlx's own
     * default of 0.25 when the key is absent (TextModelArgs.__post_init__).
     * The truncation is deliberate: mlx writes int(), not round(). */
    float prf = 0.25f;
    (void)sg_st_config_f32(s, "partial_rotary_factor", &prf);
    if (!(prf > 0.0f) || prf > 1.0f) {
        return (sg_err){"model: config.json partial_rotary_factor is out of range"};
    }
    cfg.rope_dim = (uint32_t)((double)cfg.head_dim * (double)prf);
    /* mlx's DecoderLayer default; the real checkpoints all state it. */
    cfg.full_attn_interval = 4;
    (void)sg_st_config_u32(s, "full_attention_interval", &cfg.full_attn_interval);
    if (cfg.full_attn_interval == 0) {
        return (sg_err){"model: config.json full_attention_interval is zero"};
    }
    /* Optional: absent on a checkpoint with no linear-attention layer. */
    (void)sg_st_config_u32(s, "linear_num_key_heads", &cfg.n_k_heads);
    (void)sg_st_config_u32(s, "linear_num_value_heads", &cfg.n_v_heads);
    (void)sg_st_config_u32(s, "linear_key_head_dim", &cfg.head_k_dim);
    (void)sg_st_config_u32(s, "linear_value_head_dim", &cfg.head_v_dim);
    (void)sg_st_config_u32(s, "linear_conv_kernel_dim", &cfg.conv_kernel);

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
    uint64_t lm_dims[4] = {0, 0, 0, 0};
    uint32_t lm_nd = 0;
    bool has_lm_head = sg_st_tensor(s, "lm_head.weight", &lm_head_data, lm_dims, &lm_nd);
    cfg.tied_embeddings = !has_lm_head;
    /* Presence is still the decision (note 7), but when config.json states
     * tie_word_embeddings the two must agree. mlx's sanitize POPS
     * lm_head.weight when the flag is true, so a checkpoint shipping a stale
     * lm_head.weight alongside tie_word_embeddings=true would make surge use
     * that tensor and mlx use the embedding table: different logits, no
     * error anywhere. */
    {
        bool tie = false;
        if (sg_st_config_bool(s, "tie_word_embeddings", &tie) && tie == has_lm_head) {
            fprintf(stderr, "model: config.json tie_word_embeddings=%s but "
                            "lm_head.weight is %s\n", tie ? "true" : "false",
                    has_lm_head ? "present" : "absent");
            return (sg_err){"model: tie_word_embeddings contradicts lm_head.weight"};
        }
    }
    /* Attention biases are not mapped anywhere in sg_layer_w, and
     * check_layer_groups counts only the six weight tensors, so a
     * bias-carrying checkpoint would load clean and compute attention
     * without them. Refuse it instead. */
    {
        bool bias = false;
        if (sg_st_config_bool(s, "attention_bias", &bias) && bias) {
            return (sg_err){"model: config.json attention_bias is true, which surge "
                            "does not implement (no q/k/v/o bias is mapped)"};
        }
    }

    /* Element-count validation against the config; see check_elems's comment
     * for why this is not optional and why it compares counts, not extents. */
    shape_t sh = shapes_of(&cfg);
    {
        uint64_t d[4] = {0, 0, 0, 0};
        uint32_t nd = 0;
        sg_err se;
        if (!st_dense_dims(s, "embed_tokens.weight", d, &nd)) {
            return (sg_err){"model: safetensors embed_tokens.weight has no shape"};
        }
        se = check_elems("embed_tokens.weight", tensor_elems(d, nd), sh.vocab * sh.hidden);
        if (sg_failed(se)) return se;
        if (!st_dense_dims(s, "norm.weight", d, &nd)) {
            return (sg_err){"model: safetensors norm.weight has no shape"};
        }
        se = check_elems("norm.weight", tensor_elems(d, nd), sh.hidden);
        if (sg_failed(se)) return se;
        if (has_lm_head) {
            se = check_elems("lm_head.weight", tensor_elems(lm_dims, lm_nd),
                             sh.vocab * sh.hidden);
            if (sg_failed(se)) return se;
        }
    }

    sg_layer_w *layers = calloc(cfg.n_layers, sizeof(*layers));
    if (!layers) return (sg_err){"model: out of memory"};

    /* Defaults matter: a model with no linear-attention layer at all leaves
     * these untouched, and F32 is the dtype the 2B uses. */
    sg_tensor_type a_type = SG_T_F32, norm_type = SG_T_F32;
    bool a_type_seen = false, norm_type_seen = false;
    bool unsanitized_conv1d = false;
    char suffix[160];
    /* Same element-count validation as the GGUF loader; st_dense_dims reads
     * the shape the index recorded, which st.c has already checked against
     * the tensor's byte length. */
#define ECHK(_elems) do { \
        uint64_t _d[4] = {0, 0, 0, 0}; uint32_t _nd = 0; \
        if (!st_dense_dims(s, suffix, _d, &_nd)) { \
            free(layers); return (sg_err){"model: safetensors tensor has no shape"}; } \
        sg_err _e = check_elems(suffix, tensor_elems(_d, _nd), (_elems)); \
        if (sg_failed(_e)) { free(layers); return _e; } \
    } while (0)
#define REQ(fmt, field, elems, errmsg) do { \
        int _r = snprintf(suffix, sizeof suffix, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof suffix) { free(layers); return (sg_err){"model: safetensors tensor name too long"}; } \
        const void *_p = st_dense_tensor(s, suffix); \
        if (!_p) { free(layers); return (sg_err){errmsg}; } \
        ECHK(elems); \
        lw->field = _p; \
    } while (0)
#define OPT(fmt, field, elems) do { \
        int _r = snprintf(suffix, sizeof suffix, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof suffix) { free(layers); return (sg_err){"model: safetensors tensor name too long"}; } \
        lw->field = st_dense_tensor(s, suffix); \
        if (lw->field) ECHK(elems); \
    } while (0)
/* sg_model carries ONE ssm_a_type and ONE ssm_norm_type for the whole model,
 * and the forward reads every layer's tensor at that width. A checkpoint that
 * mixed the dtypes between its own layers would therefore have all but the
 * last such layer read at 2x or 1/2 width, silently. Reject the mix. */
#define OPT_ANY(fmt, field, type_out, seen_flag, elems) do { \
        int _r = snprintf(suffix, sizeof suffix, fmt, i); \
        if (_r <= 0 || (size_t)_r >= sizeof suffix) { free(layers); return (sg_err){"model: safetensors tensor name too long"}; } \
        sg_tensor_type _t = SG_T_F32; \
        lw->field = st_dense_tensor_any(s, suffix, &_t); \
        if (lw->field) { \
            if ((seen_flag) && _t != (type_out)) { \
                fprintf(stderr, "model: %s is type %d but an earlier layer's was %d; " \
                        "surge records one dtype per model\n", suffix, (int)_t, \
                        (int)(type_out)); \
                free(layers); \
                return (sg_err){"model: a DeltaNet tensor's dtype differs between layers"}; \
            } \
            (type_out) = _t; (seen_flag) = true; ECHK(elems); \
        } \
    } while (0)

    for (uint32_t i = 0; i < cfg.n_layers; i++) {
        sg_layer_w *lw = &layers[i];

        REQ("layers.%u.input_layernorm.weight", ln1, sh.hidden,
            "model: safetensors missing layers.N.input_layernorm.weight");
        REQ("layers.%u.post_attention_layernorm.weight", ln2, sh.hidden,
            "model: safetensors missing layers.N.post_attention_layernorm.weight");
        REQ("layers.%u.mlp.gate_proj.weight", gate_proj, sh.ffn * sh.hidden,
            "model: safetensors missing layers.N.mlp.gate_proj.weight");
        REQ("layers.%u.mlp.up_proj.weight", up_proj, sh.ffn * sh.hidden,
            "model: safetensors missing layers.N.mlp.up_proj.weight");
        REQ("layers.%u.mlp.down_proj.weight", down_proj, sh.hidden * sh.ffn,
            "model: safetensors missing layers.N.mlp.down_proj.weight");

        /* Optional: NULL on a linear-attention layer (see file header note 2). */
        OPT("layers.%u.self_attn.q_proj.weight", q_proj, 2 * sh.attn_w * sh.hidden);
        OPT("layers.%u.self_attn.k_proj.weight", k_proj, sh.kv_w * sh.hidden);
        OPT("layers.%u.self_attn.v_proj.weight", v_proj, sh.kv_w * sh.hidden);
        OPT("layers.%u.self_attn.o_proj.weight", o_proj, sh.hidden * sh.attn_w);
        OPT("layers.%u.self_attn.q_norm.weight", q_norm, sh.head_dim);
        OPT("layers.%u.self_attn.k_norm.weight", k_norm, sh.head_dim);

        /* Present only on a linear-attention layer (see file header note 8).
         * A_log and norm.weight are F32 in this checkpoint while every other
         * weight in the same layer is BF16 (note 9). */
        if (!sh.have_ssm_dims) {
            int _r = snprintf(suffix, sizeof suffix,
                              "layers.%u.linear_attn.A_log", i);
            if (_r > 0 && (size_t)_r < sizeof suffix
                && st_dense_tensor_any(s, suffix, NULL)) {
                free(layers);
                return (sg_err){"model: config.json has linear-attention layers but "
                                "is missing the linear_* head dimensions"};
            }
        }
        OPT("layers.%u.linear_attn.in_proj_qkv.weight", ssm_in_qkv,
            sh.conv_dim * sh.hidden);
        OPT("layers.%u.linear_attn.in_proj_z.weight", ssm_in_z,
            sh.value_dim * sh.hidden);
        OPT("layers.%u.linear_attn.in_proj_b.weight", ssm_in_b, sh.n_v * sh.hidden);
        OPT("layers.%u.linear_attn.in_proj_a.weight", ssm_in_a, sh.n_v * sh.hidden);
        OPT_ANY("layers.%u.linear_attn.A_log", ssm_a, a_type, a_type_seen, sh.n_v);
        OPT("layers.%u.linear_attn.dt_bias", ssm_dt_bias, sh.n_v);
        OPT("layers.%u.linear_attn.conv1d.weight", ssm_conv1d, sh.conv_dim * sh.conv_k);
        OPT_ANY("layers.%u.linear_attn.norm.weight", ssm_norm, norm_type, norm_type_seen, sh.head_v_dim);
        OPT("layers.%u.linear_attn.out_proj.weight", ssm_out, sh.hidden * sh.value_dim);

        /* mlx's first norm-shift trigger: conv1d.weight still in HF's
         * [conv_dim, 1, kernel] layout rather than mlx's post-sanitize
         * [conv_dim, kernel, 1] (see file header note 12). Either way the
         * memory order is [conv_dim][kernel], which is what
         * sg_ref_conv1d_causal wants, so this is read purely as a signal. */
        if (lw->ssm_conv1d) {
            uint64_t cd[4] = {0, 0, 0, 0};
            uint32_t cnd = 0;
            int _r = snprintf(suffix, sizeof suffix,
                              "layers.%u.linear_attn.conv1d.weight", i);
            if (_r > 0 && (size_t)_r < sizeof suffix
                && st_dense_dims(s, suffix, cd, &cnd)
                && cnd >= 1 && cd[cnd - 1] != 1) {
                unsanitized_conv1d = true;
            }
        }

        sg_err ge = check_layer_groups(lw, i);
        if (sg_failed(ge)) { free(layers); return ge; }
    }
#undef REQ
#undef OPT
#undef OPT_ANY
#undef ECHK

    m->cfg = cfg;
    m->tok_emb = tok_emb;
    m->out_norm = out_norm;
    m->lm_head = has_lm_head ? (const void *)lm_head_data : tok_emb;
    m->layers = layers;
    /* The embedding table (and every matmul weight) is BF16 here. This is NOT
     * a claim about every pointer in sg_layer_w: ssm_a_log and ssm_norm come
     * from sg_st_tensor_f32 and are F32 even in this checkpoint. See the
     * per-field dtype table in surge.h's sg_layer_w comment. */
    m->wtype = SG_T_BF16;
    /* Safetensors keeps mlx's own semantics: ssm_a is A_log verbatim and the
     * value heads are in mlx's grouped order (see notes 3 and 4 in surge.h). */
    m->ssm_a_form = SG_SSM_A_LOG;
    m->v_heads_tiled = false;
    m->ssm_a_type = a_type;
    m->ssm_norm_type = norm_type;
    /* Everything reachable through st_dense_tensor came from sg_st_tensor,
     * which returns BF16 and nothing else, so this is true by construction
     * rather than by assumption. */
    m->dense_type = SG_T_BF16;
    /* mlx's TextModel.sanitize condition, reproduced (see file header
     * note 12). Both triggers hold on the real 2B. */
    m->norms_are_residual = unsanitized_conv1d || st_has_mtp_head(s);
    return SG_OK;
}

void sg_model_free(sg_model *m) {
    if (!m) return;
    free(m->layers);
    m->layers = NULL;
}
