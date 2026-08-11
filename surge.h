#ifndef SURGE_H
#define SURGE_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct { const char *msg; } sg_err;

#define SG_OK ((sg_err){0})

static inline bool sg_failed(sg_err e) { return e.msg != NULL; }

/* GGUF v3 reader (Task 2) */

typedef enum { SG_GGUF_U8=0, SG_GGUF_I8, SG_GGUF_U16, SG_GGUF_I16, SG_GGUF_U32,
  SG_GGUF_I32, SG_GGUF_F32, SG_GGUF_BOOL, SG_GGUF_STR, SG_GGUF_ARR, SG_GGUF_U64,
  SG_GGUF_I64, SG_GGUF_F64 } sg_gguf_kv_type;
typedef enum { SG_T_F32=0, SG_T_F16=1, SG_T_Q8_0=8, SG_T_BF16=30 } sg_tensor_type;
typedef struct { const char *name; sg_tensor_type type; uint32_t n_dims;
  uint64_t dims[4]; const void *data; uint64_t nbytes; } sg_tensor;
typedef struct sg_gguf sg_gguf;              /* opaque */
sg_err sg_gguf_open(const char *path, sg_gguf **out);
void sg_gguf_close(sg_gguf *g);
/* metadata lookups return false when the key is absent */
bool sg_gguf_get_u32(const sg_gguf *g, const char *key, uint32_t *out);
bool sg_gguf_get_f32(const sg_gguf *g, const char *key, float *out);
bool sg_gguf_get_str(const sg_gguf *g, const char *key, const char **out);
bool sg_gguf_get_arr(const sg_gguf *g, const char *key, sg_gguf_kv_type *elem_type,
                     const void **data, uint64_t *count);
/* Element accessor for SG_GGUF_STR arrays. sg_gguf_get_arr's data pointer for a
 * string array is the raw packed region (u64 length + bytes per element, no NUL
 * terminators); this walks it and returns an arena-backed NUL-terminated copy of
 * element i. The first call for a given key builds a cached index (one O(count)
 * pass); subsequent calls are O(1). Returns false if key is absent, not a string
 * array, or i is out of bounds. */
bool sg_gguf_get_arr_str(const sg_gguf *g, const char *key, uint64_t i, const char **out);
/* Get scalar value by index; returns false if out of bounds or not a scalar.
 * Pass NULL for outputs you don't need. int_out gets sign-extended int64. */
bool sg_gguf_kv_scalar_at(const sg_gguf *g, uint64_t i, sg_gguf_kv_type *type_out,
                          int64_t *int_out, double *float_out, bool *bool_out);
uint32_t sg_gguf_version(const sg_gguf *g);
uint64_t sg_gguf_kv_count(const sg_gguf *g);
bool sg_gguf_kv_at(const sg_gguf *g, uint64_t i, const char **key_out, sg_gguf_kv_type *type_out);
const sg_tensor *sg_gguf_tensor(const sg_gguf *g, const char *name);
uint64_t sg_gguf_tensor_count(const sg_gguf *g);
const sg_tensor *sg_gguf_tensor_at(const sg_gguf *g, uint64_t i);

/* Byte-level BPE tokenizer (Task 4), built from tokenizer.ggml.* GGUF metadata. */

typedef struct sg_tok sg_tok;
sg_err sg_tok_from_gguf(const sg_gguf *g, sg_tok **out);
void sg_tok_free(sg_tok *t);
sg_err sg_tok_encode(const sg_tok *t, const char *utf8, int32_t **ids, uint64_t *n);
/* Decode appends the bytes for ids[0..n) to buf (capacity buf_cap). Returns the
 * number of bytes written, or -1 if buf_cap is too small or an id is out of range. */
int64_t sg_tok_decode(const sg_tok *t, const int32_t *ids, uint64_t n,
                      char *buf, uint64_t buf_cap);
int32_t sg_tok_eos(const sg_tok *t);

/* Read-only safetensors bf16 loader + config.json (Task 5). */

typedef struct sg_st sg_st;
/* Opens model_dir: reads config.json plus every tensor shard (via
 * model.safetensors.index.json's weight_map when present, otherwise the
 * single *.safetensors file in model_dir). */
sg_err sg_st_open(const char *model_dir, sg_st **out);
void sg_st_close(sg_st *s);
/* bf16 tensor view; name uses the checkpoint's own names. Only BF16-dtype
 * tensors are retrievable; returns false if name is absent, not BF16, or
 * not 2-byte aligned in the file (see sg_st_read_f32 for that last case). */
bool sg_st_tensor(const sg_st *s, const char *name, const uint16_t **data,
                  uint64_t dims[4], uint32_t *n_dims);
/* Copies `count` elements starting at element `first` out of a tensor of
 * EITHER dtype into `out`, widening bf16 to f32. Unlike the two pointer
 * accessors this works on a tensor at any byte offset, because it memcpy's
 * rather than casting.
 *
 * That matters more than it sounds: the safetensors format guarantees no
 * alignment for the data section, and real writers exploit that. The mlx
 * 8-bit repack of Qwen3.6-27B starts its six shards' data sections at
 * 67061 / 49536 / 49742 / 50283 / 50184 / 12735, leaving 1143 of its 2180
 * tensors -- ordinary BF16 layer weights included -- on odd byte offsets.
 * sg_st_tensor and sg_st_tensor_f32 return false for those (handing out a
 * misaligned typed pointer would be UB); this is how to read them. Returns
 * false if the name is absent or the range is out of bounds. */
bool sg_st_read_f32(const sg_st *s, const char *name, uint64_t first,
                    uint64_t count, float *out);
/* f32 tensor view, same contract as sg_st_tensor but for F32-dtype tensors.
 * Real Qwen3.5 checkpoints store two per-linear-attention-layer tensors as
 * F32 while everything around them is BF16 (linear_attn.A_log and
 * linear_attn.norm.weight -- mlx's cast_predicate explicitly keeps A_log out
 * of any cast), so a bf16-only accessor cannot reach them. Returns false if
 * name is absent or not F32. */
bool sg_st_tensor_f32(const sg_st *s, const char *name, const float **data,
                      uint64_t dims[4], uint32_t *n_dims);
/* config.json lookups: top level first, then inside "text_config" if absent. */
bool sg_st_config_u32(const sg_st *s, const char *key, uint32_t *out);
bool sg_st_config_f32(const sg_st *s, const char *key, float *out);
/* Same nesting rules, for a JSON boolean. Needed because config.json states
 * tie_word_embeddings and attention_bias as `true`/`false`, which the two
 * numeric accessors correctly refuse; without this they read as "absent" and
 * a contradiction with the tensor layout goes unnoticed. */
bool sg_st_config_bool(const sg_st *s, const char *key, bool *out);

/* Config extraction + weight-name mapping for the qwen3_5/qwen35 dense-attention
 * subset (Task 6). Real Qwen3.5/3.6 checkpoints are a hybrid of full (softmax)
 * attention layers and linear-attention (gated Delta-Net / SSM) layers,
 * interleaved every full_attention_interval layers (4 in both checkpoints this
 * project targets); only full-attention layers carry the q/k/v/o + qk-norm
 * tensors sg_layer_w names below, so those six pointers are NULL on a
 * linear-attention layer (its state lives in tensors this loader does not map,
 * e.g. GGUF's blk.N.ssm_*). gate_proj/up_proj/down_proj/ln1/ln2 are populated
 * for every layer, attention or linear, since both layer kinds share the same
 * MLP and pre/post norms. */
typedef struct {
    uint32_t n_layers, n_heads, n_kv_heads, head_dim, hidden, ffn_hidden, vocab;
    float rope_theta, rms_eps;
    bool tied_embeddings;

    /* Task 8 additions, needed by the forward pass and by nothing before it.
     *
     * rope_dim is mlx's int(head_dim * partial_rotary_factor), i.e. the
     * number of leading head_dim elements RoPE actually rotates (64 of 256
     * in both checkpoints; the GGUF states it outright as
     * <arch>.rope.dimension_count).
     *
     * full_attn_interval mirrors qwen3_5.py's DecoderLayer rule: layer L is
     * a full-attention layer iff (L + 1) % full_attn_interval == 0, and a
     * gated-DeltaNet layer otherwise. It is a CROSS-CHECK, not the dispatch:
     * both loaders still decide layer kind by tensor presence, and
     * sg_ref_state_new rejects a model whose tensor layout and config
     * disagree rather than silently trusting one of them.
     *
     * The five DeltaNet dims are zero on a model with no linear-attention
     * layer. key_dim = n_k_heads*head_k_dim, value_dim = n_v_heads*head_v_dim
     * and conv_dim = 2*key_dim + value_dim are derived, not stored. */
    uint32_t rope_dim;
    uint32_t full_attn_interval;
    uint32_t n_k_heads, n_v_heads, head_k_dim, head_v_dim, conv_kernel;
} sg_cfg;

/* Per-layer weight pointers. Exactly one of the two attention groups is
 * populated on any given layer, decided by tensor presence (never by a
 * config layer-type list): the six q/k/v/o + qk-norm pointers on a
 * full-attention layer, the nine ssm_* pointers on a linear-attention
 * (gated DeltaNet) layer, and the shared MLP/norm pointers on both.
 *
 * ssm_* field -> checkpoint tensor name (verified against both real files):
 *   ssm_in_qkv   blk.N.attn_qkv.weight      linear_attn.in_proj_qkv.weight
 *   ssm_in_z     blk.N.attn_gate.weight     linear_attn.in_proj_z.weight
 *   ssm_in_b     blk.N.ssm_beta.weight      linear_attn.in_proj_b.weight
 *   ssm_in_a     blk.N.ssm_alpha.weight     linear_attn.in_proj_a.weight
 *   ssm_a        blk.N.ssm_a                linear_attn.A_log
 *   ssm_dt_bias  blk.N.ssm_dt.bias          linear_attn.dt_bias
 *   ssm_conv1d   blk.N.ssm_conv1d.weight    linear_attn.conv1d.weight
 *   ssm_norm     blk.N.ssm_norm.weight      linear_attn.norm.weight
 *   ssm_out      blk.N.ssm_out.weight       linear_attn.out_proj.weight
 * (GGUF names are relative to blk.N; safetensors names to
 * model[.language_model].layers.N.)
 *
 * FOUR THINGS A CALLER MUST NOT ASSUME. Items 1, 2 and 4 were verified by
 * comparing the GGUF against an HF safetensors copy of THE SAME MODEL
 * (/Users/macmini/models/qwen36-27b-8bit, an mlx 8-bit repack of
 * Qwen3.6-27B) tensor by tensor; item 3 against the 2B and the 27B.
 *
 * 1. sg_model.wtype does NOT describe these pointers. It is the embedding
 *    table's type and nothing more. The actual per-field dtypes are:
 *
 *      field                                   GGUF (27B)  safetensors (2B)
 *      q/k/v/o_proj, gate/up/down_proj,
 *      ssm_in_qkv, ssm_in_z, ssm_in_a/b,
 *      ssm_out                                 Q8_0        BF16
 *      ln1, ln2, q_norm, k_norm                F32         BF16
 *      ssm_conv1d, ssm_dt_bias                 F32         BF16
 *      ssm_a, ssm_norm                         F32         F32 or BF16 (!)
 *
 *    Note in particular that ssm_conv1d and ssm_dt_bias differ BY SOURCE,
 *    and that ssm_a/ssm_norm are NOT reliably F32 on the safetensors side:
 *    they are F32 in the Qwen3.5-2B bf16 checkpoint (mlx's cast_predicate
 *    exempts A_log from casting) but BF16 in the mlx 8-bit repack of the
 *    27B. That is why sg_model carries ssm_a_type / ssm_norm_type: the
 *    loader probes both dtypes and records which one it found. Dispatching
 *    on wtype for any of these reads the wrong format. Until the forward
 *    pass carries a full per-tensor type (Task 8), the rule is: matmul
 *    weights follow wtype, ssm_a and ssm_norm follow their recorded types,
 *    everything else in the table above is what the table says.
 *
 * 2. The RMSNorm weights are NOT on the same scale across the two sources.
 *    mlx's TextModel.sanitize adds 1.0 to every 1-D weight named
 *    input_layernorm / post_attention_layernorm / model.norm / q_norm /
 *    k_norm when the checkpoint carries mtp.* tensors or an unsanitized
 *    conv1d weight -- the real 2B safetensors checkpoint carries both, so
 *    mlx shifts them at load time. The GGUF converter baked the shift in
 *    already. So ln1/ln2/q_norm/k_norm/out_norm from sg_model_from_st are
 *    "residual" (centred near 0) and need +1.0 before use, while the same
 *    fields from sg_model_from_gguf are already absolute. ssm_norm is NOT in
 *    mlx's shift list and needs no adjustment from either source.
 *
 *    Evidence, SAME MODEL on both sides (Qwen3.6-27B-Q8_0.gguf against
 *    /Users/macmini/models/qwen36-27b-8bit, an mlx repack of the same
 *    weights, so its norms are already post-sanitize):
 *
 *      tensor                     |gguf - hf|   |gguf - (hf + 1)|
 *      blk.0.attn_norm.weight       3.906e-03       1.004
 *      blk.3.attn_q_norm.weight     3.906e-03       1.004
 *      output_norm.weight           7.812e-03       1.008
 *      blk.0.ssm_norm.weight        0.000e+00       1.000
 *
 *    The GGUF value IS the absolute weight to bf16 grid noise and is a full
 *    1.0 away from the residual one; ssm_norm matches EXACTLY, which is what
 *    "not in the shift list" looks like on both sides. On the raw
 *    (unrepacked) 2B the same tensors are residual: mean
 *    layers.0.input_layernorm.weight +0.0956 and
 *    layers.3.self_attn.q_norm.weight +0.4231, against the GGUF's +0.9762
 *    and +1.2217 for the corresponding tensors -- but note those two means
 *    are from DIFFERENT checkpoints and only the table above is same-model.
 *
 *    Task 8 applies this, dispatching on sg_model.norms_are_residual, and
 *    does the addition in f32 AFTER widening, which is transformers'
 *    Qwen3_5RMSNorm semantics and deliberately NOT mlx-lm's (see the comment
 *    on ref.c's wwiden for the measured consequence). tests/test_model.c
 *    pins the per-source discrepancy, and tests/fixtures/mini_fwd stores
 *    residual norms on purpose so the rule is exercised in a plain
 *    `make check`.
 *
 * 3. ssm_a DOES NOT HOLD THE SAME QUANTITY IN BOTH SOURCES. On the
 *    safetensors side it is mlx's A_log verbatim. On the GGUF side the
 *    converter has already applied -exp() to it, so blk.N.ssm_a stores
 *    -exp(A_log) and llama.cpp multiplies it straight into softplus with no
 *    exp and no negation. Consequences for the decay gate:
 *
 *      safetensors:  decay = exp(-exp(ssm_a) * softplus(a + dt_bias))
 *      GGUF:         decay = exp(     ssm_a  * softplus(a + dt_bias))
 *
 *    Reading the GGUF value as A_log would compute exp(-exp(-exp(A_log)*..))
 *    which is wrong AND silently plausible-looking (still in (0,1)), so this
 *    is dispatched explicitly on sg_model.ssm_a_form rather than assumed.
 *    Verified elementwise, not inferred: for every linear-attention layer of
 *    the 27B, GGUF blk.L.ssm_a equals -exp(HF A_log) under the reindexing in
 *    item 4, to 1.8e-6 (bf16 round-trip noise). All 2304 GGUF ssm_a values
 *    are strictly negative, which alone rules out the A_log reading, since
 *    A_log is genuinely mixed-sign in both HF checkpoints.
 *
 * 4. THE VALUE-HEAD TO KEY-HEAD MAP DIFFERS BY SOURCE. The GGUF converter
 *    reorders the value-head rows of in_proj_qkv / in_proj_z / in_proj_a /
 *    in_proj_b / A_log / dt_bias, the conv1d channels, and the out_proj
 *    columns, so that llama.cpp can consume them with a TILED repeat:
 *
 *      safetensors (mlx/HF):  k_head = v_head / (n_v_heads / n_k_heads)
 *      GGUF:                  k_head = v_head % n_k_heads
 *
 *    Use sg_ssm_k_head() with sg_model.v_heads_tiled rather than open-coding
 *    either one. This only bites when n_v_heads != n_k_heads, which is why
 *    the 2B (16 and 16) cannot expose it and the 27B (16 k, 48 v) hits it on
 *    every linear-attention layer. Verified by recovering the permutation:
 *    GGUF ssm_a matches -exp(HF A_log) to 1.8e-6 under the tiled reindex and
 *    is off by up to 57 under the identity ordering. */
typedef struct {
    const void *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm,
               *gate_proj, *up_proj, *down_proj, *ln1, *ln2;
    /* ssm_a is A_log on the safetensors path and -exp(A_log) on the GGUF
     * path; see note 3 above and sg_model.ssm_a_form. Named ssm_a, not
     * ssm_a_log, precisely because it is not always a log. */
    const void *ssm_in_qkv, *ssm_in_z, *ssm_in_b, *ssm_in_a, *ssm_a,
               *ssm_dt_bias, *ssm_conv1d, *ssm_norm, *ssm_out;
} sg_layer_w; /* per layer; the unused attention group's fields are NULL */

/* What sg_layer_w.ssm_a actually stores, so the decay gate can dispatch
 * instead of guessing. See note 3 on sg_layer_w. */
typedef enum {
    SG_SSM_A_LOG = 0,      /* A_log:      decay = exp(-exp(v) * softplus(..)) */
    SG_SSM_A_NEG_EXP = 1   /* -exp(A_log): decay = exp(     v  * softplus(..)) */
} sg_ssm_a_form;

typedef struct {
    sg_cfg cfg;
    const void *tok_emb, *out_norm, *lm_head; /* lm_head == tok_emb when cfg.tied_embeddings */
    sg_layer_w *layers;  /* cfg.n_layers entries, owned; free with sg_model_free */
    /* The EMBEDDING TABLE's dtype, which is also the matmul weights' dtype.
     * It does NOT describe the norm/conv/bias tensors -- see the dtype table
     * in sg_layer_w's comment above before dispatching on this. */
    sg_tensor_type wtype;

    /* Per-source DeltaNet semantics. All three are properties of the file
     * format, not of the individual layer, so they live here rather than in
     * sg_layer_w. Zeroed (A_log / grouped / F32) on a model with no
     * linear-attention layers. */
    sg_ssm_a_form ssm_a_form;   /* SG_SSM_A_NEG_EXP for GGUF, SG_SSM_A_LOG for st */
    bool v_heads_tiled;         /* true for GGUF: k_head = v_head % n_k_heads */
    sg_tensor_type ssm_a_type;    /* SG_T_F32 or SG_T_BF16; probed at load */
    sg_tensor_type ssm_norm_type; /* SG_T_F32 or SG_T_BF16; probed at load */

    /* Dtype of every SMALL tensor that is not a matmul weight and not one of
     * the two probed above: ln1, ln2, q_norm, k_norm, out_norm, ssm_conv1d,
     * ssm_dt_bias. SG_T_F32 from the GGUF loader (verified per tensor while
     * mapping, not assumed) and SG_T_BF16 from the safetensors loader (where
     * it is true by construction: st_dense_tensor goes through sg_st_tensor,
     * which returns BF16 and nothing else). This is the third row of the
     * dtype table in sg_layer_w's comment, made machine-readable so the
     * forward pass does not have to dispatch on wtype -- which describes the
     * MATMUL weights only. */
    sg_tensor_type dense_type;

    /* True when the RMSNorm weights in ln1/ln2/q_norm/k_norm/out_norm are
     * RESIDUAL and need +1.0 before use; false when they are absolute.
     *
     * This is note 2 on sg_layer_w turned into a flag. mlx's
     * TextModel.sanitize adds 1.0 to exactly those five (and NOT to
     * ssm_norm) when the checkpoint carries mtp.* tensors or an unsanitized
     * conv1d.weight, so a checkpoint that mlx would shift is one surge must
     * shift too. The safetensors loader reproduces mlx's own condition
     * (unsanitized conv1d, i.e. conv1d.weight's last dim != 1, or an mtp.*
     * tensor); the GGUF converter baked the shift in already, so the GGUF
     * loader always reports false. See note 2 on sg_layer_w for the
     * same-model measurements that establish this, and ref.c's wwiden for
     * the fact that surge performs the addition in f32 (transformers'
     * semantics) rather than in bf16 (mlx-lm's). */
    bool norms_are_residual;
} sg_model;

/* The value-head to key-head map, which differs between the two sources
 * (see note 4 on sg_layer_w). n_k_heads must be nonzero and must divide
 * n_v_heads. Always route through this rather than open-coding a division
 * or a modulo: the two agree whenever n_v_heads == n_k_heads, so a wrong
 * choice is invisible on the 2B and wrong on every 27B layer. */
static inline uint32_t sg_ssm_k_head(uint32_t v_head, uint32_t n_k_heads,
                                     uint32_t n_v_heads, bool tiled) {
    /* n_v_heads < n_k_heads would make the grouped divisor zero, so guard on
     * the divisor rather than only on n_k_heads. */
    if (n_k_heads == 0 || n_v_heads < n_k_heads) return 0;
    return tiled ? (v_head % n_k_heads) : (v_head / (n_v_heads / n_k_heads));
}

sg_err sg_model_from_gguf(const sg_gguf *g, sg_model *m);
sg_err sg_model_from_st(const sg_st *s, sg_model *m);
void sg_model_free(sg_model *m);

/* Scalar CPU reference ops (Task 7). Correctness bedrock: every one of
 * these is validated against a fixture generated by running the real
 * mlx-lm qwen3_5 implementation (tools/make_fixtures.py, tests/test_ref_ops.c).
 * They are deliberately plain scalar loops with double-precision
 * accumulators; speed is Task 9's problem. Matrices are row-major
 * [rows, cols] and every matvec computes y = W x. */

/* y = x * rsqrt(mean(x^2) + eps) * w, in place. w may be NULL, which means
 * "no weight" (mx.fast.rms_norm(x, None, eps) -- the DeltaNet q/k path). */
void sg_ref_rmsnorm(float *x, const float *w, uint32_t n, float eps);
/* Full-width RoPE over one head: sg_ref_rope_partial with rope_dim == head_dim. */
void sg_ref_rope(float *q, uint32_t head_dim, uint32_t pos, float theta);
/* Partial RoPE over one head, in place: rotates only the first rope_dim of
 * head_dim and leaves [rope_dim, head_dim) untouched. Half-split
 * (non-interleaved) pairing, matching mlx's nn.RoPE(traditional=False):
 * for i < rope_dim/2, with h = rope_dim/2 and a = pos * theta^(-2i/rope_dim),
 *   q'[i]   = q[i]cos(a) - q[i+h]sin(a)
 *   q'[i+h] = q[i]sin(a) + q[i+h]cos(a)
 * rope_dim must be even and <= head_dim. */
void sg_ref_rope_partial(float *q, uint32_t head_dim, uint32_t rope_dim,
                         uint32_t pos, float theta);
void sg_ref_matvec_f32(const float *w, const float *x, float *y,
                       uint32_t rows, uint32_t cols);
void sg_ref_matvec_bf16(const uint16_t *w, const float *x, float *y,
                        uint32_t rows, uint32_t cols);
/* Q8_0 rows: each row is cols/32 blocks of {f16 scale, int8 q[32]}, 34
 * bytes per block, laid out contiguously row by row. cols must be a
 * multiple of 32. */
void sg_ref_matvec_q8(const void *w, const float *x, float *y,
                      uint32_t rows, uint32_t cols);
void sg_ref_softmax(float *x, uint32_t n);
void sg_ref_swiglu(float *gate, const float *up, uint32_t n); /* gate = silu(gate)*up */
void sg_ref_silu(float *x, uint32_t n);
/* Attention output gate: x[i] *= sigmoid(gate[i]) (Qwen3NextAttention
 * multiplies the attention output by sigmoid of the second half of q_proj
 * before o_proj). */
void sg_ref_gate_sigmoid(float *x, const float *gate, uint32_t n);
float sg_ref_sigmoid(float x);
float sg_ref_softplus(float x);   /* log(1 + e^x), computed stably */
/* Gated DeltaNet decay gate for one head, in the two forms the two
 * checkpoint families store (see note 3 on sg_layer_w, and dispatch on
 * sg_model.ssm_a_form rather than on which source you think you have):
 *
 *   SG_SSM_A_LOG      (safetensors/mlx) exp(-exp(a_log) * softplus(a + dt_bias))
 *   SG_SSM_A_NEG_EXP  (GGUF)            exp(     neg_a  * softplus(a + dt_bias))
 *
 * The two are the same function composed differently: feeding
 * neg_a = -exp(a_log) into the second gives the first. tests/test_ref_ops.c
 * asserts that identity across the whole mlx decay fixture, so the GGUF form
 * inherits the fixture's validation instead of being taken on trust. */
float sg_ref_delta_decay(float a_log, float a, float dt_bias);
float sg_ref_delta_decay_neg_a(float neg_a, float a, float dt_bias);

/* One step of the per-channel causal depthwise conv with carried state.
 * x[channels] is the current token's input, replaced in place by the conv
 * output. w is [channels, ksize] row-major; w[c][ksize-1] multiplies the
 * current token and w[c][0] the oldest carried sample. state_in and
 * state_out are [ksize-1, channels] row-major ring-free buffers holding the
 * previous ksize-1 tokens oldest-first; state_out is the state after this
 * token. state_in == state_out is allowed (update in place); state_out may
 * be NULL to discard. Matches mlx's nn.Conv1d(groups=conv_dim, padding=0)
 * applied to concat(state, x) along the time axis. */
void sg_ref_conv1d_causal(float *x, const float *w, const float *state_in,
                          float *state_out, uint32_t channels, uint32_t ksize);

/* One token of the gated delta rule for one value head, in place on S.
 * S is [dv, dk] row-major, q and k are [dk], v and out are [dv]:
 *   S     <- decay * S
 *   kv[j]  = sum_i S[j][i] * k[i]                (j over dv)
 *   d[j]   = (v[j] - kv[j]) * beta
 *   S[j][i] += k[i] * d[j]
 *   out[j] = sum_i S[j][i] * q[i]
 * This is exactly mlx_lm.models.gated_delta's _gated_delta_step_ops (and
 * its metal kernel) for one (batch, head). The caller supplies decay
 * (= sg_ref_delta_decay) and beta (= sigmoid of the in_proj_b output), and
 * is responsible for the q/k RMS-norm-and-scale that qwen3_5.py applies
 * before this call. */
void sg_ref_delta_step(float *S, const float *q, const float *k, const float *v,
                       float beta, float decay, float *out,
                       uint32_t dk, uint32_t dv);

/* ---------------------------------------------------------------------
 * The reference forward pass (Task 8)
 * ---------------------------------------------------------------------
 *
 * One scalar-C token step over the whole hybrid model, composed from the
 * sg_ref_* ops above and nothing else. Everything is f32 (weights are
 * widened from bf16 or dequantized from Q8_0 on the fly) with double
 * accumulators inside the ops; there is no batching and no fused kernel.
 *
 * sg_ref_state owns the PER-LAYER UNION STATE, which is what makes this a
 * hybrid rather than a transformer:
 *
 *   full-attention layer  ->  f32 K and V caches, [max_ctx, n_kv_heads, head_dim]
 *   gated-DeltaNet layer  ->  a conv tail [conv_kernel-1, conv_dim] plus a
 *                             delta-rule state S [n_v_heads, head_v_dim, head_k_dim]
 *
 * Only the attention layers grow with context; a DeltaNet layer's state is
 * a fixed size no matter how long the sequence gets.
 *
 * It also owns f32 copies of every RMSNorm weight, which is where
 * sg_model.norms_are_residual is applied (+1.0 on ln1/ln2/q_norm/k_norm/
 * out_norm, never on ssm_norm) -- the checkpoint mmap is read-only, so the
 * shift needs owned storage.
 *
 * sg_ref_state_new validates the model against the config before allocating:
 * that every layer's tensor group agrees with full_attn_interval, that
 * rope_dim is even and <= head_dim, and that the DeltaNet dims are present
 * and consistent when any DeltaNet layer exists.
 *
 * sg_ref_forward runs ONE token at position pos and returns a pointer to
 * cfg.vocab logits owned by the state (valid until the next call). Positions
 * must be presented in order starting at 0; feeding the same state a
 * position it has already seen is rejected, because the caches are
 * append-only. */
typedef struct sg_ref_state sg_ref_state;
sg_err sg_ref_state_new(const sg_model *m, uint32_t max_ctx, sg_ref_state **out);
void sg_ref_state_free(sg_ref_state *st);
sg_err sg_ref_forward(sg_ref_state *st, const sg_model *m, int32_t token,
                      uint32_t pos, const float **logits);
/* Rewinds the caches to position 0 without reallocating, so one state can
 * run several independent sequences. */
void sg_ref_state_reset(sg_ref_state *st);

/* ---------------------------------------------------------------------
 * Metal per-op kernels (Task 9), implemented in src/metal.m + src/kernels.metal
 * ---------------------------------------------------------------------
 *
 * Every kernel here is a DECODE-STEP op: one token, batch of one. Weights
 * are bf16 (or f32 for the small tensors), activations are f32, and every
 * reduction has a FIXED SHAPE that does not depend on thread scheduling:
 * each thread accumulates a strided slice in index order into a private
 * register, then a binary tree over threadgroup memory folds the 256
 * partials in a fixed order. No atomics, no simd_* reduction intrinsic
 * (whose fold order the Metal spec does not pin down), no read-modify-write
 * of a shared accumulator. So a kernel run twice on the same bytes produces
 * the same bytes, which tests/test_metal_ops.c asserts 100 runs deep and
 * which is what makes Task 10's byte-exact gate reachable.
 *
 * The kernels do NOT reproduce ref.c's double-precision accumulators: Metal
 * has no f64. Parity against the ref ops is therefore ~1e-7 relative on
 * these sizes, not exact. The one place that gap would have been large is
 * RoPE, where ref computes the angle and its sine/cosine in double at
 * positions up to 262143; the fix is that the HOST precomputes cos/sin in
 * double and uploads them as f32, so the kernel only does the rotation
 * (see k_rope's `b` buffer below).
 *
 * All buffers are f32 unless stated. A buffer handle is whatever
 * sg_gpu_wrap / sg_gpu_alloc returned, never a raw pointer.
 *
 * kernel            a                     b                          out
 * k_rmsnorm         x[n]                  w[n] (or NULL)             x'[n]
 * k_rope            x[head_dim]           cos[rope_dim/2],sin[..]    x'[head_dim]
 * k_matvec_bf16     w[rows*cols] (bf16)   x[cols]                    y[rows]
 * k_matvec_f32      w[rows*cols]          x[cols]                    y[rows]
 * k_softmax         x[n]                  -                          p[n]
 * k_swiglu          gate[n]               up[n]                      silu(gate)*up
 * k_silu            x[n]                  -                          silu(x)
 * k_gate_sigmoid    x[n]                  gate[n]                    x*sigmoid(gate)
 * k_attn_decode     q[heads*q_stride]     k_cache | v_cache          ctx[heads*head_dim]
 * k_conv1d_step     x[channels]           w[channels*ksize]          y[channels] | state[(ksize-1)*channels]
 * k_delta_step      S[dv*dk] (in AND out) q[dk] | k[dk] | v[dv]      y[dv]
 * k_rmsnorm_gated   y[heads*dv]           z[heads*dv] | w[dv]        silu(z)*rms_norm(y,w)
 *
 * Task M5.3 adds three tiled GEMM kernels, Y[N,M] = X[N,K] @ W[M,K]^T, for
 * projecting a whole CHUNK of N tokens through one weight matrix in a single
 * dispatch (the M5 prefill tasks' job; the decode-step table above is
 * unchanged and none of it reaches these). Note the buffer order here is
 * (X, W, Y), the REVERSE of k_matvec_*'s (W, X, Y):
 *
 * kernel            a                     b                          out
 * k_matmul_bf16     X[N*K] f32            W[M*K] (bf16)              Y[N*M] f32
 * k_matmul_f32      X[N*K] f32            W[M*K] f32                 Y[N*M] f32
 * k_matmul_q8       X[N*K] f32            W Q8_0, M*(K/32) blocks    Y[N*M] f32
 *
 * `|` means "concatenated in one buffer". params[] per kernel:
 *
 *   k_rmsnorm        [0]=n [1]=eps bits (f32 bit pattern) [2]=1 if b holds w
 *   k_rope           [0]=head_dim [1]=rope_dim (even, 2..head_dim)
 *   k_matvec_*       [0]=rows [1]=cols
 *   k_softmax        [0]=n
 *   k_swiglu/k_silu/k_gate_sigmoid  [0]=n
 *   k_attn_decode    [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=seq_len
 *                    [4]=q_stride (head_dim, or 2*head_dim when the queries
 *                    are interleaved with the attention output gate)
 *                    [5]=v_cache offset in FLOATS inside b
 *                    [6]=softmax scale bits (host computes 1/sqrt(head_dim))
 *   k_conv1d_step    [0]=channels [1]=ksize
 *   k_delta_step     [0]=dk [1]=dv [2]=beta bits [3]=decay bits
 *   k_rmsnorm_gated  [0]=dv [1]=n_heads [2]=eps bits
 *   k_matmul_*       [0]=N [1]=M [2]=K (K must be a nonzero multiple of 32
 *                    for k_matmul_q8; N and M need not be tile-aligned)
 *
 * Aliasing: `out` must not alias `a` or `b`, with two documented exceptions
 * that mirror the ref ops' own in-place contract: k_conv1d_step's carried
 * state lives in out[channels ..] and is read then rewritten, and
 * k_delta_step updates S (buffer `a`) in place.
 */
typedef struct sg_gpu sg_gpu;

/* Opens the default Metal device, its command queue and kernels.metallib,
 * and builds every pipeline state up front (so a missing or stale metallib
 * fails here rather than mid-decode). Returns an error on a machine with no
 * Metal device; callers that must stay portable should treat that as "skip
 * the GPU path", not as a fatal error.
 *
 * The metallib is looked up in this order: $SURGE_METALLIB, the
 * SG_METALLIB_PATH the Makefile bakes in, ./src/kernels.metallib, and
 * kernels.metallib next to the running executable. */
sg_err sg_gpu_init(sg_gpu **out);
void sg_gpu_free(sg_gpu *g);

/* Wraps caller memory (typically a checkpoint mmap) with no copy.
 *
 * Metal requires a PAGE-ALIGNED base, which a tensor inside a GGUF never
 * is, so this rounds the base DOWN to a page boundary and remembers the
 * offset inside the returned handle; the whole rounded range
 * [base & ~(page-1), base + nbytes rounded up to a page) must be readable,
 * which holds for a file mapping and for any page-granular allocation.
 * The wrapped memory must outlive the handle and must not move.
 *
 * `ptr` must additionally be 4-byte aligned, because the remembered offset
 * is handed to setBuffer:offset:, whose granularity is 4 bytes. GGUF's
 * 32-byte tensor alignment satisfies that; a safetensors tensor at an odd
 * byte offset (see sg_st_read_f32's note) does not, and has to be copied
 * rather than wrapped. */
sg_err sg_gpu_wrap(sg_gpu *g, const void *ptr, uint64_t nbytes, void **buf_out);
/* Fresh shared-storage buffer, zero-filled. *host_out (optional) receives a
 * CPU pointer to its contents: the M-series GPU is unified-memory, so this
 * is the same memory the kernels see, with no upload or download step. */
sg_err sg_gpu_alloc(sg_gpu *g, uint64_t nbytes, void **buf_out, void **host_out);
void sg_gpu_buf_free(void *buf);
/* CPU pointer to a buffer's contents (offset applied), or NULL. */
void *sg_gpu_buf_host(void *buf);

/* One-shot dispatch: encode `kernel`, commit, wait. `b` may be NULL for the
 * kernels whose table row above has no second input. Synchronous by design;
 * Task 10 is what batches a whole layer into one command buffer. */
sg_err sg_gpu_run_op(sg_gpu *g, const char *kernel, void *a, void *b, void *out,
                     const uint32_t params[8]);

/* One-shot dispatch for k_attn_decode_f16 (Task M5.2's fp16-KV attention
 * kernel), same synchronous commit-and-wait contract as sg_gpu_run_op above.
 * It needs THREE device buffer inputs (q, a separate k, a separate v) where
 * every kernel sg_gpu_run_op reaches needs at most two, so it does not fit
 * that function's (a, b, out) shape and gets this dedicated entry point
 * instead. q is f32 [n_heads, q_stride]; k and v are f16
 * [seq, n_kv_heads, head_dim] SEPARATE buffers (the sg_kv layout, not one
 * combined buffer with a v_cache offset like k_attn_decode's b); out is f32
 * [n_heads, head_dim]. params: [0]=n_heads [1]=n_kv_heads [2]=head_dim
 * [3]=seq_len [4]=q_stride [5]=softmax scale bits (f32 bit pattern). */
sg_err sg_gpu_run_attn_decode_f16(sg_gpu *g, void *q, void *k, void *v, void *out,
                                  const uint32_t params[8]);

/* ---------------------------------------------------------------------
 * The full Metal decode path (Task 10)
 * ---------------------------------------------------------------------
 *
 * The GPU twin of sg_ref_state + sg_ref_forward: same hybrid structure, same
 * per-layer dispatch rule, same union state, same norm-shift rule, same
 * source flags (ssm_a_form, v_heads_tiled). One token per call, every layer
 * encoded into ONE command buffer, committed once and waited on once.
 *
 * WHAT IS AND IS NOT REPRODUCIBLE. These are not bit-identical to
 * sg_ref_forward and cannot be: ref.c accumulates in double and Metal has no
 * f64, so the gap is ~1e-7 per op and compounds across 24 layers. The M2 gate
 * is therefore byte-exact GREEDY TOKENS, never byte-exact logits, and the
 * argmax must be computed the same way on both sides (lowest index wins an
 * exact tie) so a genuinely close position cannot flip on convention alone.
 * What IS exact: every run of the GPU path on the same input produces the
 * same bytes (Task 9's determinism property), so a divergence is always a
 * real numerical difference and never a scheduling artifact.
 *
 * THREE HOST-SIDE STEPS, all deliberate: the embedding lookup (the token id
 * is known before the command buffer opens, so ref.c's wrow runs verbatim),
 * the RoPE cos/sin table (computed in double per position and uploaded as
 * f32 -- see sg_ref_rope_partial's note on why an f32 angle is not good
 * enough), and the norm-weight widening plus the +1.0 residual shift, which
 * happens once at load into surge-owned buffers exactly as ref.c does it.
 *
 * WEIGHT DTYPES. bf16 and f32 matmul weights are wrapped with no copy;
 * Q8_0 is rejected with a clear error (it is M3's job). sg_gpu_wrap needs a
 * 4-byte-aligned base: every tensor of the Qwen3.5-2B bf16 checkpoint
 * satisfies that (its safetensors data section starts at byte 76656 and all
 * 632 tensor offsets are multiples of 4), but a checkpoint whose tensors sit
 * at odd offsets -- the mlx 8-bit 27B repack does, see sg_st_read_f32 -- will
 * fail here rather than bind a misaligned pointer, and needs its weights
 * copied into owned buffers first.
 *
 * The KV cache dtype is chosen by the SURGE_KV_DTYPE env var at
 * sg_gpu_state_new time: f16 (Task M5.2, the default) or f32 (the pre-M5.2
 * path, kept so the M2 byte-exact-token gate has an unchanged oracle to run
 * against). The f16 path allocates SEPARATE per-layer K and V buffers
 * through sg_kv (Task M5.1, see the section below), [max_ctx, n_kv_heads,
 * head_dim] each, head-interleaved; the f32 path keeps the original one
 * combined buffer per full-attention layer, [2, max_ctx, n_kv_heads,
 * head_dim]. Either way it is sized from max_ctx, so pass the actual run
 * length plus whatever margin you want -- NOT max_position_embeddings,
 * which is 262144 on this model family and would be a huge cache for a
 * 64-token decode. (The f16 path additionally inherits sg_kv's own cap
 * ceiling, SG_KV_CAP_MAX == 262144, which matches this project's 256K
 * target and so is not a practical limit here.)
 *
 * Order of use: sg_gpu_init -> sg_gpu_load_model -> sg_gpu_state_new ->
 * sg_gpu_forward per token with pos = 0, 1, 2, ... A second sequence needs
 * sg_gpu_state_reset first; feeding a position out of order is an error,
 * because the caches are append-only. */
sg_err sg_gpu_load_model(sg_gpu *g, const sg_model *m);
sg_err sg_gpu_state_new(sg_gpu *g, const sg_model *m, uint32_t max_ctx);
void sg_gpu_state_reset(sg_gpu *g);
/* Runs one token and returns cfg.vocab logits in host memory, owned by the
 * gpu and valid until the next call. */
sg_err sg_gpu_forward(sg_gpu *g, const sg_model *m, int32_t token, uint32_t pos,
                      const float **logits);

/* ---------------------------------------------------------------------
 * Standalone KV-cache + DeltaNet-state module (Task M5.1, src/kv.c)
 * ---------------------------------------------------------------------
 *
 * The per-layer decode state of the hybrid model, pulled out of metal.m into
 * a module that imports NO Metal or Foundation, so it links into pure-C test
 * binaries (it lives in LIB_SRC). It operates on opaque GPU buffer handles
 * (the same void* handles sg_gpu_alloc returns) and never dereferences them
 * itself; it only sizes, allocates, zeroes and hands them out.
 *
 * LAYOUT (per the M5 plan and the verified 27B config):
 *   full-attention layer  ((L+1) % full_attn_interval == 0):
 *       SEPARATE K and V buffers, each [cap, n_kv_heads, head_dim],
 *       head-interleaved (position-major, then kv_head, then dim),
 *       dtype = kv_dtype (SG_T_F16 or SG_T_F32). These GROW with context.
 *   gated-DeltaNet layer (every other layer):
 *       conv tail [conv_kernel-1, conv_dim] f32  (conv_dim derived, see below)
 *       state S   [n_v_heads, head_v_dim, head_k_dim] f32
 *       Both are FIXED size, independent of context length.
 *
 * conv_dim is not an sg_cfg field: it is derived exactly as ref.c/metal.m do,
 * conv_dim = 2*(n_k_heads*head_k_dim) + (n_v_heads*head_v_dim).
 *
 * The layer kind is decided by the config rule (L+1) % full_attn_interval == 0,
 * which is the cross-check sg_ref_state_new / sg_gpu_state_new use; sg_kv is
 * built from an sg_cfg alone (no weight pointers) so it must use that rule.
 *
 * ALLOCATION IS INJECTED. src/kv.c cannot call sg_gpu_alloc directly without
 * dragging the Metal symbols into every pure-C test link. Instead metal.m
 * registers sg_gpu_alloc / sg_gpu_buf_free / sg_gpu_buf_host here at init via
 * sg_kv_set_backend; a pure-C test registers a malloc-backed backend. The
 * size math (sg_kv_bytes / sg_kv_state_bytes) and the f16 round-trip helpers
 * are pure functions that need no backend and allocate nothing, so CI can
 * assert the 16 GiB size WITHOUT allocating it. */

#define SG_KV_CAP_DEFAULT 131072u  /* used when sg_kv_new is passed cap == 0 */
#define SG_KV_CAP_MAX     262144u  /* hard reject above this */

typedef struct sg_kv sg_kv;

/* Pure size math, no allocation. sg_kv_bytes is the total bytes of the
 * GROWABLE K+V attention cache across all full-attention layers at `cap`
 * positions in `dtype` (SG_T_F16 -> 2 bytes, SG_T_F32 -> 4). Returns 0 on an
 * invalid dtype, an implausible dimension (> 1<<24, matching ref.c's ceiling)
 * or a u64 overflow. sg_kv_state_bytes is the total bytes of the FIXED
 * DeltaNet state (conv tail + S, always f32) across all DeltaNet layers. */
uint64_t sg_kv_bytes(const sg_cfg *c, uint32_t cap, sg_tensor_type dtype);
uint64_t sg_kv_state_bytes(const sg_cfg *c);

/* IEEE-754 binary16 round-trip, round-to-nearest-even, matching Metal's
 * `half` and the compiler's _Float16 cast bit-for-bit on finite values. Pure,
 * testable with no GPU. sg_f32_to_f16 flushes overflow to +/-Inf and returns
 * a quiet NaN (0x7E00 | sign) for a NaN input. */
uint16_t sg_f32_to_f16(float f);
float    sg_f16_to_f32(uint16_t h);

/* Allocation backend, injected so src/kv.c carries no Metal link dependency.
 * The signatures match sg_gpu_alloc / sg_gpu_buf_free / sg_gpu_buf_host, which
 * is what metal.m registers; a pure-C caller registers a malloc-backed one.
 * The registered alloc must zero-fill (sg_kv_reset relies on it and so does
 * a freshly created DeltaNet state). */
typedef sg_err (*sg_kv_alloc_fn)(sg_gpu *g, uint64_t nbytes, void **buf, void **host);
typedef void   (*sg_kv_free_fn)(void *buf);
typedef void  *(*sg_kv_host_fn)(void *buf);
void sg_kv_set_backend(sg_kv_alloc_fn alloc, sg_kv_free_fn free_, sg_kv_host_fn host);

/* Allocates the per-layer buffers through the registered backend. `g` is
 * passed straight to the backend's alloc (NULL is fine for a malloc backend).
 * cap == 0 means SG_KV_CAP_DEFAULT; cap > SG_KV_CAP_MAX is hard-rejected.
 * kv_dtype must be SG_T_F16 or SG_T_F32 (K/V only; the DeltaNet state is
 * always f32). Logs per-layer and total bytes. */
sg_err sg_kv_new(sg_gpu *g, const sg_cfg *c, uint32_t cap,
                 sg_tensor_type kv_dtype, sg_kv **out);
void sg_kv_free(sg_kv *kv);
/* Rewinds to position 0: zeroes every conv tail and S (read unconditionally by
 * the DeltaNet scan) and leaves the K/V caches untouched (nothing reads past
 * `used`, and they are 16 GiB). */
void sg_kv_reset(sg_kv *kv);
/* Moves `used` forward by n positions. Rejects used+n > cap (overflow). The
 * cache is append-only and this is the only mutator of `used`, so there is no
 * way to write a position out of order. */
sg_err sg_kv_advance(sg_kv *kv, uint32_t n);
uint32_t sg_kv_used(const sg_kv *kv);
uint32_t sg_kv_cap(const sg_kv *kv);

/* Buffer-handle getters. Return the opaque handle (as from sg_gpu_alloc), or
 * NULL when the layer is the wrong kind (k/v NULL on a DeltaNet layer,
 * conv/s NULL on a full-attention layer) or `layer` is out of range. */
void *sg_kv_k(const sg_kv *kv, uint32_t layer);
void *sg_kv_v(const sg_kv *kv, uint32_t layer);
void *sg_kv_conv(const sg_kv *kv, uint32_t layer);
void *sg_kv_s(const sg_kv *kv, uint32_t layer);

/* ---------------------------------------------------------------------
 * Bench math (Task B1, src/bench.c)
 * ---------------------------------------------------------------------
 *
 * PURE C, no Metal, no GPU, no Foundation -- safe to build and test while
 * the GPU is busy running the 256K comparison loop. Two curve-fit helpers
 * over a per-token cumulative-wall-time series, a leaderboard-row struct,
 * and formatters matching the live doc's table
 * (/Users/macmini/projects/llm-rnd/docs/256k_comparison.md): decode t/s
 * is measured by SLOPE (least-squares fit of token index against wall
 * time) as the headline number, with the mlx-lm-style endpoint average
 * also exposed so B6 can cross-check the two against each other.
 *
 * t_wall_cum[i] is the WALL-CLOCK TIME (seconds, monotonic, any epoch) at
 * which token i finished; t_wall_cum[0] is the time of the FIRST
 * generated token, not zero. `warmup` excludes indices [0, warmup) from
 * both fits: the first token after prefill carries a one-time
 * command-buffer/dispatch transient that is not representative of
 * steady-state decode, so both helpers only look at [warmup, n). */

/* Least-squares slope of token INDEX (y) against cumulative wall time (x)
 * over i in [warmup, n): fits index = slope*time + b and returns slope, in
 * tokens/second. Needs at least 2 points after warmup; returns 0.0 if
 * n <= warmup, n - warmup < 2, t_wall_cum is NULL, or the fit is
 * degenerate (every remaining timestamp identical). Double accumulators
 * throughout, mean-centered (not the textbook one-pass normal-equation
 * form) so the result stays accurate at realistic epoch-scale offsets
 * (~1e9, e.g. raw gettimeofday() seconds), which the one-pass form loses
 * to catastrophic cancellation; see the comment in bench.c. */
double sg_bench_slope(const double *t_wall_cum, uint32_t n, uint32_t warmup);

/* The mlx-lm-style average: (n-1-warmup) tokens divided by the wall time
 * between t_wall_cum[warmup] and t_wall_cum[n-1], endpoint-to-endpoint, no
 * fit. Returns 0.0 on the same degenerate inputs as sg_bench_slope
 * (t_wall_cum NULL, n == 0, warmup >= n-1, or t_wall_cum[n-1] <=
 * t_wall_cum[warmup]). */
double sg_bench_avg_tps(const double *t_wall_cum, uint32_t n, uint32_t warmup);

/* Default warmup when the caller has no opinion: max(1, round(0.02*n_gen)).
 * Always >= 1, so index 0's dispatch-cost transient is never counted. */
uint32_t sg_bench_default_warmup(uint32_t n_gen);

/* One leaderboard row: everything sg_bench_format_md_row / _json need to
 * emit a comparable line, and everything B5's CLI measures in one run.
 * Fixed-size char arrays (no ownership, no free), so a row is copyable and
 * embeddable in a fixed-size log struct. Every char array MUST be
 * NUL-terminated by whoever fills the row in (snprintf into it, never a
 * raw strcpy/memcpy of untrusted length) -- the two formatters defend
 * against a missing NUL with a bounded %s precision matching each array's
 * capacity, but that is a second line of defense, not a substitute. */
typedef struct {
    char model[64];
    char engine[64];
    double prefill_tps;        /* < 0 means "not measured" -> "-" in the md row */
    double decode_tps_slope;   /* sg_bench_slope's result */
    double decode_tps_avg;     /* sg_bench_avg_tps's result */
    double peak_ram_gib;       /* process phys_footprint peak, Task B2 */
    double gpu_alloc_gib;      /* Metal currentAllocatedSize peak, Task B2 */
    uint32_t recall_hits;      /* NIAH direct-retrieval hits, Task B4 */
    uint32_t recall_total;     /* 8 in this project's NIAH needle set */
    uint32_t assoc_hits;       /* NIAH associative-recall hits, Task B4 */
    uint64_t n_prompt_tok;     /* tokens actually ingested */
    uint32_t n_gen;            /* tokens generated */
    double wall_s;             /* total wall time, prefill+decode, seconds */
    double gemm_tflops;        /* fresh GEMM gate reading before this run */
    bool ingestion_ok;         /* Task B3's truncation guard result */
    char status[16];           /* "DONE" or "VOID"; sg_bench_finalize_status sets it */
    char log_id[96];           /* e.g. ctx256k_qwen27b_surge_20260811_120000 */
} sg_bench_row;

/* Formats the 8-column pipe row matching the live doc's header:
 *   | model | engine | 256K prefill t/s | decode t/s | peak RAM | recall | wall | status |
 * In that order: model, engine, prefill_tps ("%.0f", or "-" if < 0),
 * decode_tps_slope ("%.2f"), peak_ram_gib ("%.1f GiB"),
 * "recall_hits/recall_total", wall_s as whole minutes ("%ld min",
 * lround(wall_s/60)), status. Writes via snprintf (truncates rather than
 * overflowing buf); a NULL row or zero cap writes nothing / an empty
 * string. */
void sg_bench_format_md_row(const sg_bench_row *row, char *buf, size_t cap);

/* A flat JSON object of every field, key names matching the struct field
 * names verbatim. Numbers at "%.6g" (round-trips a double to display
 * precision, not bit-exact); ingestion_ok as a JSON bool; string fields
 * (model, engine, status, log_id) are JSON-escaped (quote, backslash, and
 * control bytes as \u00XX) so the output stays valid JSON even if one of
 * them ever contains a quote. snprintf-truncates into cap like the md
 * formatter. */
void sg_bench_format_json(const sg_bench_row *row, char *buf, size_t cap);

/* Applies the leaderboard admission rule in place: status = "DONE" iff
 * gemm_tflops > 20.5 AND ingestion_ok, else "VOID". This is the ONE place
 * that rule lives; callers (B5's CLI, B7's recipe) must call this rather
 * than setting status themselves. */
void sg_bench_finalize_status(sg_bench_row *row);

/* ---------------------------------------------------------------------
 * Prompt ingestion + truncation guard (Task B3, src/bench.c)
 * ---------------------------------------------------------------------
 *
 * PURE C, no Metal, no GPU, no Foundation -- same safety property as the
 * rest of bench.c. Two pieces: a whole-file reader for prompt files (e.g.
 * /Users/macmini/models/niah_256k_prompt.txt), and the VOID/PASS guard
 * that mirrors bench_niah_mlx.py's prompt_tokens==n_built check -- B5's
 * CLI runs this after tokenizing a prompt file and refuses to emit a
 * non-VOID row unless it passes. Tokenizer/GGUF logic (turning bytes into
 * token ids) is out of scope here; that is B5's job. */

#define SG_BENCH_MAX_FILE_BYTES ((uint64_t)3 * 1024 * 1024 * 1024)  /* 3 GiB */

/* Reads path fully into a malloc'd buffer, NUL-terminated one byte past the
 * last file byte (so the result is usable as a C string without an extra
 * copy). *out and *len are only set on success; the caller owns *out and
 * must free() it. Rejects (returns a failed sg_err, *out left NULL) an
 * empty file, a file whose size exceeds SG_BENCH_MAX_FILE_BYTES, or any
 * open/stat/read failure. *len is the byte length actually read, NOT
 * counting the added NUL. */
sg_err sg_bench_read_file(const char *path, char **out, size_t *len);

/* The ingestion/truncation guard: *ok is set true iff BOTH
 *   (n_ids <= max_ctx)                         -- the prompt was not
 *                                                  truncated to fit the
 *                                                  model's context cap, and
 *   (expect_min <= n_ids <= expect_max)         -- the tokenizer produced a
 *                                                  count in the expected
 *                                                  range for this prompt
 *                                                  (catches a silently wrong
 *                                                  tokenizer/BOS setting),
 * else *ok is set false. A row built from a run where *ok is false is VOID
 * regardless of any other measurement (see sg_bench_finalize_status). A
 * NULL ok is a no-op (nothing is written, nothing crashes). */
void sg_bench_check_ingestion(uint64_t n_ids, uint32_t max_ctx, uint64_t expect_min,
                              uint64_t expect_max, bool *ok);

/* ---------------------------------------------------------------------
 * NIAH recall scorer (Task B4, src/bench.c)
 * ---------------------------------------------------------------------
 *
 * PURE C, no Metal, no GPU, no Foundation -- same safety property as the
 * rest of bench.c. Ground truth in the 256K NIAH prompt
 * (/Users/macmini/models/niah_256k_prompt.txt) is a set of "needle" pairs
 * (city, code) buried in filler text, each written EXACTLY as:
 *
 *     IMPORTANT RECORD: the secret access code for <City> is <DIGITS>.
 *
 * where <City> is a single capitalized word (first letter uppercase, then
 * letters, no spaces) and <DIGITS> is a run of 8 or more digits
 * immediately followed by '.'. The prompt's trailing question line lists
 * the city names again WITHOUT codes ("... cities from the text above:
 * Reykjavik, Ouagadougou, ..."); extraction keys on the FULL anchor
 * phrase (not a bare "8-digit number" heuristic) specifically so that
 * line cannot be mistaken for a needle and so filler digit runs elsewhere
 * in the text never inflate ground truth. */

#define SG_BENCH_NEEDLE_CITY_MAX 32   /* longest city name + NUL, e.g. "Ouagadougou" */
#define SG_BENCH_NEEDLE_CODE_MAX 24   /* longest code digit run + NUL (spec: "8+ digits") */
#define SG_BENCH_MAX_NEEDLES 16       /* fixed cap on sg_bench_extract_needles' output array */

typedef struct {
    char city[SG_BENCH_NEEDLE_CITY_MAX];   /* NUL-terminated, letters only */
    char code[SG_BENCH_NEEDLE_CODE_MAX];   /* NUL-terminated, digits only */
} sg_bench_needle;

/* Scans prompt for every occurrence of the anchor phrase
 *     "IMPORTANT RECORD: the secret access code for <City> is <DIGITS>."
 * and writes each (city, code) pair into out[0..*n_out). out must have
 * room for at least `cap` entries; SG_BENCH_MAX_NEEDLES is the project's
 * standing cap (8 real needles in the live prompt, headroom to 16). A
 * candidate is accepted only when the FULL phrase matches: the anchor
 * literal, then an uppercase letter followed by zero or more further
 * letters (the city, stopped at the first non-letter), then " is ", then
 * 8 or more digits (the code), then a literal '.' immediately after the
 * last digit. Anything that does not match that exact shape (including
 * the trailing question line, which names cities but never followed by
 * "is <digits>.", and a lowercase-first or non-letter "city") is silently
 * skipped, not counted. A failed candidate never causes a later, valid
 * anchor occurrence to be missed (scanning resumes one byte past where
 * each attempt started, not past however much of the failed candidate it
 * consumed). Scanning never mutates *prompt and never reads past its NUL
 * terminator.
 *
 * On success (SG_OK), *n_out is the number of pairs written (<= cap); if
 * more than `cap` matches exist in prompt, scanning stops at cap and a
 * notice is printed to stderr (*n_out == cap, not an error). Returns a
 * failed sg_err (nothing written, *n_out left at 0) if prompt/out/n_out is
 * NULL or cap == 0. */
sg_err sg_bench_extract_needles(const char *prompt, sg_bench_needle *out, uint32_t cap,
                                 uint32_t *n_out);

/* Scores a generated answer `gen` against the ground-truth needles:
 *
 *   *retrieval_hits = count of needles[i] whose code appears ANYWHERE in
 *       gen as a raw substring (adjacent punctuation like "13072624." or
 *       "(13072624)" does not block the match; it is a plain substring
 *       search, not a tokenized one).
 *
 *   *assoc_hits = count of needles[i] whose code AND whose city both
 *       appear as substrings of the SAME LINE of gen (gen is split on
 *       '\n'; a trailing '\r' on a line is trimmed so CRLF input still
 *       associates correctly). This is stricter than retrieval: a code
 *       attached to the wrong city on its line does not count, even
 *       though it still counts toward retrieval_hits.
 *
 * gen is read-only throughout (line splitting is done by tracking
 * offsets, not by copying or writing a NUL into gen -- no strtok, no
 * mutation of the input). *retrieval_hits and *assoc_hits are set to 0
 * first and left at 0 if gen/needles is NULL or n_needles == 0; either
 * out-param may be NULL to skip that one. A needle with an empty code (or
 * empty city, for the association half) never matches. */
void sg_bench_score_niah(const char *gen, const sg_bench_needle *needles, uint32_t n_needles,
                          uint32_t *retrieval_hits, uint32_t *assoc_hits);

#endif
