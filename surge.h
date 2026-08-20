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
     * and conv_dim = 2*key_dim + value_dim are derived, not stored.
     *
     * attn_output_gate (Task P1): true iff a full-attention layer's q_proj is
     * DOUBLE width (n_heads*head_dim*2) with a sigmoid output gate folded
     * into the second half, as the qwen3_5/qwen35 hybrid's Qwen3NextAttention
     * does. Plain dense qwen3 (e.g. Qwen3-4B-Instruct-2507) has NO such gate:
     * q_proj is single width and o_proj consumes the raw attention output.
     * This flag makes that difference explicit in the config rather than
     * hardcoded, so the loader, ref.c and metal.m can all size/dispatch the
     * q buffer and the gate step from one place instead of assuming every
     * full-attention layer is gated. */
    uint32_t rope_dim;
    uint32_t full_attn_interval;
    uint32_t n_k_heads, n_v_heads, head_k_dim, head_v_dim, conv_kernel;
    bool attn_output_gate;
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

/* Split-K attention combine (Task P2.0): log-sum-exp rescaling that merges
 * n_parts partial attention results over disjoint, contiguous key ranges into
 * the same result a single-pass softmax-then-weighted-V-sum would produce.
 * This is the CPU correctness oracle for the flash-decoding-style Metal
 * kernel that will later dispatch many threadgroups per head instead of one;
 * that kernel is a later task, this is pure C and proves the math first.
 *
 * Each partition i reports:
 *   m[i]      max score in that partition (-INFINITY if it has zero keys)
 *   s[i]      sum of exp(score - m[i]) over the partition's keys (0.0 if
 *             empty)
 *   acc[i][d] sum of exp(score - m[i]) * v[key][d] over the partition's keys
 *             (all zero if empty); acc is laid out [n_parts][head_dim]
 *             row-major
 *
 * Combine:
 *   M      = max_i m[i]
 *   S      = sum_i s[i] * exp(m[i] - M)
 *   out[d] = ( sum_i acc[i][d] * exp(m[i] - M) ) / S
 *
 * DETERMINISM: partitions are folded in strictly increasing index order in
 * every one of the three passes above (no sort, no reassociation), matching
 * kernels.metal's fixed-shape reduction rule (src/kernels.metal:7-27) so the
 * eventual Metal kernel this proves out has a byte-identical CPU oracle. All
 * accumulation is double precision with one round to float at the end,
 * matching this file's precision policy.
 *
 * K == 1 IS AN IDENTITY: with one partition, M == m[0] exactly, so
 * exp(m[0]-M) == exp(0.0) == 1.0 exactly (an IEEE-754 exact special case) and
 * out[d] reduces to acc[0][d]/s[0] bit-for-bit -- no rounding is introduced
 * by the combine itself.
 *
 * EMPTY PARTITIONS need no special case as long as at least one partition is
 * non-empty: m[i] - M is -INFINITY - finite == -INFINITY (never NaN), and
 * exp(-INFINITY) == 0.0, so an empty partition's s[i]*0.0 and acc[i][d]*0.0
 * terms vanish cleanly on their own.
 *
 * ALL-EMPTY is the one genuine special case (every partition empty, or
 * n_parts == 0): M would be -INFINITY and M - M would be NaN, so this is
 * detected up front and DEFINED rather than left to produce a NaN: out[d] =
 * 0.0 for every d. This represents "attention over zero keys", which has no
 * textbook softmax value (0/0); zero is the documented convention.
 *
 * NaN and +/-INFINITY in m[] are handled the same way sg_ref_softmax (above)
 * handles a non-finite max, since this function exists specifically to
 * consume partials from a Metal split-K kernel during bring-up, which is
 * exactly when a NaN or +INFINITY partition max can show up:
 *
 *   - A NaN anywhere in m[] (at any index, not only m[0]) makes every
 *     out[d] NaN: "there is no defensible distribution to invent, so
 *     propagate rather than manufacture one" -- the same choice
 *     sg_ref_softmax makes and for the same reason. A silently-manufactured
 *     0.0 would be the worse failure, since it looks like a valid answer
 *     instead of surfacing the corruption.
 *   - +INFINITY in m[] (at least one partition reports +INFINITY as its own
 *     max, distinct from -INFINITY, which means empty) makes the
 *     partition(s) tied at that +INFINITY dominate completely, weight
 *     exactly 1.0 each, every other partition weight exactly 0.0 -- the
 *     same "puts all mass on the +inf entries" limit sg_ref_softmax's own
 *     m == +INFINITY branch computes, just via the general log-sum-exp
 *     formula instead of a separate hit-count loop (see attn_combine_weight
 *     in ref.c). A single +INFINITY partition reduces to that partition
 *     alone (an implicit K == 1: out[d] = acc_k[d]/s_k); several tied
 *     +INFINITY partitions combine among only themselves via their own
 *     (s_i, acc_i), exactly as if the other, non-tied partitions were not
 *     there. Finite: no output is ever NaN or +/-INFINITY from this case
 *     alone (unlike the NaN case above).
 *
 * NULL handling mirrors the matvec functions above: a NULL out or head_dim
 * == 0 is a no-op. A NULL m, s or acc with n_parts > 0 is a caller contract
 * violation and is also a no-op (out left untouched) -- NOT the same as the
 * documented all-empty case, which requires valid (possibly all-degenerate)
 * arrays. n_parts == 0 needs no valid m/s/acc at all (nothing is ever
 * indexed) and always writes the defined zero output. */
void sg_ref_attn_combine(const float *m, const float *s, const float *acc,
                         uint32_t n_parts, uint32_t head_dim, float *out);

/* Decode-attention CPU oracle (Task P2.1): the per-op reference for the
 * coming Metal split-K decode kernel, isolating exactly the attention core
 * `k_attn_decode_f16` computes (src/kernels.metal:455-509). `attn_layer`
 * (src/ref.c, static) cannot serve this role: it is a whole hybrid-layer
 * function (projections, qk-norm, RoPE, KV write, attention, gate, o_proj),
 * not a standalone attention-core primitive.
 *
 * LAYOUT (matches sg_gpu_run_attn_decode_f16 above exactly, just f32 kc/vc
 * in place of f16, since this is the pure-C oracle, not a GPU dispatch):
 *   q       [n_heads, q_stride] -- only q[h][0 .. head_dim) is the query.
 *           When q_stride == 2*head_dim (the hybrid model's attention output
 *           gate, Task P1) q[h][head_dim .. q_stride) is the gate and is
 *           NEVER read by this function. q_stride == head_dim (dense qwen3)
 *           has no gate at all.
 *   kc, vc  [seq, n_kv_heads, head_dim], head-interleaved (the sg_kv
 *           layout, see the sg_kv section below), already widened to f32.
 *   out     [n_heads, head_dim].
 *
 * GQA: query head h reads kv head h / (n_heads / n_kv_heads), matching
 * src/kernels.metal:471-472 exactly, including its fallback to kv head 0
 * when n_heads < n_kv_heads (repeat == 0).
 *
 * NUMERICS: max-subtracted softmax, double accumulation, one round to float
 * at the end -- the same convention sg_ref_attn_combine (above) and the rest
 * of this file use. sg_ref_attn_decode is exactly the n_parts == 1 case of
 * sg_ref_attn_decode_splitk below: a single partition spanning the whole
 * [0, seq) range IS a single pass over every key, so both share one static
 * core in ref.c rather than risk two independently written implementations
 * disagreeing by a rounding bit.
 *
 * NULL/degenerate handling mirrors sg_ref_attn_combine: `out == NULL` or
 * `head_dim == 0` or `n_heads == 0` is a no-op (out untouched); `n_kv_heads
 * == 0` is also a no-op (matches k_attn_decode_f16's own dispatch guard,
 * which returns before doing anything when n_kv == 0). NULL q, kc or vc
 * while n_heads > 0 is a caller contract violation, out left untouched.
 * `seq == 0` is well-defined, NOT a no-op: every head then attends over zero
 * keys, sg_ref_attn_combine's documented all-empty convention, so out[d] ==
 * 0.0 for every d.
 *
 * KNOWN DIVERGENCE FROM k_attn_decode_f16 AT seq==0 (review finding, P2.1
 * fix round 1): unlike the n_kv_heads==0/head_dim==0/n_heads==0 no-ops
 * above, seq==0 is NOT a no-op on the Metal kernel side. k_attn_decode_f16
 * (src/kernels.metal:469) folds `seq == 0u` into its own single early-return
 * guard and leaves `out` COMPLETELY UNWRITTEN there -- it does NOT write
 * zeros. This oracle instead treats seq==0 as the well-defined "attention
 * over zero keys" case and writes an explicit 0.0 per the all-empty
 * convention above. A future byte-for-byte Metal-vs-oracle comparison at
 * seq==0 MUST account for this divergence (e.g. pre-zero the Metal output
 * buffer before dispatch, or exclude seq==0 from that specific comparison),
 * or it will spuriously disagree despite both sides being correct by their
 * own documented contract. */
void sg_ref_attn_decode(const float *q, const float *kc, const float *vc,
                        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                        uint32_t seq, uint32_t q_stride, float scale, float *out);

/* Split-K sibling of sg_ref_attn_decode above: partitions [0, seq) into
 * n_parts contiguous, disjoint ranges by the FIXED, data-independent rule
 * `t0 = i*seq/n_parts, t1 = (i+1)*seq/n_parts` (integer division, i in
 * [0, n_parts), computed with a 64-bit intermediate to avoid overflow --
 * the same rule Task P2.0's test file used for its own partition splitter).
 * This tiles the sequence exactly (partition i's t1 equals partition i+1's
 * t0, and the last partition's t1 is exactly seq, so there is no gap and no
 * overlap) and produces a ragged last partition whenever n_parts does not
 * divide seq evenly. A partition can be empty (t0 == t1, guaranteed for the
 * higher-index partitions whenever n_parts > seq); its triple is
 * sg_ref_attn_combine's documented m=-INFINITY/s=0/acc=0 encoding, not a
 * special case here.
 *
 * Per (head, partition), in strictly increasing head order and then
 * strictly increasing partition order (the determinism rule,
 * src/kernels.metal:7-27), computes the partial triple (m, s, acc[head_dim])
 * exactly as sg_ref_attn_combine's own contract defines it, then combines
 * that head's n_parts triples with ONE call to sg_ref_attn_combine (reused,
 * never reimplemented). n_parts == 1 covers [0, seq) in a single partition,
 * which is why sg_ref_attn_decode (n_parts hardcoded to 1 internally) is
 * bit-identical to this function called with n_parts == 1.
 *
 * Same layout, GQA rule, numerics and out/head_dim/n_heads/n_kv_heads/NULL
 * conventions as sg_ref_attn_decode above -- INCLUDING its documented
 * seq==0 divergence from k_attn_decode_f16 (that function leaves `out`
 * unwritten at seq==0; this one writes an explicit 0.0) -- plus: `n_parts
 * == 0` is also well-defined, not a no-op (sg_ref_attn_combine's own
 * n_parts==0 convention: out[d] == 0.0 for every d, since every head then
 * has zero partitions to combine). */
void sg_ref_attn_decode_splitk(const float *q, const float *kc, const float *vc,
                               uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                               uint32_t seq, uint32_t q_stride, float scale,
                               uint32_t n_parts, float *out);

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

/* Metal-only (src/metal.m; Task B2). Bytes currently allocated by this
 * device across every live buffer (MTLDevice.currentAllocatedSize) -- a
 * snapshot, not a peak. Task B2's peak-memory probe samples this alongside
 * sg_proc_phys_footprint (below) into an sg_mem_tracker to build one. 0 if g
 * or g->dev is NULL. */
uint64_t sg_gpu_current_alloc_bytes(const sg_gpu *g);

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

/* One-shot dispatch for k_attn_prefill (Task M5.4's full-attention tiled
 * prefill kernel), the same synchronous three-device-input contract as
 * sg_gpu_run_attn_decode_f16 but over a CHUNK of n query tokens. q is f32
 * [n, n_heads, q_stride]; k and v are f16 [base+n, n_kv_heads, head_dim]
 * SEPARATE buffers (the sg_kv layout) that ALREADY hold base+n positions (the
 * caller stores the chunk's own K/V into the cache before this runs); out is
 * f32 [n, n_heads, head_dim]. Query token t (absolute position base+t) attends
 * causally over cache positions 0..base+t only, so no query ever reads a
 * strictly-future key. params: [0]=n_heads [1]=n_kv_heads [2]=head_dim
 * [3]=base [4]=n [5]=q_stride [6]=softmax scale bits (f32 bit pattern). */
sg_err sg_gpu_run_attn_prefill(sg_gpu *g, void *q, void *k, void *v, void *out,
                               const uint32_t params[8]);

/* One-shot dispatches for the SPLIT-K decode-attention pair (Task P2.2,
 * k_attn_decode_splitk_partial + k_attn_decode_splitk_combine), the Metal twin
 * of sg_ref_attn_decode_splitk / sg_ref_attn_combine above. Same synchronous
 * commit-and-wait contract as the entries above; each is its own dispatch
 * because the partial kernel takes SIX device buffers and the combine FOUR,
 * where sg_gpu_run_op carries at most three.
 *
 * WHY: k_attn_decode_f16 dispatches exactly n_heads threadgroups, so at 32
 * heads only 32 of the 80 GPU cores are scheduled and one threadgroup walks the
 * whole 262,144-key sequence. Split-K makes the grid n_heads x n_splits.
 *
 * WIRED INTO DECODE (Task P2.3, after the P2.2 gates passed on hardware).
 * sg_gpu_forward's attention now dispatches this PAIR, by hand, inside its one
 * open command buffer, whenever the fp16 KV path is live and the sequence is at
 * least SG_TG * 4 == 1024 keys; below that, and on the f32 KV path, and with
 * SURGE_ATTN_SPLITK=0, it dispatches the incumbent k_attn_decode_f16 as before.
 * The entry points here are still the per-op test's oracles and are unchanged;
 * the decode path does NOT call them (they commit and wait, which is exactly
 * what a batched encoder must not do).
 *
 * PARTITIONING is sg_ref_attn_decode_splitk's rule verbatim: split i covers
 * [i*seq/n_splits, (i+1)*seq/n_splits), integer division in 64-bit. It tiles
 * [0, seq) exactly and yields empty splits whenever n_splits > seq; an empty
 * split emits sg_ref_attn_combine's documented m = -INFINITY, s = 0, acc = 0
 * encoding, which the combine consumes with no special case.
 *
 * LAYOUT. q f32 [n_heads, q_stride] (only q[h][0 .. head_dim) is the query;
 * the hybrid model's gate half is never read); k, v f16
 * [seq, n_kv_heads, head_dim] SEPARATE buffers, the sg_kv layout, exactly as
 * sg_gpu_run_attn_decode_f16 takes them; m, s f32 [n_heads, n_splits]; acc f32
 * [n_heads, n_splits, head_dim] (the UNNORMALIZED weighted-V sums); out f32
 * [n_heads, head_dim]. GQA: query head h reads kv head
 * h / (n_heads / n_kv_heads).
 *
 * GQA DOMAIN, and which side is authoritative (P2.2 review finding 2). These
 * entry points REJECT n_heads % n_kv_heads != 0, which includes every
 * n_heads < n_kv_heads shape (with n_heads > 0, n_heads < n_kv_heads can never
 * divide evenly). sg_ref_attn_decode / sg_ref_attn_decode_splitk accept that
 * shape and fall back to kv head 0, so the CPU oracle's domain is a strict
 * SUPERSET of this pair's. THE METAL DOMAIN IS THE AUTHORITATIVE ONE for
 * anything that will be dispatched: on every shape these functions accept, the
 * two agree; on the shapes only the oracle accepts, there is nothing to compare
 * because the dispatch is refused up front, not silently computed differently.
 * k_attn_decode_splitk_partial does carry the same `repeat == 0` fallback the
 * oracle has, so the kernel would compute the oracle's answer, but check_params
 * makes that branch unreachable through these entry points and it must not be
 * read as a promise that the shape is supported.
 *
 * OCCUPANCY, which is the whole point of splitting (P2.2 review finding 4). A
 * threadgroup is SG_TG (256) lanes wide and a split's keys are handed out one
 * per lane, so a split shorter than 256 keys leaves lanes idle: at seq 200 with
 * n_splits 257 the per-split length is 1 and 255 of 256 lanes do nothing.
 * Roughly n_splits <= seq / SG_TG keeps every lane fed, and above that the
 * split count buys threadgroups at the cost of lanes.
 *
 * THE MEASURED DEFAULT (Task P2.3a's sweep, `make bench-splitk`), which the
 * decode path uses: n_splits = clamp(seq / SG_TG, 4, 1024). Sweeping n_splits
 * over {1..1024} at seq 8192 / 32768 / 131072 / 262144, on both the real 27B
 * decode shape (24 heads, 4 kv, head_dim 256) and the real 4B dense shape (32
 * heads, 8 kv, head_dim 128), the fastest value was seq / SG_TG in SEVEN of
 * those eight cells (32, 128, 512, 1024 respectively), i.e. the top of this
 * band: give every split exactly SG_TG keys so no lane idles. At 262144 the
 * pair beat k_attn_decode_f16 by 15.9x on the 27B shape and 21.9x on the 4B
 * shape.
 *
 * THE EIGHTH CELL IS A REPRODUCED COUNTEREXAMPLE, not noise: the 27B shape at
 * seq 8192 is fastest at n_splits 16, not the closed form's 32. Task P2.3's
 * re-sweep read 16 -> 5.226x against 32 -> 4.965x and the P2.3 review's
 * independent one (--reps 20) read 16 -> 5.447x against 32 -> 5.180x, so the
 * roughly 5 percent gap is real. The same review saw the 27B curve go
 * non-monotonic at 32768 as well (16 -> 9.699x above 32 -> 9.129x, before
 * 128 -> 10.783x won the cell). The 4B shape at 8192 does peak at the closed
 * form. The closed form is KEPT as the default anyway, deliberately: it is the
 * top of the occupancy band above rather than a fitted constant, the curve is
 * shallow near the optimum, and a single 5 percent outlier at one shape and one
 * depth (where attention is not the decode bottleneck) does not pay for a
 * shape-specific special case that would then need its own sweep and its own
 * gate. Do not read the closed form as a proven optimum everywhere: re-measure
 * before extending the policy. Below
 * seq 1024 the floor of that clamp is what binds instead of seq / SG_TG, so the
 * splits fall under SG_TG keys and the closed form stops being the measured
 * optimum; the decode path keeps the incumbent kernel there rather than
 * extrapolating below the shortest sequence that was measured.
 *
 * ONE params ARRAY SERVES BOTH CALLS: [0]=n_heads [1]=n_kv_heads [2]=head_dim
 * [3]=seq [4]=q_stride [5]=softmax scale bits (f32 bit pattern) [6]=n_splits
 * (params[7] unused). The combine reads only [0], [2] and [6]. n_splits is a
 * dispatch parameter and is never derived from data, which is what keeps the
 * partition boundaries (and so every summation order) fixed.
 *
 * REJECTED: a NULL argument; n_kv_heads == 0 or n_heads not a multiple of it;
 * q_stride < head_dim; n_heads, head_dim or n_splits == 0; a buffer smaller
 * than the layout above; an output overlapping an input or another output.
 *
 * seq == 0 IS ACCEPTED and is NOT the k_attn_decode_f16 divergence sg_ref_
 * attn_decode documents: every split is then empty, so the pair writes the
 * oracle's own out[d] = 0.0 rather than leaving `out` untouched. These kernels
 * therefore match sg_ref_attn_decode_splitk at every seq, including 0.
 *
 * SCORE SCRATCH (P2.2 review finding 1, the constraint that matters when this
 * is wired into decode). The partial kernel needs one private score row of
 * ceil(seq/n_splits) floats per (query head, split). It does NOT share the
 * process-wide `sg_gpu.scratch` that k_attn_decode, k_attn_decode_f16,
 * k_attn_prefill and sg_gpu_forward's encoder all bind at offset 0: it owns a
 * SEPARATE allocation, grown on demand exactly like that one. That is
 * deliberate and structural. The shared buffer is grown, never partitioned, and
 * every user binds it at offset 0, so two different users inside ONE open
 * command buffer would be addressing the same bytes: split-K's
 * n_heads*n_splits*span rows against another kernel's n_heads*seq rows. A
 * separate buffer means the batched decode path can dispatch the partial
 * alongside anything else without that question arising at all, rather than
 * relying on a rule someone has to remember. Both scratches are released by
 * sg_gpu_free and both are counted by sg_gpu_current_alloc_bytes (which reads
 * the device's own currentAllocatedSize), so the peak-memory probe sees them.
 * The two split-K kernels themselves never conflict: only the partial touches
 * scratch, and each (head, split) threadgroup owns a disjoint row. */
sg_err sg_gpu_run_attn_splitk_partial(sg_gpu *g, void *q, void *k, void *v,
                                      void *m, void *s, void *acc,
                                      const uint32_t params[8]);
sg_err sg_gpu_run_attn_splitk_combine(sg_gpu *g, void *m, void *s, void *acc,
                                      void *out, const uint32_t params[8]);

/* THE GQA-SHARED PARTIAL (Task P2.4), a drop-in alternative to
 * sg_gpu_run_attn_splitk_partial above: same arguments, same params array,
 * same buffer sizes, same score scratch, same [n_heads, n_splits] output
 * layout, and the SAME OUTPUT BYTES. It dispatches
 * k_attn_decode_splitk_partial_gqa, whose grid is (n_splits, n_kv_heads)
 * rather than (n_splits, n_heads).
 *
 * WHY: GQA maps repeat = n_heads / n_kv_heads query heads onto one kv head, so
 * in the per-head kernel the `repeat` threadgroups of a group each stream the
 * SAME K and V slices out of memory. On the real 4B decode shape (32 heads, 8
 * kv) that is 4x the unique bytes; on the 27B shape (24 heads, 4 kv), 6x. This
 * kernel reads each K/V element once and applies it to all `repeat` query
 * vectors, holding `repeat` independent running results per thread.
 *
 * BYTE-IDENTICAL BY CONSTRUCTION, not to a tolerance: the per-head dot
 * products, the per-head fixed-shape tg_max/tg_sum folds and the per-head acc
 * sums keep their operands and their order exactly. Anything else is a bug in
 * the kernel. (The pair of kernels still agrees with sg_ref_attn_decode only
 * to float rounding, as before; it is the two METAL partials that must match
 * bit for bit.)
 *
 * SAME DOMAIN AND SAME REJECTIONS as sg_gpu_run_attn_splitk_partial, checked
 * by the same code. n_heads % n_kv_heads == 0 (already required) is what makes
 * the groups tile the query heads exactly. A group wider than the kernel's
 * SG_SPLITK_GQA_MAX (8) is accepted and answered correctly, one head at a
 * time, but buys no reuse, so the decode path routes those shapes to the
 * per-head kernel instead.
 *
 * THIS IS THE SHIPPED DECODE PARTIAL SINCE TASK P4.0 (2026-08-18). sg_gpu_forward
 * dispatches it by default wherever the group band and the occupancy floor admit
 * it; SURGE_ATTN_SPLITK_GQA=0 (see sg_gpu_state_new below) pins the per-head
 * partial instead, which is what keeps the A/B runnable on one binary. */
sg_err sg_gpu_run_attn_splitk_partial_gqa(sg_gpu *g, void *q, void *k, void *v,
                                          void *m, void *s, void *acc,
                                          const uint32_t params[8]);

/* THE ONLINE-SOFTMAX GQA PARTIAL (Task P2.8), the third alternative for the
 * same job: same arguments, same params array, same buffer sizes, same
 * [n_heads, n_splits] m/s and [n_heads, n_splits, head_dim] acc layout, same
 * (n_splits, n_kv_heads) grid as the GQA partial above, and the SAME COMBINE
 * consumes its output. It dispatches k_attn_decode_splitk_partial_gqa_online.
 *
 * WHAT IS DIFFERENT INSIDE. The two partials above are four-pass kernels: they
 * write every score of the split into a device-memory row and then walk it three
 * more times (maximum, exponentiate, accumulate). This one keeps a running
 * (m, s, acc) per head and updates it as each key is visited, rescaling by
 * exp(m_old - m_new) whenever the running maximum moves. So the score row is
 * gone, and with it the whole reason the split-K score scratch exists: THIS
 * ENTRY POINT NEITHER GROWS NOR BINDS g's split-K scratch buffer. (The buffer is
 * still allocated by sg_gpu_state_new, because the other two arms still need it.
 * If the online kernel ever became the only partial, that allocation would go
 * away entirely: it is sized n_heads * (max_ctx + n_splits) floats, which at the
 * 27B's 24 heads and 262144 context is about 25 MB, the largest scratch surge
 * allocates for decode.)
 *
 * NOT BYTE-IDENTICAL TO THE OTHER TWO, AND THAT IS INHERENT. Streaming changes
 * the ORDER in which the exponentials are summed, so `s` and the `acc` sums
 * differ in the last bits from the four-pass kernels whenever a split spans more
 * than one 256-key tile. What IS exact: `m` (a maximum is order-independent),
 * the empty-split m=-INFINITY/s=0/acc=0 encoding, and in fact the whole triple
 * when a split fits in a single tile, because each thread then holds exactly one
 * key and the folds are the same trees over the same lanes. The bar for this
 * kernel is therefore the P2.2 bar (agreement with sg_ref_attn_decode_splitk to
 * float rounding, plus determinism, plus byte-exact greedy tokens end to end),
 * NOT the memcmp the GQA partial above can promise.
 *
 * DETERMINISM IS UNCHANGED. Each thread streams its own fixed subset of the
 * split's keys, cross-thread combination is only ever a fixed-shape tree with a
 * data-independent stride schedule, and every acc slot has exactly one writer.
 * Repeated dispatches on identical inputs are byte-identical.
 *
 * SAME DOMAIN AND SAME REJECTIONS as the two entry points above, checked by the
 * same code. head_dim > 256 is ACCEPTED and answered correctly, but the kernel
 * then re-streams the split once per 256-wide band of output dims (that is where
 * the running accumulator has to live: one register per head per thread, which
 * needs one output dim per thread), so it re-reads K and the decode path routes
 * those shapes to a four-pass kernel instead. Same for a GQA group wider than 8.
 *
 * NOT THE DECODE DEFAULT. sg_gpu_forward dispatches it only under
 * SURGE_ATTN_SPLITK_ONLINE=1 (see sg_gpu_state_new below): it was written while
 * the GPU was held by a benchmark, so no timing and no accuracy figure had been
 * observed when it landed. */
sg_err sg_gpu_run_attn_splitk_partial_gqa_online(sg_gpu *g, void *q, void *k, void *v,
                                                 void *m, void *s, void *acc,
                                                 const uint32_t params[8]);

/* TWO OBSERVATION POINTS FOR THE GQA GATE (P2.4 fix round 1). Both are
 * read-only diagnostics: no kernel reads them, no buffer size or dispatch shape
 * depends on them, so they cannot change any computed output.
 *
 * They exist because the end-to-end A/B is VACUOUS WITHOUT THEM. The two
 * partials are contracted to write the same bytes, so "SURGE_ATTN_SPLITK_GQA=0
 * and =1 produce byte-identical logits" is ALSO exactly what you see if the GQA
 * kernel is never selected: a later narrowing of the group band, an
 * unset flag, a lost dispatch. Without a way to observe WHICH kernel ran, that
 * gate stays green while the traffic saving silently disappears. P2.3 solved the
 * same hazard by asserting its threshold in both directions.
 *
 * sg_gpu_splitk_gqa_selected answers the POLICY question for a (n_heads,
 * n_kv_heads, seq) triple by calling the same internal predicate the decode
 * encoder consults, so a test of the group band (2 to 8), of the repeat == 1
 * decline, of the not-a-multiple decline or of P2.7's threadgroup floor tests
 * the real rule rather than a copy of it. It reflects the CURRENT state's
 * SURGE_ATTN_SPLITK_GQA, KV dtype and SURGE_SPLITK_GQA_CAP, so it answers false
 * for every shape on a state that turned the switch off with =0 (since P4.0 that
 * is the non-default arm, and it is also how a caller can prove the override
 * still works). A NULL g answers false.
 *
 * `seq` is required because the answer DEPENDS on it (task P2.7): the GQA kernel
 * gives one threadgroup the whole GQA group, which divides the grid by `repeat`,
 * and below a measured floor of 128 threadgroups
 * (sg_gpu_splitk_gqa_n_splits_at(g, seq) * n_kv_heads) that costs more than the
 * traffic it saves, so the same shape is selected at depth and declined at short
 * context. Measured both ways on both real shapes; the table is in metal.m next
 * to splitk_gqa_use and in docs/17082026_splitk_gqa_threadgroups.md.
 *
 * sg_gpu_splitk_dispatch_counts reports how many split-K partial dispatches
 * sg_gpu_forward has ENCODED since the last sg_gpu_state_new, split by kernel:
 * `per_head` counts k_attn_decode_splitk_partial, `gqa` counts
 * k_attn_decode_splitk_partial_gqa. Steps below the split-K threshold dispatch
 * k_attn_decode_f16 and increment neither. The one-shot entry points above are
 * NOT counted; this is about what the decode path chose. Either pointer may be
 * NULL. With the GQA switch on, a run that starts short is EXPECTED to increment
 * both, and where it stops incrementing `per_head` is where the P2.7 floor is. */
bool sg_gpu_splitk_gqa_selected(const sg_gpu *g, uint32_t n_heads,
                                uint32_t n_kv_heads, uint32_t seq);
void sg_gpu_splitk_dispatch_counts(const sg_gpu *g, uint64_t *per_head, uint64_t *gqa);

/* THE SAME TWO OBSERVATION POINTS FOR THE ONLINE ARM (Task P2.8), and they carry
 * the vacuity argument above even more directly: the online partial is contracted
 * to agree with the four-pass ones on greedy TOKENS, not on bits, so an A/B that
 * only compares tokens passes both when the kernel ran and when it was never
 * selected. Read-only diagnostics; nothing computed depends on them.
 *
 * sg_gpu_splitk_online_selected calls the same internal predicate the decode
 * encoder consults. It needs `head_dim` as well as the GQA triple, because the
 * online kernel's own extra condition is head_dim <= 256: that is where its
 * running accumulator stops fitting in one register per head per thread. It
 * reflects the current state's SURGE_ATTN_SPLITK_ONLINE, so it answers false for
 * every shape while the switch is off (the default). NULL g answers false.
 *
 * Everything else the GQA policy requires still applies to this arm, through the
 * same shared predicate: the group must be in [2, 8] and the dispatch must clear
 * P2.7's floor of 128 threadgroups. When BOTH kernel switches are on, the online
 * arm is the one that runs and sg_gpu_splitk_gqa_selected answers false, so the
 * two never both claim the same dispatch.
 *
 * sg_gpu_splitk_online_dispatches counts what sg_gpu_forward ENCODED since the
 * last sg_gpu_state_new, exactly like sg_gpu_splitk_dispatch_counts (whose `gqa`
 * counter still means the FOUR-PASS GQA partial only, so the existing exact-count
 * gates keep their meaning). One-shot entry points are not counted. */
bool sg_gpu_splitk_online_selected(const sg_gpu *g, uint32_t n_heads,
                                   uint32_t n_kv_heads, uint32_t head_dim,
                                   uint32_t seq);
uint64_t sg_gpu_splitk_online_dispatches(const sg_gpu *g);

/* Task P2.5: the GQA partial's OWN split count, the diagnostic counterpart of
 * sg_gpu_splitk_gqa_selected above. sg_gpu_forward's decode encoder computes
 * n_splits for whichever kernel it just selected (per-head: the measured
 * default above, splitk_n_splits(seq), unchanged since P2.3; GQA: this
 * function), so a test of the measured table in metal.m's
 * splitk_gqa_n_splits is a test of the value the encoder actually dispatches
 * with, not a copy of it.
 *
 * n_splits_gqa = clamp(min(seq / SG_TG, 256), 4, 1024), measured task P2.5
 * (`./tests/bench_splitk.bin --seqs 8192,32768,131072,262144 --gqa`): mean
 * regret 0.5%, worst 2.6% against the per-point measured optimum, the best
 * of four candidates scored. The rejected "half the per-head optimum"
 * (seq / (2*SG_TG)) scored worst at 3.8%/13.4%: the true optimum SATURATES
 * near 256 rather than continuing to scale with seq, so a value that keeps
 * climbing and is merely halved eventually overshoots almost as badly as
 * never capping at all. Full table: metal.m (splitk_gqa_n_splits) and
 * docs/17082026_splitk_gqa_threadgroups.md.
 *
 * Pure function of seq; does not read sg_gpu state (there is no on/off
 * switch to gate here, unlike sg_gpu_splitk_gqa_selected -- the decode path
 * only reaches this once the GQA kernel is already selected). Always
 * <= splitk_n_splits(seq) for the same seq, which is why no split-K buffer
 * changed size for this task: it can only shrink the grid the GQA kernel
 * dispatches, never exceed what the per-head policy already sized.
 *
 * Task P2.6 made the cap overridable per state (SURGE_SPLITK_GQA_CAP, see
 * sg_gpu_state_new). This entry point keeps reporting the SHIPPED table, i.e.
 * the compiled 256, because that is the measured policy the table above is
 * about. Use sg_gpu_splitk_gqa_n_splits_at for what a given state dispatches. */
uint32_t sg_gpu_splitk_gqa_n_splits(uint32_t seq);

/* Task P2.6: the same policy AS A PARTICULAR STATE WILL DISPATCH IT, plus the
 * two values a gate needs to prove the two decode arms really do partition the
 * keys differently.
 *
 * sg_gpu_splitk_gqa_n_splits_at(g, seq) is the only honest answer to "how many
 * splits will the GQA arm use at this seq", since that depends on g's resolved
 * cap. It calls the same internal policy with the same resolved cap the decode
 * encoder uses, so a test of it is a test of the dispatch, not of a copy.
 *
 * sg_gpu_splitk_gqa_cap(g) is that resolved cap: the accepted
 * SURGE_SPLITK_GQA_CAP override, or the compiled 256 when none was given (also
 * for a NULL g, or a gpu with no live state). This is how a gate asserts an
 * override was PARSED rather than quietly dropped.
 *
 * sg_gpu_splitk_n_splits(seq) is the PER-HEAD arm's split count, the value the
 * decode path hands the per-head partial in p[6]. Unchanged since P2.3 and
 * never affected by the cap; exposed so a gate can assert the DIVERGENCE
 * (per-head != GQA) instead of only one side of it. Pure function of seq.
 *
 * The invariant across all three: sg_gpu_splitk_gqa_n_splits_at(g, seq) <=
 * sg_gpu_splitk_n_splits(seq) for every state and every seq, at any cap. */
uint32_t sg_gpu_splitk_gqa_n_splits_at(const sg_gpu *g, uint32_t seq);
uint32_t sg_gpu_splitk_gqa_cap(const sg_gpu *g);
uint32_t sg_gpu_splitk_n_splits(uint32_t seq);

/* One-shot dispatches for the gated-DeltaNet chunked-scan prefill kernels (Task
 * M5.5), each the same synchronous commit-and-wait contract as the entries
 * above, extended to the extra device buffer the kernel needs beyond (a, b,
 * out). The within-chunk scan is SEQUENTIAL, so each kernel is BIT-IDENTICAL to
 * its per-token decode sibling looped over the chunk with the recurrent state
 * (conv tail or S) threaded; these entry points are the per-op test's oracles
 * and enc_gdn_prefill dispatches the same kernels by hand. All buffers f32.
 *
 * k_conv1d_chunk: causal depthwise conv over `channels` for a chunk, threading
 *   the conv tail. x [n_tok, channels], w [channels, ksize], out [n_tok,
 *   channels], state [ksize-1, channels] (in AND out). params: [0]=channels
 *   [1]=ksize [2]=n_tok.
 * k_delta_gates_chunk: the alpha/beta gates for a chunk. a [n_tok, n], b
 *   [n_tok, n], gates [n_tok, 2n] out ([beta;decay] per token), adt [ssm_a(n),
 *   dt_bias(n)]. params: [0]=n [1]=neg_exp [2]=n_tok.
 * k_delta_chunk: the delta rule for a chunk through every value head, threading
 *   S. S [n_v, dv, dk] (in AND out), qkv [n_tok, conv_dim], out [n_tok,
 *   value_dim], gates [n_tok, 2*n_v]. params: [0]=dk [1]=dv [2]=n_v [3]=n_k
 *   [4]=key_dim [5]=tiled [6]=n_tok [7]=conv_dim.
 * k_delta_multi (single token): the per-op oracle for k_delta_chunk. S
 *   [n_v, dv, dk] (in AND out), qkv [conv_dim], out [value_dim], gates [2*n_v].
 *   params: [0]=dk [1]=dv [2]=n_v [3]=n_k [4]=key_dim [5]=tiled.
 * k_rmsnorm_gated_chunk: gated output RMSNorm for a chunk. y [n_tok, heads*dv],
 *   z [n_tok, heads*dv], out [n_tok, heads*dv], w [dv] (shared norm weight).
 *   params: [0]=dv [1]=heads [2]=eps bits [3]=n_tok. */
sg_err sg_gpu_run_conv1d_chunk(sg_gpu *g, void *x, void *w, void *out, void *state,
                               const uint32_t params[8]);
sg_err sg_gpu_run_delta_gates_chunk(sg_gpu *g, void *a, void *b, void *gates, void *adt,
                                    const uint32_t params[8]);
sg_err sg_gpu_run_delta_chunk(sg_gpu *g, void *S, void *qkv, void *out, void *gates,
                              const uint32_t params[8]);
sg_err sg_gpu_run_delta_multi(sg_gpu *g, void *S, void *qkv, void *out, void *gates,
                              const uint32_t params[8]);
sg_err sg_gpu_run_rmsnorm_gated_chunk(sg_gpu *g, void *y, void *z, void *out, void *w,
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
 * The DECODE-STEP ATTENTION KERNEL is chosen by the SURGE_ATTN_SPLITK env var,
 * also read at sg_gpu_state_new time (Task P2.3). Default 1: sg_gpu_forward
 * dispatches the split-K pair (k_attn_decode_splitk_partial +
 * k_attn_decode_splitk_combine) with the measured
 * n_splits = clamp(seq / SG_TG, 4, 1024), for every step whose sequence has
 * reached SG_TG * 4 == 1024 keys. SURGE_ATTN_SPLITK=0 pins the incumbent
 * k_attn_decode_f16 for every step, which is what an A/B measurement of the two
 * needs; the f32 KV path always uses the incumbent, since the split-K kernels
 * read half-typed K/V. The two kernels partition the sum over keys differently,
 * so they agree to float rounding, not bit for bit: expect the same generated
 * token ids, not the same logit bits. Each choice is independently
 * deterministic (rerunning either one reproduces its own logits exactly).
 *
 * WHICH SPLIT-K PARTIAL runs is a second, independent env var read at the same
 * point (Task P2.4): SURGE_ATTN_SPLITK_GQA. DEFAULT 1 SINCE TASK P4.0
 * (2026-08-18): the GQA-shared k_attn_decode_splitk_partial_gqa, which reads each
 * K/V element once per GQA GROUP instead of once per query head and is contracted
 * to produce the SAME BYTES (unlike the split-K/incumbent choice above, which
 * changes rounding). 0 pins the per-head k_attn_decode_splitk_partial P2.3
 * shipped. It is ignored unless split-K itself is on, for GQA groups outside
 * [2, 8], where it would buy nothing, and below the measured occupancy floor of
 * 128 threadgroups, where the collapsed grid costs more than the traffic it saves
 * (the per-head partial runs there instead, so a single long run legitimately
 * dispatches BOTH kernels). The default moved only after byte-identity, the
 * greedy-token gate where the two split policies diverge, and that floor were all
 * gated on hardware; 1.74x over the per-head partial at the 27B 262144 shape.
 *
 * SURGE_ATTN_SPLITK_ONLINE (Task P2.8) is a THIRD kernel choice at the same
 * point, and a peer of SURGE_ATTN_SPLITK_GQA rather than a modifier of it.
 * Default 0. 1 selects k_attn_decode_splitk_partial_gqa_online, the
 * online-softmax (streaming) form of the GQA partial: one pass over the keys
 * with a running (m, s, acc) instead of a score row in device memory, so it
 * never touches the split-K score scratch. It obeys the same GQA group band and
 * the same P2.7 threadgroup floor, and additionally requires head_dim <= 256
 * (where its running accumulator stops fitting in registers); shapes outside
 * that go to a four-pass partial. When both this and SURGE_ATTN_SPLITK_GQA are
 * 1, this one wins.
 *
 * UNLIKE THE GQA CHOICE, IT CHANGES THE LAST BITS: streaming reorders the
 * exponential sums, so expect the same generated token ids and logits that agree
 * to float rounding, the same relationship SURGE_ATTN_SPLITK=0/1 has. And unlike
 * the two switches above, an unusable value is REJECTED rather than warned about
 * and ignored, for SURGE_SPLITK_GQA_CAP's reason below: the gate for this switch
 * is an A/B, and an A/B whose "on" arm was silently never turned on passes
 * vacuously. Accepted values are exactly "0" and "1".
 *
 * SURGE_SPLITK_GQA_CAP (Task P2.6) overrides the GQA split policy's measured
 * saturation cap, 256 (see sg_gpu_splitk_gqa_n_splits above). IT IS A GATE AND
 * RETUNING KNOB, NOT A RUN OPTION: the GQA and per-head policies diverge only
 * from seq SG_TG * (cap + 1) on, so at the shipped 256 the divergence begins at
 * 65792 keys and no test suite can reach it, while SURGE_SPLITK_GQA_CAP=4 puts
 * the identical mechanism at seq 1280. Unlike the three env vars above, an
 * unusable value is REJECTED (sg_gpu_state_new returns an error) rather than
 * warned about and ignored: the gate that depends on it would otherwise pass
 * vacuously with both arms picking the same n_splits. Accepted values are plain
 * integers in [4, 1024], the same occupancy band the policy clamps to (a leading
 * sign or space is an error, not a value: strtol would accept " 4" and "+4" and
 * this parser deliberately does not). Leaving it unset is the shipped, measured
 * behaviour. Note that ACCEPTED does not imply OBSERVABLE at the top of the
 * band: at cap 1024 the divergence point is 262400, past SG_KV_CAP_MAX, so that
 * override can never change a dispatch; only caps below 1024 do anything, and
 * only above seq SG_TG * (cap + 1).
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
/* Positions currently held in the decode/prefill state (g->used): 0 after
 * sg_gpu_state_new / sg_gpu_state_reset, n_tokens after sg_gpu_prefill, and
 * pos+1 after sg_gpu_forward(pos). Returns 0 for a NULL gpu or one with no live
 * state. This is the public read of the same counter sg_kv_used exposes on the
 * kv module; the M5.7 long-context gate uses it to assert the used counter
 * reaches 262144 after a full-cap ingest without reaching into the opaque
 * sg_gpu. */
uint32_t sg_gpu_used(const sg_gpu *g);

/* ---------------------------------------------------------------------
 * Chunked prompt prefill (Task M5.6)
 * ---------------------------------------------------------------------
 *
 * Ingests a whole prompt (`tokens[0..n_tokens)`, absolute positions
 * 0..n_tokens-1) in chunks of at most `chunk_size` tokens (0 selects the
 * SG_PREFILL_CHUNK_DEFAULT of 1024), ONE Metal command buffer per chunk,
 * every layer of the model encoded into that command buffer: full-attention
 * layers through the M5.4 tiled prefill kernels, gated-DeltaNet layers through
 * the M5.5 chunked-scan kernels. Only the FINAL chunk's LAST row runs
 * out_norm + lm_head; earlier rows and earlier chunks never touch lm_head, so
 * this returns just the last position's cfg.vocab logits, in host memory owned
 * by the gpu and valid until the next sg_gpu_forward / sg_gpu_prefill call.
 *
 * This is the fast path replacement for feeding the prompt one token at a time
 * through sg_gpu_forward: after sg_gpu_prefill(g, m, tokens, n, chunk, &lg), a
 * subsequent sg_gpu_forward(g, m, next, pos=n, ...) continues decoding from the
 * prefilled state and produces the same tokens as the one-at-a-time path.
 *
 * STATE. Prefill starts from a clean state (it resets used to 0, resets the
 * sg_kv DeltaNet conv/S carriers, and zeroes the ad hoc decode conv/S
 * buffers), then, at the end, BRIDGES each DeltaNet layer's final conv tail and
 * S matrix out of the sg_kv carriers into the L->conv_buf / L->ssm the decode
 * path reads, and leaves g->used == n_tokens with g->kv holding the
 * full-attention K/V for positions 0..n_tokens-1. So decode picks up exactly
 * where prefill left off.
 *
 * REQUIRES the fp16 KV path (the default; SURGE_KV_DTYPE=f32 has no sg_kv
 * cache and returns an error here). tokens must all be in [0, vocab) and
 * n_tokens must be >= 1 and <= max_ctx. */
#define SG_PREFILL_CHUNK_DEFAULT 1024u
sg_err sg_gpu_prefill(sg_gpu *g, const sg_model *m, const int32_t *tokens,
                      uint32_t n_tokens, uint32_t chunk_size,
                      const float **out_last_logits);

/* ---------------------------------------------------------------------
 * Prefill duty-cycle: keeping the compositor alive (Task B8)
 * ---------------------------------------------------------------------
 *
 * sg_gpu_prefill normally submits one chunk command buffer after another with
 * no gap, so on a long prefill the GPU never idles. Nothing else on the machine
 * that needs the GPU gets a turn, including WindowServer.
 *
 * WHY THIS EXISTS (corrected 2026-08-15). B8 was originally justified as dodging a
 * Mac Studio M3 firmware GPU limiter believed to clamp to 338 MHz after 3-4 minutes of
 * sustained load. Telemetry from the 30-hour 256K run does not support that: clock is
 * FLAT across a burst (early bursts 712/723/716/710/701 MHz at t+0/30/60/90/120s, late
 * bursts 573/591/590/591/588, constant bin counts), rest length does not predict the
 * next burst's clock (r = +0.017 over 376 burst pairs), and 338 MHz appears in 27 of
 * 7724 loaded samples. Clock tracks CONTEXT LENGTH (r = -0.575), set at burst start,
 * which is long-context prefill becoming memory-bandwidth bound.
 *
 * The rest is still needed, for a different reason: a saturated GPU starves the
 * compositor. WindowServer was watchdog-killed twice on 2026-08-14, taking the GUI
 * session with it, while surge-bench held the GPU. Yielding periodically is what
 * prevents that. See docs/15082026_prefill_duty_cycle_plan.md.
 *
 * sg_gpu_set_prefill_rest arms a duty cycle on `g`. The budget test is
 * PREDICTIVE: sg_gpu_prefill rests when the accumulated GPU-busy time plus an
 * ESTIMATE of the next chunk's (the previous chunk's time scaled by
 * PF_EST_MARGIN) would cross work_budget_ms, rather than after it already has.
 * Testing after the fact let a burst run to budget plus one whole chunk, and
 * chunk cost grows with context: 367 of 367 bursts in the 2026-08-14 256K run
 * overran a 150 s budget, median 199.5 s, worst 332.9 s.
 *
 * sg_gpu_prefill accumulates the wall time of each chunk's
 * commit..waitUntilCompleted span, and once that accumulated time PLUS the
 * estimate above would reach work_budget_ms -- provided at least one chunk still
 * remains -- sleeps rest_ms with NO command buffer in flight, so the GPU goes
 * genuinely idle and anything else needing the GPU can run, then resumes and
 * resets the accumulator.
 *
 * Note that span is WALL time, not GPU-busy time. On a contended GPU it also
 * counts time other processes held the device, so the accumulator inflates and
 * the duty cycle rests more often than the work alone warrants. That errs
 * toward yielding, which is the safe direction here, but it means budget values
 * tuned on an idle machine do not transfer directly to a busy one.
 *
 * DISABLED is the default (both fields 0 from sg_gpu_init's calloc) and is
 * also what either argument being 0 selects: no accumulator ever crosses
 * work_budget_ms and sg_gpu_prefill never sleeps, so its OUTPUT (gen_ids,
 * logits, KV/decode state, the used-position counter) is byte-identical to
 * before this task -- the chunk loop still takes two extra clock_gettime
 * reads per chunk for the (unused, when disabled) work-time accounting, but
 * those feed nothing that reaches the computed result. When enabled, the
 * rest is PURE TIMING the same way: it reads/writes no buffer, model state,
 * or KV/accumulator that feeds the computed logits, so a rested prefill and
 * an unrested prefill on the same (model, tokens, chunk_size) produce
 * byte-identical gen_ids; only wall-clock time differs.
 *
 * Settings persist on `g` across calls until changed again; pass either
 * argument as 0 to disable. sg_gpu_prefill_rest_ms reads back the total
 * milliseconds actually slept during the MOST RECENT sg_gpu_prefill call
 * (reset to 0 at the start of every call, including a disabled one), which
 * callers such as surge-bench use to separate compute time from idle time
 * in their reported throughput. */
void sg_gpu_set_prefill_rest(sg_gpu *g, uint32_t work_budget_ms, uint32_t rest_ms);
uint64_t sg_gpu_prefill_rest_ms(const sg_gpu *g);

/*
 * sg_gpu_set_prefill_max_burst sets a TARGET ceiling, in milliseconds, for how
 * long any one prefill command buffer holds the GPU. 0 (the default) disables
 * it: one command buffer per chunk, exactly as before.
 *
 * A target, not a cap, and not a promise of a single overrun. The check is
 * reactive: each overrunning submission runs in full and only then halves the
 * segment, so a 64-layer sweep can overrun at 64, then 32, then 16, converging
 * over up to log2(layers) overruns rather than stopping after one. A segment
 * already at the 1-layer floor cannot shrink further and will keep overrunning.
 * Pick a ceiling well under the watchdog window so the overruns along the way
 * still land inside it.
 *
 * Separate mechanism from the duty-cycle rest, for a problem the rest cannot
 * reach. The rest yields the GPU between chunks; it cannot help once a single
 * chunk's command buffer runs longer than macOS lets WindowServer go without
 * rendering (80 s). At 220k context one 256-token chunk measured ~130 s, and
 * WindowServer was watchdog-killed twice on 2026-08-14. When a submission
 * overruns the ceiling, the layer sweep is split across more command buffers.
 *
 * Cannot change computed output. Command buffer boundaries carry no state: the
 * same kernels run with the same arguments in the same order, and buffers
 * committed in sequence on one queue execute in order. This is why the split
 * may adapt mid-run, which changing `chunk` could not safely do.
 */
void sg_gpu_set_prefill_max_burst(sg_gpu *g, uint32_t max_burst_ms);

/*
 * Command buffers submitted by the most recent sg_gpu_prefill call, reset at
 * the start of every call. With segmentation off this equals the chunk count.
 * Exposed so a test can prove segmentation engaged, rather than asserting
 * output parity against a run where nothing actually split.
 */
uint64_t sg_gpu_prefill_segments(const sg_gpu *g);

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
 * Shared greedy decode (src/greedy.c)
 * ---------------------------------------------------------------------
 *
 * The ONE argmax the greedy drivers use. `surge` (cli_metal.c) and
 * `surge-bench` (cli_bench.c) both call this, so their decoded token ids
 * cannot drift: lowest index wins an exact tie (strict >, scanning upward
 * from 0), matching the CPU reference's rule. PURE C, no GPU. Returns 0 for
 * a NULL v or n == 0. */
uint32_t sg_argmax_f32(const float *v, uint32_t n);

/* ---------------------------------------------------------------------
 * Decode pacing + clamp detection (Task P3.0, src/sched.c)
 * ---------------------------------------------------------------------
 *
 * PURE C, no Metal, no GPU, no Foundation. The DECISION LOGIC is a function
 * of (step timings, work budget, rest) with no clock read and no sleep
 * inside it, so it is unit-exact under `make debug` (SURGE_NO_METAL +
 * ASan/UBSan) with no GPU anywhere in the picture. Only
 * sg_decode_pace_step's optional nanosleep touches the outside world.
 *
 * WHY THIS EXISTS. Decode throughput on this machine is partly a function
 * of how recently the GPU was busy. P2.9 (2026-08-18) ran THREE IDENTICAL
 * decode arms on the same binary and prompt and measured 38.47, 25.40 and
 * 16.71 tok/s; after 150 s of idle it re-ran the same three arms and the
 * order REVERSED, at 39.72, 41.16 and 34.66 tok/s. Nothing about the
 * kernels changed between those runs. That 2.4x spread on identical work is
 * why P2.9 declared surge-bench decode tok/s unusable for A/B ranking.
 * Prefill has a mitigation for the related problem (B8's duty cycle, which
 * yields the GPU between chunks); decode had none. This is decode's.
 *
 * TWO PARTS, and they are separable:
 *
 *   (1) A DUTY CYCLE, the direct analogue of sg_gpu_set_prefill_rest: after
 *       work_budget_ms of accumulated decode step time, rest rest_ms with
 *       nothing in flight. OFF unless both fields are > 0, exactly like
 *       B8's, because a decode path that silently inserted sleeps would
 *       corrupt every timing number this project has recorded.
 *
 *   (2) A CLAMP DETECTOR, which needs no configuration and never sleeps on
 *       its own. surge cannot read GPU frequency, so the signal is the only
 *       thing surge can observe: per-step decode wall time rising against
 *       its own baseline, PERSISTENTLY. See sg_decode_pace_decide for the
 *       exact rule and, importantly, for what else can trip it.
 *
 * NOT A GPU OBJECT. Unlike B8's prefill rest, which lives on `sg_gpu`
 * because the chunk loop it paces is inside sg_gpu_prefill, there is no
 * decode loop inside src/metal.m: sg_gpu_forward is ONE step and the
 * per-token loop belongs to the caller (src/cli_bench.c, src/cli_metal.c).
 * So the pacer is a caller-owned value, and src/metal.m is not involved.
 *
 * The struct is transparent (like sg_mem_tracker) so it needs no allocator
 * and a test can inspect its state directly rather than inferring it. Treat
 * the fields as read-only outside src/sched.c; use the accessors. */

/* Upper bound on baseline_n, so the sample ring is inline and the whole
 * pacer is a plain value with no ownership. */
#define SG_PACE_MAX_BASELINE 32u

/* Defaults sg_decode_pace_init installs. See sg_decode_pace_decide for the
 * reasoning behind each, and docs/18082026_decode_pacing.md for the
 * measurement the ratio was chosen against. */
#define SG_PACE_DEF_BASELINE_N 8u
#define SG_PACE_DEF_CONFIRM_N  3u
#define SG_PACE_DEF_CLAMP_RATIO 1.5

/* DEFAULT 1, meaning a confirmed clamp changes no rest schedule at all: the
 * detector REPORTS by default and drives nothing. That default is deliberate
 * and it is the conservative one. A divisor d > 1 makes a confirmed clamp
 * drop the rest threshold to work_budget_ms / d, so rests get up to d times
 * denser while the signal persists.
 *
 * Why it is not on by default. The detector cannot identify a cause (see
 * sg_decode_pace_decide), so every false positive it has would be multiplied
 * into d times the idle time, spent for nothing. And the obvious rationale
 * for escalating -- "rest harder and the clock comes back" -- is the one this
 * project's own telemetry does NOT support: over 376 burst pairs in the
 * 2026-08-14 256K run, rest length did not predict the next burst's clock
 * (r = +0.017), which is why B8's rationale in this header was corrected away
 * from clock recovery and toward compositor protection.
 *
 * The rationale that does survive is contention. If the slowdown is another
 * process taking a share of the GPU, yielding more is the right response and
 * is exactly what B8 yields for. An operator who knows that is the case can
 * set d; surge will not guess it.
 *
 * The divisor form (rather than "rest after every step while clamped") is
 * what bounds the cost: at most one rest_ms per work_budget_ms / d of work,
 * so the worst-case idle fraction a confirmed clamp can add is
 * rest_ms / (work_budget_ms / d + rest_ms), computable before the run. */
#define SG_PACE_DEF_CLAMP_DIV 1u

typedef struct {
    /* --- configuration (sg_decode_pace_init / sg_decode_pace_tune) --- */
    uint32_t work_budget_ms;  /* 0 => pacing disabled (never sleeps) */
    uint32_t rest_ms;         /* 0 => pacing disabled (never sleeps) */
    uint32_t baseline_n;      /* steps per baseline, 1..SG_PACE_MAX_BASELINE */
    uint32_t confirm_n;       /* consecutive steps to confirm/clear a clamp */
    double   clamp_ratio;     /* "over" means step_ms > baseline_ms * ratio */
    uint32_t clamp_div;       /* 1 => a clamp changes no rest schedule */

    /* --- detector state, not configuration --- */
    bool     baseline_seeded; /* baseline came from sg_decode_pace_set_baseline
                               * rather than from this run's own first steps.
                               * Deliberately NOT among the fields
                               * sg_decode_pace_reset preserves: a reset means
                               * "measure the next phase fresh", and silently
                               * keeping a baseline seeded for the previous
                               * phase would be the opposite of that. */

    /* --- state --- */
    double   work_acc_ms;     /* decode step time since the last rest */
    double   baseline_ms;     /* 0.0 until baseline_n samples have arrived */
    double   samples[SG_PACE_MAX_BASELINE];
    uint32_t n_samples;       /* samples collected toward the baseline */
    uint32_t over_run;        /* consecutive over-threshold steps */
    uint32_t under_run;       /* consecutive under-threshold steps */
    uint32_t clamp_events;    /* times the detector LATCHED (edges, not steps) */
    uint32_t clamp_steps;     /* steps observed while latched */
    uint32_t rests;           /* rests emitted */
    uint64_t rest_total_ms;   /* accumulated rest, ms */
    uint64_t steps;           /* real step measurements fed in; a non-finite
                               * or non-positive step is dropped before this
                               * is incremented, so it is not "calls made" */
    bool     clamped;         /* detector latched right now */
} sg_decode_pacer;

/* Zeroes all state, installs the SG_PACE_DEF_* detector defaults, and sets
 * the duty cycle. Either argument 0 leaves pacing DISABLED: no threshold is
 * ever crossed, sg_decode_pace_decide always returns 0, and
 * sg_decode_pace_step never sleeps. The detector still runs when disabled
 * (it costs a few doubles per step and no syscall), so clamp_events is
 * meaningful on an unpaced run too, which is the point: it tells a bench row
 * whether its own decode measurement was taken under a rising step time.
 * A NULL pacer is a no-op, matching this file's defensive-NULL convention. */
void sg_decode_pace_init(sg_decode_pacer *p, uint32_t work_budget_ms, uint32_t rest_ms);

/* Overrides the detector's four constants and resets the detector's state
 * (baseline, sample ring, run counters, clamped flag) so the new settings
 * take effect from the next step rather than being mixed with samples
 * gathered under the old ones. Leaves the duty cycle and the accumulated
 * totals (rest_total_ms, rests, steps, clamp_events) alone. Call it AFTER
 * sg_decode_pace_init, which installs the defaults and would otherwise
 * overwrite these. Out-of-range arguments are CLAMPED, not rejected:
 * baseline_n into [1, SG_PACE_MAX_BASELINE], confirm_n and clamp_div to at
 * least 1, clamp_ratio to at least 1.0 (a ratio below 1 would make an
 * at-baseline step "over" its own baseline, latching the detector on every
 * clean run). Exists so a test can force the detector to fire on a short
 * synthetic series, so the ratio can be retuned against fresh measurement
 * without touching this header, and so an operator who has established that
 * a slowdown is GPU contention can opt into clamp_div > 1. */
void sg_decode_pace_tune(sg_decode_pacer *p, uint32_t baseline_n, uint32_t confirm_n,
                         double clamp_ratio, uint32_t clamp_div);

/* Seeds the detector's baseline from OUTSIDE the run instead of measuring it
 * from the run's own first steps, and exists to cover the detector's worst
 * blind spot.
 *
 * THE BLIND SPOT. A self-measured baseline defines this run's own opening
 * steps as normal BY CONSTRUCTION. So a run that is already slow when it
 * starts -- the GPU still in a low clock state from whatever ran before it,
 * which is precisely the P2.9 case that motivated this task (its second and
 * third arms were slow from their first token, at 25.40 and 16.71 tok/s
 * against 38.47) -- reports ZERO clamp events, because nothing about it ever
 * rises. Within-run detection cannot see a constant. Seeding a baseline
 * measured on a known-idle machine is what makes that case visible.
 *
 * baseline_ms > 0 (and finite) installs it and marks the detector live
 * immediately, so the very first step is classified rather than spent
 * measuring. baseline_ms <= 0 (or non-finite) reverts to self-measurement
 * and clears the detector. Either way the run counters and the latch reset.
 *
 * The second half of the mitigation needs no API: surge-bench reports the
 * effective baseline as decode_baseline_ms in its JSON row, so comparing
 * that field ACROSS runs shows a clamped-from-start run for what it is even
 * when no baseline was seeded. */
void sg_decode_pace_set_baseline(sg_decode_pacer *p, double baseline_ms);

/* THE POLICY. Pure: no clock read, no sleep, no Metal. Call once per decode
 * step with that step's measured wall time in ms; returns the number of ms
 * the caller should now rest (0 = do not rest), and adds that to
 * rest_total_ms on the caller's promise to honour it. Use
 * sg_decode_pace_step in a real loop, which cannot break that promise; use
 * this directly in tests, where the whole point is to drive the policy from
 * a synthetic series with no wall clock involved.
 *
 * THE CLAMP SIGNAL, exactly. The baseline is the MEDIAN of the FIRST
 * baseline_n steps fed in after init/tune/reset (or whatever
 * sg_decode_pace_set_baseline was given), and is then FIXED. A step is
 * "over" if step_ms > baseline_ms * clamp_ratio. The detector LATCHES
 * (clamp_events++, clamped = true) only after confirm_n CONSECUTIVE over
 * steps, and CLEARS after confirm_n consecutive not-over steps. Any single
 * step of the other kind resets the corresponding run counter, so one slow
 * step -- a page fault, a scheduler preemption, a mem sample -- cannot latch
 * it and one fast step cannot clear it.
 *
 * WHY MEDIAN: the first decode steps after a prefill include first-touch
 * and pipeline warm-up and can be several times the steady-state cost. A
 * mean over baseline_n=8 would inherit that; a median cannot unless most of
 * the window is affected. And the failure direction matters: a baseline
 * biased HIGH desensitises the detector (fewer rests than warranted), a
 * baseline biased LOW would insert rests that cost throughput for nothing,
 * so the estimator is chosen to fail toward the former.
 *
 * WHAT ELSE TRIPS IT (the honest part). The signal is "sustained slowdown",
 * which is NOT specific to a firmware clock clamp. Everything below fires it
 * too, and surge cannot tell them apart from inside a decode loop:
 *   - another process taking a share of the GPU,
 *   - thermal throttling,
 *   - legitimate drift as decode extends the KV cache the attention kernels
 *     read (small within one decode run of a few hundred tokens, but real,
 *     and unbounded if a caller feeds a long enough run against one fixed
 *     baseline),
 *   - a caller feeding step times that include non-GPU work whose cost grows
 *     (which is why the wiring in src/cli_bench.c times sg_gpu_forward ALONE
 *     and not the whole loop iteration).
 * The detector's claim is therefore only "step time has persistently risen
 * against this run's own baseline", which is exactly what makes a decode
 * measurement untrustworthy. It is not a claim about a cause.
 *
 * WHAT IT CANNOT SEE, which matters more. It detects a RISE, so it cannot
 * see a run that was slow before its first token. See
 * sg_decode_pace_set_baseline for that blind spot, why it is the motivating
 * case rather than a corner one, and the two ways to cover it.
 *
 * WHAT A CONFIRMED CLAMP DOES. By default (clamp_div == 1) nothing: it is
 * counted and reported and the rest schedule is unchanged. With clamp_div
 * > 1 the rest threshold drops to work_budget_ms / clamp_div while the
 * signal persists. Nothing here is claimed to un-clamp anything: recovery
 * from the M3 GPU clock limiter was observed to need 60-120 s of idle,
 * orders of magnitude more than a per-token rest, and this project's own
 * burst telemetry found no correlation between rest length and the next
 * burst's clock. See SG_PACE_DEF_CLAMP_DIV.
 *
 * CANNOT CHANGE OUTPUT. The pacer reads only step times and writes only its
 * own counters. Nothing it touches feeds a logit, a KV entry, or a token id,
 * and resting happens with no work in flight. A paced decode and an unpaced
 * decode over the same (model, prompt, n) emit byte-identical gen_ids; only
 * wall time differs. tests/test_cli_bench.sh asserts that on a real model.
 *
 * A NULL pacer returns 0. A non-finite or non-positive step_ms is IGNORED
 * (not counted, not accumulated, not fed to the baseline) rather than
 * poisoning the baseline with it. */
uint32_t sg_decode_pace_decide(sg_decode_pacer *p, double step_ms);

/* sg_decode_pace_decide, then actually sleep whatever it returned. This is
 * the entry point a real decode loop uses, and the reason it exists is that
 * rest_total_ms must not be able to drift from reality: a caller cannot
 * accidentally take the accounting without taking the sleep. Returns the ms
 * slept (0 = none). The sleep is restarted on EINTR so a signal cannot cut
 * it short and leave the accounting overstated. */
uint32_t sg_decode_pace_step(sg_decode_pacer *p, double step_ms);

/* Accumulated rest in ms across every step since init (NOT reset per call,
 * unlike sg_gpu_prefill_rest_ms, which resets at the start of each prefill:
 * a pacer instance IS the decode phase, so there is no enclosing call whose
 * boundary would define a reset). sg_decode_pace_reset zeroes it. */
uint64_t sg_decode_pace_rest_ms(const sg_decode_pacer *p);

/* Number of times the detector LATCHED, i.e. rising edges, not steps spent
 * clamped (sg_decode_pace_clamp_steps is that). > 0 on a run means the run's
 * own step time persistently rose, so its throughput number mixes two clock
 * states and should not be compared against another run's. */
uint32_t sg_decode_pace_clamp_events(const sg_decode_pacer *p);
uint32_t sg_decode_pace_clamp_steps(const sg_decode_pacer *p);

/* Detector state right now, the baseline it is measuring against (0.0 until
 * baseline_n steps have been fed), and the number of rests emitted. */
bool sg_decode_pace_clamped(const sg_decode_pacer *p);
double sg_decode_pace_baseline_ms(const sg_decode_pacer *p);
uint32_t sg_decode_pace_rests(const sg_decode_pacer *p);

/* Zeroes every counter and the detector state, keeping the configuration
 * (both duty-cycle fields and all FOUR detector constants: baseline_n,
 * confirm_n, clamp_ratio, clamp_div). A SEEDED baseline is NOT kept -- a
 * reset means "measure the next phase fresh", and carrying a baseline seeded
 * for the previous phase into it would be the opposite of that. For a caller
 * that runs several decode phases through one pacer and wants each phase's
 * baseline measured at that phase's own context length. */
void sg_decode_pace_reset(sg_decode_pacer *p);

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
    double prefill_wall_s;     /* prefill-phase wall time, seconds, Task B6:
                                 * independently timed (not derived from
                                 * wall_s), so prefill_wall_s + decode_wall_s
                                 * closing wall_s is a real cross-check, not
                                 * a tautology. */
    double decode_wall_s;      /* decode-phase wall time, seconds, Task B6:
                                 * own now_s()..now_s() span around the
                                 * decode loop, independent of the per-token
                                 * t_wall_cum series used for the slope fit. */
    /* Command buffers the prefill submitted. Equals the chunk count unless
     * --prefill-max-burst-ms split the layer sweep. Reported so a run's log
     * shows whether segmentation engaged. */
    unsigned long long prefill_segments;
    double prefill_rest_s;     /* Task B8: total time slept by the duty cycle
                                 * GPU-clamp duty cycle, seconds, INCLUDED in
                                 * prefill_wall_s (a subset of it, not
                                 * additional). 0 when the feature is disabled
                                 * or never triggered. */
    double prefill_compute_tps; /* Task B8: n_prompt_tok / (prefill_wall_s -
                                 * prefill_rest_s) -- the fair full-clock
                                 * kernel-speed number with duty-cycle idle
                                 * time excluded, as opposed to prefill_tps
                                 * (which is wall-clock and so falls when
                                 * duty-cycling is active). < 0 if the
                                 * denominator is <= 0 (not measured, or the
                                 * whole prefill was spent resting). */
    /* Task P3.0, the decode-side mirror of the two fields above. Same
     * discipline, same reason: with pacing armed, decode_tps_slope and
     * decode_tps_avg are wall-clock and so FALL, and these are what let a
     * reader separate compute from idle. */
    double decode_rest_s;      /* total time the decode duty cycle slept,
                                * seconds, a SUBSET of decode_wall_s (not
                                * additional). 0 when pacing is disabled,
                                * which is the default. */
    double decode_compute_tps; /* n_gen / (decode_wall_s - decode_rest_s),
                                * the fair full-clock decode rate with the
                                * pacing idle excluded. < 0 if the
                                * denominator is <= 0. */
    uint32_t decode_rests;     /* rests the duty cycle emitted. Reported next
                                * to decode_rest_s so a test can check the
                                * accounting IDENTITY (rest_s == rests *
                                * rest_ms) without depending on how long a
                                * decode step happens to take on the machine
                                * running it, and can bound the count by the
                                * number of per-token pacing points, which
                                * catches a rest emitted from the wrong place
                                * in the loop. */
    /* Detector output. These are populated on EVERY run, paced or not,
     * because they cost no wall time and their job is to tell a reader
     * whether the decode number next to them is trustworthy. */
    uint32_t decode_clamp_events; /* times per-step decode time persistently
                                   * rose against this run's own baseline.
                                   * > 0 means the run mixes two GPU clock
                                   * states and its decode tok/s should not
                                   * be compared with another run's. */
    double decode_baseline_ms;    /* the baseline those events were measured
                                   * against, ms; 0 if decode was too short
                                   * to establish one. Comparable ACROSS
                                   * runs, which is the only way to see a run
                                   * that was already slow at its first
                                   * token: within-run detection cannot see a
                                   * constant (see sg_decode_pace_set_baseline). */
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
 * formatter, so SIZE THE BUFFER for the worst case rather than the typical
 * one: about 1.5 KB of escaped strings (every byte of model/engine/status/
 * log_id becoming a 6-byte \u00XX) plus about 0.8 KB of keys and numbers.
 * src/cli_bench.c uses 4096; anything under ~3 KB can silently drop the
 * closing brace on a row with long, escape-heavy string fields. */
void sg_bench_format_json(const sg_bench_row *row, char *buf, size_t cap);

/* The GEMM-gate floor: a row is admissible only when its measured GEMM
 * throughput clears this (strictly). One named constant so the pre-load
 * short-circuit in surge-bench and the post-run status can never desync from
 * a bare literal. */
#define SG_BENCH_GEMM_MIN_TFLOPS 20.5

/* The leaderboard admission rule, as a pure predicate: true iff
 * gemm_tflops > SG_BENCH_GEMM_MIN_TFLOPS AND ingestion_ok (false for a NULL
 * row). This is the ONE place the rule's LOGIC lives; sg_bench_finalize_status
 * is defined in terms of it, and surge-bench calls it for its pre-load
 * VOID short-circuit so the two cannot drift. */
bool sg_bench_admitted(const sg_bench_row *row);

/* Applies the admission rule in place: status = "DONE" iff sg_bench_admitted,
 * else "VOID". This is the ONE place that WRITES status; callers (B5's CLI,
 * B7's recipe) must call this rather than setting status themselves. */
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

/* ---------------------------------------------------------------------
 * Peak-memory probe (Task B2, src/bench.c + src/metal.m)
 * ---------------------------------------------------------------------
 *
 * surge-bench (B5) needs a peak-RAM column comparable to mlx-lm/llama.cpp's
 * in the 256K comparison: the highest value seen, over a run, of two
 * signals -- the Metal device's currently allocated bytes
 * (sg_gpu_current_alloc_bytes, Metal-only, declared above next to the rest
 * of the buffer API) and the process's physical footprint
 * (sg_proc_phys_footprint, below). phys_footprint is process-wide (code,
 * stacks, every allocation, not just Metal's) and is the number that
 * ultimately lands in sg_bench_row.peak_ram_gib, but tracking both catches
 * more: a Metal-buffer leak that never shows up in phys_footprint (or an
 * ordinary CPU-side leak next to a flat GPU allocator) is invisible to
 * either signal alone.
 *
 * sg_mem_tracker is PURE C and unit-testable with no GPU: feed it numbers,
 * it maxes them. The mach/Metal SAMPLING (sg_proc_phys_footprint here,
 * sg_gpu_current_alloc_bytes in src/metal.m) is deliberately kept OUTSIDE
 * the tracker, so tests/test_gpu_mem.c's tracker-math checks run
 * deterministically under `make debug` (SURGE_NO_METAL) with no Metal and
 * no mach call anywhere in that path, while the live probes stay
 * guarded/live like the rest of the Metal surface. */

/* Process physical footprint, in bytes: mach
 * task_info(mach_task_self(), TASK_VM_INFO, ...)'s
 * task_vm_info_data_t.phys_footprint. This is the same number Activity
 * Monitor's "Memory" column and `footprint`/`vmmap` report: resident pages
 * actually charged to this process (compressed memory counted, clean
 * file-backed pages generally not). PURE C: mach syscalls only, no Metal,
 * no Foundation. Returns 0 if task_info fails, or if the running kernel only
 * supports the REV0 task_vm_info revision that predates phys_footprint
 * (never crashes and never reads an uninitialized field; a 0 reading is a
 * visibly wrong number a caller can catch, not a fabricated one).
 *
 * NOT reliably greater than sg_gpu_current_alloc_bytes for a REAL model.
 * The task-load path (sg_gpu_load_model) wraps checkpoint weights with
 * newBufferWithBytesNoCopy (no copy, see sg_gpu_wrap), and Metal's
 * currentAllocatedSize counts a no-copy wrap at its full DECLARED length
 * the instant it is created, regardless of how many of its pages are
 * actually resident -- confirmed with a standalone 512 MiB mmap+wrap probe
 * (reported size == the wrap length exactly, before touching a single
 * page) and live against the real 2B (SURGE_GATE_MODEL=
 * /Users/macmini/models/qwen35-2b): immediately after sg_gpu_load_model,
 * sg_gpu_current_alloc_bytes read 3,768,385,536 bytes (the model's ~3.5 GiB
 * of wrapped bf16 weights) while this function read only 10,683,616 bytes,
 * since almost none of that mmap had been touched yet. The ordering DOES
 * hold for tests/fixtures/mini_fwd (hidden=32, so its wrapped weights are a
 * few KB, negligible next to the process's own baseline footprint) and
 * should also hold again once decode has actually run (every matvec/GEMM
 * kernel reads the weight rows it uses, which pages them in). tests/
 * test_gpu_mem.c's phys_footprint_exceeds_gpu_alloc_after_load gate is
 * therefore mini-fixture-only, not SURGE_GATE_MODEL-driven; a second,
 * SURGE_GATE_MODEL-gated test observes (does not assert an ordering on) the
 * real-model numbers instead. */
uint64_t sg_proc_phys_footprint(void);

/* A running max over two signals, sampled together: each call updates
 * peak = max(peak, max(current_alloc, phys_footprint)). PURE C, no Metal,
 * no mach -- feed it numbers, it maxes them, which is what makes it
 * unit-exact under `make debug` with no GPU anywhere in the picture.
 * sg_mem_tracker_reset zeroes peak; sg_mem_tracker_sample updates it;
 * sg_mem_tracker_peak reads it. A NULL tracker is a no-op / reads back 0,
 * matching this file's other defensive-NULL convention. */
typedef struct { uint64_t peak; } sg_mem_tracker;
void sg_mem_tracker_reset(sg_mem_tracker *t);
void sg_mem_tracker_sample(sg_mem_tracker *t, uint64_t current_alloc, uint64_t phys_footprint);
uint64_t sg_mem_tracker_peak(const sg_mem_tracker *t);

#endif
