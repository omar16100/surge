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
 * tensors are retrievable; returns false if name is absent or not BF16. */
bool sg_st_tensor(const sg_st *s, const char *name, const uint16_t **data,
                  uint64_t dims[4], uint32_t *n_dims);
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
 *   ssm_a_log    blk.N.ssm_a                linear_attn.A_log
 *   ssm_dt_bias  blk.N.ssm_dt.bias          linear_attn.dt_bias
 *   ssm_conv1d   blk.N.ssm_conv1d.weight    linear_attn.conv1d.weight
 *   ssm_norm     blk.N.ssm_norm.weight      linear_attn.norm.weight
 *   ssm_out      blk.N.ssm_out.weight       linear_attn.out_proj.weight
 * (GGUF names are relative to blk.N; safetensors names to
 * model[.language_model].layers.N.)
 *
 * TWO THINGS A CALLER MUST NOT ASSUME (both verified against the real
 * Qwen3.6-27B-Q8_0.gguf and Qwen3.5-2B safetensors checkpoints):
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
 *      ssm_a_log, ssm_norm                     F32         F32
 *
 *    Note in particular that ssm_conv1d and ssm_dt_bias differ BY SOURCE,
 *    and that ssm_a_log/ssm_norm are F32 even in the bf16 checkpoint (mlx's
 *    cast_predicate deliberately exempts A_log). Dispatching on wtype for
 *    any of these reads the wrong format. Until the forward pass carries a
 *    per-tensor type (Task 8), the rule is: matmul weights follow wtype,
 *    everything else in the table above is what the table says.
 *
 * 2. The RMSNorm weights are NOT on the same scale across the two sources.
 *    mlx's TextModel.sanitize adds 1.0 to every 1-D weight named
 *    input_layernorm / post_attention_layernorm / model.norm / q_norm /
 *    k_norm when the checkpoint carries mtp.* tensors or an unsanitized
 *    conv1d weight -- the real 2B safetensors checkpoint carries both, so
 *    mlx shifts them at load time. The GGUF converter baked the shift in
 *    already. Measured means: ST layers.0.input_layernorm.weight +0.0956 and
 *    layers.3.self_attn.q_norm.weight +0.4231, versus GGUF
 *    blk.0.attn_norm.weight +0.9762 and blk.3.attn_q_norm.weight +1.2217.
 *    So ln1/ln2/q_norm/k_norm/out_norm from sg_model_from_st are "residual"
 *    (centred near 0) and need +1.0 before use, while the same fields from
 *    sg_model_from_gguf are already absolute. ssm_norm is NOT in mlx's shift
 *    list and needs no adjustment from either source. Task 8 owns applying
 *    this; tests/test_model.c pins the discrepancy so it cannot be
 *    rediscovered the hard way. */
typedef struct {
    const void *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm,
               *gate_proj, *up_proj, *down_proj, *ln1, *ln2;
    const void *ssm_in_qkv, *ssm_in_z, *ssm_in_b, *ssm_in_a, *ssm_a_log,
               *ssm_dt_bias, *ssm_conv1d, *ssm_norm, *ssm_out;
} sg_layer_w; /* per layer; the unused attention group's fields are NULL */

typedef struct {
    sg_cfg cfg;
    const void *tok_emb, *out_norm, *lm_head; /* lm_head == tok_emb when cfg.tied_embeddings */
    sg_layer_w *layers;  /* cfg.n_layers entries, owned; free with sg_model_free */
    /* The EMBEDDING TABLE's dtype, which is also the matmul weights' dtype.
     * It does NOT describe the norm/conv/bias tensors -- see the dtype table
     * in sg_layer_w's comment above before dispatching on this. */
    sg_tensor_type wtype;
} sg_model;

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
/* Gated DeltaNet decay gate for one head: exp(-exp(a_log) * softplus(a + dt_bias)). */
float sg_ref_delta_decay(float a_log, float a, float dt_bias);

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

#endif
