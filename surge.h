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

typedef struct {
    const void *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm,
               *gate_proj, *up_proj, *down_proj, *ln1, *ln2;
} sg_layer_w; /* per layer; attention-only fields NULL on a linear-attention layer */

typedef struct {
    sg_cfg cfg;
    const void *tok_emb, *out_norm, *lm_head; /* lm_head == tok_emb when cfg.tied_embeddings */
    sg_layer_w *layers;  /* cfg.n_layers entries, owned; free with sg_model_free */
    sg_tensor_type wtype;
} sg_model;

sg_err sg_model_from_gguf(const sg_gguf *g, sg_model *m);
sg_err sg_model_from_st(const sg_st *s, sg_model *m);
void sg_model_free(sg_model *m);

#endif
