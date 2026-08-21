/* kv.c - standalone KV-cache + DeltaNet-state module (Task M5.1).
 *
 * PURE C. No Metal, no Foundation, no GPU header. This file is in LIB_SRC and
 * links into pure-C test binaries, so it must not name a Metal symbol at all.
 * It allocates GPU buffers through an injected backend (sg_kv_set_backend);
 * metal.m registers sg_gpu_alloc/free/host, a pure-C test registers a
 * malloc-backed one. The size math and the f16 round-trip are pure functions
 * that allocate nothing and need no backend, which is how CI asserts the
 * 16 GiB K+V size WITHOUT allocating it.
 *
 * See surge.h for the layout and the design rationale. Everything here is a
 * faithful extraction of the per-layer state that sg_ref_state / sg_gpu_state
 * already carry, restated over opaque handles with SEPARATE K and V buffers
 * (M2's combined [2,ctx,...] buffer becomes two buffers here, per the M5 plan).
 */

#include "surge.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Same ceiling ref.c/metal.m use: far above any real width (the 27B's largest
 * is conv_dim 10240) and far below a u64 wrap, so a garbage config is rejected
 * before it can size an allocation from a wrapped product. */
#define KV_WIDTH_MAX ((uint64_t)1u << 24)

/* u64 overflow-checked mul/add, mirroring the Metal host layer's mul_ck/add_ck
 * (src/metal_internal.h since task R2). */
static bool kv_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}
static bool kv_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static uint64_t kv_dtype_size(sg_tensor_type t) {
    switch (t) {
        case SG_T_F16: return 2;
        case SG_T_F32: return 4;
        default:       return 0;   /* unsupported for a KV buffer */
    }
}

/* Is layer L a full-attention layer? The config rule, matching the cross-check
 * in sg_ref_state_new / sg_gpu_state_new. full_attn_interval 0 would divide by
 * zero, so treat it as "no full-attention layers" (sg_kv_new rejects it). */
static bool kv_is_attn(const sg_cfg *c, uint32_t layer) {
    if (c->full_attn_interval == 0) return false;
    return ((layer + 1u) % c->full_attn_interval) == 0u;
}

/* conv_dim is derived, not stored (see surge.h sg_cfg):
 *   conv_dim = 2*(n_k_heads*head_k_dim) + (n_v_heads*head_v_dim).
 * Each component is bounded by the width ceiling BEFORE the doubling and sum,
 * exactly as ref.c bounds key_dim/value_dim separately: a single u32*u32 fits
 * in u64, but 2*key_dim can wrap, and a wrapped-small conv_dim would slip past
 * the ceiling check and size an undersized buffer. Returns false on a zero or
 * implausible width. */
static bool kv_conv_dim(const sg_cfg *c, uint64_t *out) {
    uint64_t key_dim   = (uint64_t)c->n_k_heads * c->head_k_dim;
    uint64_t value_dim = (uint64_t)c->n_v_heads * c->head_v_dim;
    if (key_dim > KV_WIDTH_MAX || value_dim > KV_WIDTH_MAX) return false;
    uint64_t conv_dim = 2 * key_dim + value_dim;   /* <= 3<<24, cannot wrap */
    if (conv_dim == 0 || conv_dim > KV_WIDTH_MAX) return false;
    *out = conv_dim;
    return true;
}

/* --------------------------------------------------------------------
 * Pure size math (no allocation, no backend)
 * -------------------------------------------------------------------- */

/* Bytes of ONE K (or V) buffer for a full-attention layer: cap positions,
 * each n_kv_heads*head_dim elements of `esz` bytes. Returns false on an
 * implausible width or a u64 overflow (so the caller reports 0). */
static bool kv_kv_buf_bytes(const sg_cfg *c, uint32_t cap, uint64_t esz,
                            uint64_t *out) {
    uint64_t kv_width = (uint64_t)c->n_kv_heads * c->head_dim;
    if (kv_width == 0 || kv_width > KV_WIDTH_MAX) return false;
    uint64_t per_pos;
    if (!kv_mul(kv_width, esz, &per_pos)) return false;
    return kv_mul(per_pos, cap, out);
}

uint64_t sg_kv_bytes(const sg_cfg *c, uint32_t cap, sg_tensor_type dtype) {
    if (!c || c->n_layers == 0) return 0;
    /* The module hard-rejects a larger cap, so sizing one is meaningless and a
     * real number here could mislead a caller into allocating past the max.
     * cap == 0 honestly sizes 0 positions -> 0 bytes (the SG_KV_CAP_DEFAULT
     * substitution is sg_kv_new's job, not the pure math's). */
    if (cap > SG_KV_CAP_MAX) return 0;
    uint64_t esz = kv_dtype_size(dtype);
    if (esz == 0) return 0;

    uint64_t one_buf;
    if (!kv_kv_buf_bytes(c, cap, esz, &one_buf)) return 0;

    /* K and V per full-attention layer. */
    uint64_t per_layer;
    if (!kv_mul(one_buf, 2, &per_layer)) return 0;

    uint64_t total = 0;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        if (!kv_is_attn(c, l)) continue;
        if (!kv_add(total, per_layer, &total)) return 0;
    }
    return total;
}

/* Bytes of the fixed DeltaNet state for ONE layer (conv tail + S, f32).
 * Returns false on an implausible dimension or a u64 overflow. */
static bool kv_state_layer_bytes(const sg_cfg *c, uint64_t *out) {
    uint64_t conv_dim;
    if (!kv_conv_dim(c, &conv_dim)) return false;
    if (c->conv_kernel == 0) return false;

    /* conv tail [conv_kernel-1, conv_dim] */
    uint64_t conv_rows = (uint64_t)c->conv_kernel - 1;
    uint64_t conv_elems;
    if (!kv_mul(conv_rows, conv_dim, &conv_elems)) return false;

    /* S [n_v_heads, head_v_dim, head_k_dim] */
    uint64_t s_elems = c->n_v_heads;
    if (!kv_mul(s_elems, c->head_v_dim, &s_elems)) return false;
    if (!kv_mul(s_elems, c->head_k_dim, &s_elems)) return false;

    uint64_t elems;
    if (!kv_add(conv_elems, s_elems, &elems)) return false;
    return kv_mul(elems, 4, out);   /* f32 */
}

uint64_t sg_kv_state_bytes(const sg_cfg *c) {
    if (!c || c->n_layers == 0) return 0;
    uint64_t per_layer;
    if (!kv_state_layer_bytes(c, &per_layer)) return 0;

    uint64_t total = 0;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        if (kv_is_attn(c, l)) continue;
        if (!kv_add(total, per_layer, &total)) return 0;
    }
    return total;
}

/* --------------------------------------------------------------------
 * f16 round-trip (round-to-nearest-even, matches Metal `half` / _Float16)
 * -------------------------------------------------------------------- */

uint16_t sg_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);                    /* memcpy pun: strict-alias/UBSan clean */
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e    = (int32_t)((x >> 23) & 0xFFu);   /* biased exponent */
    uint32_t m    = x & 0x7FFFFFu;                  /* 23-bit mantissa */

    if (e == 0xFF) {                     /* Inf or NaN */
        if (m) return (uint16_t)(sign | 0x7E00u);   /* quiet NaN */
        return (uint16_t)(sign | 0x7C00u);          /* Inf */
    }

    int32_t E = e - 127;                 /* unbiased exponent */

    if (E >= 16) {                       /* magnitude >= 2^16 overflows to Inf */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (E >= -14) {                      /* normalized half, exp field 1..30 */
        uint32_t hi    = m >> 13;                 /* 10 kept bits */
        uint32_t lower = m & 0x1FFFu;             /* 13 dropped bits */
        uint32_t base  = ((uint32_t)(E + 15) << 10) | hi;
        /* round-to-nearest-even; a carry ripples correctly into the exponent
         * field and, at the top, produces 0x7C00 (Inf) for a rounded overflow. */
        if (lower > 0x1000u || (lower == 0x1000u && (hi & 1u))) base += 1u;
        return (uint16_t)(sign | base);
    }
    if (E < -25) {                       /* magnitude < 2^-25 rounds to 0 */
        return (uint16_t)sign;
    }
    /* subnormal: value = (1.m) * 2^E, quantized to q * 2^-24, q in [0, 1024].
     * q = round((m | 2^23) >> s) with s = -(E+1) in [14, 24]; a round-up from
     * 0x3FF to 0x400 yields the smallest normal (0x0400), which is correct. */
    uint32_t mant = m | 0x800000u;       /* 24-bit 1.m */
    int32_t  s    = -(E + 1);            /* shift in [14, 24] */
    uint32_t q    = mant >> s;
    uint32_t rem  = mant & ((1u << s) - 1u);
    uint32_t halfr = 1u << (s - 1);
    if (rem > halfr || (rem == halfr && (q & 1u))) q += 1u;
    return (uint16_t)(sign | q);
}

float sg_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;                    /* +/- 0 */
        } else {
            /* subnormal: normalize into a f32 normal */
            uint32_t fe = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; fe--; }
            mant &= 0x3FFu;
            f = sign | (fe << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);   /* Inf / NaN */
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

/* --------------------------------------------------------------------
 * Injected allocation backend
 * -------------------------------------------------------------------- */

static sg_kv_alloc_fn kv_backend_alloc = NULL;
static sg_kv_free_fn  kv_backend_free  = NULL;
static sg_kv_host_fn  kv_backend_host  = NULL;

void sg_kv_set_backend(sg_kv_alloc_fn alloc, sg_kv_free_fn free_, sg_kv_host_fn host) {
    kv_backend_alloc = alloc;
    kv_backend_free  = free_;
    kv_backend_host  = host;
}

/* --------------------------------------------------------------------
 * sg_kv object
 * -------------------------------------------------------------------- */

struct sg_kv {
    sg_cfg cfg;
    uint32_t cap;
    uint32_t used;
    sg_tensor_type dtype;   /* K/V dtype (F16 or F32) */

    /* Bound at creation so free/reset use the same backend even if the global
     * is later re-registered. */
    sg_kv_free_fn free_fn;
    sg_kv_host_fn host_fn;

    /* Per-layer handles; NULL when the layer is the other kind. */
    void **k;       /* [n_layers] full-attention K */
    void **v;       /* [n_layers] full-attention V */
    void **conv;    /* [n_layers] DeltaNet conv tail */
    void **s;       /* [n_layers] DeltaNet state S */

    uint64_t conv_bytes;   /* per DeltaNet layer */
    uint64_t s_bytes;      /* per DeltaNet layer */
};

void sg_kv_free(sg_kv *kv) {
    if (!kv) return;
    if (kv->free_fn) {
        for (uint32_t l = 0; l < kv->cfg.n_layers; l++) {
            if (kv->k)    kv->free_fn(kv->k[l]);
            if (kv->v)    kv->free_fn(kv->v[l]);
            if (kv->conv) kv->free_fn(kv->conv[l]);
            if (kv->s)    kv->free_fn(kv->s[l]);
        }
    }
    free(kv->k);
    free(kv->v);
    free(kv->conv);
    free(kv->s);
    free(kv);
}

sg_err sg_kv_new(sg_gpu *g, const sg_cfg *c, uint32_t cap,
                 sg_tensor_type kv_dtype, sg_kv **out) {
    if (!out) return (sg_err){"kv: sg_kv_new got a NULL out"};
    *out = NULL;
    if (!c) return (sg_err){"kv: sg_kv_new got a NULL cfg"};
    if (!kv_backend_alloc || !kv_backend_free || !kv_backend_host) {
        return (sg_err){"kv: no allocation backend registered (call sg_kv_set_backend)"};
    }
    if (c->n_layers == 0) return (sg_err){"kv: cfg has zero layers"};
    if (c->full_attn_interval == 0) return (sg_err){"kv: full_attn_interval is zero"};

    uint64_t esz = kv_dtype_size(kv_dtype);
    if (esz == 0) return (sg_err){"kv: kv_dtype must be SG_T_F16 or SG_T_F32"};

    if (cap == 0) cap = SG_KV_CAP_DEFAULT;
    if (cap > SG_KV_CAP_MAX) {
        return (sg_err){"kv: cap exceeds SG_KV_CAP_MAX (262144)"};
    }

    /* Count kinds and size each buffer up front, u64-guarded, before touching
     * the allocator. */
    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        if (kv_is_attn(c, l)) n_attn++; else n_gdn++;
    }

    uint64_t one_kv = 0;   /* bytes of one K or V buffer */
    if (n_attn > 0) {
        if (!kv_kv_buf_bytes(c, cap, esz, &one_kv) || one_kv == 0) {
            return (sg_err){"kv: attention KV size is zero, implausible or overflows"};
        }
    }
    uint64_t conv_bytes = 0, s_bytes = 0, conv_dim = 0;
    if (n_gdn > 0) {
        if (c->n_v_heads == 0 || c->head_v_dim == 0 || c->head_k_dim == 0
            || c->conv_kernel == 0) {
            return (sg_err){"kv: model has DeltaNet layers but no DeltaNet dims"};
        }
        if (!kv_conv_dim(c, &conv_dim)) {
            return (sg_err){"kv: conv_dim is zero or implausibly large"};
        }
        uint64_t conv_elems, s_elems = c->n_v_heads;
        if (!kv_mul((uint64_t)c->conv_kernel - 1, conv_dim, &conv_elems)
            || !kv_mul(s_elems, c->head_v_dim, &s_elems)
            || !kv_mul(s_elems, c->head_k_dim, &s_elems)
            || !kv_mul(conv_elems, 4, &conv_bytes)
            || !kv_mul(s_elems, 4, &s_bytes)) {
            return (sg_err){"kv: DeltaNet state size overflows"};
        }
        /* A DeltaNet layer with a degenerate (zero-row) conv tail is still
         * valid; guard only against a zero S, which would be a bad config. */
        if (s_bytes == 0) return (sg_err){"kv: DeltaNet state S size is zero"};
    }

    sg_kv *kv = calloc(1, sizeof *kv);
    if (!kv) return (sg_err){"kv: out of memory"};
    kv->cfg = *c;
    kv->cap = cap;
    kv->used = 0;
    kv->dtype = kv_dtype;
    kv->free_fn = kv_backend_free;
    kv->host_fn = kv_backend_host;
    kv->conv_bytes = conv_bytes;
    kv->s_bytes = s_bytes;

    kv->k    = calloc(c->n_layers, sizeof *kv->k);
    kv->v    = calloc(c->n_layers, sizeof *kv->v);
    kv->conv = calloc(c->n_layers, sizeof *kv->conv);
    kv->s    = calloc(c->n_layers, sizeof *kv->s);
    if (!kv->k || !kv->v || !kv->conv || !kv->s) {
        sg_kv_free(kv);
        return (sg_err){"kv: out of memory"};
    }

    for (uint32_t l = 0; l < c->n_layers; l++) {
        sg_err e;
        if (kv_is_attn(c, l)) {
            e = kv_backend_alloc(g, one_kv, &kv->k[l], NULL);
            if (sg_failed(e)) { sg_kv_free(kv); return e; }
            e = kv_backend_alloc(g, one_kv, &kv->v[l], NULL);
            if (sg_failed(e)) { sg_kv_free(kv); return e; }
        } else {
            if (conv_bytes > 0) {
                e = kv_backend_alloc(g, conv_bytes, &kv->conv[l], NULL);
                if (sg_failed(e)) { sg_kv_free(kv); return e; }
            }
            e = kv_backend_alloc(g, s_bytes, &kv->s[l], NULL);
            if (sg_failed(e)) { sg_kv_free(kv); return e; }
        }
    }

    /* Detailed logging: per-layer (per-kind, since uniform) and totals. Both
     * totals equal sg_kv_bytes / sg_kv_state_bytes by construction; the
     * per-kind byte counts above are already u64-guarded, so the sums here
     * cannot wrap for a config that got this far. */
    uint64_t kv_total    = (uint64_t)n_attn * one_kv * 2;
    uint64_t state_total = (uint64_t)n_gdn * (conv_bytes + s_bytes);
    fprintf(stderr,
            "kv: cap=%u dtype=%s layers=%u (full-attn=%u deltanet=%u)\n",
            cap, kv_dtype == SG_T_F16 ? "f16" : "f32",
            c->n_layers, n_attn, n_gdn);
    if (n_attn > 0) {
        fprintf(stderr,
                "kv: per full-attn layer: K=%llu V=%llu bytes "
                "(kv_width=%llu esz=%llu)\n",
                (unsigned long long)one_kv, (unsigned long long)one_kv,
                (unsigned long long)((uint64_t)c->n_kv_heads * c->head_dim),
                (unsigned long long)esz);
    }
    if (n_gdn > 0) {
        fprintf(stderr,
                "kv: per deltanet layer: conv=%llu S=%llu bytes "
                "(conv_dim=%llu conv_kernel=%u)\n",
                (unsigned long long)conv_bytes, (unsigned long long)s_bytes,
                (unsigned long long)conv_dim, c->conv_kernel);
    }
    fprintf(stderr,
            "kv: totals: K+V=%llu (%.2f GiB) deltanet=%llu grand=%llu bytes\n",
            (unsigned long long)kv_total,
            (double)kv_total / (1024.0 * 1024.0 * 1024.0),
            (unsigned long long)state_total,
            (unsigned long long)(kv_total + state_total));

    *out = kv;
    return SG_OK;
}

void sg_kv_reset(sg_kv *kv) {
    if (!kv) return;
    kv->used = 0;
    /* Zero conv + S (read unconditionally by the DeltaNet scan); leave K/V. */
    for (uint32_t l = 0; l < kv->cfg.n_layers; l++) {
        if (kv->conv && kv->conv[l] && kv->conv_bytes) {
            void *h = kv->host_fn(kv->conv[l]);
            if (h) memset(h, 0, (size_t)kv->conv_bytes);
        }
        if (kv->s && kv->s[l] && kv->s_bytes) {
            void *h = kv->host_fn(kv->s[l]);
            if (h) memset(h, 0, (size_t)kv->s_bytes);
        }
    }
}

sg_err sg_kv_advance(sg_kv *kv, uint32_t n) {
    if (!kv) return (sg_err){"kv: sg_kv_advance got a NULL kv"};
    /* used + n in u64 so the sum itself cannot wrap before the compare. */
    uint64_t next = (uint64_t)kv->used + n;
    if (next > kv->cap) {
        return (sg_err){"kv: advance would exceed cap"};
    }
    kv->used = (uint32_t)next;
    return SG_OK;
}

uint32_t sg_kv_used(const sg_kv *kv) { return kv ? kv->used : 0; }
uint32_t sg_kv_cap(const sg_kv *kv)  { return kv ? kv->cap  : 0; }

void *sg_kv_k(const sg_kv *kv, uint32_t layer) {
    if (!kv || layer >= kv->cfg.n_layers) return NULL;
    return kv->k[layer];
}
void *sg_kv_v(const sg_kv *kv, uint32_t layer) {
    if (!kv || layer >= kv->cfg.n_layers) return NULL;
    return kv->v[layer];
}
void *sg_kv_conv(const sg_kv *kv, uint32_t layer) {
    if (!kv || layer >= kv->cfg.n_layers) return NULL;
    return kv->conv[layer];
}
void *sg_kv_s(const sg_kv *kv, uint32_t layer) {
    if (!kv || layer >= kv->cfg.n_layers) return NULL;
    return kv->s[layer];
}
