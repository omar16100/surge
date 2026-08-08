/* gguf.c - GGUF v3 file reader.
 *
 * Maps a .gguf file read-only and parses the metadata table and tensor
 * directory into arrays owned by the sg_gguf handle. Tensor and array
 * payloads are not copied; they are pointers into the mapping, which stays
 * mapped until sg_gguf_close(). String scalars and names get NUL-terminated
 * copies in a small pool freed at close.
 *
 * Every read is bounds-checked against the mapped file size before use.
 * This reader does not print; callers decide how to surface sg_err.
 */
#include "surge.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" read as a little-endian u32 */

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
} rd_t;

typedef union {
    uint8_t u8; int8_t i8;
    uint16_t u16; int16_t i16;
    uint32_t u32; int32_t i32;
    uint64_t u64; int64_t i64;
    float f32; double f64;
    bool b;
    const char *str; /* arena copy, NUL-terminated */
    struct { sg_gguf_kv_type elem_type; uint64_t count; const void *data; } arr;
} sg_kv_value;

typedef struct {
    char *key; /* arena copy, NUL-terminated */
    sg_gguf_kv_type type;
    sg_kv_value v;
} sg_kv;

struct sg_gguf {
    void *map;
    size_t map_size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    sg_kv *kvs;
    sg_tensor *tensors;
    char **arena;
    size_t arena_len;
    size_t arena_cap;
};

static uint64_t align_up(uint64_t v, uint64_t a) {
    if (a == 0) return v;
    uint64_t rem = v % a;
    if (rem == 0) return v;
    uint64_t pad = a - rem;
    if (v > UINT64_MAX - pad) return UINT64_MAX; /* forces the caller's size check to fail */
    return v + pad;
}

/* Fixed-width bounds-checked reads. Host is little-endian Apple Silicon,
 * matching the GGUF on-disk encoding, so a raw memcpy is correct here. */
#define RD_DEF(name, ty) \
    static sg_err name(rd_t *r, ty *out, const char *what) { \
        if (r->pos > r->size || sizeof(ty) > r->size - r->pos) return (sg_err){what}; \
        memcpy(out, r->base + r->pos, sizeof(ty)); \
        r->pos += sizeof(ty); \
        return SG_OK; \
    }
RD_DEF(rd_u8, uint8_t)
RD_DEF(rd_i8, int8_t)
RD_DEF(rd_u16, uint16_t)
RD_DEF(rd_i16, int16_t)
RD_DEF(rd_u32, uint32_t)
RD_DEF(rd_i32, int32_t)
RD_DEF(rd_u64, uint64_t)
RD_DEF(rd_i64, int64_t)
RD_DEF(rd_f32, float)
RD_DEF(rd_f64, double)
#undef RD_DEF

static sg_err rd_str_raw(rd_t *r, const char **out, uint64_t *len_out,
                          const char *what_len, const char *what_bytes) {
    uint64_t n;
    sg_err e = rd_u64(r, &n, what_len);
    if (sg_failed(e)) return e;
    if (n > r->size - r->pos) return (sg_err){what_bytes};
    *out = (const char *)(r->base + r->pos);
    *len_out = n;
    r->pos += n;
    return SG_OK;
}

static sg_err arena_add(sg_gguf *g, const char *bytes, uint64_t len, char **out) {
    char *s = malloc((size_t)len + 1);
    if (!s) return (sg_err){"gguf: out of memory"};
    if (len) memcpy(s, bytes, (size_t)len);
    s[len] = '\0';

    if (g->arena_len == g->arena_cap) {
        size_t new_cap = g->arena_cap ? g->arena_cap * 2 : 8;
        char **na = realloc(g->arena, new_cap * sizeof(*na));
        if (!na) { free(s); return (sg_err){"gguf: out of memory"}; }
        g->arena = na;
        g->arena_cap = new_cap;
    }
    g->arena[g->arena_len++] = s;
    *out = s;
    return SG_OK;
}

static sg_err elem_fixed_size(sg_gguf_kv_type t, uint64_t *out) {
    switch (t) {
        case SG_GGUF_U8: case SG_GGUF_I8: case SG_GGUF_BOOL:
            *out = 1; return SG_OK;
        case SG_GGUF_U16: case SG_GGUF_I16:
            *out = 2; return SG_OK;
        case SG_GGUF_U32: case SG_GGUF_I32: case SG_GGUF_F32:
            *out = 4; return SG_OK;
        case SG_GGUF_U64: case SG_GGUF_I64: case SG_GGUF_F64:
            *out = 8; return SG_OK;
        case SG_GGUF_STR: case SG_GGUF_ARR:
        default:
            return (sg_err){"gguf: invalid array element type"};
    }
}

static sg_err tensor_elem_count(uint32_t n_dims, const uint64_t *dims, uint64_t *out) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < n_dims; i++) {
        uint64_t d = dims[i];
        if (d != 0 && n > UINT64_MAX / d) return (sg_err){"gguf: tensor dims overflow"};
        n *= d;
    }
    *out = n;
    return SG_OK;
}

static sg_err tensor_nbytes(sg_tensor_type type, uint32_t n_dims, const uint64_t *dims,
                             uint64_t *out) {
    uint64_t n;
    sg_err e = tensor_elem_count(n_dims, dims, &n);
    if (sg_failed(e)) return e;
    switch (type) {
        case SG_T_F32:
            if (n > UINT64_MAX / 4) return (sg_err){"gguf: tensor size overflow"};
            *out = n * 4;
            return SG_OK;
        case SG_T_F16:
        case SG_T_BF16:
            if (n > UINT64_MAX / 2) return (sg_err){"gguf: tensor size overflow"};
            *out = n * 2;
            return SG_OK;
        case SG_T_Q8_0:
            if (n % 32 != 0) {
                return (sg_err){"gguf: q8_0 tensor element count not a multiple of 32"};
            }
            *out = (n / 32) * 34;
            return SG_OK;
        default:
            return (sg_err){"gguf: unsupported tensor type"};
    }
}

static sg_err parse_kv_array(rd_t *r, sg_kv_value *v) {
    uint32_t elem_type_raw;
    sg_err e = rd_u32(r, &elem_type_raw, "gguf: truncated at kv array elem_type");
    if (sg_failed(e)) return e;
    if (elem_type_raw > SG_GGUF_F64 || elem_type_raw == SG_GGUF_ARR) {
        return (sg_err){"gguf: invalid array element type"};
    }
    sg_gguf_kv_type elem_type = (sg_gguf_kv_type)elem_type_raw;

    uint64_t count;
    e = rd_u64(r, &count, "gguf: truncated at kv array count");
    if (sg_failed(e)) return e;

    if (elem_type == SG_GGUF_STR) {
        uint64_t start = r->pos;
        for (uint64_t i = 0; i < count; i++) {
            const char *p;
            uint64_t len;
            e = rd_str_raw(r, &p, &len, "gguf: truncated at kv array string length",
                           "gguf: truncated at kv array string bytes");
            if (sg_failed(e)) return e;
        }
        v->arr.elem_type = elem_type;
        v->arr.count = count;
        v->arr.data = r->base + start;
        return SG_OK;
    }

    uint64_t esz;
    e = elem_fixed_size(elem_type, &esz);
    if (sg_failed(e)) return e;
    if (count != 0 && esz > UINT64_MAX / count) return (sg_err){"gguf: array size overflow"};
    uint64_t total = esz * count;
    if (total > r->size - r->pos) return (sg_err){"gguf: truncated at kv array data"};

    v->arr.elem_type = elem_type;
    v->arr.count = count;
    v->arr.data = r->base + r->pos;
    r->pos += total;
    return SG_OK;
}

static sg_err parse_kv_value(sg_gguf *g, rd_t *r, sg_gguf_kv_type type, sg_kv_value *v) {
    switch (type) {
        case SG_GGUF_U8: {
            uint8_t x; sg_err e = rd_u8(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->u8 = x; return SG_OK;
        }
        case SG_GGUF_I8: {
            int8_t x; sg_err e = rd_i8(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->i8 = x; return SG_OK;
        }
        case SG_GGUF_U16: {
            uint16_t x; sg_err e = rd_u16(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->u16 = x; return SG_OK;
        }
        case SG_GGUF_I16: {
            int16_t x; sg_err e = rd_i16(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->i16 = x; return SG_OK;
        }
        case SG_GGUF_U32: {
            uint32_t x; sg_err e = rd_u32(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->u32 = x; return SG_OK;
        }
        case SG_GGUF_I32: {
            int32_t x; sg_err e = rd_i32(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->i32 = x; return SG_OK;
        }
        case SG_GGUF_F32: {
            float x; sg_err e = rd_f32(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->f32 = x; return SG_OK;
        }
        case SG_GGUF_U64: {
            uint64_t x; sg_err e = rd_u64(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->u64 = x; return SG_OK;
        }
        case SG_GGUF_I64: {
            int64_t x; sg_err e = rd_i64(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->i64 = x; return SG_OK;
        }
        case SG_GGUF_F64: {
            double x; sg_err e = rd_f64(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->f64 = x; return SG_OK;
        }
        case SG_GGUF_BOOL: {
            uint8_t x; sg_err e = rd_u8(r, &x, "gguf: truncated at kv value");
            if (sg_failed(e)) return e; v->b = (x != 0); return SG_OK;
        }
        case SG_GGUF_STR: {
            const char *p; uint64_t len;
            sg_err e = rd_str_raw(r, &p, &len, "gguf: truncated at kv value length",
                                  "gguf: truncated at kv value bytes");
            if (sg_failed(e)) return e;
            char *copy;
            e = arena_add(g, p, len, &copy);
            if (sg_failed(e)) return e;
            v->str = copy;
            return SG_OK;
        }
        case SG_GGUF_ARR:
            return parse_kv_array(r, v);
    }
    return (sg_err){"gguf: invalid kv type"};
}

static sg_err parse_kv(sg_gguf *g, rd_t *r, sg_kv *kv) {
    const char *name_ptr;
    uint64_t name_len;
    sg_err e = rd_str_raw(r, &name_ptr, &name_len, "gguf: truncated at kv key length",
                          "gguf: truncated at kv key bytes");
    if (sg_failed(e)) return e;
    char *key_copy;
    e = arena_add(g, name_ptr, name_len, &key_copy);
    if (sg_failed(e)) return e;
    kv->key = key_copy;

    uint32_t type_raw;
    e = rd_u32(r, &type_raw, "gguf: truncated at kv type");
    if (sg_failed(e)) return e;
    if (type_raw > SG_GGUF_F64) return (sg_err){"gguf: invalid kv type"};
    kv->type = (sg_gguf_kv_type)type_raw;

    return parse_kv_value(g, r, kv->type, &kv->v);
}

static sg_err parse_tensor_info(sg_gguf *g, rd_t *r, sg_tensor *t, uint64_t *offset_out) {
    const char *name_ptr;
    uint64_t name_len;
    sg_err e = rd_str_raw(r, &name_ptr, &name_len, "gguf: truncated at tensor name length",
                          "gguf: truncated at tensor name bytes");
    if (sg_failed(e)) return e;
    char *name_copy;
    e = arena_add(g, name_ptr, name_len, &name_copy);
    if (sg_failed(e)) return e;
    t->name = name_copy;

    e = rd_u32(r, &t->n_dims, "gguf: truncated at tensor n_dims");
    if (sg_failed(e)) return e;
    if (t->n_dims == 0 || t->n_dims > 4) {
        return (sg_err){"gguf: tensor n_dims must be between 1 and 4"};
    }
    for (uint32_t d = 0; d < t->n_dims; d++) {
        e = rd_u64(r, &t->dims[d], "gguf: truncated at tensor dims");
        if (sg_failed(e)) return e;
    }

    uint32_t type_raw;
    e = rd_u32(r, &type_raw, "gguf: truncated at tensor type");
    if (sg_failed(e)) return e;
    t->type = (sg_tensor_type)type_raw;

    e = rd_u64(r, offset_out, "gguf: truncated at tensor offset");
    if (sg_failed(e)) return e;

    return SG_OK;
}

static const sg_kv *find_kv(const sg_gguf *g, const char *key) {
    if (!g || !key) return NULL;
    for (uint64_t i = 0; i < g->kv_count; i++) {
        if (strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    }
    return NULL;
}

sg_err sg_gguf_open(const char *path, sg_gguf **out) {
    if (out) *out = NULL;
    if (!path || !out) return (sg_err){"gguf: invalid arguments"};

    int fd = open(path, O_RDONLY);
    if (fd < 0) return (sg_err){"gguf: failed to open file"};

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return (sg_err){"gguf: fstat failed"};
    }
    if (st.st_size <= 0) {
        close(fd);
        return (sg_err){"gguf: empty file"};
    }

    size_t size = (size_t)st.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* the mapping outlives the fd */
    if (map == MAP_FAILED) return (sg_err){"gguf: mmap failed"};

    sg_gguf *g = calloc(1, sizeof(*g));
    if (!g) {
        munmap(map, size);
        return (sg_err){"gguf: out of memory"};
    }
    g->map = map;
    g->map_size = size;

    uint64_t *offs = NULL;
    sg_err e;
    rd_t r = { .base = (const uint8_t *)map, .size = size, .pos = 0 };

    uint32_t magic;
    e = rd_u32(&r, &magic, "gguf: truncated at magic");
    if (sg_failed(e)) goto fail;
    if (magic != GGUF_MAGIC) { e = (sg_err){"gguf: bad magic"}; goto fail; }

    e = rd_u32(&r, &g->version, "gguf: truncated at version");
    if (sg_failed(e)) goto fail;
    if (g->version != 3) { e = (sg_err){"gguf: unsupported version"}; goto fail; }

    e = rd_u64(&r, &g->tensor_count, "gguf: truncated at tensor_count");
    if (sg_failed(e)) goto fail;

    e = rd_u64(&r, &g->kv_count, "gguf: truncated at kv_count");
    if (sg_failed(e)) goto fail;

    /* A corrupt count could otherwise request a multi-terabyte allocation;
     * every kv/tensor entry needs at least a few bytes on disk. */
    if (g->kv_count > g->map_size || g->tensor_count > g->map_size) {
        e = (sg_err){"gguf: implausible tensor or kv count"};
        goto fail;
    }

    if (g->kv_count > 0) {
        g->kvs = calloc(g->kv_count, sizeof(*g->kvs));
        if (!g->kvs) { e = (sg_err){"gguf: out of memory"}; goto fail; }
    }
    for (uint64_t i = 0; i < g->kv_count; i++) {
        e = parse_kv(g, &r, &g->kvs[i]);
        if (sg_failed(e)) goto fail;
    }

    if (g->tensor_count > 0) {
        g->tensors = calloc(g->tensor_count, sizeof(*g->tensors));
        if (!g->tensors) { e = (sg_err){"gguf: out of memory"}; goto fail; }
        offs = calloc(g->tensor_count, sizeof(*offs));
        if (!offs) { e = (sg_err){"gguf: out of memory"}; goto fail; }
    }
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        e = parse_tensor_info(g, &r, &g->tensors[i], &offs[i]);
        if (sg_failed(e)) goto fail;
    }

    {
        uint32_t alignment = 32;
        uint32_t align_kv;
        if (sg_gguf_get_u32(g, "general.alignment", &align_kv) && align_kv != 0) {
            alignment = align_kv;
        }

        uint64_t data_start = align_up(r.pos, alignment);
        if (data_start > g->map_size) {
            e = (sg_err){"gguf: truncated at data section"};
            goto fail;
        }

        for (uint64_t i = 0; i < g->tensor_count; i++) {
            sg_tensor *t = &g->tensors[i];
            uint64_t nbytes;
            e = tensor_nbytes(t->type, t->n_dims, t->dims, &nbytes);
            if (sg_failed(e)) goto fail;

            uint64_t off = offs[i];
            if (off > g->map_size - data_start) {
                e = (sg_err){"gguf: truncated at tensor data"};
                goto fail;
            }
            uint64_t abs_off = data_start + off;
            if (nbytes > g->map_size - abs_off) {
                e = (sg_err){"gguf: truncated at tensor data"};
                goto fail;
            }
            t->data = (const uint8_t *)g->map + abs_off;
            t->nbytes = nbytes;
        }
    }

    free(offs);
    *out = g;
    return SG_OK;

fail:
    free(offs);
    sg_gguf_close(g);
    return e;
}

void sg_gguf_close(sg_gguf *g) {
    if (!g) return;
    if (g->map) munmap(g->map, g->map_size);
    for (size_t i = 0; i < g->arena_len; i++) free(g->arena[i]);
    free(g->arena);
    free(g->kvs);
    free(g->tensors);
    free(g);
}

bool sg_gguf_get_u32(const sg_gguf *g, const char *key, uint32_t *out) {
    const sg_kv *kv = find_kv(g, key);
    if (!kv || kv->type != SG_GGUF_U32 || !out) return false;
    *out = kv->v.u32;
    return true;
}

bool sg_gguf_get_f32(const sg_gguf *g, const char *key, float *out) {
    const sg_kv *kv = find_kv(g, key);
    if (!kv || kv->type != SG_GGUF_F32 || !out) return false;
    *out = kv->v.f32;
    return true;
}

bool sg_gguf_get_str(const sg_gguf *g, const char *key, const char **out) {
    const sg_kv *kv = find_kv(g, key);
    if (!kv || kv->type != SG_GGUF_STR || !out) return false;
    *out = kv->v.str;
    return true;
}

bool sg_gguf_get_arr(const sg_gguf *g, const char *key, sg_gguf_kv_type *elem_type,
                     const void **data, uint64_t *count) {
    const sg_kv *kv = find_kv(g, key);
    if (!kv || kv->type != SG_GGUF_ARR) return false;
    if (elem_type) *elem_type = kv->v.arr.elem_type;
    if (data) *data = kv->v.arr.data;
    if (count) *count = kv->v.arr.count;
    return true;
}

const sg_tensor *sg_gguf_tensor(const sg_gguf *g, const char *name) {
    if (!g || !name) return NULL;
    for (uint64_t i = 0; i < g->tensor_count; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

uint32_t sg_gguf_version(const sg_gguf *g) {
    return g ? g->version : 0;
}

uint64_t sg_gguf_kv_count(const sg_gguf *g) {
    return g ? g->kv_count : 0;
}

bool sg_gguf_kv_at(const sg_gguf *g, uint64_t i, const char **key_out, sg_gguf_kv_type *type_out) {
    if (!g || i >= g->kv_count) return false;
    if (key_out) *key_out = g->kvs[i].key;
    if (type_out) *type_out = g->kvs[i].type;
    return true;
}

uint64_t sg_gguf_tensor_count(const sg_gguf *g) {
    return g ? g->tensor_count : 0;
}

const sg_tensor *sg_gguf_tensor_at(const sg_gguf *g, uint64_t i) {
    if (!g || i >= g->tensor_count) return NULL;
    return &g->tensors[i];
}
