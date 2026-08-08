/* st.c - read-only safetensors bf16/f32 loader + config.json.
 *
 * Maps every *.safetensors shard for a model directory read-only and parses
 * each shard's JSON header (u64 header_len, then header_len bytes of JSON
 * mapping tensor name -> {dtype, shape, data_offsets:[begin,end]}, offsets
 * relative to the byte right after the header) into a flat tensor
 * directory. BF16 tensors are retrievable via sg_st_tensor() and F32 ones
 * via sg_st_tensor_f32(); other dtypes are bounds-checked but not indexed.
 * F32 indexing was added in Task 7: the real Qwen3.5 checkpoint stores
 * linear_attn.A_log and linear_attn.norm.weight as F32 amid otherwise-BF16
 * layer weights (mlx's cast_predicate deliberately exempts A_log), so a
 * bf16-only index cannot reach a complete gated-DeltaNet layer.
 * Shard filenames come from model.safetensors.index.json's weight_map when
 * present (never a hardcoded "model-NNNNN-of-NNNNN.safetensors" pattern,
 * since real checkpoints don't always follow it), otherwise model_dir must
 * contain exactly one *.safetensors file.
 *
 * config.json is read fully into memory and parsed with the same JSON
 * scanner; sg_st_config_* look a key up at the top level first, then inside
 * a "text_config" object if absent (some HF configs nest the text model's
 * hyperparameters there).
 *
 * JSON scanner: a minimal, read-only, non-streaming recursive-descent
 * parser for objects/arrays/strings/numbers/true/false/null. String escapes
 * are limited to \" \\ \n \t; anything else is a hard error, not a
 * misparse. Nesting (objects+arrays combined) is capped at 64 levels.
 * Numbers are copied into a small bounded stack buffer and converted with
 * strtod (always) and strtoll (when the literal has no '.' or exponent, so
 * shape/data_offsets integers survive exactly rather than round-tripping
 * through a double).
 *
 * Every read is bounds-checked against the mapped/read buffer before use.
 * This reader does not print; callers decide how to surface sg_err.
 */
#include "surge.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ST_PATH_MAX 4096
#define ST_JSON_MAX_DEPTH 64

/* ---------------------------------------------------------------------
 * Minimal read-only JSON scanner: builds a small in-memory value tree.
 * --------------------------------------------------------------------- */

typedef enum { JV_NULL, JV_BOOL, JV_NUM, JV_STR, JV_ARR, JV_OBJ } jv_kind;
typedef struct jv jv;

typedef struct {
    char *key;          /* owned, NUL-terminated */
    uint32_t key_len;
    jv *val;             /* owned */
} jv_member;

struct jv {
    jv_kind kind;
    union {
        bool b;
        struct {
            double d;
            int64_t i;
            bool has_i; /* true if the literal was a plain integer (no '.'/'e'/'E') */
        } num;
        struct {
            char *ptr;   /* owned, NUL-terminated, escapes already decoded */
            uint32_t len;
        } str;
        struct {
            jv **items;  /* owned array of owned elements */
            uint32_t count;
            uint32_t cap;
        } arr;
        struct {
            jv_member *members; /* owned array */
            uint32_t count;
            uint32_t cap;
        } obj;
    } as;
};

typedef struct {
    const char *base;
    uint64_t size;
    uint64_t pos;
    int depth;
} jp_t;

static void jv_free(jv *v);

static void skip_ws(jp_t *p) {
    while (p->pos < p->size) {
        char c = p->base[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
}

/* Decodes a JSON string starting at p->pos (which must point at the opening
 * quote) into an owned, NUL-terminated buffer. Escapes beyond \" \\ \n \t
 * are a hard error. */
static sg_err jp_parse_raw_string(jp_t *p, char **out_str, uint32_t *out_len) {
    *out_str = NULL;
    *out_len = 0;
    p->pos++; /* consume opening '"' */

    uint32_t cap = 16;
    char *buf = malloc(cap);
    if (!buf) return (sg_err){"st: out of memory"};
    uint32_t len = 0;

    for (;;) {
        if (p->pos >= p->size) { free(buf); return (sg_err){"st: unterminated json string"}; }
        unsigned char c = (unsigned char)p->base[p->pos];
        char decoded;
        if (c == '"') { p->pos++; break; }
        if (c == '\\') {
            p->pos++;
            if (p->pos >= p->size) { free(buf); return (sg_err){"st: unterminated json string escape"}; }
            switch (p->base[p->pos]) {
                case '"': decoded = '"'; break;
                case '\\': decoded = '\\'; break;
                case 'n': decoded = '\n'; break;
                case 't': decoded = '\t'; break;
                default: free(buf); return (sg_err){"st: unsupported json escape sequence"};
            }
            p->pos++;
        } else {
            decoded = (char)c;
            p->pos++;
        }
        if (len >= UINT32_MAX - 1) { free(buf); return (sg_err){"st: json string too long"}; }
        if (len == cap) {
            if (cap > UINT32_MAX / 2) { free(buf); return (sg_err){"st: json string too long"}; }
            uint32_t ncap = cap * 2;
            char *nb = realloc(buf, ncap);
            if (!nb) { free(buf); return (sg_err){"st: out of memory"}; }
            buf = nb;
            cap = ncap;
        }
        buf[len++] = decoded;
    }

    char *final = realloc(buf, (size_t)len + 1);
    if (!final) { free(buf); return (sg_err){"st: out of memory"}; }
    final[len] = '\0';
    *out_str = final;
    *out_len = len;
    return SG_OK;
}

static sg_err jp_parse_string(jp_t *p, jv **out) {
    *out = NULL;
    char *buf = NULL;
    uint32_t len = 0;
    sg_err e = jp_parse_raw_string(p, &buf, &len);
    if (sg_failed(e)) return e;
    jv *v = calloc(1, sizeof(*v));
    if (!v) { free(buf); return (sg_err){"st: out of memory"}; }
    v->kind = JV_STR;
    v->as.str.ptr = buf;
    v->as.str.len = len;
    *out = v;
    return SG_OK;
}

static sg_err jp_parse_literal(jp_t *p, const char *lit, jv_kind kind, bool bval, jv **out) {
    *out = NULL;
    size_t len = strlen(lit);
    if (p->size - p->pos < len || memcmp(p->base + p->pos, lit, len) != 0) {
        return (sg_err){"st: invalid json literal"};
    }
    jv *v = calloc(1, sizeof(*v));
    if (!v) return (sg_err){"st: out of memory"};
    v->kind = kind;
    if (kind == JV_BOOL) v->as.b = bval;
    p->pos += len;
    *out = v;
    return SG_OK;
}

static sg_err jp_parse_number(jp_t *p, jv **out) {
    *out = NULL;
    uint64_t start = p->pos;
    if (p->base[p->pos] == '-') p->pos++;
    if (p->pos >= p->size || !isdigit((unsigned char)p->base[p->pos])) {
        return (sg_err){"st: invalid json number"};
    }
    while (p->pos < p->size && isdigit((unsigned char)p->base[p->pos])) p->pos++;
    bool is_float_form = false;
    if (p->pos < p->size && p->base[p->pos] == '.') {
        is_float_form = true;
        p->pos++;
        if (p->pos >= p->size || !isdigit((unsigned char)p->base[p->pos])) {
            return (sg_err){"st: invalid json number"};
        }
        while (p->pos < p->size && isdigit((unsigned char)p->base[p->pos])) p->pos++;
    }
    if (p->pos < p->size && (p->base[p->pos] == 'e' || p->base[p->pos] == 'E')) {
        is_float_form = true;
        p->pos++;
        if (p->pos < p->size && (p->base[p->pos] == '+' || p->base[p->pos] == '-')) p->pos++;
        if (p->pos >= p->size || !isdigit((unsigned char)p->base[p->pos])) {
            return (sg_err){"st: invalid json number"};
        }
        while (p->pos < p->size && isdigit((unsigned char)p->base[p->pos])) p->pos++;
    }

    uint64_t span = p->pos - start;
    char buf[64];
    if (span >= sizeof(buf)) return (sg_err){"st: json number literal too long"};
    memcpy(buf, p->base + start, span);
    buf[span] = '\0';

    jv *v = calloc(1, sizeof(*v));
    if (!v) return (sg_err){"st: out of memory"};
    v->kind = JV_NUM;

    char *endp = NULL;
    v->as.num.d = strtod(buf, &endp);
    if (endp != buf + span) { free(v); return (sg_err){"st: invalid json number"}; }

    if (!is_float_form) {
        errno = 0;
        char *endi = NULL;
        long long iv = strtoll(buf, &endi, 10);
        if (endi == buf + span && errno == 0) {
            v->as.num.has_i = true;
            v->as.num.i = (int64_t)iv;
        }
    }

    *out = v;
    return SG_OK;
}

static sg_err jp_parse_value(jp_t *p, jv **out);

static sg_err jp_parse_object(jp_t *p, jv **out) {
    *out = NULL;
    p->depth++;
    if (p->depth > ST_JSON_MAX_DEPTH) { p->depth--; return (sg_err){"st: json nesting too deep"}; }
    p->pos++; /* consume '{' */

    jv *obj = calloc(1, sizeof(*obj));
    if (!obj) { p->depth--; return (sg_err){"st: out of memory"}; }
    obj->kind = JV_OBJ;

    skip_ws(p);
    if (p->pos < p->size && p->base[p->pos] == '}') {
        p->pos++;
        *out = obj;
        p->depth--;
        return SG_OK;
    }

    for (;;) {
        skip_ws(p);
        if (p->pos >= p->size || p->base[p->pos] != '"') {
            jv_free(obj); p->depth--; return (sg_err){"st: expected string key in json object"};
        }
        char *key = NULL;
        uint32_t key_len = 0;
        sg_err e = jp_parse_raw_string(p, &key, &key_len);
        if (sg_failed(e)) { jv_free(obj); p->depth--; return e; }

        skip_ws(p);
        if (p->pos >= p->size || p->base[p->pos] != ':') {
            free(key); jv_free(obj); p->depth--; return (sg_err){"st: expected : in json object"};
        }
        p->pos++;
        skip_ws(p);

        jv *val = NULL;
        e = jp_parse_value(p, &val);
        if (sg_failed(e)) { free(key); jv_free(obj); p->depth--; return e; }

        if (obj->as.obj.count == obj->as.obj.cap) {
            if (obj->as.obj.cap > UINT32_MAX / 2) {
                free(key); jv_free(val); jv_free(obj); p->depth--;
                return (sg_err){"st: too many json object members"};
            }
            uint32_t ncap = obj->as.obj.cap ? obj->as.obj.cap * 2 : 4;
            jv_member *nm = realloc(obj->as.obj.members, (size_t)ncap * sizeof(*nm));
            if (!nm) {
                free(key); jv_free(val); jv_free(obj); p->depth--;
                return (sg_err){"st: out of memory"};
            }
            obj->as.obj.members = nm;
            obj->as.obj.cap = ncap;
        }
        jv_member *m = &obj->as.obj.members[obj->as.obj.count++];
        m->key = key;
        m->key_len = key_len;
        m->val = val;

        skip_ws(p);
        if (p->pos >= p->size) { jv_free(obj); p->depth--; return (sg_err){"st: unterminated json object"}; }
        char c = p->base[p->pos];
        if (c == ',') { p->pos++; continue; }
        if (c == '}') { p->pos++; break; }
        jv_free(obj); p->depth--; return (sg_err){"st: expected , or } in json object"};
    }

    *out = obj;
    p->depth--;
    return SG_OK;
}

static sg_err jp_parse_array(jp_t *p, jv **out) {
    *out = NULL;
    p->depth++;
    if (p->depth > ST_JSON_MAX_DEPTH) { p->depth--; return (sg_err){"st: json nesting too deep"}; }
    p->pos++; /* consume '[' */

    jv *arr = calloc(1, sizeof(*arr));
    if (!arr) { p->depth--; return (sg_err){"st: out of memory"}; }
    arr->kind = JV_ARR;

    skip_ws(p);
    if (p->pos < p->size && p->base[p->pos] == ']') {
        p->pos++;
        *out = arr;
        p->depth--;
        return SG_OK;
    }

    for (;;) {
        jv *elem = NULL;
        sg_err e = jp_parse_value(p, &elem);
        if (sg_failed(e)) { jv_free(arr); p->depth--; return e; }

        if (arr->as.arr.count == arr->as.arr.cap) {
            if (arr->as.arr.cap > UINT32_MAX / 2) {
                jv_free(elem); jv_free(arr); p->depth--;
                return (sg_err){"st: too many json array elements"};
            }
            uint32_t ncap = arr->as.arr.cap ? arr->as.arr.cap * 2 : 4;
            jv **ni = realloc(arr->as.arr.items, (size_t)ncap * sizeof(*ni));
            if (!ni) { jv_free(elem); jv_free(arr); p->depth--; return (sg_err){"st: out of memory"}; }
            arr->as.arr.items = ni;
            arr->as.arr.cap = ncap;
        }
        arr->as.arr.items[arr->as.arr.count++] = elem;

        skip_ws(p);
        if (p->pos >= p->size) { jv_free(arr); p->depth--; return (sg_err){"st: unterminated json array"}; }
        char c = p->base[p->pos];
        if (c == ',') { p->pos++; skip_ws(p); continue; }
        if (c == ']') { p->pos++; break; }
        jv_free(arr); p->depth--; return (sg_err){"st: expected , or ] in json array"};
    }

    *out = arr;
    p->depth--;
    return SG_OK;
}

static sg_err jp_parse_value(jp_t *p, jv **out) {
    *out = NULL;
    skip_ws(p);
    if (p->pos >= p->size) return (sg_err){"st: unexpected end of json"};
    char c = p->base[p->pos];
    if (c == '{') return jp_parse_object(p, out);
    if (c == '[') return jp_parse_array(p, out);
    if (c == '"') return jp_parse_string(p, out);
    if (c == 't') return jp_parse_literal(p, "true", JV_BOOL, true, out);
    if (c == 'f') return jp_parse_literal(p, "false", JV_BOOL, false, out);
    if (c == 'n') return jp_parse_literal(p, "null", JV_NULL, false, out);
    if (c == '-' || (c >= '0' && c <= '9')) return jp_parse_number(p, out);
    return (sg_err){"st: unexpected character in json"};
}

static void jv_free(jv *v) {
    if (!v) return;
    switch (v->kind) {
        case JV_STR:
            free(v->as.str.ptr);
            break;
        case JV_ARR:
            for (uint32_t i = 0; i < v->as.arr.count; i++) jv_free(v->as.arr.items[i]);
            free(v->as.arr.items);
            break;
        case JV_OBJ:
            for (uint32_t i = 0; i < v->as.obj.count; i++) {
                free(v->as.obj.members[i].key);
                jv_free(v->as.obj.members[i].val);
            }
            free(v->as.obj.members);
            break;
        default:
            break;
    }
    free(v);
}

/* Parses a whole buffer as a single JSON document; the top-level value must
 * be an object (true of both config.json and a safetensors header). */
static sg_err jp_parse_document(const char *base, uint64_t size, jv **out) {
    jp_t p = { .base = base, .size = size, .pos = 0, .depth = 0 };
    sg_err e = jp_parse_value(&p, out);
    if (sg_failed(e)) return e;
    if ((*out)->kind != JV_OBJ) {
        jv_free(*out);
        *out = NULL;
        return (sg_err){"st: expected a json object at the top level"};
    }
    /* The whole buffer must be consumed: valid JSON followed by trailing
     * garbage (still within header_len, or after config.json's top-level
     * object) must be rejected, not silently accepted as if the garbage
     * weren't there. */
    skip_ws(&p);
    if (p.pos != p.size) {
        jv_free(*out);
        *out = NULL;
        return (sg_err){"st: trailing content after json value"};
    }
    return SG_OK;
}

static jv *jv_obj_get(const jv *obj, const char *key) {
    if (!obj || obj->kind != JV_OBJ) return NULL;
    uint32_t klen = (uint32_t)strlen(key);
    for (uint32_t i = 0; i < obj->as.obj.count; i++) {
        jv_member *m = &obj->as.obj.members[i];
        if (m->key_len == klen && memcmp(m->key, key, klen) == 0) return m->val;
    }
    return NULL;
}

/* Only accepts values that parsed as a plain (non-float-form) non-negative
 * integer literal, which is what safetensors shape/data_offsets entries
 * always are; a float there means a corrupt/foreign header. */
static bool jv_num_u64(const jv *v, uint64_t *out) {
    if (!v || v->kind != JV_NUM || !v->as.num.has_i || v->as.num.i < 0) return false;
    *out = (uint64_t)v->as.num.i;
    return true;
}

/* ---------------------------------------------------------------------
 * Small filesystem/path helpers.
 * --------------------------------------------------------------------- */

static sg_err join_path(const char *dir, const char *name, char out[ST_PATH_MAX]) {
    int n = snprintf(out, ST_PATH_MAX, "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= ST_PATH_MAX) return (sg_err){"st: path too long"};
    return SG_OK;
}

static char *st_owned_copy(const char *s, size_t len) {
    char *c = malloc(len + 1);
    if (!c) return NULL;
    memcpy(c, s, len);
    c[len] = '\0';
    return c;
}

static void free_str_list(char **names, uint32_t n) {
    if (!names) return;
    for (uint32_t i = 0; i < n; i++) free(names[i]);
    free(names);
}

static sg_err read_file_fully(const char *path, char **out_buf, uint64_t *out_size) {
    *out_buf = NULL;
    *out_size = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return (sg_err){"st: failed to open file"};
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return (sg_err){"st: fseek failed"}; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return (sg_err){"st: ftell failed"}; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return (sg_err){"st: fseek failed"}; }

    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return (sg_err){"st: out of memory"}; }
    size_t n = sz > 0 ? fread(buf, 1, (size_t)sz, f) : 0;
    fclose(f);
    if (n != (size_t)sz) { free(buf); return (sg_err){"st: short read"}; }
    buf[sz] = '\0';

    *out_buf = buf;
    *out_size = (uint64_t)sz;
    return SG_OK;
}

/* ---------------------------------------------------------------------
 * sg_st: shard mmaps + flat tensor directory + parsed config.json.
 * --------------------------------------------------------------------- */

typedef struct {
    void *map;
    uint64_t map_size;
} st_shard;

typedef struct {
    char *name;          /* owned, NUL-terminated; the checkpoint's own name */
    uint32_t n_dims;
    uint64_t dims[4];
    const void *data;    /* pointer into the owning shard's mmap */
    uint64_t nbytes;
    bool is_f32;         /* false => BF16 (sg_st_tensor), true => F32 */
} st_tensor;

struct sg_st {
    st_shard *shards;
    uint32_t n_shards;

    st_tensor *tensors;
    uint64_t n_tensors;
    uint64_t tensors_cap;

    jv *cfg_root;
};

/* Collects the shard filenames model_dir actually references: from
 * model.safetensors.index.json's weight_map when present (deduplicated,
 * in first-seen order), else the single *.safetensors file in model_dir. */
static sg_err collect_shard_filenames(const char *model_dir, char ***out_names, uint32_t *out_n) {
    *out_names = NULL;
    *out_n = 0;

    char idx_path[ST_PATH_MAX];
    sg_err e = join_path(model_dir, "model.safetensors.index.json", idx_path);
    if (sg_failed(e)) return e;

    struct stat idx_st;
    if (stat(idx_path, &idx_st) == 0) {
        char *buf = NULL;
        uint64_t sz = 0;
        e = read_file_fully(idx_path, &buf, &sz);
        if (sg_failed(e)) return e;

        jv *root = NULL;
        sg_err pe = jp_parse_document(buf, sz, &root);
        free(buf);
        if (sg_failed(pe)) return pe;

        const jv *wm = jv_obj_get(root, "weight_map");
        if (!wm || wm->kind != JV_OBJ || wm->as.obj.count == 0) {
            jv_free(root);
            return (sg_err){"st: index.json missing a non-empty weight_map"};
        }

        char **names = NULL;
        uint32_t n = 0, cap = 0;
        for (uint32_t i = 0; i < wm->as.obj.count; i++) {
            const jv *fv = wm->as.obj.members[i].val;
            if (!fv || fv->kind != JV_STR) {
                jv_free(root); free_str_list(names, n);
                return (sg_err){"st: weight_map value is not a string"};
            }
            bool dup = false;
            for (uint32_t j = 0; j < n; j++) {
                if (strcmp(names[j], fv->as.str.ptr) == 0) { dup = true; break; }
            }
            if (dup) continue;

            if (n == cap) {
                if (cap > UINT32_MAX / 2) {
                    jv_free(root); free_str_list(names, n);
                    return (sg_err){"st: too many weight_map entries"};
                }
                uint32_t ncap = cap ? cap * 2 : 4;
                char **nn = realloc(names, (size_t)ncap * sizeof(*nn));
                if (!nn) { jv_free(root); free_str_list(names, n); return (sg_err){"st: out of memory"}; }
                names = nn;
                cap = ncap;
            }
            char *copy = st_owned_copy(fv->as.str.ptr, fv->as.str.len);
            if (!copy) { jv_free(root); free_str_list(names, n); return (sg_err){"st: out of memory"}; }
            names[n++] = copy;
        }
        jv_free(root);

        *out_names = names;
        *out_n = n;
        return SG_OK;
    }

    /* No index.json: expect exactly one *.safetensors file in model_dir. */
    DIR *d = opendir(model_dir);
    if (!d) return (sg_err){"st: failed to open model_dir"};

    char **names = NULL;
    uint32_t n = 0, cap = 0;
    struct dirent *de;
    static const char suffix[] = ".safetensors";
    size_t suffix_len = sizeof(suffix) - 1;
    while ((de = readdir(d)) != NULL) {
        size_t nlen = strlen(de->d_name);
        if (nlen <= suffix_len) continue;
        if (strcmp(de->d_name + nlen - suffix_len, suffix) != 0) continue;

        if (n == cap) {
            if (cap > UINT32_MAX / 2) {
                closedir(d); free_str_list(names, n);
                return (sg_err){"st: too many .safetensors files in model_dir"};
            }
            uint32_t ncap = cap ? cap * 2 : 4;
            char **nn = realloc(names, (size_t)ncap * sizeof(*nn));
            if (!nn) { closedir(d); free_str_list(names, n); return (sg_err){"st: out of memory"}; }
            names = nn;
            cap = ncap;
        }
        char *copy = st_owned_copy(de->d_name, nlen);
        if (!copy) { closedir(d); free_str_list(names, n); return (sg_err){"st: out of memory"}; }
        names[n++] = copy;
    }
    closedir(d);

    if (n != 1) {
        free_str_list(names, n);
        return (sg_err){"st: expected exactly one .safetensors file when "
                         "model.safetensors.index.json is absent"};
    }

    *out_names = names;
    *out_n = n;
    return SG_OK;
}

/* Validates and, if dtype is BF16, appends one tensor entry for header
 * member `name` (a name -> {dtype,shape,data_offsets} object). Non-BF16
 * tensors are bounds-checked but not indexed (not retrievable via
 * sg_st_tensor). data_offsets are relative to the byte after the header;
 * data_start is that absolute file offset, map/map_size describe the shard. */
static sg_err st_add_tensor(sg_st *s, const char *name, uint32_t name_len, const jv *val,
                             const void *map, uint64_t map_size, uint64_t data_start) {
    if (!val || val->kind != JV_OBJ) return (sg_err){"st: tensor entry is not a json object"};

    const jv *dtype_v = jv_obj_get(val, "dtype");
    const jv *shape_v = jv_obj_get(val, "shape");
    const jv *off_v = jv_obj_get(val, "data_offsets");
    if (!dtype_v || dtype_v->kind != JV_STR) return (sg_err){"st: tensor missing string dtype"};
    if (!shape_v || shape_v->kind != JV_ARR) return (sg_err){"st: tensor missing array shape"};
    if (!off_v || off_v->kind != JV_ARR || off_v->as.arr.count != 2) {
        return (sg_err){"st: tensor missing 2-element data_offsets"};
    }
    /* Validate every shape entry regardless of rank (a malformed header must
     * still be rejected, not silently skipped); only the first 4 are kept,
     * since sg_st_tensor's interface is a fixed dims[4]. Real checkpoints do
     * carry rank>4 tensors (e.g. a vision conv weight's [out,in,t,h,w]) -
     * those are still bounds-checked below but not indexed, same treatment
     * as a non-BF16 dtype, rather than failing the whole shard's load. */
    uint64_t dims[4] = {0, 0, 0, 0};
    uint32_t n_dims = shape_v->as.arr.count;
    for (uint32_t i = 0; i < n_dims; i++) {
        uint64_t d;
        if (!jv_num_u64(shape_v->as.arr.items[i], &d)) {
            return (sg_err){"st: tensor shape entry is not a non-negative integer"};
        }
        if (i < 4) dims[i] = d;
    }

    uint64_t begin, end;
    if (!jv_num_u64(off_v->as.arr.items[0], &begin) || !jv_num_u64(off_v->as.arr.items[1], &end)) {
        return (sg_err){"st: tensor data_offsets entry is not a non-negative integer"};
    }
    if (end < begin) return (sg_err){"st: tensor data_offsets end precedes begin"};
    if (end > map_size - data_start) return (sg_err){"st: tensor data out of bounds"};

    bool is_bf16 = (dtype_v->as.str.len == 4 && memcmp(dtype_v->as.str.ptr, "BF16", 4) == 0);
    /* F32 is indexed too (retrievable via sg_st_tensor_f32, never via
     * sg_st_tensor): real Qwen3.5 checkpoints keep two per-linear-attention
     * -layer tensors in F32 while the rest of the layer is BF16
     * (linear_attn.A_log and linear_attn.norm.weight), so a bf16-only index
     * cannot reach the DeltaNet weights at all. Every other dtype stays
     * validated-but-not-indexed as before. */
    bool is_f32 = (dtype_v->as.str.len == 3 && memcmp(dtype_v->as.str.ptr, "F32", 3) == 0);
    if ((!is_bf16 && !is_f32) || n_dims > 4) return SG_OK; /* validated, not indexed */

    uint64_t elem_size = is_f32 ? 4 : 2;
    uint64_t elem_count = 1;
    for (uint32_t i = 0; i < n_dims; i++) {
        uint64_t d = dims[i];
        if (d != 0 && elem_count > UINT64_MAX / d) return (sg_err){"st: tensor dims overflow"};
        elem_count *= d;
    }
    if (elem_count > UINT64_MAX / elem_size) return (sg_err){"st: tensor size overflow"};
    uint64_t expected_bytes = elem_count * elem_size;
    uint64_t nbytes = end - begin;
    uint64_t abs_off = data_start + begin; /* begin <= map_size - data_start, so no overflow */

    /* Natural alignment is required so the mmap pointer can be handed out as
     * a uint16_t or float pointer directly.
     *
     * The two failure modes are treated differently by dtype, on purpose:
     *
     *  - BF16 keeps its original hard-error behaviour. This loader has always
     *    been "the bf16 loader", so a bf16 tensor it cannot represent is a
     *    file it cannot honestly load.
     *  - F32 only started being indexed in Task 7, and before that ANY
     *    non-BF16 tensor was validated-but-not-indexed and could never fail
     *    an open. Hard-erroring on it now would newly reject checkpoints that
     *    loaded fine before, including for a caller that only ever wanted the
     *    bf16 tensors: the safetensors spec does not require 8 + header_len
     *    to be a multiple of 4, so a legal shard can put every F32 tensor at
     *    a 2-aligned offset. So an F32 tensor that is misaligned or whose
     *    byte length disagrees with its shape is skipped, not fatal --
     *    exactly the pre-Task-7 contract for a dtype this loader cannot
     *    represent. It then reads as absent through sg_st_tensor_f32.
     *
     * Both real checkpoints are clean on both counts (the 2B's data section
     * starts at 76656, and all 36 of its F32 tensors are 4-aligned). */
    if (is_f32) {
        if (nbytes != expected_bytes || abs_off % elem_size != 0) {
            return SG_OK; /* validated, not indexed */
        }
    } else {
        if (nbytes != expected_bytes) {
            return (sg_err){"st: bf16 tensor byte length does not match its shape"};
        }
        if (abs_off % elem_size != 0) {
            return (sg_err){"st: bf16 tensor offset is not 2-byte aligned"};
        }
    }

    if (s->n_tensors == s->tensors_cap) {
        uint64_t ncap = s->tensors_cap ? s->tensors_cap * 2 : 16;
        st_tensor *nt = realloc(s->tensors, (size_t)ncap * sizeof(*nt));
        if (!nt) return (sg_err){"st: out of memory"};
        s->tensors = nt;
        s->tensors_cap = ncap;
    }
    char *name_copy = st_owned_copy(name, name_len);
    if (!name_copy) return (sg_err){"st: out of memory"};

    st_tensor *t = &s->tensors[s->n_tensors++];
    t->name = name_copy;
    t->n_dims = n_dims;
    memcpy(t->dims, dims, sizeof(dims));
    t->data = (const void *)((const uint8_t *)map + abs_off);
    t->nbytes = nbytes;
    t->is_f32 = is_f32;
    return SG_OK;
}

/* Opens and mmaps one shard file, parses its header, and appends its BF16
 * tensors to s->tensors. Registers the mapping in s->shards (bumping
 * s->n_shards) before walking the header, so sg_st_close() unmaps it even
 * if a later tensor in this same shard is malformed. */
static sg_err st_open_shard(sg_st *s, const char *model_dir, const char *filename) {
    char path[ST_PATH_MAX];
    sg_err e = join_path(model_dir, filename, path);
    if (sg_failed(e)) return e;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return (sg_err){"st: failed to open safetensors file"};
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return (sg_err){"st: fstat failed"}; }
    if (st.st_size < 8) { close(fd); return (sg_err){"st: safetensors file too small for header"}; }
    uint64_t map_size = (uint64_t)st.st_size;
    void *map = mmap(NULL, (size_t)map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* the mapping outlives the fd */
    if (map == MAP_FAILED) return (sg_err){"st: mmap failed"};

    /* Host is little-endian Apple Silicon, matching the safetensors
     * on-disk u64 header_len encoding, so a raw memcpy is correct here. */
    uint64_t header_len;
    memcpy(&header_len, map, 8);
    if (header_len > map_size - 8) {
        munmap(map, map_size);
        return (sg_err){"st: safetensors header_len exceeds file size"};
    }

    jv *hdr = NULL;
    sg_err pe = jp_parse_document((const char *)map + 8, header_len, &hdr);
    if (sg_failed(pe)) { munmap(map, map_size); return pe; }

    uint64_t data_start = 8 + header_len;

    uint32_t shard_idx = s->n_shards;
    s->shards[shard_idx].map = map;
    s->shards[shard_idx].map_size = map_size;
    s->n_shards++; /* sg_st_close() now owns unmapping this on any later failure */

    sg_err werr = SG_OK;
    for (uint32_t i = 0; i < hdr->as.obj.count; i++) {
        jv_member *m = &hdr->as.obj.members[i];
        static const char meta_key[] = "__metadata__";
        if (m->key_len == sizeof(meta_key) - 1 && memcmp(m->key, meta_key, m->key_len) == 0) {
            continue; /* metadata block, not a tensor */
        }
        werr = st_add_tensor(s, m->key, m->key_len, m->val, map, map_size, data_start);
        if (sg_failed(werr)) break;
    }
    jv_free(hdr);
    return werr;
}

sg_err sg_st_open(const char *model_dir, sg_st **out) {
    if (out) *out = NULL;
    if (!model_dir || !out) return (sg_err){"st: invalid arguments"};

    sg_st *s = calloc(1, sizeof(*s));
    if (!s) return (sg_err){"st: out of memory"};

    sg_err e;

    /* config.json */
    {
        char path[ST_PATH_MAX];
        e = join_path(model_dir, "config.json", path);
        if (sg_failed(e)) goto fail;

        char *buf = NULL;
        uint64_t sz = 0;
        e = read_file_fully(path, &buf, &sz);
        if (sg_failed(e)) goto fail;

        jv *root = NULL;
        sg_err pe = jp_parse_document(buf, sz, &root);
        free(buf);
        if (sg_failed(pe)) { e = pe; goto fail; }
        s->cfg_root = root;
    }

    /* shard filenames + shards */
    {
        char **names = NULL;
        uint32_t n_names = 0;
        e = collect_shard_filenames(model_dir, &names, &n_names);
        if (sg_failed(e)) goto fail;

        s->shards = calloc(n_names, sizeof(*s->shards));
        if (!s->shards) {
            free_str_list(names, n_names);
            e = (sg_err){"st: out of memory"};
            goto fail;
        }

        for (uint32_t i = 0; i < n_names; i++) {
            e = st_open_shard(s, model_dir, names[i]);
            if (sg_failed(e)) { free_str_list(names, n_names); goto fail; }
        }
        free_str_list(names, n_names);
    }

    *out = s;
    return SG_OK;

fail:
    sg_st_close(s);
    return e;
}

void sg_st_close(sg_st *s) {
    if (!s) return;
    for (uint32_t i = 0; i < s->n_shards; i++) {
        if (s->shards[i].map) munmap(s->shards[i].map, s->shards[i].map_size);
    }
    free(s->shards);
    for (uint64_t i = 0; i < s->n_tensors; i++) free(s->tensors[i].name);
    free(s->tensors);
    jv_free(s->cfg_root);
    free(s);
}

/* Shared lookup for both typed accessors; want_f32 selects which dtype the
 * caller is asking for, so a name that exists at the other dtype reports
 * "absent" exactly as the header promises. */
static const st_tensor *st_find(const sg_st *s, const char *name, bool want_f32) {
    if (!s || !name) return NULL;
    for (uint64_t i = 0; i < s->n_tensors; i++) {
        if (s->tensors[i].is_f32 != want_f32) continue;
        if (strcmp(s->tensors[i].name, name) == 0) return &s->tensors[i];
    }
    return NULL;
}

bool sg_st_tensor(const sg_st *s, const char *name, const uint16_t **data,
                  uint64_t dims[4], uint32_t *n_dims) {
    const st_tensor *t = st_find(s, name, false);
    if (!t) return false;
    if (data) *data = (const uint16_t *)t->data;
    if (dims) memcpy(dims, t->dims, sizeof(t->dims));
    if (n_dims) *n_dims = t->n_dims;
    return true;
}

bool sg_st_tensor_f32(const sg_st *s, const char *name, const float **data,
                      uint64_t dims[4], uint32_t *n_dims) {
    const st_tensor *t = st_find(s, name, true);
    if (!t) return false;
    if (data) *data = (const float *)t->data;
    if (dims) memcpy(dims, t->dims, sizeof(t->dims));
    if (n_dims) *n_dims = t->n_dims;
    return true;
}

/* key looked up at the top level first, then inside "text_config" if absent
 * there (some HF configs, e.g. Qwen3.5's, nest the text model's
 * hyperparameters under "text_config"), then inside a "rope_parameters"
 * object -- itself either at the top level or inside "text_config" -- as a
 * last resort. That last tier exists because Qwen3.5's real config.json
 * nests rope_theta two levels deep (text_config.rope_parameters.rope_theta)
 * rather than as a flat text_config.rope_theta key; checked last so a key
 * that resolves at a shallower level always wins. */
static const jv *st_config_lookup(const sg_st *s, const char *key) {
    if (!s || !s->cfg_root || !key) return NULL;
    jv *v = jv_obj_get(s->cfg_root, key);
    if (v) return v;

    jv *tc = jv_obj_get(s->cfg_root, "text_config");
    if (tc && tc->kind == JV_OBJ) {
        v = jv_obj_get(tc, key);
        if (v) return v;
    }

    jv *rp = jv_obj_get(s->cfg_root, "rope_parameters");
    if (rp && rp->kind == JV_OBJ) {
        v = jv_obj_get(rp, key);
        if (v) return v;
    }
    if (tc && tc->kind == JV_OBJ) {
        jv *tc_rp = jv_obj_get(tc, "rope_parameters");
        if (tc_rp && tc_rp->kind == JV_OBJ) {
            v = jv_obj_get(tc_rp, key);
            if (v) return v;
        }
    }
    return NULL;
}

bool sg_st_config_u32(const sg_st *s, const char *key, uint32_t *out) {
    if (!out) return false;
    const jv *v = st_config_lookup(s, key);
    if (!v || v->kind != JV_NUM) return false;
    if (v->as.num.has_i) {
        if (v->as.num.i < 0 || v->as.num.i > (int64_t)UINT32_MAX) return false;
        *out = (uint32_t)v->as.num.i;
        return true;
    }
    double d = v->as.num.d;
    if (d < 0 || d > (double)UINT32_MAX || d != (double)(uint32_t)d) return false;
    *out = (uint32_t)d;
    return true;
}

bool sg_st_config_f32(const sg_st *s, const char *key, float *out) {
    if (!out) return false;
    const jv *v = st_config_lookup(s, key);
    if (!v || v->kind != JV_NUM) return false;
    *out = v->as.num.has_i ? (float)v->as.num.i : (float)v->as.num.d;
    return true;
}
