/* fixture.h - reader for the "SURGEOPS" fixture container that
 * tools/make_fixtures.py writes (tests/fixtures/ops.bin,
 * tests/fixtures/hybrid_ops.bin). The format is documented in that script;
 * this is its C side, shared by tests/test_ref_ops.c (ref ops vs the
 * fixtures) and tests/test_metal_ops.c (Metal ops vs the ref ops on the same
 * fixture inputs).
 *
 * Everything is `static inline` on purpose: a test that uses only half of
 * these must not trip -Wunused-function under -Werror.
 */
#ifndef SURGE_TEST_FIXTURE_H
#define SURGE_TEST_FIXTURE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FX_F32 0u
#define FX_U16 1u
#define FX_U8  2u
#define FX_DIR_ENTRY 36u

/* Allocation failure in a test is not a test failure to report, it is a
 * reason to stop: soldiering on through a NULL would turn an OOM into a
 * misleading segfault inside an op. */
static inline void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "FATAL: out of memory (%zu bytes)\n", n); exit(2); }
    return p;
}

static inline void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) { fprintf(stderr, "FATAL: out of memory (%zu x %zu)\n", n, sz); exit(2); }
    return p;
}

typedef struct {
    uint8_t *buf;
    uint64_t size;
    uint32_t n_records;
} fixture;

static inline bool fx_open(fixture *f, const char *path) {
    memset(f, 0, sizeof *f);
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return false; }
    long sz = ftell(fp);
    if (sz < 16) { fclose(fp); return false; }
    rewind(fp);
    uint8_t *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return false; }
    fclose(fp);
    if (memcmp(buf, "SURGEOPS", 8) != 0) { free(buf); return false; }
    uint32_t ver;
    memcpy(&ver, buf + 8, 4);
    if (ver != 1) { free(buf); return false; }
    memcpy(&f->n_records, buf + 12, 4);
    if (16 + (uint64_t)f->n_records * FX_DIR_ENTRY > (uint64_t)sz) { free(buf); return false; }
    f->buf = buf;
    f->size = (uint64_t)sz;
    return true;
}

static inline void fx_close(fixture *f) { free(f->buf); memset(f, 0, sizeof *f); }

/* Returns the payload pointer for `name`, or NULL. dtype/dims/count are
 * optional outputs; count is the element count (payload bytes / elem size).
 *
 * Every structural property the callers rely on is checked here, so that a
 * truncated or edited fixture is a loud exit rather than an out-of-bounds
 * read inside an op: the name must be NUL-terminated inside the buffer, the
 * payload must lie inside the buffer, the payload length must be exactly
 * prod(dims) * elem_size (so a caller that sizes a read from the record's
 * DIMS can never outrun the record's BYTES), and the payload must be
 * naturally aligned for its element type (the getters hand out float* and
 * uint16_t*; a misaligned one is UB and a UBSan failure). */
static inline const void *fx_find(const fixture *f, const char *name, uint32_t *dtype,
                                  uint64_t dims[4], uint64_t *count) {
    for (uint32_t i = 0; i < f->n_records; i++) {
        const uint8_t *e = f->buf + 16 + (uint64_t)i * FX_DIR_ENTRY;
        uint32_t v[9];
        memcpy(v, e, sizeof v);
        uint32_t name_off = v[0], dt = v[1], n_dims = v[2];
        uint32_t data_off = v[7], nbytes = v[8];

        if (name_off >= f->size) return NULL;
        const char *nm = (const char *)f->buf + name_off;
        if (memchr(nm, '\0', (size_t)(f->size - name_off)) == NULL) return NULL;
        if (strcmp(nm, name) != 0) continue;

        if (dt > FX_U8 || n_dims > 4) return NULL;
        uint32_t esz = (dt == FX_F32) ? 4u : (dt == FX_U16 ? 2u : 1u);
        if ((uint64_t)data_off + nbytes > f->size) return NULL;
        if (data_off % esz != 0) return NULL;

        uint64_t elems = 1;
        for (uint32_t d = 0; d < n_dims; d++) elems *= v[3 + d];
        if (elems > UINT32_MAX / esz || elems * esz != nbytes) return NULL;

        if (dtype) *dtype = dt;
        if (dims) {
            for (uint32_t d = 0; d < 4; d++) dims[d] = (d < n_dims) ? v[3 + d] : 0;
        }
        if (count) *count = elems;
        return f->buf + data_off;
    }
    return NULL;
}

/* Every getter aborts loudly rather than returning garbage: a missing or
 * structurally invalid record means the committed fixture and this test
 * disagree about the fixture's contents, which is never something to
 * soldier on through.
 *
 * `want` is the number of elements the caller is about to read. Pass 0 only
 * when the caller is DISCOVERING the length from this very record (and then
 * asks for `count`); anywhere the length comes from somewhere else -- a
 * scalar record, another array's dims, a computed rows*cols -- state it, so
 * the read is bounded by this record's own payload rather than by another
 * record's idea of how big it should be. */
static inline const float *fx_f32(const fixture *f, const char *name, uint64_t want,
                                  uint64_t *count) {
    uint32_t dt = 0;
    uint64_t n = 0;
    const void *p = fx_find(f, name, &dt, NULL, &n);
    if (!p || dt != FX_F32) {
        fprintf(stderr, "FATAL: fixture record '%s' missing, malformed, or not f32\n", name);
        exit(2);
    }
    if (want != 0 && n < want) {
        fprintf(stderr, "FATAL: fixture record '%s' has %llu f32 elements, "
                        "caller needs %llu\n", name,
                        (unsigned long long)n, (unsigned long long)want);
        exit(2);
    }
    if (count) *count = n;
    return (const float *)p;
}

static inline const uint16_t *fx_u16(const fixture *f, const char *name, uint64_t want,
                                     uint64_t *count) {
    uint32_t dt = 0;
    uint64_t n = 0;
    const void *p = fx_find(f, name, &dt, NULL, &n);
    if (!p || dt != FX_U16) {
        fprintf(stderr, "FATAL: fixture record '%s' missing, malformed, or not u16\n", name);
        exit(2);
    }
    if (want != 0 && n < want) {
        fprintf(stderr, "FATAL: fixture record '%s' has %llu u16 elements, "
                        "caller needs %llu\n", name,
                        (unsigned long long)n, (unsigned long long)want);
        exit(2);
    }
    if (count) *count = n;
    return (const uint16_t *)p;
}

static inline const void *fx_u8(const fixture *f, const char *name, uint64_t want) {
    uint32_t dt = 0;
    uint64_t n = 0;
    const void *p = fx_find(f, name, &dt, NULL, &n);
    if (!p || dt != FX_U8) {
        fprintf(stderr, "FATAL: fixture record '%s' missing, malformed, or not u8\n", name);
        exit(2);
    }
    if (want != 0 && n < want) {
        fprintf(stderr, "FATAL: fixture record '%s' has %llu bytes, caller needs %llu\n",
                name, (unsigned long long)n, (unsigned long long)want);
        exit(2);
    }
    return p;
}

static inline uint32_t fx_dim(const fixture *f, const char *name) {
    uint64_t n = 0;
    const float *p = fx_f32(f, name, 1, &n);
    if (n != 1) { fprintf(stderr, "FATAL: '%s' is not a scalar\n", name); exit(2); }
    if (!(p[0] >= 0.0f && p[0] <= 1e9f)) {
        fprintf(stderr, "FATAL: '%s' = %g is not a plausible dimension\n", name, (double)p[0]);
        exit(2);
    }
    return (uint32_t)(p[0] + 0.5f);
}

static inline float fx_scalar(const fixture *f, const char *name) {
    uint64_t n = 0;
    const float *p = fx_f32(f, name, 1, &n);
    if (n != 1) { fprintf(stderr, "FATAL: '%s' is not a scalar\n", name); exit(2); }
    return p[0];
}

#endif
