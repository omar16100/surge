#include "tinytest.h"
#include "../surge.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void gguf_open_close(void) {
    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open("tests/fixtures/mini.gguf", &g);
    tt_assert(!sg_failed(e), "open mini.gguf should succeed: %s", e.msg ? e.msg : "");
    tt_assert(g != NULL, "handle should be non-NULL on success");
    if (!g) return;

    tt_assert(sg_gguf_tensor_count(g) == 2, "tensor_count should be 2");

    uint32_t block_count = 0;
    tt_assert(sg_gguf_get_u32(g, "qwen3_5.block_count", &block_count),
              "block_count key should be present");
    tt_assert(block_count == 2, "block_count should be 2");

    const char *arch = NULL;
    tt_assert(sg_gguf_get_str(g, "general.architecture", &arch),
              "architecture key should be present");
    tt_assert(arch != NULL && strcmp(arch, "qwen3_5") == 0,
              "architecture should be qwen3_5");

    sg_gguf_kv_type elem_type;
    const void *arr_data = NULL;
    uint64_t arr_count = 0;
    tt_assert(sg_gguf_get_arr(g, "test.arr", &elem_type, &arr_data, &arr_count),
              "test.arr key should be present");
    tt_assert(elem_type == SG_GGUF_U32, "test.arr elem_type should be U32");
    tt_assert(arr_count == 3, "test.arr count should be 3");
    if (arr_data && arr_count == 3) {
        /* sg_gguf_get_arr hands back a pointer straight into the mmap'd
         * file, and GGUF array payloads carry no alignment guarantee at all
         * (this one lands on a 2-byte boundary in the fixture), so the
         * elements must be memcpy'd out rather than read through a
         * uint32_t*. Casting works on arm64 but is UB, and UBSan flags it. */
        uint32_t vals[3];
        memcpy(vals, arr_data, sizeof vals);
        tt_assert(vals[0] == 1 && vals[1] == 2 && vals[2] == 3,
                  "test.arr values should be 1,2,3");
    }

    uint32_t missing_u32 = 0;
    tt_assert(!sg_gguf_get_u32(g, "does.not.exist", &missing_u32),
              "missing key should return false for get_u32");
    const char *missing_str = NULL;
    tt_assert(!sg_gguf_get_str(g, "does.not.exist", &missing_str),
              "missing key should return false for get_str");

    const sg_tensor *tf32 = sg_gguf_tensor(g, "t.f32");
    tt_assert(tf32 != NULL, "t.f32 tensor should be found");
    if (tf32) {
        tt_assert(tf32->n_dims == 1, "t.f32 n_dims should be 1");
        tt_assert(tf32->dims[0] == 4, "t.f32 dims[0] should be 4");
        tt_assert(tf32->nbytes == 16, "t.f32 nbytes should be 16");
        const float *fdata = (const float *)tf32->data;
        tt_assert(fdata != NULL && fdata[2] == 3.0f, "t.f32 data[2] should be 3.0");
    }

    const sg_tensor *tq8 = sg_gguf_tensor(g, "t.q8");
    tt_assert(tq8 != NULL, "t.q8 tensor should be found");
    if (tq8) {
        tt_assert(tq8->nbytes == 34, "t.q8 nbytes should be 34");
    }

    tt_assert(sg_gguf_tensor(g, "no.such.tensor") == NULL,
              "missing tensor lookup should return NULL");

    sg_gguf_close(g);
}

static void gguf_metadata_enumeration(void) {
    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open("tests/fixtures/mini.gguf", &g);
    tt_assert(!sg_failed(e), "open mini.gguf should succeed: %s", e.msg ? e.msg : "");
    tt_assert(g != NULL, "handle should be non-NULL on success");
    if (!g) return;

    uint32_t version = sg_gguf_version(g);
    tt_assert(version == 3, "version should be 3");

    uint64_t kv_count = sg_gguf_kv_count(g);
    tt_assert(kv_count == 5, "kv_count should be 5 (arch, block_count, arr, i32, bool)");

    const char *arch_key;
    sg_gguf_kv_type arch_type;
    tt_assert(sg_gguf_kv_at(g, 0, &arch_key, &arch_type),
              "first kv should be accessible");
    tt_assert(arch_type == SG_GGUF_STR, "first key should be STR type");
    tt_assert(strcmp(arch_key, "general.architecture") == 0,
              "first key should be general.architecture");

    const char *i32_key;
    sg_gguf_kv_type i32_type;
    tt_assert(sg_gguf_kv_at(g, 3, &i32_key, &i32_type),
              "fourth kv should be accessible");
    tt_assert(i32_type == SG_GGUF_I32, "fourth key should be I32 type");
    tt_assert(strcmp(i32_key, "test.i32") == 0,
              "fourth key should be test.i32");

    int64_t i32_val = 0;
    sg_gguf_kv_scalar_at(g, 3, NULL, &i32_val, NULL, NULL);
    tt_assert(i32_val == -42, "i32 value should be -42");

    const char *bool_key;
    sg_gguf_kv_type bool_type;
    tt_assert(sg_gguf_kv_at(g, 4, &bool_key, &bool_type),
              "fifth kv should be accessible");
    tt_assert(bool_type == SG_GGUF_BOOL, "fifth key should be BOOL type");
    tt_assert(strcmp(bool_key, "test.bool") == 0,
              "fifth key should be test.bool");

    bool bool_val = false;
    sg_gguf_kv_scalar_at(g, 4, NULL, NULL, NULL, &bool_val);
    tt_assert(bool_val == true, "bool value should be true");

    tt_assert(!sg_gguf_kv_at(g, 999, NULL, NULL),
              "out of bounds access should return false");

    sg_gguf_close(g);
}

static void gguf_open_nonexistent(void) {
    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open("/nonexistent/path/mini.gguf", &g);
    tt_assert(sg_failed(e), "opening a nonexistent path should fail");
    tt_assert(g == NULL, "handle should stay NULL on failure");
}

static void gguf_open_truncated(void) {
    FILE *src = fopen("tests/fixtures/mini.gguf", "rb");
    tt_assert(src != NULL, "fixture should be readable for truncation test");
    if (!src) return;

    unsigned char buf[40];
    size_t n = fread(buf, 1, sizeof(buf), src);
    fclose(src);
    tt_assert(n == sizeof(buf), "fixture should be at least 40 bytes");

    char path[64];
    snprintf(path, sizeof(path), "/tmp/surge_gguf_trunc_%d.gguf", (int)getpid());

    FILE *dst = fopen(path, "wb");
    tt_assert(dst != NULL, "truncated fixture should be writable");
    if (!dst) return;
    fwrite(buf, 1, n, dst);
    fclose(dst);

    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open(path, &g);
    tt_assert(sg_failed(e), "opening a truncated file should fail, not crash");
    tt_assert(g == NULL, "handle should stay NULL on truncated open failure");

    remove(path);
}

int main(void) {
    tt_run("gguf_open_close", gguf_open_close);
    tt_run("gguf_metadata_enumeration", gguf_metadata_enumeration);
    tt_run("gguf_open_nonexistent", gguf_open_nonexistent);
    tt_run("gguf_open_truncated", gguf_open_truncated);
    return tt_report();
}
