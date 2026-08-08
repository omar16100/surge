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
        const uint32_t *vals = (const uint32_t *)arr_data;
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
    tt_run("gguf_open_nonexistent", gguf_open_nonexistent);
    tt_run("gguf_open_truncated", gguf_open_truncated);
    return tt_report();
}
