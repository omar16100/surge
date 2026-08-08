/* test_st.c - tests for the safetensors bf16 loader + config.json (st.c).
 *
 * Fixture asserts run unconditionally against tests/fixtures/mini_st/
 * (produced by tools/make_fixtures.py write_mini_safetensors()): a
 * single-shard model dir with one bf16 tensor "w" [2,3] and
 * config.json {"hidden_size": 3}.
 *
 * Real-model asserts are gated on env SURGE_ST (path to a directory with
 * config.json + model.safetensors.index.json + shards, e.g.
 * /Users/macmini/models/qwen35-2b) and auto-skip with a notice when unset,
 * so plain `make check` stays green without the multi-GB model on disk.
 */
#include "tinytest.h"
#include "../surge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* bf16(0.5)=0x3f00 bf16(1)=0x3f80 bf16(2)=0x4000 bf16(3)=0x4040
 * bf16(4)=0x4080 bf16(5)=0x40a0 - precomputed via truncated-round-to-nearest
 * float32->bf16 (exact for these values; verified against the fixture
 * generator's own _f32_to_bf16_bytes()). */
static const uint16_t MINI_W_BF16[6] = {0x3f00, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0};

static void st_fixture_open_close(void) {
    sg_st *s = NULL;
    sg_err e = sg_st_open("tests/fixtures/mini_st", &s);
    tt_assert(!sg_failed(e), "open mini_st should succeed: %s", e.msg ? e.msg : "");
    tt_assert(s != NULL, "handle should be non-NULL on success");
    if (!s) return;

    const uint16_t *data = NULL;
    uint64_t dims[4] = {99, 99, 99, 99};
    uint32_t n_dims = 99;
    bool found = sg_st_tensor(s, "w", &data, dims, &n_dims);
    tt_assert(found, "tensor w should be found");
    if (found) {
        tt_assert(n_dims == 2, "w n_dims should be 2, got %u", n_dims);
        tt_assert(dims[0] == 2 && dims[1] == 3,
                  "w dims should be [2,3], got [%llu,%llu]",
                  (unsigned long long)dims[0], (unsigned long long)dims[1]);
        tt_assert(dims[2] == 0 && dims[3] == 0,
                  "w unused dims slots should be 0");
        tt_assert(data != NULL, "w data pointer should be non-NULL");
        if (data) {
            for (int i = 0; i < 6; i++) {
                tt_assert(data[i] == MINI_W_BF16[i],
                          "w bf16[%d] should be 0x%04x, got 0x%04x",
                          i, MINI_W_BF16[i], data[i]);
            }
        }
    }

    tt_assert(!sg_st_tensor(s, "does.not.exist", NULL, NULL, NULL),
              "missing tensor lookup should return false");

    uint32_t hidden_size = 0;
    tt_assert(sg_st_config_u32(s, "hidden_size", &hidden_size),
              "hidden_size config key should be present");
    tt_assert(hidden_size == 3, "hidden_size should be 3, got %u", hidden_size);

    float hidden_size_f = 0;
    tt_assert(sg_st_config_f32(s, "hidden_size", &hidden_size_f),
              "hidden_size should also be readable as f32");
    tt_assert(hidden_size_f == 3.0f, "hidden_size as f32 should be 3.0, got %f",
              (double)hidden_size_f);

    uint32_t missing = 0;
    tt_assert(!sg_st_config_u32(s, "does.not.exist", &missing),
              "missing config key should return false");

    sg_st_close(s);
}

static void st_open_nonexistent(void) {
    sg_st *s = NULL;
    sg_err e = sg_st_open("/nonexistent/model/dir", &s);
    tt_assert(sg_failed(e), "opening a nonexistent model_dir should fail");
    tt_assert(s == NULL, "handle should stay NULL on failure");
}

static void st_open_truncated_shard(void) {
    FILE *src = fopen("tests/fixtures/mini_st/model.safetensors", "rb");
    tt_assert(src != NULL, "fixture shard should be readable for truncation test");
    if (!src) return;

    unsigned char buf[16];
    size_t n = fread(buf, 1, sizeof(buf), src);
    fclose(src);
    tt_assert(n == sizeof(buf), "fixture shard should be at least 16 bytes");

    char dir[128];
    snprintf(dir, sizeof(dir), "/tmp/surge_st_trunc_%d", (int)getpid());
    char mkcmd[192];
    snprintf(mkcmd, sizeof(mkcmd), "mkdir -p %s", dir);
    tt_assert(system(mkcmd) == 0, "mkdir for truncated fixture dir should succeed");

    char shard_path[192];
    snprintf(shard_path, sizeof(shard_path), "%s/model.safetensors", dir);
    FILE *dst = fopen(shard_path, "wb");
    tt_assert(dst != NULL, "truncated shard should be writable");
    if (dst) {
        fwrite(buf, 1, n, dst);
        fclose(dst);
    }

    char cfg_path[192];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", dir);
    FILE *cfg = fopen(cfg_path, "w");
    tt_assert(cfg != NULL, "truncated fixture config.json should be writable");
    if (cfg) {
        fputs("{\"hidden_size\": 3}", cfg);
        fclose(cfg);
    }

    sg_st *s = NULL;
    sg_err e = sg_st_open(dir, &s);
    tt_assert(sg_failed(e), "opening a truncated shard should fail, not crash");
    tt_assert(s == NULL, "handle should stay NULL on truncated open failure");

    remove(shard_path);
    remove(cfg_path);
    rmdir(dir);
}

static void st_real_model(void) {
    const char *model_dir = getenv("SURGE_ST");
    if (!model_dir || !*model_dir) {
        fprintf(stderr, "SKIP: SURGE_ST not set; skipping real-model st test "
                        "(set it to a safetensors model dir, e.g. "
                        "/Users/macmini/models/qwen35-2b)\n");
        return;
    }

    sg_st *s = NULL;
    sg_err e = sg_st_open(model_dir, &s);
    tt_assert(!sg_failed(e), "open %s should succeed: %s", model_dir, e.msg ? e.msg : "");
    if (!s) return;

    uint32_t hidden_size = 0;
    tt_assert(sg_st_config_u32(s, "hidden_size", &hidden_size),
              "hidden_size should be found (top-level or under text_config)");

    const uint16_t *data = NULL;
    uint64_t dims[4] = {0, 0, 0, 0};
    uint32_t n_dims = 0;
    bool found = sg_st_tensor(s, "model.language_model.embed_tokens.weight",
                               &data, dims, &n_dims);
    tt_assert(found, "model.language_model.embed_tokens.weight should be found");
    if (found) {
        tt_assert(n_dims == 2, "embed_tokens n_dims should be 2, got %u", n_dims);
        tt_assert(data != NULL, "embed_tokens data pointer should be non-NULL");
        if (hidden_size != 0) {
            tt_assert(dims[1] == hidden_size,
                      "embed_tokens dims[1] (%llu) should equal config hidden_size (%u)",
                      (unsigned long long)dims[1], hidden_size);
        }
    }

    tt_assert(!sg_st_tensor(s, "no.such.tensor", NULL, NULL, NULL),
              "missing tensor lookup should return false");

    sg_st_close(s);
}

int main(void) {
    tt_run("st_fixture_open_close", st_fixture_open_close);
    tt_run("st_open_nonexistent", st_open_nonexistent);
    tt_run("st_open_truncated_shard", st_open_truncated_shard);
    tt_run("st_real_model", st_real_model);
    return tt_report();
}
