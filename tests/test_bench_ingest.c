/* test_bench_ingest.c - src/bench.c: prompt ingestion + truncation guard
 * (Task B3). Pure C, no Metal, no GPU; runs under a plain `make check` and
 * under `make debug` (ASan/UBSan, SURGE_NO_METAL).
 *
 * Covers the B3 gate verbatim:
 *   (a) sg_bench_check_ingestion PASSES (ok=true) at n_ids == max_ctx;
 *       ok=false for n_ids == max_ctx+1;
 *       ok=false for n_ids just below expect_min and just above expect_max.
 *   (b) sg_bench_read_file rejects an empty file (temp file created here).
 *   (c) sg_bench_read_file returns the exact byte length for a known-content
 *       temp file (round trip), and the real byte length (1462729) for
 *       /Users/macmini/models/niah_256k_prompt.txt when that file exists on
 *       disk (skipped with a printed notice if it is absent).
 *   (d) sg_bench_read_file rejects a file over the 3 GiB cap (sparse file,
 *       size checked before any read so this stays fast).
 *   (e) NULL-argument and missing-file paths fail cleanly, no crash.
 *
 * Every successful sg_bench_read_file call here frees its buffer -- ASan
 * under `make debug` catches a leak otherwise.
 */

#include "surge.h"
#include "tinytest.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* --------------------------------------------------------------------
 * (a) sg_bench_check_ingestion boundary behavior.
 * -------------------------------------------------------------------- */
static void test_check_ingestion_bounds(void) {
    const uint32_t max_ctx = 262144;
    const uint64_t expect_min = 240000, expect_max = 262144;
    bool ok;

    /* PASS at n_ids == max_ctx (also inside [expect_min, expect_max]). */
    ok = false;
    sg_bench_check_ingestion(max_ctx, max_ctx, expect_min, expect_max, &ok);
    tt_assert(ok, "n_ids==max_ctx (%u) should PASS", max_ctx);

    /* VOID at n_ids == max_ctx + 1 (truncation-guard boundary). */
    ok = true;
    sg_bench_check_ingestion((uint64_t)max_ctx + 1, max_ctx, expect_min, expect_max, &ok);
    tt_assert(!ok, "n_ids==max_ctx+1 (%u) should be VOID", max_ctx + 1);

    /* VOID just below expect_min, with max_ctx not the limiting factor. */
    ok = true;
    sg_bench_check_ingestion(expect_min - 1, max_ctx, expect_min, expect_max, &ok);
    tt_assert(!ok, "n_ids==expect_min-1 (%llu) should be VOID",
              (unsigned long long)(expect_min - 1));

    /* VOID just above expect_max, with a max_ctx large enough that the
     * max_ctx bound alone would NOT reject it -- isolates the expect_max
     * check from the max_ctx check. */
    const uint32_t roomy_ctx = 300000;
    ok = true;
    sg_bench_check_ingestion(expect_max + 1, roomy_ctx, expect_min, expect_max, &ok);
    tt_assert(!ok, "n_ids==expect_max+1 (%llu) should be VOID",
              (unsigned long long)(expect_max + 1));

    /* Inclusive boundaries: exactly expect_min and exactly expect_max PASS. */
    ok = false;
    sg_bench_check_ingestion(expect_min, max_ctx, expect_min, expect_max, &ok);
    tt_assert(ok, "n_ids==expect_min should PASS (inclusive bound)");

    ok = false;
    sg_bench_check_ingestion(expect_max, max_ctx, expect_min, expect_max, &ok);
    tt_assert(ok, "n_ids==expect_max should PASS (inclusive bound)");

    /* NULL ok must not crash. */
    sg_bench_check_ingestion(max_ctx, max_ctx, expect_min, expect_max, NULL);
}

/* --------------------------------------------------------------------
 * (b) empty file rejected.
 * -------------------------------------------------------------------- */
static void test_read_file_rejects_empty(void) {
    char path[64];
    snprintf(path, sizeof path, "/tmp/surge_bench_ingest_empty_%d.txt", (int)getpid());

    FILE *f = fopen(path, "wb");
    tt_assert(f != NULL, "temp empty file should be creatable");
    if (!f) return;
    fclose(f);   /* zero bytes written */

    char *buf = (char *)0x1;   /* sentinel: must be reset to NULL on failure */
    size_t len = 999;
    sg_err e = sg_bench_read_file(path, &buf, &len);
    tt_assert(sg_failed(e), "empty file must be rejected");
    tt_assert(buf == NULL, "out pointer must be NULL after a rejected read");
    tt_assert(len == 0, "len must be 0 after a rejected read");

    remove(path);
}

/* --------------------------------------------------------------------
 * (c) known-content round trip, exact byte length.
 * -------------------------------------------------------------------- */
static void test_read_file_roundtrip(void) {
    char path[64];
    snprintf(path, sizeof path, "/tmp/surge_bench_ingest_roundtrip_%d.txt", (int)getpid());

    const char *content = "hello ingestion guard\nsecond line, no trailing NUL in source";
    size_t content_len = strlen(content);

    FILE *f = fopen(path, "wb");
    tt_assert(f != NULL, "temp content file should be creatable");
    if (!f) return;
    size_t written = fwrite(content, 1, content_len, f);
    fclose(f);
    tt_assert(written == content_len, "temp file should be fully written");

    char *buf = NULL;
    size_t len = 0;
    sg_err e = sg_bench_read_file(path, &buf, &len);
    tt_assert(!sg_failed(e), "round-trip read should succeed: %s",
              sg_failed(e) ? e.msg : "");
    if (!sg_failed(e)) {
        tt_assert(len == content_len, "len=%zu want=%zu", len, content_len);
        tt_assert(buf != NULL, "buf must be non-NULL on success");
        if (buf) {
            tt_assert(memcmp(buf, content, content_len) == 0,
                      "content mismatch on round trip");
            tt_assert(buf[len] == '\0', "buffer must be NUL-terminated at buf[len]");
            free(buf);
        }
    }

    remove(path);
}

/* --------------------------------------------------------------------
 * (d) oversize file (> 3 GiB) rejected. Uses a sparse file (ftruncate,
 * no actual data written) so the test stays fast; sg_bench_read_file
 * checks the size via fstat BEFORE reading, so this never allocates or
 * reads 3 GiB of anything.
 * -------------------------------------------------------------------- */
static void test_read_file_rejects_oversize(void) {
    char path[64];
    snprintf(path, sizeof path, "/tmp/surge_bench_ingest_oversize_%d.bin", (int)getpid());

    FILE *f = fopen(path, "wb");
    tt_assert(f != NULL, "temp oversize file should be creatable");
    if (!f) return;
    fclose(f);

    int fd = open(path, O_WRONLY);
    tt_assert(fd >= 0, "temp oversize file should be reopenable for truncate");
    if (fd < 0) { remove(path); return; }
    off_t oversize = (off_t)SG_BENCH_MAX_FILE_BYTES + 1;
    int rc = ftruncate(fd, oversize);
    close(fd);
    tt_assert(rc == 0, "ftruncate to %lld bytes should succeed (sparse file)",
              (long long)oversize);

    char *buf = (char *)0x1;
    size_t len = 999;
    sg_err e = sg_bench_read_file(path, &buf, &len);
    tt_assert(sg_failed(e), "file over SG_BENCH_MAX_FILE_BYTES must be rejected");
    tt_assert(buf == NULL, "out pointer must be NULL after a rejected oversize read");

    remove(path);
}

/* --------------------------------------------------------------------
 * (e) NULL arguments and a missing path fail cleanly.
 * -------------------------------------------------------------------- */
static void test_read_file_edge_cases(void) {
    char *buf = NULL;
    size_t len = 0;

    sg_err e1 = sg_bench_read_file(NULL, &buf, &len);
    tt_assert(sg_failed(e1), "NULL path must fail");

    sg_err e2 = sg_bench_read_file("/tmp/surge_bench_ingest_does_not_exist.txt", &buf, &len);
    tt_assert(sg_failed(e2), "nonexistent path must fail, not crash");
    tt_assert(buf == NULL, "out pointer must stay NULL on open failure");

    /* NULL out / NULL len: must not crash. */
    sg_err e3 = sg_bench_read_file("/tmp/surge_bench_ingest_does_not_exist.txt", NULL, &len);
    tt_assert(sg_failed(e3), "NULL out must fail cleanly");
    sg_err e4 = sg_bench_read_file("/tmp/surge_bench_ingest_does_not_exist.txt", &buf, NULL);
    tt_assert(sg_failed(e4), "NULL len must fail cleanly");
}

/* --------------------------------------------------------------------
 * (c2) the real NIAH 256K prompt file: exact byte length 1462729.
 * Runs unconditionally as file-read + guard unit tests happen above
 * regardless of this file's presence; this assertion alone is skipped
 * (with a printed notice) if the file is absent from disk.
 * -------------------------------------------------------------------- */
#define NIAH_PROMPT_PATH "/Users/macmini/models/niah_256k_prompt.txt"
#define NIAH_PROMPT_EXPECT_BYTES ((size_t)1462729)

static void test_real_niah_prompt_byte_length(void) {
    struct stat st;
    if (stat(NIAH_PROMPT_PATH, &st) != 0) {
        fprintf(stderr,
                "NOTICE: %s not found on disk; skipping the real-file byte-length "
                "assertion (SURGE_NIAH_GGUF gate is separate and not required here)\n",
                NIAH_PROMPT_PATH);
        return;
    }

    char *buf = NULL;
    size_t len = 0;
    sg_err e = sg_bench_read_file(NIAH_PROMPT_PATH, &buf, &len);
    tt_assert(!sg_failed(e), "reading the real NIAH prompt file should succeed: %s",
              sg_failed(e) ? e.msg : "");
    if (!sg_failed(e)) {
        tt_assert(len == NIAH_PROMPT_EXPECT_BYTES,
                  "niah_256k_prompt.txt len=%zu want=%zu", len, NIAH_PROMPT_EXPECT_BYTES);
        if (buf) {
            tt_assert(buf[len] == '\0', "real-file buffer must be NUL-terminated");
            free(buf);
        }
    }
}

int main(void) {
    tt_run("check_ingestion: PASS/VOID boundaries", test_check_ingestion_bounds);
    tt_run("read_file: rejects empty file", test_read_file_rejects_empty);
    tt_run("read_file: round trip, exact byte length", test_read_file_roundtrip);
    tt_run("read_file: rejects file > 3 GiB", test_read_file_rejects_oversize);
    tt_run("read_file: NULL args / missing path", test_read_file_edge_cases);
    tt_run("read_file: real niah_256k_prompt.txt == 1462729 bytes",
           test_real_niah_prompt_byte_length);
    return tt_report();
}
