/* test_bench_score.c - src/bench.c: NIAH recall scorer (Task B4). Pure C,
 * no Metal, no GPU; runs under a plain `make check` and under `make debug`
 * (ASan/UBSan, SURGE_NO_METAL).
 *
 * Covers the B4 gate verbatim:
 *   (i)   sg_bench_extract_needles pulls exactly the 8 known pairs from
 *         BOTH the real file /Users/macmini/models/niah_256k_prompt.txt
 *         (skipped with a notice if absent) and a synthetic haystack
 *         built here.
 *   (ii)  a golden correct-answer string (all 8 "City: code" lines) =>
 *         retrieval 8/8 AND assoc 8/8.
 *   (iii) codes-right-but-cities-shuffled (each code on a line with the
 *         WRONG city) => retrieval 8/8, assoc 0/8.
 *   (iv)  a 3-of-8 answer => retrieval 3/8.
 *   (v)   a filler 8-digit number that is NOT a needle code does not
 *         inflate retrieval.
 *   (vi)  codes with adjacent punctuation ("13072624." / "(13072624)")
 *         still match, for both retrieval and association.
 *
 * Also covers: extract_needles' fixed cap truncating cleanly, NULL/
 * invalid-argument handling on both functions (no crash), and that
 * near-miss text (missing trailing '.', missing "IMPORTANT RECORD: "
 * prefix, the trailing NIAH question line that names cities with no
 * codes attached, an empty city) never produces a false needle.
 */

#include "surge.h"
#include "tinytest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* The 8 real needle pairs this project's NIAH prompt generator wrote into
 * /Users/macmini/models/niah_256k_prompt.txt, in file order. The
 * extractor derives these at RUNTIME from the prompt text; this table is
 * the test's EXPECTED values only, not something bench.c reads. */
static const char *const NEEDLE_CITY[8] = {
    "Reykjavik", "Ouagadougou", "Valparaiso", "Nakhodka",
    "Timbuktu", "Kirkwall", "Ushuaia", "Yakutsk",
};
static const char *const NEEDLE_CODE[8] = {
    "13072624", "28450913", "70915533", "48221067",
    "95513380", "36628401", "81190244", "55372918",
};

static void assert_needles_match_golden(const sg_bench_needle *n, uint32_t count,
                                         const char *label) {
    tt_assert(count == 8, "%s: expected 8 needles, got %u", label, count);
    uint32_t check_n = count < 8 ? count : 8;
    for (uint32_t i = 0; i < check_n; i++) {
        tt_assert(strcmp(n[i].city, NEEDLE_CITY[i]) == 0,
                  "%s: needle[%u].city=%s want=%s", label, i, n[i].city, NEEDLE_CITY[i]);
        tt_assert(strcmp(n[i].code, NEEDLE_CODE[i]) == 0,
                  "%s: needle[%u].code=%s want=%s", label, i, n[i].code, NEEDLE_CODE[i]);
    }
}

/* --------------------------------------------------------------------
 * (i) extraction, part 1: the real file, skipped with a notice if the
 * file is not present on this machine.
 * -------------------------------------------------------------------- */
#define NIAH_PROMPT_PATH "/Users/macmini/models/niah_256k_prompt.txt"

static void test_extract_real_file(void) {
    struct stat st;
    if (stat(NIAH_PROMPT_PATH, &st) != 0) {
        fprintf(stderr,
                "NOTICE: %s not found on disk; skipping the real-file extraction "
                "assertion\n", NIAH_PROMPT_PATH);
        return;
    }

    char *buf = NULL;
    size_t len = 0;
    sg_err e = sg_bench_read_file(NIAH_PROMPT_PATH, &buf, &len);
    tt_assert(!sg_failed(e), "reading the real NIAH prompt file should succeed: %s",
              sg_failed(e) ? e.msg : "");
    if (sg_failed(e)) return;

    sg_bench_needle needles[SG_BENCH_MAX_NEEDLES];
    uint32_t n_out = 0;
    sg_err e2 = sg_bench_extract_needles(buf, needles, SG_BENCH_MAX_NEEDLES, &n_out);
    tt_assert(!sg_failed(e2), "extract_needles on the real file should succeed: %s",
              sg_failed(e2) ? e2.msg : "");
    if (!sg_failed(e2)) assert_needles_match_golden(needles, n_out, "real file");

    free(buf);
}

/* --------------------------------------------------------------------
 * (i) extraction, part 2: a synthetic haystack built here, with filler
 * text, a filler 8-digit number that is not a needle code, and a
 * trailing NIAH-style question line -- none of which must produce a
 * false needle.
 * -------------------------------------------------------------------- */
static size_t build_synthetic_haystack(char *buf, size_t cap) {
    size_t off = 0;
    int n;

    n = snprintf(buf + off, cap - off,
        "The quarterly logistics review covered warehouse throughput, fleet "
        "routing, cold-chain compliance, and vendor lead times. ");
    off += (size_t)n;

    n = snprintf(buf + off, cap - off,
        "A filler 8-digit number appears here: 99999999, unrelated to any city. ");
    off += (size_t)n;

    for (int i = 0; i < 8; i++) {
        n = snprintf(buf + off, cap - off,
            "IMPORTANT RECORD: the secret access code for %s is %s.\n",
            NEEDLE_CITY[i], NEEDLE_CODE[i]);
        off += (size_t)n;
        n = snprintf(buf + off, cap - off,
            "More filler text between records, nothing to see here. ");
        off += (size_t)n;
    }

    n = snprintf(buf + off, cap - off,
        "Extract the secret access code for each of these cities from the text "
        "above: Reykjavik, Ouagadougou, Valparaiso, Nakhodka, Timbuktu, Kirkwall, "
        "Ushuaia, Yakutsk. List each city and its code.\n");
    off += (size_t)n;

    return off;
}

static void test_extract_synthetic(void) {
    char *buf = malloc(8192);
    tt_assert(buf != NULL, "synthetic buffer alloc should succeed");
    if (!buf) return;
    build_synthetic_haystack(buf, 8192);

    sg_bench_needle needles[SG_BENCH_MAX_NEEDLES];
    uint32_t n_out = 0;
    sg_err e = sg_bench_extract_needles(buf, needles, SG_BENCH_MAX_NEEDLES, &n_out);
    tt_assert(!sg_failed(e), "extract_needles on synthetic haystack should succeed");
    if (!sg_failed(e)) assert_needles_match_golden(needles, n_out, "synthetic");

    free(buf);
}

/* --------------------------------------------------------------------
 * extract_needles: near-miss text must never produce a false needle:
 * missing trailing '.', missing "IMPORTANT RECORD: " prefix, an empty
 * city, a bare question line, a lowercase-first "city", and a code
 * shorter than the project's 8-digit minimum.
 * -------------------------------------------------------------------- */
static void test_extract_rejects_malformed(void) {
    const char *text =
        "IMPORTANT RECORD: the secret access code for Springfield is unknown.\n"
        "IMPORTANT RECORD: the secret access code for Shelbyville is 42\n"
        "the secret access code for Ogdenville is 87654321.\n"
        "Extract the secret access code for each of these cities from the text "
        "above: Springfield, Shelbyville, Ogdenville.\n"
        "IMPORTANT RECORD: the secret access code for  is 11112222.\n"
        "IMPORTANT RECORD: the secret access code for boston is 12345678.\n"
        "IMPORTANT RECORD: the secret access code for Ames is 1234567.\n";

    sg_bench_needle needles[SG_BENCH_MAX_NEEDLES];
    uint32_t n_out = 999;
    sg_err e = sg_bench_extract_needles(text, needles, SG_BENCH_MAX_NEEDLES, &n_out);
    tt_assert(!sg_failed(e), "extract_needles on malformed text should still succeed (0 matches)");
    tt_assert(n_out == 0, "malformed/near-miss text must yield 0 needles, got %u", n_out);
}

/* --------------------------------------------------------------------
 * extract_needles: the fixed-cap array truncates cleanly (no overflow,
 * no crash) and still returns the first `cap` pairs correctly.
 * -------------------------------------------------------------------- */
static void test_extract_cap_truncates(void) {
    char *buf = malloc(8192);
    tt_assert(buf != NULL, "synthetic buffer alloc should succeed");
    if (!buf) return;
    build_synthetic_haystack(buf, 8192);

    sg_bench_needle needles[3];
    uint32_t n_out = 999;
    sg_err e = sg_bench_extract_needles(buf, needles, 3, &n_out);
    tt_assert(!sg_failed(e), "extract_needles with a small cap should still succeed");
    tt_assert(n_out == 3, "cap=3 should yield exactly 3 needles, got %u", n_out);
    for (uint32_t i = 0; i < 3 && i < n_out; i++) {
        tt_assert(strcmp(needles[i].city, NEEDLE_CITY[i]) == 0,
                  "cap-truncated needle[%u].city=%s want=%s", i, needles[i].city, NEEDLE_CITY[i]);
        tt_assert(strcmp(needles[i].code, NEEDLE_CODE[i]) == 0,
                  "cap-truncated needle[%u].code=%s want=%s", i, needles[i].code, NEEDLE_CODE[i]);
    }
    free(buf);
}

/* --------------------------------------------------------------------
 * extract_needles: a malformed anchor occurrence directly abutting (no
 * separating text) a well-formed one must not cause the well-formed one
 * to be skipped. The malformed candidate's "city" scan would otherwise
 * swallow the literal "IMPORTANT" of the next occurrence if scanning
 * jumped past the whole failed match instead of stepping one byte at a
 * time; this is the regression case for that bug.
 * -------------------------------------------------------------------- */
static void test_extract_overlapping_anchor_not_skipped(void) {
    const char *text =
        "IMPORTANT RECORD: the secret access code for BadCityNoSpaceHereIMPORTANT "
        "RECORD: the secret access code for Reykjavik is 13072624.\n";

    sg_bench_needle needles[SG_BENCH_MAX_NEEDLES];
    uint32_t n_out = 999;
    sg_err e = sg_bench_extract_needles(text, needles, SG_BENCH_MAX_NEEDLES, &n_out);
    tt_assert(!sg_failed(e), "extract_needles on overlapping-anchor text should succeed");
    tt_assert(n_out == 1, "overlapping-anchor text should still yield the well-formed pair, got %u",
              n_out);
    if (n_out == 1) {
        tt_assert(strcmp(needles[0].city, "Reykjavik") == 0,
                  "overlapping-anchor city=%s want=Reykjavik", needles[0].city);
        tt_assert(strcmp(needles[0].code, "13072624") == 0,
                  "overlapping-anchor code=%s want=13072624", needles[0].code);
    }
}

/* --------------------------------------------------------------------
 * The current needle set (used to build shuffled/partial/filler test
 * fixtures below) must have no city-vs-city or code-vs-code substring
 * collisions: test_score_shuffled relies on a needle's city never being
 * found on a line via a DIFFERENT needle's city substring, and
 * test_score_filler_no_inflate relies on the filler code never being a
 * substring of (or containing as a substring) a real code. This guard
 * catches a silent break of that assumption if the needle set ever
 * changes.
 * -------------------------------------------------------------------- */
static void test_needle_set_has_no_substring_collisions(void) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i == j) continue;
            tt_assert(strstr(NEEDLE_CITY[j], NEEDLE_CITY[i]) == NULL,
                      "city collision: '%s' is a substring of '%s'",
                      NEEDLE_CITY[i], NEEDLE_CITY[j]);
            tt_assert(strstr(NEEDLE_CODE[j], NEEDLE_CODE[i]) == NULL,
                      "code collision: '%s' is a substring of '%s'",
                      NEEDLE_CODE[i], NEEDLE_CODE[j]);
        }
    }
}

/* --------------------------------------------------------------------
 * extract_needles: invalid arguments fail cleanly, never crash.
 * -------------------------------------------------------------------- */
static void test_extract_invalid_args(void) {
    sg_bench_needle needles[4];
    uint32_t n_out = 999;

    sg_err e1 = sg_bench_extract_needles(NULL, needles, 4, &n_out);
    tt_assert(sg_failed(e1), "NULL prompt must fail");
    tt_assert(n_out == 0, "n_out must be reset to 0 on failure");

    n_out = 999;
    sg_err e2 = sg_bench_extract_needles("text", NULL, 4, &n_out);
    tt_assert(sg_failed(e2), "NULL out must fail");

    n_out = 999;
    sg_err e3 = sg_bench_extract_needles("text", needles, 0, &n_out);
    tt_assert(sg_failed(e3), "cap==0 must fail");

    sg_err e4 = sg_bench_extract_needles("text", needles, 4, NULL);
    tt_assert(sg_failed(e4), "NULL n_out must fail cleanly, not crash");
}

/* --------------------------------------------------------------------
 * Scoring tests. A tiny local pair type keeps the golden/shuffled/
 * partial generators free of pointer-qualifier gymnastics.
 * -------------------------------------------------------------------- */
typedef struct { const char *city; const char *code; } line_pair;

static void build_lines(char *buf, size_t cap, const line_pair *pairs, int n) {
    size_t off = 0;
    for (int i = 0; i < n && off < cap; i++) {
        int w = snprintf(buf + off, cap - off, "%s: %s\n", pairs[i].city, pairs[i].code);
        if (w > 0) off += (size_t)w;
    }
}

static void fill_golden_needles(sg_bench_needle needles[8]) {
    for (int i = 0; i < 8; i++) {
        snprintf(needles[i].city, sizeof needles[i].city, "%s", NEEDLE_CITY[i]);
        snprintf(needles[i].code, sizeof needles[i].code, "%s", NEEDLE_CODE[i]);
    }
}

/* (ii) golden correct answer: retrieval 8/8, assoc 8/8. */
static void test_score_golden(void) {
    sg_bench_needle needles[8];
    fill_golden_needles(needles);

    line_pair golden[8];
    for (int i = 0; i < 8; i++) { golden[i].city = NEEDLE_CITY[i]; golden[i].code = NEEDLE_CODE[i]; }

    char gen[2048];
    build_lines(gen, sizeof gen, golden, 8);

    uint32_t retrieval = 0, assoc = 0;
    sg_bench_score_niah(gen, needles, 8, &retrieval, &assoc);
    tt_assert(retrieval == 8, "golden retrieval=%u want 8", retrieval);
    tt_assert(assoc == 8, "golden assoc=%u want 8", assoc);
}

/* (iii) codes right, cities shuffled (rotate by 1, so no line pairs a
 * code with its own city): retrieval 8/8, assoc 0/8. */
static void test_score_shuffled(void) {
    sg_bench_needle needles[8];
    fill_golden_needles(needles);

    line_pair shuffled[8];
    for (int i = 0; i < 8; i++) {
        shuffled[i].city = NEEDLE_CITY[(i + 1) % 8];   /* wrong city for this code */
        shuffled[i].code = NEEDLE_CODE[i];
    }

    char gen[2048];
    build_lines(gen, sizeof gen, shuffled, 8);

    uint32_t retrieval = 0, assoc = 0;
    sg_bench_score_niah(gen, needles, 8, &retrieval, &assoc);
    tt_assert(retrieval == 8, "shuffled retrieval=%u want 8 (codes are all still present)",
              retrieval);
    tt_assert(assoc == 0, "shuffled assoc=%u want 0 (no code shares a line with its city)",
              assoc);
}

/* (iv) 3-of-8 answer: retrieval 3/8 (also checks assoc==3, since the 3
 * present are correctly paired -- stricter than the gate requires, kept
 * as extra signal). */
static void test_score_partial_3of8(void) {
    sg_bench_needle needles[8];
    fill_golden_needles(needles);

    line_pair partial[3];
    for (int i = 0; i < 3; i++) { partial[i].city = NEEDLE_CITY[i]; partial[i].code = NEEDLE_CODE[i]; }

    char gen[512];
    build_lines(gen, sizeof gen, partial, 3);
    strncat(gen, "The model was unsure about the remaining cities.\n",
            sizeof gen - strlen(gen) - 1);

    uint32_t retrieval = 0, assoc = 0;
    sg_bench_score_niah(gen, needles, 8, &retrieval, &assoc);
    tt_assert(retrieval == 3, "3-of-8 retrieval=%u want 3", retrieval);
    tt_assert(assoc == 3, "3-of-8 assoc=%u want 3 (the 3 present are correctly paired)", assoc);
}

/* (v) a filler 8-digit number that is not any needle's code must not
 * inflate retrieval beyond the actually-present pairs. */
static void test_score_filler_no_inflate(void) {
    sg_bench_needle needles[8];
    fill_golden_needles(needles);

    const char *filler_code = "97531246";
    for (int i = 0; i < 8; i++) {
        tt_assert(strcmp(filler_code, NEEDLE_CODE[i]) != 0,
                  "test bug: filler code collides with a real needle code");
    }

    line_pair some[4];
    for (int i = 0; i < 4; i++) { some[i].city = NEEDLE_CITY[i]; some[i].code = NEEDLE_CODE[i]; }

    char gen[512];
    build_lines(gen, sizeof gen, some, 4);
    char filler_line[96];
    snprintf(filler_line, sizeof filler_line,
             "Also noted: %s, an unrelated filler number.\n", filler_code);
    strncat(gen, filler_line, sizeof gen - strlen(gen) - 1);

    uint32_t retrieval = 0, assoc = 0;
    sg_bench_score_niah(gen, needles, 8, &retrieval, &assoc);
    tt_assert(retrieval == 4,
              "filler-present retrieval=%u want 4 (filler code must not inflate)", retrieval);
}

/* (vi) codes with adjacent punctuation still match, for both retrieval
 * and association. */
static void test_score_punctuation_adjacent(void) {
    sg_bench_needle needles[2];
    snprintf(needles[0].city, sizeof needles[0].city, "Reykjavik");
    snprintf(needles[0].code, sizeof needles[0].code, "13072624");
    snprintf(needles[1].city, sizeof needles[1].city, "Ouagadougou");
    snprintf(needles[1].code, sizeof needles[1].code, "28450913");

    const char *gen =
        "The code for Reykjavik is (13072624).\n"
        "Ouagadougou secret code: 28450913,\n";

    uint32_t retrieval = 0, assoc = 0;
    sg_bench_score_niah(gen, needles, 2, &retrieval, &assoc);
    tt_assert(retrieval == 2, "punctuation-adjacent retrieval=%u want 2", retrieval);
    tt_assert(assoc == 2, "punctuation-adjacent assoc=%u want 2", assoc);
}

/* --------------------------------------------------------------------
 * score_niah: invalid/degenerate arguments fail cleanly, never crash.
 * -------------------------------------------------------------------- */
static void test_score_invalid_args(void) {
    sg_bench_needle needles[1];
    snprintf(needles[0].city, sizeof needles[0].city, "Reykjavik");
    snprintf(needles[0].code, sizeof needles[0].code, "13072624");

    uint32_t retrieval = 99, assoc = 99;
    sg_bench_score_niah(NULL, needles, 1, &retrieval, &assoc);
    tt_assert(retrieval == 0 && assoc == 0, "NULL gen must reset hits to 0, not crash");

    retrieval = 99; assoc = 99;
    sg_bench_score_niah("text", NULL, 1, &retrieval, &assoc);
    tt_assert(retrieval == 0 && assoc == 0, "NULL needles must reset hits to 0, not crash");

    retrieval = 99; assoc = 99;
    sg_bench_score_niah("text", needles, 0, &retrieval, &assoc);
    tt_assert(retrieval == 0 && assoc == 0, "n_needles==0 must reset hits to 0");

    /* NULL out-params must not crash. */
    sg_bench_score_niah("Reykjavik 13072624", needles, 1, NULL, NULL);

    /* Empty gen string. */
    retrieval = 99; assoc = 99;
    sg_bench_score_niah("", needles, 1, &retrieval, &assoc);
    tt_assert(retrieval == 0 && assoc == 0, "empty gen must yield 0/0");
}

int main(void) {
    tt_run("extract_needles: real niah_256k_prompt.txt -> exactly the 8 known pairs",
           test_extract_real_file);
    tt_run("extract_needles: synthetic haystack -> exactly the 8 known pairs",
           test_extract_synthetic);
    tt_run("extract_needles: malformed/near-miss text -> 0 needles",
           test_extract_rejects_malformed);
    tt_run("extract_needles: small cap truncates cleanly", test_extract_cap_truncates);
    tt_run("extract_needles: overlapping anchor is not skipped",
           test_extract_overlapping_anchor_not_skipped);
    tt_run("needle set: no city/code substring collisions",
           test_needle_set_has_no_substring_collisions);
    tt_run("extract_needles: invalid arguments fail cleanly", test_extract_invalid_args);
    tt_run("score_niah: golden answer -> retrieval 8/8, assoc 8/8", test_score_golden);
    tt_run("score_niah: shuffled cities -> retrieval 8/8, assoc 0/8", test_score_shuffled);
    tt_run("score_niah: 3-of-8 answer -> retrieval 3/8", test_score_partial_3of8);
    tt_run("score_niah: filler 8-digit number does not inflate retrieval",
           test_score_filler_no_inflate);
    tt_run("score_niah: adjacent punctuation still matches", test_score_punctuation_adjacent);
    tt_run("score_niah: invalid/degenerate arguments fail cleanly", test_score_invalid_args);
    return tt_report();
}
