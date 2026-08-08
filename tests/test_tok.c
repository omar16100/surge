/* test_tok.c - fixture-exact test for the byte-level BPE tokenizer (tok.c).
 *
 * Loads the real Qwen3.6-27B GGUF (path from env SURGE_GGUF) and asserts that
 * every case in tests/fixtures/tok_cases.jsonl (produced by
 * tools/make_fixtures.py write_tok_fixture() from the real HF tokenizer)
 * encodes to the exact id sequence and decodes back to the exact bytes.
 *
 * Auto-skips with a notice when SURGE_GGUF is unset, so plain `make check`
 * stays green without the multi-GB model file on disk.
 */
#include "tinytest.h"
#include "../surge.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *text;
    size_t text_len;
    int32_t *ids;
    uint64_t n_ids;
} tok_case;

/* ---- minimal hand-rolled JSON reader for the fixed {"text":..,"ids":[..]}
 * line shape written by write_tok_fixture(). No general JSON needed. ---- */

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static void push_byte(char **buf, size_t *len, size_t *cap, char c) {
    if (*len + 1 > *cap) {
        *cap *= 2;
        *buf = realloc(*buf, *cap);
    }
    (*buf)[(*len)++] = c;
}

static void append_utf8(char **buf, size_t *len, size_t *cap, uint32_t cp) {
    unsigned char tmp[4];
    int n;
    if (cp < 0x80) {
        tmp[0] = (unsigned char)cp; n = 1;
    } else if (cp < 0x800) {
        tmp[0] = (unsigned char)(0xC0 | (cp >> 6));
        tmp[1] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        tmp[0] = (unsigned char)(0xE0 | (cp >> 12));
        tmp[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        tmp[0] = (unsigned char)(0xF0 | (cp >> 18));
        tmp[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = (unsigned char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    for (int k = 0; k < n; k++) push_byte(buf, len, cap, (char)tmp[k]);
}

/* s must point at the opening '"'. Decodes JSON string escapes; non-escaped
 * bytes (including multi-byte UTF-8, since the fixture is written with
 * ensure_ascii=False) pass through verbatim. Returns pointer just past the
 * closing '"'. */
static const char *parse_json_string(const char *s, char **out_buf, size_t *out_len) {
    size_t cap = 64, len = 0;
    char *buf = malloc(cap);
    s++; /* opening quote */
    while (*s != '"') {
        if (*s == '\\') {
            s++;
            switch (*s) {
                case '"': push_byte(&buf, &len, &cap, '"'); s++; break;
                case '\\': push_byte(&buf, &len, &cap, '\\'); s++; break;
                case '/': push_byte(&buf, &len, &cap, '/'); s++; break;
                case 'b': push_byte(&buf, &len, &cap, '\b'); s++; break;
                case 'f': push_byte(&buf, &len, &cap, '\f'); s++; break;
                case 'n': push_byte(&buf, &len, &cap, '\n'); s++; break;
                case 'r': push_byte(&buf, &len, &cap, '\r'); s++; break;
                case 't': push_byte(&buf, &len, &cap, '\t'); s++; break;
                case 'u': {
                    s++;
                    uint32_t cp = 0;
                    for (int k = 0; k < 4; k++) cp = cp * 16 + (uint32_t)hex_val(s[k]);
                    s += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && s[0] == '\\' && s[1] == 'u') {
                        uint32_t lo = 0;
                        for (int k = 0; k < 4; k++) lo = lo * 16 + (uint32_t)hex_val(s[2 + k]);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            s += 6;
                        }
                    }
                    append_utf8(&buf, &len, &cap, cp);
                    break;
                }
                default: s++; break;
            }
        } else {
            push_byte(&buf, &len, &cap, *s);
            s++;
        }
    }
    s++; /* closing quote */
    push_byte(&buf, &len, &cap, '\0');
    len--; /* don't count the NUL in the reported length */
    *out_buf = buf;
    *out_len = len;
    return s;
}

/* s must point at '['. Parses comma-separated (possibly negative) integers
 * up to ']'. */
static const char *parse_json_int_array(const char *s, int32_t **out_ids, uint64_t *out_n) {
    size_t cap = 8, n = 0;
    int32_t *ids = malloc(cap * sizeof(*ids));
    s++; /* '[' */
    while (1) {
        while (*s == ' ' || *s == '\t') s++;
        if (*s == ']') { s++; break; }
        char *end;
        long v = strtol(s, &end, 10);
        if (n == cap) {
            cap *= 2;
            ids = realloc(ids, cap * sizeof(*ids));
        }
        ids[n++] = (int32_t)v;
        s = end;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == ',') s++;
    }
    *out_ids = ids;
    *out_n = n;
    return s;
}

static bool parse_fixture_line(const char *line, tok_case *out) {
    const char *p = strstr(line, "\"text\"");
    if (!p) return false;
    p = strchr(p, ':');
    if (!p) return false;
    p++;
    while (*p == ' ') p++;
    char *text; size_t text_len;
    p = parse_json_string(p, &text, &text_len);

    p = strstr(p, "\"ids\"");
    if (!p) { free(text); return false; }
    p = strchr(p, ':');
    if (!p) { free(text); return false; }
    p++;
    while (*p == ' ') p++;
    int32_t *ids; uint64_t n_ids;
    parse_json_int_array(p, &ids, &n_ids);

    out->text = text;
    out->text_len = text_len;
    out->ids = ids;
    out->n_ids = n_ids;
    return true;
}

static void tok_fixtures(void) {
    const char *gguf_path = getenv("SURGE_GGUF");
    if (!gguf_path || !*gguf_path) {
        fprintf(stderr, "SKIP: SURGE_GGUF not set; skipping tok fixture test "
                        "(set it to a GGUF with tokenizer.ggml.* metadata)\n");
        return;
    }

    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open(gguf_path, &g);
    tt_assert(!sg_failed(e), "open %s should succeed: %s", gguf_path, e.msg ? e.msg : "");
    if (!g) return;

    sg_tok *t = NULL;
    e = sg_tok_from_gguf(g, &t);
    tt_assert(!sg_failed(e), "sg_tok_from_gguf should succeed: %s", e.msg ? e.msg : "");
    if (!t) { sg_gguf_close(g); return; }

    int32_t eos = sg_tok_eos(t);
    tt_assert(eos >= 0, "sg_tok_eos should return a valid token id, got %d", eos);

    FILE *f = fopen("tests/fixtures/tok_cases.jsonl", "r");
    tt_assert(f != NULL, "tok_cases.jsonl should be readable");
    if (!f) { sg_tok_free(t); sg_gguf_close(g); return; }

    char line[16384];
    int case_no = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        case_no++;

        tok_case c;
        bool parsed = parse_fixture_line(line, &c);
        tt_assert(parsed, "case %d: fixture line should parse", case_no);
        if (!parsed) continue;

        int32_t *ids = NULL;
        uint64_t n_ids = 0;
        sg_err ee = sg_tok_encode(t, c.text, &ids, &n_ids);
        tt_assert(!sg_failed(ee), "case %d (%.40s): encode should succeed: %s",
                  case_no, c.text, ee.msg ? ee.msg : "");

        if (!sg_failed(ee)) {
            bool ids_match = (n_ids == c.n_ids);
            if (ids_match) {
                for (uint64_t i = 0; i < n_ids; i++) {
                    if (ids[i] != c.ids[i]) { ids_match = false; break; }
                }
            }
            tt_assert(ids_match, "case %d (%.40s): id sequence mismatch (got n=%llu want n=%llu)",
                      case_no, c.text, (unsigned long long)n_ids, (unsigned long long)c.n_ids);

            char buf[16384];
            int64_t written = sg_tok_decode(t, ids, n_ids, buf, sizeof(buf));
            tt_assert(written >= 0, "case %d (%.40s): decode should not overflow/fail",
                      case_no, c.text);
            if (written >= 0) {
                bool decode_match = ((uint64_t)written == c.text_len) &&
                                     (c.text_len == 0 || memcmp(buf, c.text, c.text_len) == 0);
                tt_assert(decode_match, "case %d (%.40s): decode should reproduce exact bytes",
                          case_no, c.text);
            }
        }

        free(ids);
        free(c.text);
        free(c.ids);
    }
    tt_assert(case_no == 24, "fixture file should have 24 cases, found %d", case_no);

    fclose(f);
    sg_tok_free(t);
    sg_gguf_close(g);
}

int main(void) {
    tt_run("tok_fixtures", tok_fixtures);
    return tt_report();
}
