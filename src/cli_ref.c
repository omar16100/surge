/* cli_ref.c - `surge-ref`, the reference forward pass on the command line.
 *
 *   surge-ref <model_dir_or_gguf> [-p "prompt" | --ids 1,2,3] [-n N]
 *             [--logits out.f32] [--max-ctx N] [--quiet]
 *
 * Two jobs, and the second one is the reason this binary exists:
 *
 * 1. Greedy decode N tokens and print them, as a human-readable smoke test.
 * 2. `--logits` dumps the logits of EVERY prompt position as raw
 *    little-endian f32, [n_prompt_positions, vocab], which is what
 *    tools/tf_compare.py compares against mlx-lm for the M1 gate. That is a
 *    TEACHER-FORCED dump: every position is conditioned on the given token
 *    ids, never on what the model would itself have produced. Free-running
 *    comparison is not an acceptable substitute -- this model class diverges
 *    across runs at depth, so a free-running mismatch says nothing.
 *
 * Tokenizer availability decides how the prompt is supplied. A GGUF carries
 * its own tokenizer metadata, so -p works. A safetensors directory does not
 * (surge's tok.c is built from tokenizer.ggml.* keys), so a safetensors
 * model needs --ids; tools/tf_compare.py produces them with the model's own
 * HF tokenizer and passes the identical list to both sides.
 *
 * Expect minutes, not milliseconds: this is a scalar C forward with double
 * accumulators and no blocking, roughly 1.4 s per position on a 2B. That is
 * the intent (Task 9 owns speed), and it is why tf_compare.py runs the 16
 * M1 prompts as concurrent processes rather than in one loop.
 */
#include "surge.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void usage(void) {
    fprintf(stderr,
            "usage: surge-ref <model_dir_or_gguf> [-p TEXT | --ids 1,2,3] [-n N]\n"
            "                 [--logits OUT.f32] [--max-ctx N] [--quiet]\n"
            "\n"
            "  -p TEXT       prompt text; needs a GGUF (safetensors dirs carry no\n"
            "                tokenizer surge can read, so use --ids there)\n"
            "  --ids LIST    comma-separated token ids, teacher-forced\n"
            "  -n N          greedy-decode N tokens after the prompt (default 0)\n"
            "  --logits P    write every prompt position's logits to P as raw f32\n"
            "                ([positions, vocab], little-endian)\n"
            "  --max-ctx N   cache size; defaults to prompt length + N\n"
            "  --quiet       suppress the progress line on stderr\n");
}

/* Parses "1,2,3" into a freshly allocated array. Rejects anything that is
 * not a clean list of non-negative integers rather than silently stopping
 * at the first bad character, because a truncated prompt would make the M1
 * comparison quietly compare different sequences. */
static int32_t *parse_ids(const char *s, uint64_t *n_out) {
    uint64_t cap = 16, n = 0;
    int32_t *ids = malloc(cap * sizeof *ids);
    if (!ids) return NULL;
    const char *p = s;
    for (;;) {
        while (*p == ' ' || *p == '\t') p++;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) { free(ids); return NULL; }
        if (n == cap) {
            cap *= 2;
            int32_t *g = realloc(ids, cap * sizeof *ids);
            if (!g) { free(ids); return NULL; }
            ids = g;
        }
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') { p++; continue; }
        /* Trailing whitespace (including ONE final newline) is fine; an
         * embedded newline is not. `--ids "$(cat ids.txt)"` on a multi-line
         * file must be an error, not a silent comparison of only the first
         * line -- that is exactly the "quietly compare different sequences"
         * failure this parser exists to prevent. */
        unsigned newlines = 0;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
            if (*p == '\n') newlines++;
            p++;
        }
        if (*p == '\0' && newlines <= 1) break;
        free(ids);
        return NULL;
    }
    *n_out = n;
    return ids;
}

static bool ends_with(const char *s, const char *suffix) {
    size_t ls = strlen(s), lx = strlen(suffix);
    return ls >= lx && strcmp(s + ls - lx, suffix) == 0;
}

static void print_ids(const sg_tok *tok, const int32_t *ids, uint64_t n) {
    if (tok) {
        char buf[8192];
        int64_t got = sg_tok_decode(tok, ids, n, buf, sizeof buf - 1);
        if (got >= 0) {
            buf[got] = '\0';
            fputs(buf, stdout);
            return;
        }
    }
    for (uint64_t i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    const char *path = argv[1];
    const char *prompt = NULL, *ids_arg = NULL, *logits_path = NULL;
    uint32_t n_gen = 0, max_ctx_arg = 0;
    bool quiet = false;

    /* A missing value must say so rather than fall through to "unknown
     * argument", and a non-numeric count must be an error rather than
     * silently parse to 0 and look like "not supplied". */
#define NEED_VALUE(flag) do { \
        if (i + 1 >= argc) { \
            fprintf(stderr, "surge-ref: %s needs a value\n", (flag)); \
            return 2; \
        } \
    } while (0)
#define PARSE_U32(flag, dst) do { \
        NEED_VALUE(flag); \
        char *_end = NULL; \
        unsigned long _v = strtoul(argv[++i], &_end, 10); \
        if (!_end || *_end != '\0' || argv[i][0] == '\0' || _v > UINT32_MAX) { \
            fprintf(stderr, "surge-ref: %s expects a non-negative integer, got '%s'\n", \
                    (flag), argv[i]); \
            return 2; \
        } \
        (dst) = (uint32_t)_v; \
    } while (0)

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0) { NEED_VALUE("-p"); prompt = argv[++i]; }
        else if (strcmp(argv[i], "--ids") == 0) { NEED_VALUE("--ids"); ids_arg = argv[++i]; }
        else if (strcmp(argv[i], "-n") == 0) PARSE_U32("-n", n_gen);
        else if (strcmp(argv[i], "--logits") == 0) { NEED_VALUE("--logits"); logits_path = argv[++i]; }
        else if (strcmp(argv[i], "--max-ctx") == 0) PARSE_U32("--max-ctx", max_ctx_arg);
        else if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else { fprintf(stderr, "surge-ref: unknown argument '%s'\n", argv[i]); usage(); return 2; }
    }
#undef NEED_VALUE
#undef PARSE_U32
    if ((prompt == NULL) == (ids_arg == NULL)) {
        fprintf(stderr, "surge-ref: pass exactly one of -p and --ids\n");
        return 2;
    }

    sg_gguf *g = NULL;
    sg_st *s = NULL;
    sg_tok *tok = NULL;
    sg_model m;
    sg_err e;

    if (ends_with(path, ".gguf")) {
        e = sg_gguf_open(path, &g);
        if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); return 1; }
        e = sg_model_from_gguf(g, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); sg_gguf_close(g); return 1; }
        if (sg_failed(sg_tok_from_gguf(g, &tok))) tok = NULL;
    } else {
        e = sg_st_open(path, &s);
        if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); return 1; }
        e = sg_model_from_st(s, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); sg_st_close(s); return 1; }
    }

    int rc = 1;
    int32_t *ids = NULL;
    uint64_t n_ids = 0;
    sg_ref_state *st = NULL;
    FILE *lf = NULL;

    if (prompt) {
        if (!tok) {
            fprintf(stderr, "surge-ref: this model carries no tokenizer surge can read; "
                            "use --ids (tools/tf_compare.py generates them)\n");
            goto done;
        }
        e = sg_tok_encode(tok, prompt, &ids, &n_ids);
        if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); goto done; }
    } else {
        ids = parse_ids(ids_arg, &n_ids);
        if (!ids) { fprintf(stderr, "surge-ref: --ids must be a comma-separated list of "
                                    "non-negative integers\n"); goto done; }
    }
    if (n_ids == 0) { fprintf(stderr, "surge-ref: the prompt is empty\n"); goto done; }
    if (n_ids + n_gen > 1000000u) {
        fprintf(stderr, "surge-ref: %llu prompt + %u generated tokens is beyond "
                        "anything this scalar reference should be asked to do\n",
                (unsigned long long)n_ids, n_gen);
        goto done;
    }

    uint32_t max_ctx = max_ctx_arg ? max_ctx_arg : (uint32_t)(n_ids + n_gen);
    if ((uint64_t)max_ctx < n_ids + n_gen) max_ctx = (uint32_t)(n_ids + n_gen);
    e = sg_ref_state_new(&m, max_ctx, &st);
    if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); goto done; }

    if (logits_path) {
        lf = fopen(logits_path, "wb");
        if (!lf) { fprintf(stderr, "surge-ref: cannot write %s\n", logits_path); goto done; }
    }

    if (!quiet) {
        fprintf(stderr, "surge-ref: %s, %u layers, vocab %u, %llu prompt tokens\n",
                path, m.cfg.n_layers, m.cfg.vocab, (unsigned long long)n_ids);
    }

    double t0 = now_s();
    const float *lg = NULL;
    int32_t next = 0;
    for (uint64_t t = 0; t < n_ids; t++) {
        e = sg_ref_forward(st, &m, ids[t], (uint32_t)t, &lg);
        if (sg_failed(e)) {
            fprintf(stderr, "surge-ref: %s\n", e.msg);
            if (lf) { fclose(lf); lf = NULL; remove(logits_path); }
            goto done;
        }
        if (lf && fwrite(lg, sizeof(float), m.cfg.vocab, lf) != m.cfg.vocab) {
            fprintf(stderr, "surge-ref: short write to %s\n", logits_path);
            /* Leave no partial dump behind: tools/tf_compare.py's --reuse
             * would otherwise happily read it as a complete one. */
            fclose(lf); lf = NULL; remove(logits_path);
            goto done;
        }
        if (!quiet) {
            fprintf(stderr, "\r  prompt %llu/%llu (%.2f s/pos)",
                    (unsigned long long)(t + 1), (unsigned long long)n_ids,
                    (now_s() - t0) / (double)(t + 1));
        }
    }
    if (!quiet) fprintf(stderr, "\n");
    if (lf) {
        if (fclose(lf) != 0) { lf = NULL; fprintf(stderr, "surge-ref: close failed\n"); goto done; }
        lf = NULL;
        if (!quiet) {
            fprintf(stderr, "surge-ref: wrote %llu x %u f32 to %s\n",
                    (unsigned long long)n_ids, m.cfg.vocab, logits_path);
        }
    }

    if (n_gen > 0) {
        int32_t *gen = malloc((size_t)n_gen * sizeof *gen);
        if (!gen) { fprintf(stderr, "surge-ref: out of memory\n"); goto done; }
        uint32_t produced = 0;
        /* lg currently holds the logits of the LAST prompt position, which
         * predict the token at position n_ids. Step i emits that token and
         * feeds it back at position n_ids + i. */
        for (uint32_t i = 0; i < n_gen; i++) {
            uint32_t arg = 0;
            for (uint32_t v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[arg]) arg = v;
            next = (int32_t)arg;
            gen[produced++] = next;
            if (tok && next == sg_tok_eos(tok)) break;
            if (i + 1 == n_gen) break;                  /* no need for its logits */
            if ((uint64_t)n_ids + i >= max_ctx) break;
            e = sg_ref_forward(st, &m, next, (uint32_t)(n_ids + i), &lg);
            if (sg_failed(e)) { fprintf(stderr, "surge-ref: %s\n", e.msg); free(gen); goto done; }
            if (!quiet) fprintf(stderr, "\r  generated %u/%u", i + 1, n_gen);
        }
        if (!quiet) fprintf(stderr, "\n");
        printf("prompt: ");
        print_ids(tok, ids, n_ids);
        printf("\ncompletion: ");
        print_ids(tok, gen, produced);
        printf("\n");
        free(gen);
    }

    rc = 0;
done:
    if (lf) fclose(lf);
    free(ids);
    sg_ref_state_free(st);
    sg_model_free(&m);
    if (tok) sg_tok_free(tok);
    if (s) sg_st_close(s);
    if (g) sg_gguf_close(g);
    return rc;
}
