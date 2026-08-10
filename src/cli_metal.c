/* cli_metal.c - `surge`, the Metal decode path on the command line.
 *
 *   surge <model_dir_or_gguf> [-p "prompt" | --ids 1,2,3] [-n N]
 *         [--ref] [--max-ctx N] [--logits OUT.f32] [--margins] [--quiet]
 *
 * Default is the Metal path (sg_gpu_load_model / sg_gpu_state_new /
 * sg_gpu_forward); `--ref` runs Task 8's scalar CPU forward instead. Both go
 * through the SAME driver loop and the SAME argmax below, which is the whole
 * point of putting them in one binary: the M2 gate is "the token ids are
 * equal", so anything other than the kernels differing between the two runs
 * would make the comparison meaningless.
 *
 * A GGUF carries its own tokenizer metadata, so -p works there. A
 * safetensors directory does not (surge's tok.c is built from
 * tokenizer.ggml.* keys), so those need --ids -- which is what the M2 gate
 * uses anyway, since it compares ids and not text.
 *
 * --logits writes one row of raw little-endian f32 logits per FORWARD PASS,
 * [positions, vocab]: the n prompt positions, then the n-1 feedback steps
 * that consume generated tokens 0..n-2. There is deliberately no row for the
 * last generated token, because nothing needs its logits and computing them
 * would cost a whole forward pass; the file therefore holds
 * n_prompt + n_gen - 1 rows, not n_prompt + n_gen. Row i's argmax is the
 * token at index i - n_prompt + 1 of gen_ids for i >= n_prompt. That is the
 * divergence-analysis tool: dump both paths, find the first row whose argmax
 * differs, and look at the two rows there.
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
            "usage: surge <model_dir_or_gguf> [-p TEXT | --ids 1,2,3] [-n N]\n"
            "             [--ref] [--max-ctx N] [--logits OUT.f32] [--margins]\n"
            "             [--quiet]\n"
            "\n"
            "  -p TEXT       prompt text; needs a GGUF (safetensors dirs carry no\n"
            "                tokenizer surge can read, so use --ids there)\n"
            "  --ids LIST    comma-separated token ids\n"
            "  -n N          greedy-decode N tokens after the prompt (default 0)\n"
            "  --ref         run the scalar CPU reference forward instead of Metal\n"
            "  --max-ctx N   cache size; defaults to prompt length + N\n"
            "  --logits P    write every position's logits to P as raw f32\n"
            "  --margins     print each generated token's top1-top2 logit gap\n"
            "  --quiet       suppress progress on stderr\n");
}

/* Parses "1,2,3" into a freshly allocated array. Rejects anything that is
 * not a clean list of non-negative integers rather than silently stopping at
 * the first bad character: a truncated prompt would make the two paths
 * compare different sequences and still look like a pass. */
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
            int32_t *gr = realloc(ids, cap * sizeof *ids);
            if (!gr) { free(ids); return NULL; }
            ids = gr;
        }
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') { p++; continue; }
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

/* THE argmax, used by both paths. Lowest index wins an exact tie, which is
 * the same rule surge-ref uses. Written once so the two paths cannot drift. */
static uint32_t argmax_f32(const float *v, uint32_t n) {
    uint32_t arg = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[arg]) arg = i;
    return arg;
}

/* The top-1 minus top-2 gap at this position. Printed under --margins
 * because it is the number that decides whether a token that DID diverge is
 * a real error or a legitimate near-tie: the two paths differ by ~1e-4 on a
 * 2B, so a gap far above that cannot flip and a gap below it can. */
static float top2_gap(const float *v, uint32_t n, uint32_t arg) {
    float second = -3.402823466e38f;
    for (uint32_t i = 0; i < n; i++) {
        if (i == arg) continue;
        if (v[i] > second) second = v[i];
    }
    return v[arg] - second;
}

static void print_ids(const int32_t *ids, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
}

static void print_text(const sg_tok *tok, const int32_t *ids, uint64_t n) {
    if (!tok) return;
    char buf[16384];
    int64_t got = sg_tok_decode(tok, ids, n, buf, sizeof buf - 1);
    if (got < 0) return;
    buf[got] = '\0';
    printf("text: %s\n", buf);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    const char *path = argv[1];
    const char *prompt = NULL, *ids_arg = NULL, *logits_path = NULL;
    uint32_t n_gen = 0, max_ctx_arg = 0;
    bool quiet = false, use_ref = false, margins = false;

#define NEED_VALUE(flag) do { \
        if (i + 1 >= argc) { \
            fprintf(stderr, "surge: %s needs a value\n", (flag)); \
            return 2; \
        } \
    } while (0)
#define PARSE_U32(flag, dst) do { \
        NEED_VALUE(flag); \
        char *_end = NULL; \
        unsigned long _v = strtoul(argv[++i], &_end, 10); \
        if (!_end || *_end != '\0' || argv[i][0] == '\0' || _v > UINT32_MAX) { \
            fprintf(stderr, "surge: %s expects a non-negative integer, got '%s'\n", \
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
        else if (strcmp(argv[i], "--ref") == 0) use_ref = true;
        else if (strcmp(argv[i], "--margins") == 0) margins = true;
        else if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else { fprintf(stderr, "surge: unknown argument '%s'\n", argv[i]); usage(); return 2; }
    }
#undef NEED_VALUE
#undef PARSE_U32
    if ((prompt == NULL) == (ids_arg == NULL)) {
        fprintf(stderr, "surge: pass exactly one of -p and --ids\n");
        return 2;
    }

    sg_gguf *gg = NULL;
    sg_st *st = NULL;
    sg_tok *tok = NULL;
    sg_model m;
    sg_err e;

    if (ends_with(path, ".gguf")) {
        e = sg_gguf_open(path, &gg);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); return 1; }
        e = sg_model_from_gguf(gg, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); sg_gguf_close(gg); return 1; }
        if (sg_failed(sg_tok_from_gguf(gg, &tok))) tok = NULL;
    } else {
        e = sg_st_open(path, &st);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); return 1; }
        e = sg_model_from_st(st, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); sg_st_close(st); return 1; }
    }

    int rc = 1;
    int32_t *ids = NULL, *gen = NULL;
    uint64_t n_ids = 0;
    sg_ref_state *rs = NULL;
    sg_gpu *gpu = NULL;
    FILE *lf = NULL;

    if (prompt) {
        if (!tok) {
            fprintf(stderr, "surge: this model carries no tokenizer surge can read; "
                            "use --ids\n");
            goto done;
        }
        e = sg_tok_encode(tok, prompt, &ids, &n_ids);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); goto done; }
    } else {
        ids = parse_ids(ids_arg, &n_ids);
        if (!ids) {
            fprintf(stderr, "surge: --ids must be a comma-separated list of "
                            "non-negative integers\n");
            goto done;
        }
    }
    if (n_ids == 0) { fprintf(stderr, "surge: the prompt is empty\n"); goto done; }
    if (n_ids + n_gen > 1000000u) {
        fprintf(stderr, "surge: %llu prompt + %u generated tokens is beyond what "
                        "this path is sized for\n", (unsigned long long)n_ids, n_gen);
        goto done;
    }

    /* max_position_embeddings is 262144 on this model family, so the cache is
     * sized from the ACTUAL run length, never from the config. */
    uint32_t max_ctx = max_ctx_arg ? max_ctx_arg : (uint32_t)(n_ids + n_gen);
    if ((uint64_t)max_ctx < n_ids + n_gen) max_ctx = (uint32_t)(n_ids + n_gen);

    if (use_ref) {
        e = sg_ref_state_new(&m, max_ctx, &rs);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); goto done; }
    } else {
        e = sg_gpu_init(&gpu);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); goto done; }
        e = sg_gpu_load_model(gpu, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); goto done; }
        e = sg_gpu_state_new(gpu, &m, max_ctx);
        if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); goto done; }
    }

    if (logits_path) {
        lf = fopen(logits_path, "wb");
        if (!lf) { fprintf(stderr, "surge: cannot write %s\n", logits_path); goto done; }
    }
    if (!quiet) {
        fprintf(stderr, "surge: %s [%s], %u layers, vocab %u, %llu prompt tokens\n",
                path, use_ref ? "ref" : "metal", m.cfg.n_layers, m.cfg.vocab,
                (unsigned long long)n_ids);
    }

#define FORWARD(tokid, pos, out) (use_ref ? sg_ref_forward(rs, &m, (tokid), (pos), (out)) \
                                          : sg_gpu_forward(gpu, &m, (tokid), (pos), (out)))

    const float *lg = NULL;
    double t_prompt = 0.0, t_gen = 0.0;
    uint32_t produced = 0;

    double t0 = now_s();
    for (uint64_t t = 0; t < n_ids; t++) {
        e = FORWARD(ids[t], (uint32_t)t, &lg);
        if (sg_failed(e)) {
            fprintf(stderr, "surge: %s\n", e.msg);
            if (lf) { fclose(lf); lf = NULL; remove(logits_path); }
            goto done;
        }
        if (lf && fwrite(lg, sizeof(float), m.cfg.vocab, lf) != m.cfg.vocab) {
            fprintf(stderr, "surge: short write to %s\n", logits_path);
            fclose(lf); lf = NULL; remove(logits_path);
            goto done;
        }
        if (!quiet && (t % 8 == 0 || t + 1 == n_ids)) {
            fprintf(stderr, "\r  prompt %llu/%llu", (unsigned long long)(t + 1),
                    (unsigned long long)n_ids);
        }
    }
    t_prompt = now_s() - t0;
    if (!quiet) fprintf(stderr, "\n");

    if (n_gen > 0) {
        gen = malloc((size_t)n_gen * sizeof *gen);
        if (!gen) { fprintf(stderr, "surge: out of memory\n"); goto done; }
        t0 = now_s();
        /* lg holds the logits of the LAST prompt position, which predict the
         * token at position n_ids. Step i emits that token and feeds it back
         * at position n_ids + i. */
        for (uint32_t i = 0; i < n_gen; i++) {
            uint32_t arg = argmax_f32(lg, m.cfg.vocab);
            if (margins) {
                fprintf(stderr, "  margin[%u] tok %u gap %.6e\n", i, arg,
                        (double)top2_gap(lg, m.cfg.vocab, arg));
            }
            gen[produced++] = (int32_t)arg;
            if (tok && (int32_t)arg == sg_tok_eos(tok)) break;
            if (i + 1 == n_gen) break;                  /* no need for its logits */
            if ((uint64_t)n_ids + i >= max_ctx) break;
            e = FORWARD((int32_t)arg, (uint32_t)(n_ids + i), &lg);
            if (sg_failed(e)) { fprintf(stderr, "surge: %s\n", e.msg); goto done; }
            if (lf && fwrite(lg, sizeof(float), m.cfg.vocab, lf) != m.cfg.vocab) {
                fprintf(stderr, "surge: short write to %s\n", logits_path);
                fclose(lf); lf = NULL; remove(logits_path);
                goto done;
            }
            if (!quiet && (i % 8 == 0 || i + 2 == n_gen)) {
                fprintf(stderr, "\r  generated %u/%u", i + 1, n_gen);
            }
        }
        t_gen = now_s() - t0;
        if (!quiet) fprintf(stderr, "\n");
    }
#undef FORWARD

    if (lf) {
        if (fclose(lf) != 0) { lf = NULL; fprintf(stderr, "surge: close failed\n"); goto done; }
        lf = NULL;
    }

    printf("path: %s\n", use_ref ? "ref" : "metal");
    printf("prompt_ids: ");
    print_ids(ids, n_ids);
    printf("\ngen_ids: ");
    print_ids(gen, produced);
    printf("\n");
    print_text(tok, gen, produced);

    if (!quiet) {
        /* The prompt is processed one token at a time here, exactly like the
         * generated tokens, so both rates measure the same decode step. */
        fprintf(stderr, "surge: prompt %llu tok in %.3f s (%.2f tok/s)\n",
                (unsigned long long)n_ids, t_prompt,
                t_prompt > 0 ? (double)n_ids / t_prompt : 0.0);
        if (produced > 1) {
            fprintf(stderr, "surge: decode %u tok in %.3f s (%.2f tok/s)\n",
                    produced, t_gen, t_gen > 0 ? (double)(produced - 1) / t_gen : 0.0);
        }
    }

    rc = 0;
done:
    if (lf) fclose(lf);
    free(ids);
    free(gen);
    sg_ref_state_free(rs);
    sg_gpu_free(gpu);
    sg_model_free(&m);
    if (tok) sg_tok_free(tok);
    if (st) sg_st_close(st);
    if (gg) sg_gguf_close(gg);
    return rc;
}
