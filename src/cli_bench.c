/* cli_bench.c - `surge-bench`, the benchmark harness (Task B5).
 *
 *   surge-bench <model_dir_or_gguf>
 *       [--prompt-file PATH | -p TEXT | --ids LIST]
 *       [--max-ctx 262144] [-n 300] [--chunk N] [--no-prefill]
 *       [--engine STR] [--model STR] [--log-id STR]
 *       [--expect-min N] [--expect-max N] [--bos | --no-bos]
 *       [--gemm-gate-tflops F] [--json PATH] [--emit-timeseries PATH]
 *       [--warmup N] [--quiet]
 *
 * This is the harness B7 runs to produce surge's row in the 256K comparison
 * (/Users/macmini/projects/llm-rnd/docs/256k_comparison.md). It wires the
 * pure-C bench math of src/bench.c (B1-B4 + the B2 peak-memory probe) to the
 * real Metal forward pass (M5 tiled prefill + the decode driver), and emits
 * one leaderboard row.
 *
 * WHAT IT MEASURES, and the admission rule (sg_bench_finalize_status):
 *   - Prefill the prompt via sg_gpu_prefill (the M5 tiled path, default chunk
 *     1024), then greedy-decode -n tokens via sg_argmax_f32 + sg_gpu_forward,
 *     the SAME argmax and the SAME prefill `surge` uses -- so surge-bench's
 *     gen_ids are byte-identical to `surge` on the same input (B5's primary
 *     gate). No chat template: the prompt is tokenized RAW, matching how the
 *     mlx-lm / llama.cpp cells tokenize the NIAH file.
 *   - Ingestion guard (sg_bench_check_ingestion): the tokenized count must sit
 *     inside [--expect-min, --expect-max] and within --max-ctx, else the row
 *     is VOID.
 *   - GEMM gate: --gemm-gate-tflops passes the externally-measured GEMM TFLOPS
 *     (the run recipe measures it before calling surge-bench). A non-VOID row
 *     needs F > 20.5 AND ingestion_ok; below/absent => VOID.
 *   - A VOID row is emitted and the process exits 3 BEFORE any GPU load, so a
 *     failed gate costs a tokenize, not a 28 GiB model load + a full run.
 *
 * PEAK RAM SEMANTICS (B2 carry-forward, load-bearing -- see the report):
 *   sg_gpu_current_alloc_bytes counts every newBufferWithBytesNoCopy-wrapped
 *   weight at its full DECLARED length from load onward (~28 GiB for the 27B
 *   Q8_0), regardless of page residency, so it is an ALLOCATED upper bound,
 *   NOT resident memory. sg_proc_phys_footprint is the resident footprint
 *   (what Activity Monitor / `footprint` report), which is the number
 *   comparable to mlx-lm / llama.cpp's peak-RAM column. This harness therefore
 *   tracks the two SEPARATELY across the whole run (after load, after prefill,
 *   periodically during decode) and reports:
 *       row.peak_ram_gib  = peak process phys_footprint  (RESIDENT, comparable)
 *       row.gpu_alloc_gib = peak Metal currentAllocatedSize (ALLOCATED bound)
 *   so surge's resident peak is never silently mis-compared to another
 *   engine's, and the allocated upper bound is still visible next to it.
 *
 * Exit codes: 0 = row DONE (admitted), 3 = VOID (missing/failed GEMM gate or
 * failed ingestion), any other nonzero = hard error (bad args, load/tokenize/
 * Metal failure).
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

#define GIB (1024.0 * 1024.0 * 1024.0)

static void usage(void) {
    fprintf(stderr,
        "usage: surge-bench <model_dir_or_gguf>\n"
        "         [--prompt-file PATH | -p TEXT | --ids LIST]\n"
        "         [--max-ctx N] [-n N] [--chunk N] [--no-prefill]\n"
        "         [--engine STR] [--model STR] [--log-id STR]\n"
        "         [--expect-min N] [--expect-max N] [--bos | --no-bos]\n"
        "         [--gemm-gate-tflops F] [--json PATH]\n"
        "         [--emit-timeseries PATH] [--warmup N] [--quiet]\n"
        "\n"
        "  --prompt-file P   raw prompt file (tokenized, no chat template)\n"
        "  -p TEXT           inline prompt text (needs a GGUF tokenizer)\n"
        "  --ids LIST        comma-separated token ids (no tokenizer; no recall)\n"
        "  --max-ctx N       KV cap (default 262144)\n"
        "  -n N              greedy-decode N tokens (default 300)\n"
        "  --chunk N         prefill chunk size (default 1024)\n"
        "  --no-prefill      serial one-token ingest instead of tiled prefill\n"
        "  --engine STR      engine label for the row (default \"surge\")\n"
        "  --model STR       model label for the row (default: basename of path)\n"
        "  --log-id STR      log id recorded in the JSON\n"
        "  --expect-min N    ingestion window lower bound (default 1)\n"
        "  --expect-max N    ingestion window upper bound (default max-ctx)\n"
        "  --bos/--no-bos    prepend a BOS token (default: model add_bos_token)\n"
        "  --gemm-gate-tflops F  externally-measured GEMM TFLOPS; F>20.5 required\n"
        "  --json PATH       also write the row as JSON to PATH\n"
        "  --emit-timeseries PATH  write per-token cumulative wall times to PATH\n"
        "  --warmup N        decode-fit warmup tokens (default ~0.02*n, >=1)\n"
        "  --quiet           suppress progress on stderr\n");
}

/* Parses "1,2,3" into a freshly allocated array; rejects anything that is not
 * a clean list of non-negative integers (a truncated list would silently
 * change the run). Mirrors cli_metal.c / cli_ref.c so --ids behaves identically
 * across the three binaries. */
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

/* basename without the directory or a trailing ".gguf" -- used as the default
 * row model label. Writes at most cap-1 bytes and always NUL-terminates. */
static void model_label_default(const char *path, char *out, size_t cap) {
    if (cap == 0) return;
    const char *base = path;
    for (const char *p = path; *p; p++) if (*p == '/') base = p + 1;
    size_t len = strlen(base);
    if (len >= 5 && strcmp(base + len - 5, ".gguf") == 0) len -= 5;
    if (len > cap - 1) len = cap - 1;
    memcpy(out, base, len);
    out[len] = '\0';
}

/* Scans the GGUF kv table for `key` and returns its scalar int / bool through
 * the generic accessor (the typed getters do not cover every int width, and
 * add_bos_token is a bool). Returns false when the key is absent. */
static bool gguf_scalar_int(const sg_gguf *g, const char *key, int64_t *out) {
    for (uint64_t i = 0; i < sg_gguf_kv_count(g); i++) {
        const char *k; sg_gguf_kv_type t;
        if (!sg_gguf_kv_at(g, i, &k, &t)) continue;
        if (strcmp(k, key) == 0) return sg_gguf_kv_scalar_at(g, i, NULL, out, NULL, NULL);
    }
    return false;
}
static bool gguf_scalar_bool(const sg_gguf *g, const char *key, bool *out) {
    for (uint64_t i = 0; i < sg_gguf_kv_count(g); i++) {
        const char *k; sg_gguf_kv_type t;
        if (!sg_gguf_kv_at(g, i, &k, &t)) continue;
        if (strcmp(k, key) == 0) return sg_gguf_kv_scalar_at(g, i, NULL, NULL, NULL, out);
    }
    return false;
}

/* Samples the two memory signals into their separate running-max trackers.
 * Kept OUTSIDE the tracker (which just maxes numbers) exactly as B2 intends,
 * so the mach / Metal reads stay on the live-only side. The two trackers stay
 * distinct so peak_ram (resident) is never conflated with gpu_alloc
 * (allocated upper bound). */
static void sample_mem(const sg_gpu *gpu, sg_mem_tracker *phys, sg_mem_tracker *alloc) {
    uint64_t a = sg_gpu_current_alloc_bytes(gpu);
    uint64_t p = sg_proc_phys_footprint();
    sg_mem_tracker_sample(phys, 0, p);   /* max(0, p)  == phys peak  */
    sg_mem_tracker_sample(alloc, a, 0);  /* max(a, 0)  == alloc peak */
}

static void print_ids(const int32_t *ids, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
}

/* Emits the row to stdout (md) and, if json_path, to json_path. Returns 0 on
 * success, nonzero if the JSON file could not be written. */
static int emit_row(const sg_bench_row *row, const char *json_path) {
    char md[512];
    sg_bench_format_md_row(row, md, sizeof md);
    printf("%s\n", md);

    if (json_path) {
        char js[2048];
        sg_bench_format_json(row, js, sizeof js);
        FILE *jf = fopen(json_path, "wb");
        if (!jf) { fprintf(stderr, "surge-bench: cannot write %s\n", json_path); return 1; }
        fprintf(jf, "%s\n", js);
        if (fclose(jf) != 0) { fprintf(stderr, "surge-bench: close failed on %s\n", json_path); return 1; }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 2; }
    const char *path = argv[1];
    const char *prompt = NULL, *ids_arg = NULL, *prompt_file = NULL;
    const char *engine = "surge", *model_name = NULL, *log_id = NULL;
    const char *json_path = NULL, *ts_path = NULL;
    uint32_t n_gen = 300, max_ctx_arg = 0, chunk_size = SG_PREFILL_CHUNK_DEFAULT;
    uint32_t warmup_arg = 0;
    bool has_warmup = false, no_prefill = false, quiet = false;
    uint64_t expect_min = 0, expect_max = 0;
    bool has_expect_min = false, has_expect_max = false;
    double gemm_tflops = 0.0;
    /* bos_choice: -1 = follow model default, 0 = --no-bos, 1 = --bos. */
    int bos_choice = -1;

#define NEED_VALUE(flag) do { \
        if (i + 1 >= argc) { fprintf(stderr, "surge-bench: %s needs a value\n", (flag)); return 2; } \
    } while (0)
#define PARSE_U32(flag, dst) do { \
        NEED_VALUE(flag); \
        char *_end = NULL; \
        unsigned long _v = strtoul(argv[++i], &_end, 10); \
        if (!_end || *_end != '\0' || argv[i][0] == '\0' || _v > UINT32_MAX) { \
            fprintf(stderr, "surge-bench: %s expects a non-negative integer, got '%s'\n", (flag), argv[i]); \
            return 2; \
        } \
        (dst) = (uint32_t)_v; \
    } while (0)
#define PARSE_U64(flag, dst) do { \
        NEED_VALUE(flag); \
        char *_end = NULL; \
        unsigned long long _v = strtoull(argv[++i], &_end, 10); \
        if (!_end || *_end != '\0' || argv[i][0] == '\0') { \
            fprintf(stderr, "surge-bench: %s expects a non-negative integer, got '%s'\n", (flag), argv[i]); \
            return 2; \
        } \
        (dst) = (uint64_t)_v; \
    } while (0)

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0) { NEED_VALUE("-p"); prompt = argv[++i]; }
        else if (strcmp(argv[i], "--prompt-file") == 0) { NEED_VALUE("--prompt-file"); prompt_file = argv[++i]; }
        else if (strcmp(argv[i], "--ids") == 0) { NEED_VALUE("--ids"); ids_arg = argv[++i]; }
        else if (strcmp(argv[i], "-n") == 0) PARSE_U32("-n", n_gen);
        else if (strcmp(argv[i], "--max-ctx") == 0) PARSE_U32("--max-ctx", max_ctx_arg);
        else if (strcmp(argv[i], "--chunk") == 0) PARSE_U32("--chunk", chunk_size);
        else if (strcmp(argv[i], "--no-prefill") == 0) no_prefill = true;
        else if (strcmp(argv[i], "--engine") == 0) { NEED_VALUE("--engine"); engine = argv[++i]; }
        else if (strcmp(argv[i], "--model") == 0) { NEED_VALUE("--model"); model_name = argv[++i]; }
        else if (strcmp(argv[i], "--log-id") == 0) { NEED_VALUE("--log-id"); log_id = argv[++i]; }
        else if (strcmp(argv[i], "--expect-min") == 0) { PARSE_U64("--expect-min", expect_min); has_expect_min = true; }
        else if (strcmp(argv[i], "--expect-max") == 0) { PARSE_U64("--expect-max", expect_max); has_expect_max = true; }
        else if (strcmp(argv[i], "--bos") == 0) bos_choice = 1;
        else if (strcmp(argv[i], "--no-bos") == 0) bos_choice = 0;
        else if (strcmp(argv[i], "--gemm-gate-tflops") == 0) {
            NEED_VALUE("--gemm-gate-tflops");
            char *end = NULL;
            gemm_tflops = strtod(argv[++i], &end);
            if (!end || *end != '\0' || argv[i][0] == '\0') {
                fprintf(stderr, "surge-bench: --gemm-gate-tflops expects a number, got '%s'\n", argv[i]);
                return 2;
            }
        }
        else if (strcmp(argv[i], "--json") == 0) { NEED_VALUE("--json"); json_path = argv[++i]; }
        else if (strcmp(argv[i], "--emit-timeseries") == 0) { NEED_VALUE("--emit-timeseries"); ts_path = argv[++i]; }
        else if (strcmp(argv[i], "--warmup") == 0) { PARSE_U32("--warmup", warmup_arg); has_warmup = true; }
        else if (strcmp(argv[i], "--quiet") == 0) quiet = true;
        else { fprintf(stderr, "surge-bench: unknown argument '%s'\n", argv[i]); usage(); return 2; }
    }
#undef NEED_VALUE
#undef PARSE_U32
#undef PARSE_U64

    int n_inputs = (prompt != NULL) + (ids_arg != NULL) + (prompt_file != NULL);
    if (n_inputs != 1) {
        fprintf(stderr, "surge-bench: pass exactly one of -p, --prompt-file, --ids\n");
        return 2;
    }

    /* ---- open the model + tokenizer (GGUF) / config (safetensors) ---- */
    sg_gguf *gg = NULL;
    sg_st *st = NULL;
    sg_tok *tok = NULL;
    sg_model m;
    sg_err e;

    if (ends_with(path, ".gguf")) {
        e = sg_gguf_open(path, &gg);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); return 1; }
        e = sg_model_from_gguf(gg, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); sg_gguf_close(gg); return 1; }
        if (sg_failed(sg_tok_from_gguf(gg, &tok))) tok = NULL;
    } else {
        e = sg_st_open(path, &st);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); return 1; }
        e = sg_model_from_st(st, &m);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); sg_st_close(st); return 1; }
    }

    int rc = 1;                 /* hard error unless we reach DONE (0) / VOID (3) */
    int32_t *ids = NULL, *gen = NULL;
    uint64_t n_ids = 0;
    char *file_buf = NULL;
    double *t_wall = NULL;
    sg_gpu *gpu = NULL;
    const char *prompt_text = NULL;   /* text for NIAH extraction (not for --ids) */

    /* ---- resolve the prompt tokens ---- */
    if (prompt_file) {
        size_t flen = 0;
        e = sg_bench_read_file(prompt_file, &file_buf, &flen);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
        if (!tok) {
            fprintf(stderr, "surge-bench: --prompt-file needs a GGUF tokenizer; this model "
                            "carries none, use --ids\n");
            goto done;
        }
        e = sg_tok_encode(tok, file_buf, &ids, &n_ids);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
        prompt_text = file_buf;
    } else if (prompt) {
        if (!tok) {
            fprintf(stderr, "surge-bench: -p needs a GGUF tokenizer; this model carries none, "
                            "use --ids\n");
            goto done;
        }
        e = sg_tok_encode(tok, prompt, &ids, &n_ids);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
        prompt_text = prompt;
    } else {
        ids = parse_ids(ids_arg, &n_ids);
        if (!ids) {
            fprintf(stderr, "surge-bench: --ids must be a comma-separated list of "
                            "non-negative integers\n");
            goto done;
        }
    }

    /* ---- BOS: only meaningful for tokenized text; --ids is literal ---- */
    bool bos_enabled = false;
    if (prompt_text) {
        bool bos_default = false;
        if (gg) gguf_scalar_bool(gg, "tokenizer.ggml.add_bos_token", &bos_default);
        bos_enabled = (bos_choice == -1) ? bos_default : (bos_choice == 1);
        if (bos_enabled) {
            int64_t bos_id = -1;
            if (!gg || !gguf_scalar_int(gg, "tokenizer.ggml.bos_token_id", &bos_id) || bos_id < 0) {
                fprintf(stderr, "surge-bench: --bos requested but this model has no "
                                "tokenizer.ggml.bos_token_id\n");
                goto done;
            }
            int32_t *grown = realloc(ids, (n_ids + 1) * sizeof *ids);
            if (!grown) { fprintf(stderr, "surge-bench: out of memory\n"); goto done; }
            ids = grown;
            memmove(ids + 1, ids, n_ids * sizeof *ids);
            ids[0] = (int32_t)bos_id;
            n_ids++;
        }
    } else if (bos_choice != -1) {
        fprintf(stderr, "surge-bench: --bos/--no-bos has no effect with --ids (ignored)\n");
    }

    if (n_ids == 0) { fprintf(stderr, "surge-bench: the prompt is empty\n"); goto done; }
    fprintf(stderr, "bench: n_prompt_tokens=%llu (bos=%s)\n",
            (unsigned long long)n_ids, bos_enabled ? "on" : "off");

    /* ---- context cap (M5.7 semantics: --max-ctx is a HARD cap) ----
     * PREFLIGHT, mirroring cli_metal.c: a prompt (plus the tokens it asks to
     * generate) that does not fit the cap is HARD-REJECTED here (exit 1, no
     * gen_ids), never truncated mid-run and then reported DONE. This keeps the
     * "surge-bench gen_ids == surge" guarantee in B7's regime (prompt near the
     * 262144 cap, -n 300): both binaries refuse the same runs and generate the
     * full -n on the ones they accept, so neither ever truncates. B7's recipe
     * must therefore leave >= n_gen tokens of headroom under --max-ctx. */
    uint32_t max_ctx = max_ctx_arg ? max_ctx_arg : 262144u;
    if (n_ids > max_ctx) {
        fprintf(stderr, "surge-bench: prompt is %llu tokens but --max-ctx is %u; "
                        "the prompt exceeds the context cap\n",
                (unsigned long long)n_ids, max_ctx);
        goto done;
    }
    if ((uint64_t)n_ids + n_gen > max_ctx) {
        fprintf(stderr, "surge-bench: %llu prompt + %u generated tokens exceeds "
                        "--max-ctx %u; raise --max-ctx or lower -n\n",
                (unsigned long long)n_ids, n_gen, max_ctx);
        goto done;
    }

    /* ---- build the row skeleton + the two gates ---- */
    sg_bench_row row;
    memset(&row, 0, sizeof row);
    char default_model[64];
    model_label_default(path, default_model, sizeof default_model);
    snprintf(row.model, sizeof row.model, "%s", model_name ? model_name : default_model);
    snprintf(row.engine, sizeof row.engine, "%s", engine);
    if (log_id) snprintf(row.log_id, sizeof row.log_id, "%s", log_id);
    row.n_prompt_tok = n_ids;
    row.n_gen = 0;               /* set to the ACTUAL produced count after decode */
    row.prefill_tps = -1.0;      /* "-" until measured */
    row.recall_total = 0;
    row.gemm_tflops = gemm_tflops;

    uint64_t emin = has_expect_min ? expect_min : 1;
    uint64_t emax = has_expect_max ? expect_max : (uint64_t)max_ctx;
    sg_bench_check_ingestion(n_ids, max_ctx, emin, emax, &row.ingestion_ok);

    /* VOID short-circuit: a failed gate costs a tokenize, not a model load.
     * Uses the SAME predicate sg_bench_finalize_status applies, so the pre-load
     * decision and the final status can never disagree. */
    if (!sg_bench_admitted(&row)) {
        sg_bench_finalize_status(&row);   /* sets status = "VOID" */
        rc = (emit_row(&row, json_path) == 0) ? 3 : 1;
        goto done;
    }

    /* ---- load the GPU model + state ---- */
    e = sg_gpu_init(&gpu);
    if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
    e = sg_gpu_load_model(gpu, &m);
    if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
    e = sg_gpu_state_new(gpu, &m, max_ctx);
    if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }

    sg_mem_tracker mt_phys, mt_alloc;
    sg_mem_tracker_reset(&mt_phys);
    sg_mem_tracker_reset(&mt_alloc);
    sample_mem(gpu, &mt_phys, &mt_alloc);      /* after load */

    if (!quiet) {
        fprintf(stderr, "surge-bench: %s [metal], %u layers, vocab %u, %llu prompt tokens, "
                        "gemm=%.2f TFLOPS\n",
                path, m.cfg.n_layers, m.cfg.vocab, (unsigned long long)n_ids, row.gemm_tflops);
    }

    /* ---- prefill ---- */
    const float *lg = NULL;
    double t_run_start = now_s();
    double t0 = t_run_start;
    if (!no_prefill) {
        e = sg_gpu_prefill(gpu, &m, ids, (uint32_t)n_ids, chunk_size, &lg);
        if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
        if (!quiet) fprintf(stderr, "  prefill %llu tokens, chunk %u\n",
                            (unsigned long long)n_ids, chunk_size);
    } else {
        for (uint64_t t = 0; t < n_ids; t++) {
            e = sg_gpu_forward(gpu, &m, ids[t], (uint32_t)t, &lg);
            if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
        }
    }
    double t_prefill = now_s() - t0;
    row.prefill_tps = t_prefill > 0.0 ? (double)n_ids / t_prefill : -1.0;
    sample_mem(gpu, &mt_phys, &mt_alloc);      /* after prefill */

    /* ---- decode (shared driver: sg_argmax_f32 + sg_gpu_forward) ----
     * The stopping rules below are byte-identical to cli_metal.c's, so with the
     * shared argmax and shared prefill the emitted gen_ids match `surge`. */
    uint32_t produced = 0;
    if (n_gen > 0) {
        gen = malloc((size_t)n_gen * sizeof *gen);
        t_wall = malloc((size_t)n_gen * sizeof *t_wall);
        if (!gen || !t_wall) { fprintf(stderr, "surge-bench: out of memory\n"); goto done; }
        double t_dec_start = now_s();
        for (uint32_t i = 0; i < n_gen; i++) {
            uint32_t arg = sg_argmax_f32(lg, m.cfg.vocab);
            gen[produced] = (int32_t)arg;
            t_wall[produced] = now_s() - t_dec_start;   /* cumulative decode wall time */
            produced++;
            if ((produced & 31u) == 0) sample_mem(gpu, &mt_phys, &mt_alloc);
            if (tok && (int32_t)arg == sg_tok_eos(tok)) break;
            if (i + 1 == n_gen) break;                  /* last token needs no logits */
            if ((uint64_t)n_ids + i >= max_ctx) break;
            e = sg_gpu_forward(gpu, &m, (int32_t)arg, (uint32_t)(n_ids + i), &lg);
            if (sg_failed(e)) { fprintf(stderr, "surge-bench: %s\n", e.msg); goto done; }
            if (!quiet && (i % 32 == 0 || i + 2 == n_gen)) {
                fprintf(stderr, "\r  generated %u/%u", i + 1, n_gen);
            }
        }
        if (!quiet) fprintf(stderr, "\n");
    }
    row.wall_s = now_s() - t_run_start;
    row.n_gen = produced;   /* the ACTUAL number generated (EOS may stop early) */
    sample_mem(gpu, &mt_phys, &mt_alloc);      /* after decode */

    /* ---- decode-by-slope + mlx-style average ---- */
    uint32_t warmup = has_warmup ? warmup_arg : sg_bench_default_warmup(produced);
    row.decode_tps_slope = sg_bench_slope(t_wall, produced, warmup);
    row.decode_tps_avg = sg_bench_avg_tps(t_wall, produced, warmup);

    /* ---- peak RAM: resident (comparable) vs allocated (upper bound) ---- */
    uint64_t peak_phys = sg_mem_tracker_peak(&mt_phys);
    uint64_t peak_alloc = sg_mem_tracker_peak(&mt_alloc);
    row.peak_ram_gib = (double)peak_phys / GIB;
    row.gpu_alloc_gib = (double)peak_alloc / GIB;
    fprintf(stderr, "bench: peak resident (phys_footprint) %.2f GiB; peak allocated "
                    "(gpu currentAllocatedSize, upper bound, NOT resident) %.2f GiB\n",
            row.peak_ram_gib, row.gpu_alloc_gib);

    /* ---- recall (text input only) ---- */
    if (prompt_text && tok && produced > 0) {
        sg_bench_needle needles[SG_BENCH_MAX_NEEDLES];
        uint32_t n_needles = 0;
        sg_bench_extract_needles(prompt_text, needles, SG_BENCH_MAX_NEEDLES, &n_needles);
        row.recall_total = n_needles;

        size_t gcap = (size_t)produced * 64 + 1024;
        char *gen_text = malloc(gcap);
        if (gen_text) {
            /* Pass gcap - 1: sg_tok_decode permits written == its cap exactly,
             * so bounding it to gcap-1 keeps the gen_text[got]='\0' below inside
             * the gcap-byte allocation (never one past it). */
            int64_t got = sg_tok_decode(tok, gen, produced, gen_text, gcap - 1);
            if (got >= 0) {
                gen_text[got] = '\0';
                uint32_t rh = 0, ah = 0;
                sg_bench_score_niah(gen_text, needles, n_needles, &rh, &ah);
                row.recall_hits = rh;
                row.assoc_hits = ah;
            } else {
                fprintf(stderr, "surge-bench: generated text did not fit the decode buffer; "
                                "recall left at 0\n");
            }
            free(gen_text);
        }
    }

    /* ---- optional per-token timeseries for B6 ---- */
    if (ts_path && produced > 0) {
        FILE *tf = fopen(ts_path, "wb");
        if (!tf) { fprintf(stderr, "surge-bench: cannot write %s\n", ts_path); goto done; }
        fprintf(tf, "# token_index\tcumulative_wall_s\n");
        for (uint32_t i = 0; i < produced; i++) fprintf(tf, "%u\t%.9f\n", i, t_wall[i]);
        if (fclose(tf) != 0) { fprintf(stderr, "surge-bench: close failed on %s\n", ts_path); goto done; }
    }

    /* ---- finalize + emit ---- */
    sg_bench_finalize_status(&row);      /* DONE here (both gates already passed) */

    printf("prompt_ids: ");
    print_ids(ids, n_ids);
    printf("\ngen_ids: ");
    print_ids(gen, produced);
    printf("\n");

    rc = (emit_row(&row, json_path) == 0) ? 0 : 1;

done:
    free(ids);
    free(gen);
    free(t_wall);
    free(file_buf);
    sg_gpu_free(gpu);
    sg_model_free(&m);
    if (tok) sg_tok_free(tok);
    if (st) sg_st_close(st);
    if (gg) sg_gguf_close(gg);
    return rc;
}
