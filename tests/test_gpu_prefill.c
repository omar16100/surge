/* test_gpu_prefill.c - the Task M5.6 chunked prefill path against the Task 10
 * serial Metal decode path, whole-model, on the mini hybrid fixture.
 *
 * sg_gpu_prefill wires the M5.4 full-attention prefill kernels and the M5.5
 * DeltaNet chunked-scan kernels across ALL layers, one command buffer per
 * chunk. This is the FIRST end-to-end numeric exercise of enc_attn_prefill and
 * enc_gdn_prefill, so the oracle here is the serial forward (sg_gpu_forward
 * fed one token at a time), NOT the CPU reference: test_gpu_fwd.c already pins
 * the serial GPU path to ref.c, so a divergence here is on the prefill wiring.
 *
 * Both paths run with the DEFAULT f16 KV cache (prefill requires it: the full-
 * attention prefill kernel reads g->kv's fp16 K/V and the DeltaNet prefill
 * kernels read its conv/S carriers, neither of which exists on the f32 path).
 * The only numerical difference between the two paths is GEMM-vs-matvec
 * reassociation in the projections (the attn/conv/delta/rmsnorm chunk kernels
 * are documented bit-identical to their per-token siblings), so the last-logit
 * bar is the task's 2e-4, not test_gpu_fwd's 1e-4.
 *
 * The SAME 4-layer hybrid ships twice (bf16 safetensors + f32 gguf); its
 * full_attention_interval is 4, so layer 3 is full-attention and layers 0..2
 * are gated-DeltaNet -- ONE full-attn and THREE DeltaNet layers, both kinds
 * driven by prefill. Each model is asserted to carry both kinds.
 *
 * Gates:
 *   1. chunk {1,2,3}: prefill last-position argmax == serial last-position
 *      argmax, worst relative logit gap < 2e-4.
 *   2. prefill-then-decode gen_ids == serial-then-decode gen_ids (the real
 *      state-bridging check: decode must continue from the prefilled state).
 *   3. determinism: prefill reruns (which reset state internally) are
 *      byte-identical.
 */
#ifdef SURGE_NO_METAL

#include <stdio.h>
int main(void) {
    fprintf(stderr, "SKIP test_gpu_prefill: built with -DSURGE_NO_METAL "
                    "(Metal and the ASan/UBSan run do not mix)\n");
    return 0;
}

#else

#include "tinytest.h"
#include "../surge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MINI_DIR "tests/fixtures/mini_fwd"
#define N_GEN 16u

/* M5.7 long-context gate constants. N_GATE_GEN is the 32-token greedy decode
 * the depth-equivalence gate compares byte-for-byte; SG_KV_CAP_MAX (262144,
 * from surge.h) is the full-ingest depth. */
#define N_GATE_GEN 32u

static sg_gpu *g_gpu;
static double g_worst_rel = 0.0;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

static int32_t *read_ids_file(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char buf[8192];
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = '\0';
    size_t cap = 64, n = 0;
    int32_t *ids = xmalloc(cap * sizeof *ids);
    const char *p = buf;
    while (*p) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) { p++; continue; }
        if (n == cap) { cap *= 2; ids = realloc(ids, cap * sizeof *ids); }
        ids[n++] = (int32_t)v;
        p = end;
    }
    *n_out = n;
    return ids;
}

/* THE argmax, byte-for-byte the one cli_metal.c and test_gpu_fwd.c use:
 * lowest index wins an exact tie, so a genuinely close position cannot flip on
 * convention alone. */
static uint32_t argmax_f32(const float *v, uint32_t n) {
    uint32_t arg = 0;
    for (uint32_t i = 1; i < n; i++) if (v[i] > v[arg]) arg = i;
    return arg;
}

/* One model, all three prefill gates. `label` names the checkpoint format so a
 * failure says which convention broke. */
static void run_gates(const char *label, sg_model *m, const int32_t *ids, size_t n_ids) {
    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t i = 0; i < m->cfg.n_layers; i++) {
        if (m->layers[i].q_proj) n_attn++; else n_gdn++;
    }
    tt_assert(n_attn > 0 && n_gdn > 0,
              "%s: prefill gate must exercise BOTH a full-attn and a DeltaNet layer "
              "(full-attn=%u deltanet=%u)", label, n_attn, n_gdn);

    uint32_t vocab = m->cfg.vocab;
    uint32_t max_ctx = (uint32_t)n_ids + N_GEN;

    sg_err e = sg_gpu_load_model(g_gpu, m);
    tt_assert(!sg_failed(e), "%s: sg_gpu_load_model: %s", label, e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;
    e = sg_gpu_state_new(g_gpu, m, max_ctx);
    tt_assert(!sg_failed(e), "%s: sg_gpu_state_new: %s", label, e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    /* --- Serial oracle: feed the prompt one token at a time, snapshot the
     * last-position logits, then greedily decode N_GEN tokens. --- */
    float *serial_last = xmalloc(vocab * sizeof *serial_last);
    int32_t serial_gen[N_GEN];
    {
        sg_gpu_state_reset(g_gpu);
        const float *lg = NULL;
        for (uint32_t t = 0; t < n_ids; t++) {
            e = sg_gpu_forward(g_gpu, m, ids[t], t, &lg);
            tt_assert(!sg_failed(e), "%s: serial sg_gpu_forward %u: %s",
                      label, t, e.msg ? e.msg : "ok");
            if (sg_failed(e)) { free(serial_last); return; }
        }
        memcpy(serial_last, lg, vocab * sizeof *serial_last);
        for (uint32_t i = 0; i < N_GEN; i++) {
            uint32_t arg = argmax_f32(lg, vocab);
            serial_gen[i] = (int32_t)arg;
            if (i + 1 == N_GEN) break;
            e = sg_gpu_forward(g_gpu, m, (int32_t)arg, (uint32_t)n_ids + i, &lg);
            tt_assert(!sg_failed(e), "%s: serial decode %u: %s", label, i, e.msg ? e.msg : "ok");
            if (sg_failed(e)) { free(serial_last); return; }
        }
    }

    /* --- Gate 1: chunk {1,2,3} last-position parity against the serial oracle. --- */
    for (uint32_t chunk = 1; chunk <= 3; chunk++) {
        const float *pl = NULL;
        e = sg_gpu_prefill(g_gpu, m, ids, (uint32_t)n_ids, chunk, &pl);
        tt_assert(!sg_failed(e), "%s chunk %u: sg_gpu_prefill: %s",
                  label, chunk, e.msg ? e.msg : "ok");
        if (sg_failed(e)) continue;

        double scale = 0.0, err = 0.0;
        for (uint32_t i = 0; i < vocab; i++) {
            double a = fabs((double)serial_last[i]);
            double d = fabs((double)pl[i] - (double)serial_last[i]);
            if (a > scale) scale = a;
            if (d > err) err = d;
        }
        double rel = (scale > 0.0) ? err / scale : err;
        if (rel > g_worst_rel) g_worst_rel = rel;

        tt_assert(argmax_f32(pl, vocab) == argmax_f32(serial_last, vocab),
                  "%s chunk %u: prefill last-argmax %u != serial last-argmax %u",
                  label, chunk, argmax_f32(pl, vocab), argmax_f32(serial_last, vocab));
        tt_assert(rel < 2e-4,
                  "%s chunk %u: worst last-logit relative gap %.3e (bar 2e-4)",
                  label, chunk, rel);
    }

    /* --- Gate 2: prefill-then-decode gen_ids must equal serial-then-decode.
     * This is the state-bridging proof: after prefill the DeltaNet conv/S state
     * has been copied from sg_kv into the decode buffers and g->used == n_ids,
     * so decode continues correctly. --- */
    {
        const float *lg = NULL;
        e = sg_gpu_prefill(g_gpu, m, ids, (uint32_t)n_ids, 2, &lg);
        tt_assert(!sg_failed(e), "%s: sg_gpu_prefill (gate2): %s", label, e.msg ? e.msg : "ok");
        if (!sg_failed(e)) {
            int32_t pre_gen[N_GEN];
            for (uint32_t i = 0; i < N_GEN; i++) {
                uint32_t arg = argmax_f32(lg, vocab);
                pre_gen[i] = (int32_t)arg;
                if (i + 1 == N_GEN) break;
                e = sg_gpu_forward(g_gpu, m, (int32_t)arg, (uint32_t)n_ids + i, &lg);
                tt_assert(!sg_failed(e), "%s: post-prefill decode %u: %s",
                          label, i, e.msg ? e.msg : "ok");
                if (sg_failed(e)) break;
            }
            uint32_t match = 0, first_bad = N_GEN;
            for (uint32_t i = 0; i < N_GEN; i++) {
                if (pre_gen[i] == serial_gen[i]) match++;
                else if (first_bad == N_GEN) first_bad = i;
            }
            tt_assert(match == N_GEN,
                      "%s: prefill+decode gen_ids match serial+decode at %u/%u "
                      "(first diff at step %u)", label, match, N_GEN, first_bad);
            if (match != N_GEN) {
                fprintf(stderr, "   %s serial gen:", label);
                for (uint32_t i = 0; i < N_GEN; i++) fprintf(stderr, " %d", serial_gen[i]);
                fprintf(stderr, "\n   %s prefil gen:", label);
                for (uint32_t i = 0; i < N_GEN; i++) fprintf(stderr, " %d", pre_gen[i]);
                fprintf(stderr, "\n");
            }
        }
    }

    /* --- Gate 3: determinism. Two prefill runs (each resets state internally)
     * must produce byte-identical last-position logits AND, threaded through
     * the bridged state, byte-identical decode gen_ids. --- */
    {
        const float *p1 = NULL;
        e = sg_gpu_prefill(g_gpu, m, ids, (uint32_t)n_ids, 3, &p1);
        tt_assert(!sg_failed(e), "%s: sg_gpu_prefill (gate3 run1): %s", label, e.msg ? e.msg : "ok");
        float *snap = xmalloc(vocab * sizeof *snap);
        int32_t gen1[N_GEN];
        if (!sg_failed(e)) {
            memcpy(snap, p1, vocab * sizeof *snap);
            const float *lg = p1;
            for (uint32_t i = 0; i < N_GEN; i++) {
                gen1[i] = (int32_t)argmax_f32(lg, vocab);
                if (i + 1 == N_GEN) break;
                e = sg_gpu_forward(g_gpu, m, gen1[i], (uint32_t)n_ids + i, &lg);
                if (sg_failed(e)) break;
            }
        }
        const float *p2 = NULL;
        e = sg_gpu_prefill(g_gpu, m, ids, (uint32_t)n_ids, 3, &p2);
        tt_assert(!sg_failed(e), "%s: sg_gpu_prefill (gate3 run2): %s", label, e.msg ? e.msg : "ok");
        if (!sg_failed(e)) {
            uint32_t identical = 0;
            for (uint32_t i = 0; i < vocab; i++) if (p2[i] == snap[i]) identical++;
            tt_assert(identical == vocab,
                      "%s: prefill rerun matched %u/%u last-logits bit-exactly",
                      label, identical, vocab);
            int32_t gen2[N_GEN];
            const float *lg = p2;
            uint32_t gen_same = 0;
            for (uint32_t i = 0; i < N_GEN; i++) {
                gen2[i] = (int32_t)argmax_f32(lg, vocab);
                if (gen2[i] == gen1[i]) gen_same++;
                if (i + 1 == N_GEN) break;
                e = sg_gpu_forward(g_gpu, m, gen2[i], (uint32_t)n_ids + i, &lg);
                if (sg_failed(e)) break;
            }
            tt_assert(gen_same == N_GEN,
                      "%s: prefill+decode gen_ids reproducible at %u/%u", label, gen_same, N_GEN);
        }
        free(snap);
    }

    free(serial_last);
}

static void mini_st_prefill(void) {
    size_t n_ids = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    tt_assert(ids && n_ids > 1, "read %s/ids.txt", MINI_DIR);
    if (!ids || n_ids < 2) { free(ids); return; }

    sg_st *s = NULL;
    sg_err e = sg_st_open(MINI_DIR, &s);
    tt_assert(!sg_failed(e), "sg_st_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); return; }

    sg_model m;
    e = sg_model_from_st(s, &m);
    tt_assert(!sg_failed(e), "sg_model_from_st: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_st_close(s); free(ids); return; }

    tt_assert(m.wtype == SG_T_BF16, "safetensors mini should have bf16 matmuls");
    tt_assert(!m.v_heads_tiled, "safetensors mini should use grouped value heads");

    run_gates("mini/safetensors", &m, ids, n_ids);

    sg_model_free(&m);
    sg_st_close(s);
    free(ids);
}

static void mini_gguf_prefill(void) {
    size_t n_ids = 0;
    int32_t *ids = read_ids_file(MINI_DIR "/ids.txt", &n_ids);
    tt_assert(ids && n_ids > 1, "read %s/ids.txt", MINI_DIR);
    if (!ids || n_ids < 2) { free(ids); return; }

    sg_gguf *gg = NULL;
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { free(ids); return; }

    sg_model m;
    e = sg_model_from_gguf(gg, &m);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) { sg_gguf_close(gg); free(ids); return; }

    tt_assert(m.wtype == SG_T_F32, "gguf mini should have f32 matmuls");
    tt_assert(m.v_heads_tiled, "gguf mini should use tiled value heads");
    tt_assert(m.cfg.n_v_heads != m.cfg.n_k_heads,
              "the gguf mini must have n_v_heads != n_k_heads or the tiled map "
              "is untested (%u vs %u)", m.cfg.n_v_heads, m.cfg.n_k_heads);

    run_gates("mini/gguf", &m, ids, n_ids);

    sg_model_free(&m);
    sg_gguf_close(gg);
    free(ids);
}

/* ============================ M5.7 long-context gate ======================= *
 *
 * A REAL model (SURGE_GATE_MODEL, e.g. /Users/macmini/models/qwen35-2b, the 2B
 * bf16 qwen3_5 safetensors, the M1/M2 gate model) driven ENTIRELY through the C
 * API, because the id sequences (up to 262144) do not fit through the CLI's
 * argv. When SURGE_GATE_MODEL is unset the whole gate SKIPs cleanly, so
 * `make check` stays mini-only and hermetic (mirrors test_gpu_fwd.c's
 * SURGE_GGUF pattern).
 *
 *   (A) DEPTH EQUIVALENCE at 8192 / 16384 / 32768: prefill(prompt)+decode 32
 *       tokens must equal serial-forward(prompt one token at a time)+decode 32,
 *       with 0 token-id mismatches at each depth. State is reset between the two
 *       runs and the SAME argmax_f32 is used both sides. The serial reference at
 *       32k is ~32800 forwards and slow; that is expected for a gated (not
 *       make-check) run.
 *
 *   (B) 262144 INGEST: fill the whole 262144-position context and decode at
 *       ~256K depth. SG_KV_CAP_MAX == 262144 leaves NO cache slot to append
 *       after a full-cap prefill, so the context is filled as 262112 prefill +
 *       32 decode, which reaches the used counter == 262144 while still
 *       exercising decode at depth. That satisfies every clause of the gate:
 *       used == 262144, no Metal fault, non-degenerate final (prefill-position)
 *       logits, and 32 non-degenerate decoded tokens (finite each step, ids in
 *       range, not one id repeated). Valid-UTF-8 coherence on real text is B7's
 *       job (the 27B carries a surge-readable tokenizer; this 2B safetensors
 *       carries none), so the interim capability check here is non-degenerate
 *       token ids, per the plan's 2B-interim allowance.
 *
 * SURGE_GATE_SKIP_A=1 skips (A) (already proven; its 32k serial reference is
 * ~40 min under the box's power limiter) so a re-run can exercise only (B).
 */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Deterministic 64-bit LCG (Knuth's MMIX constants), token ids in [0, vocab).
 * Seeded per call so a given (seed, vocab) always yields the same sequence,
 * which is what lets the prefill and serial paths ingest the SAME ids. */
static void lcg_fill_ids(int32_t *ids, uint32_t n, uint32_t vocab, uint64_t seed) {
    uint64_t s = seed;
    for (uint32_t i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        ids[i] = (int32_t)((s >> 33) % vocab);
    }
}

/* Non-degenerate == all finite, NOT all-equal, argmax in range. A broken deep
 * prefill tends to produce NaN/Inf or a flat (all-equal) logit vector. */
static bool logits_nondegenerate(const float *v, uint32_t n) {
    bool finite = true, all_equal = true;
    for (uint32_t i = 0; i < n; i++) if (!isfinite(v[i])) { finite = false; break; }
    for (uint32_t i = 1; i < n; i++) if (v[i] != v[0]) { all_equal = false; break; }
    return finite && !all_equal && argmax_f32(v, n) < n;
}

static void gate_real_model(void) {
    const char *path = getenv("SURGE_GATE_MODEL");
    if (!path) {
        fprintf(stderr, "   SKIP: set SURGE_GATE_MODEL to a real model "
                        "(e.g. /Users/macmini/models/qwen35-2b) to run the M5.7 "
                        "long-context gate\n");
        return;
    }

    /* A .gguf path uses the GGUF loader; anything else is a safetensors
     * directory (the 2B interim model). */
    sg_gguf *gg = NULL;
    sg_st *st = NULL;
    sg_model m;
    sg_err e;
    size_t plen = strlen(path);
    bool is_gguf = plen >= 5 && strcmp(path + plen - 5, ".gguf") == 0;
    if (is_gguf) {
        e = sg_gguf_open(path, &gg);
        tt_assert(!sg_failed(e), "gate: sg_gguf_open %s: %s", path, e.msg ? e.msg : "ok");
        if (sg_failed(e)) return;
        e = sg_model_from_gguf(gg, &m);
        tt_assert(!sg_failed(e), "gate: sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
        if (sg_failed(e)) { sg_gguf_close(gg); return; }
    } else {
        e = sg_st_open(path, &st);
        tt_assert(!sg_failed(e), "gate: sg_st_open %s: %s", path, e.msg ? e.msg : "ok");
        if (sg_failed(e)) return;
        e = sg_model_from_st(st, &m);
        tt_assert(!sg_failed(e), "gate: sg_model_from_st: %s", e.msg ? e.msg : "ok");
        if (sg_failed(e)) { sg_st_close(st); return; }
    }

    uint32_t vocab = m.cfg.vocab;
    fprintf(stderr, "   gate model: %s (%u layers, vocab %u, full_attn_interval %u)\n",
            path, m.cfg.n_layers, vocab, m.cfg.full_attn_interval);

    e = sg_gpu_load_model(g_gpu, &m);
    tt_assert(!sg_failed(e), "gate: sg_gpu_load_model: %s", e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        /* One deterministic id stream, reused at every depth (each depth takes
         * the first `depth` ids). SG_KV_CAP_MAX ids so gate B uses it whole. */
        int32_t *ids = xmalloc((size_t)SG_KV_CAP_MAX * sizeof *ids);
        lcg_fill_ids(ids, SG_KV_CAP_MAX, vocab, 0x9E3779B97F4A7C15ULL);

        /* ---- (A) depth equivalence: prefill+decode 32 == serial+decode 32 --- */
        static const uint32_t depths[] = { 8192u, 16384u, 32768u };
        bool skip_a = getenv("SURGE_GATE_SKIP_A") != NULL;
        if (skip_a) fprintf(stderr, "   [A] SKIPPED (SURGE_GATE_SKIP_A set)\n");
        for (uint32_t d = 0; !skip_a && d < sizeof depths / sizeof depths[0]; d++) {
            uint32_t depth = depths[d];
            e = sg_gpu_state_new(g_gpu, &m, depth + N_GATE_GEN);
            tt_assert(!sg_failed(e), "gate A depth %u: sg_gpu_state_new: %s",
                      depth, e.msg ? e.msg : "ok");
            if (sg_failed(e)) continue;

            /* Prefill path. sg_gpu_prefill resets state internally; the explicit
             * reset makes the "reset between the two runs" contract visible. */
            double t0 = now_s();
            sg_gpu_state_reset(g_gpu);
            const float *lg = NULL;
            e = sg_gpu_prefill(g_gpu, &m, ids, depth, 1024, &lg);
            tt_assert(!sg_failed(e), "gate A depth %u: sg_gpu_prefill: %s",
                      depth, e.msg ? e.msg : "ok");
            if (sg_failed(e)) continue;
            int32_t pre_gen[N_GATE_GEN];
            for (uint32_t i = 0; i < N_GATE_GEN; i++) {
                pre_gen[i] = (int32_t)argmax_f32(lg, vocab);
                if (i + 1 == N_GATE_GEN) break;
                e = sg_gpu_forward(g_gpu, &m, pre_gen[i], depth + i, &lg);
                tt_assert(!sg_failed(e), "gate A depth %u: prefill-decode %u: %s",
                          depth, i, e.msg ? e.msg : "ok");
                if (sg_failed(e)) break;
            }
            if (sg_failed(e)) continue;
            double t_prefill = now_s() - t0;

            /* Serial oracle: the prompt one token at a time, then decode 32. */
            t0 = now_s();
            sg_gpu_state_reset(g_gpu);
            lg = NULL;
            for (uint32_t t = 0; t < depth; t++) {
                e = sg_gpu_forward(g_gpu, &m, ids[t], t, &lg);
                if (sg_failed(e)) break;
            }
            tt_assert(!sg_failed(e), "gate A depth %u: serial forward: %s",
                      depth, e.msg ? e.msg : "ok");
            if (sg_failed(e)) continue;
            int32_t ser_gen[N_GATE_GEN];
            for (uint32_t i = 0; i < N_GATE_GEN; i++) {
                ser_gen[i] = (int32_t)argmax_f32(lg, vocab);
                if (i + 1 == N_GATE_GEN) break;
                e = sg_gpu_forward(g_gpu, &m, ser_gen[i], depth + i, &lg);
                if (sg_failed(e)) break;
            }
            tt_assert(!sg_failed(e), "gate A depth %u: serial decode: %s",
                      depth, e.msg ? e.msg : "ok");
            if (sg_failed(e)) continue;
            double t_serial = now_s() - t0;

            uint32_t mism = 0, first_bad = N_GATE_GEN;
            for (uint32_t i = 0; i < N_GATE_GEN; i++) {
                if (pre_gen[i] != ser_gen[i]) {
                    mism++;
                    if (first_bad == N_GATE_GEN) first_bad = i;
                }
            }
            tt_assert(mism == 0, "gate A depth %u: %u/%u token mismatches "
                      "(first at step %u)", depth, mism, N_GATE_GEN, first_bad);
            fprintf(stderr, "   [A] depth %6u: %u/%u tokens match "
                    "(prefill+decode %.2fs, serial+decode %.2fs)\n",
                    depth, N_GATE_GEN - mism, N_GATE_GEN, t_prefill, t_serial);
        }

        /* ---- (B) 262144 ingest: fill the whole 262144-position cache and
         * decode at ~256K depth in ONE run. SG_KV_CAP_MAX == 262144 leaves no
         * slot to append after a full-cap prefill, so the context is filled as
         * (cap - 32) prefill + 32 decode, reaching used == cap. SURGE_GATE_B_CAP
         * overrides the cap for a cheap smoke test of this used-counter logic;
         * the gate itself (unset) uses the full SG_KV_CAP_MAX == 262144. ---- */
        uint32_t bcap = SG_KV_CAP_MAX;
        const char *bcap_env = getenv("SURGE_GATE_B_CAP");
        if (bcap_env) {
            unsigned long v = strtoul(bcap_env, NULL, 10);
            if (v > N_GATE_GEN && v <= SG_KV_CAP_MAX) bcap = (uint32_t)v;
        }
        e = sg_gpu_state_new(g_gpu, &m, bcap);
        tt_assert(!sg_failed(e), "gate B: sg_gpu_state_new(%u): %s",
                  bcap, e.msg ? e.msg : "ok");
        if (!sg_failed(e)) {
            uint32_t pdepth = bcap - N_GATE_GEN;
            double t0 = now_s();
            const float *lg = NULL;
            e = sg_gpu_prefill(g_gpu, &m, ids, pdepth, 1024, &lg);
            tt_assert(!sg_failed(e), "gate B: sg_gpu_prefill(%u): %s",
                      pdepth, e.msg ? e.msg : "ok");
            double t_ingest = now_s() - t0;
            if (!sg_failed(e)) {
                tt_assert(sg_gpu_used(g_gpu) == pdepth,
                          "gate B: used counter %u != %u after prefill",
                          sg_gpu_used(g_gpu), pdepth);
                tt_assert(logits_nondegenerate(lg, vocab),
                          "gate B: final prefill-position logits are degenerate "
                          "(non-finite, all-equal, or argmax out of range)");
                fprintf(stderr, "   [B] ingest %u (prefill): used=%u, final logits "
                        "non-degenerate (argmax %u), prefill %.2fs\n",
                        pdepth, sg_gpu_used(g_gpu), argmax_f32(lg, vocab), t_ingest);

                /* Decode 32 at depth -> used reaches bcap (262144 for the gate). */
                t0 = now_s();
                int32_t gen[N_GATE_GEN];
                bool all_finite = true, all_in_range = true;
                for (uint32_t i = 0; i < N_GATE_GEN; i++) {
                    for (uint32_t k = 0; k < vocab; k++) {
                        if (!isfinite(lg[k])) { all_finite = false; break; }
                    }
                    uint32_t arg = argmax_f32(lg, vocab);
                    if (arg >= vocab) all_in_range = false;
                    gen[i] = (int32_t)arg;
                    /* Feed EVERY decoded token back (all 32 forwards, positions
                     * pdepth..pdepth+31) so the whole bcap-position cache is
                     * filled and the used counter reaches exactly bcap (262144
                     * for the gate). Unlike
                     * an ordinary greedy loop (which skips the final feedback
                     * because it does not need that token's logits), here the
                     * position advance is the point; the last forward's logits
                     * are simply unused. */
                    e = sg_gpu_forward(g_gpu, &m, gen[i], pdepth + i, &lg);
                    tt_assert(!sg_failed(e), "gate B: decode %u: %s", i, e.msg ? e.msg : "ok");
                    if (sg_failed(e)) break;
                }
                double t_decode = now_s() - t0;
                uint32_t distinct = 1;
                for (uint32_t i = 1; i < N_GATE_GEN; i++) {
                    bool seen = false;
                    for (uint32_t j = 0; j < i; j++) if (gen[j] == gen[i]) { seen = true; break; }
                    if (!seen) distinct++;
                }
                tt_assert(all_finite, "gate B: a decode step produced non-finite logits");
                tt_assert(all_in_range, "gate B: a decode argmax was out of vocab range");
                tt_assert(distinct >= 2, "gate B: 32 decoded tokens collapsed to a single "
                          "repeated id (%u distinct)", distinct);
                if (!sg_failed(e)) {
                    tt_assert(sg_gpu_used(g_gpu) == bcap,
                              "gate B: used counter %u != %u after 32-token decode",
                              sg_gpu_used(g_gpu), bcap);
                }
                fprintf(stderr, "   [B] decode %u at depth: %u distinct ids, finite + "
                        "in-range, used=%u (== %u), decode %.2fs\n",
                        N_GATE_GEN, distinct, sg_gpu_used(g_gpu), bcap, t_decode);
            }
        }

        free(ids);
    }

    sg_model_free(&m);
    if (st) sg_st_close(st);
    if (gg) sg_gguf_close(gg);
}

int main(void) {
    sg_err e = sg_gpu_init(&g_gpu);
    if (sg_failed(e)) {
        fprintf(stderr, "SKIP test_gpu_prefill: %s\n", e.msg);
        return 0;
    }

    /* Prefill requires the f16 KV path; set it explicitly so this test does not
     * depend on whatever the process default happens to be. */
    setenv("SURGE_KV_DTYPE", "f16", 1);
    tt_run("mini_st_prefill", mini_st_prefill);
    tt_run("mini_gguf_prefill", mini_gguf_prefill);
    /* M5.7: the real-model long-context gate. SKIPs cleanly (make check stays
     * mini-only and hermetic) unless SURGE_GATE_MODEL points at a real model. */
    tt_run("gate_real_model", gate_real_model);
    unsetenv("SURGE_KV_DTYPE");

    fprintf(stderr, "worst prefill-vs-serial last-logit relative gap: %.3e\n", g_worst_rel);
    sg_gpu_free(g_gpu);
    return tt_report();
}

#endif /* SURGE_NO_METAL */
