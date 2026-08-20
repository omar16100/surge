/* test_gpu_mem.c - Task B2: peak-memory probe.
 *
 * Two tiers, deliberately split so the pure-C tracker math is exercised by
 * BOTH `make check` and `make debug` (SURGE_NO_METAL, ASan/UBSan) while the
 * mach/Metal SAMPLING stays live-and-guarded like the rest of the Metal
 * surface:
 *
 *   TIER 1 (always runs): sg_mem_tracker's max() logic, unit-exact over a
 *   known sequence, feeding it plain numbers -- no Metal, no mach syscall,
 *   no GPU.
 *
 *   TIER 2 (SKIPs under SURGE_NO_METAL; live otherwise): sg_proc_phys_
 *   footprint and sg_gpu_current_alloc_bytes against a real Metal device --
 *   gate (2) loads a model and checks phys_footprint exceeds the Metal
 *   allocation alone; gate (3) allocates a modest fp16 KV cache through
 *   sg_kv_new and checks the tracked peak grows by at least 90% of the
 *   computed budget.
 *
 * Unlike test_gpu_fwd.c / test_metal_ops.c, the whole file is NOT a bare
 * skip stub under -DSURGE_NO_METAL: only tier 2 is #ifndef-guarded out,
 * because tier 1 needs to keep running (and proving the tracker byte-exact)
 * under ASan/UBSan too. sg_proc_phys_footprint / sg_mem_tracker_* live in
 * src/bench.c (LIB_SRC, links into every test binary); sg_gpu_current_
 * alloc_bytes lives in src/metal.m and is therefore unreachable (and
 * untested) here under SURGE_NO_METAL, exactly like every other Metal
 * symbol. See the Makefile's METAL_HYBRID_TESTS rule, which links this file
 * against LIB_SRC in both modes but only pulls in $(METAL_M) (metal.m +
 * metal_prefill.m + metal_validate.m, three translation units since task R3)
 * / the frameworks when Metal is actually being built.
 */
#include "tinytest.h"
#include "../surge.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* --------------------------------------------------------------------
 * Tier 1: the pure-C tracker. No Metal, no mach, runs under check AND debug.
 * -------------------------------------------------------------------- */

/* The gate's own known sequence, verbatim: (10,20)->20, (5,5)->still 20,
 * (100,3)->100, (0,0)->still 100, then reset->0. */
static void tracker_max_is_unit_exact(void) {
    sg_mem_tracker t;
    sg_mem_tracker_reset(&t);
    tt_assert(sg_mem_tracker_peak(&t) == 0, "fresh/reset tracker peak is 0, got %llu",
              (unsigned long long)sg_mem_tracker_peak(&t));

    sg_mem_tracker_sample(&t, 10, 20);
    tt_assert(sg_mem_tracker_peak(&t) == 20,
              "(10,20) -> peak 20, got %llu", (unsigned long long)sg_mem_tracker_peak(&t));

    sg_mem_tracker_sample(&t, 5, 5);
    tt_assert(sg_mem_tracker_peak(&t) == 20,
              "(5,5) -> peak stays 20, got %llu", (unsigned long long)sg_mem_tracker_peak(&t));

    sg_mem_tracker_sample(&t, 100, 3);
    tt_assert(sg_mem_tracker_peak(&t) == 100,
              "(100,3) -> peak 100, got %llu", (unsigned long long)sg_mem_tracker_peak(&t));

    sg_mem_tracker_sample(&t, 0, 0);
    tt_assert(sg_mem_tracker_peak(&t) == 100,
              "(0,0) -> peak stays 100, got %llu", (unsigned long long)sg_mem_tracker_peak(&t));

    sg_mem_tracker_reset(&t);
    tt_assert(sg_mem_tracker_peak(&t) == 0,
              "reset -> peak 0, got %llu", (unsigned long long)sg_mem_tracker_peak(&t));
}

/* Order-independence within one sample() call: max(current_alloc,
 * phys_footprint) must not silently assume one argument is always the
 * larger. Feed the two orderings of an unequal pair and confirm both land
 * on the larger value. */
static void tracker_sample_order_independent(void) {
    sg_mem_tracker t;
    sg_mem_tracker_reset(&t);
    sg_mem_tracker_sample(&t, 7, 42);
    tt_assert(sg_mem_tracker_peak(&t) == 42, "(7,42) -> 42, got %llu",
              (unsigned long long)sg_mem_tracker_peak(&t));

    sg_mem_tracker_reset(&t);
    sg_mem_tracker_sample(&t, 42, 7);
    tt_assert(sg_mem_tracker_peak(&t) == 42, "(42,7) -> 42, got %llu",
              (unsigned long long)sg_mem_tracker_peak(&t));
}

/* NULL-safety: matches this project's defensive-NULL convention throughout
 * (sg_kv_*, sg_bench_*). Must not crash and must read back 0. */
static void tracker_null_is_safe(void) {
    sg_mem_tracker_reset(NULL);
    sg_mem_tracker_sample(NULL, 123, 456);
    tt_assert(sg_mem_tracker_peak(NULL) == 0, "NULL tracker peak reads 0");
}

#ifndef SURGE_NO_METAL
/* --------------------------------------------------------------------
 * Tier 2: live Metal + mach. Compiled out entirely under -DSURGE_NO_METAL.
 * -------------------------------------------------------------------- */

#define MINI_DIR "tests/fixtures/mini_fwd"

static sg_gpu *g_gpu;

/* Model + checkpoint backing memory for everything this file loads into
 * g_gpu, kept alive until g_gpu itself is freed (see main()'s cleanup
 * order). Two contracts this satisfies that per-function-scoped locals
 * would not:
 *   (1) sg_gpu_wrap's (surge.h): wrapped memory must outlive the Metal
 *       handle wrapping it -- sg_gpu_load_model wraps every matmul weight
 *       straight out of the checkpoint's mmap with no copy.
 *   (2) sg_gpu_load_model stores g->model = m, the RAW POINTER passed in
 *       (not a copy of the struct), and dereferences it on every later
 *       call that touches this gpu (gpu_wrap_w's g->model->wtype during
 *       the load itself, and the `m != g->model` identity checks in
 *       sg_gpu_state_new/_forward/_prefill) -- so the sg_model struct
 *       itself, not just its checkpoint bytes, must outlive g_gpu's use of
 *       it.
 * Both are latent rather than live bugs with today's call sequence
 * (nothing here calls sg_gpu_forward/_state_new after a load, and
 * gpu_unload only NULLs g->model rather than dereferencing it before the
 * next sg_gpu_load_model overwrites it), but test_gpu_fwd.c /
 * test_gpu_prefill.c's established pattern is to keep a loaded model alive
 * for the whole window g_gpu can reach it. Matching that here removes the
 * hazard outright instead of relying on which calls happen not to trip it.
 * Static storage duration zero-initializes all of these, so freeing them
 * unconditionally at the end (sg_model_free / sg_gguf_close / sg_st_close
 * all tolerate a never-populated/NULL argument) is safe even when a load
 * never ran or failed partway through. */
static sg_model g_mini_model;
static sg_gguf *g_mini_gg;
static sg_model g_real_model;
static sg_gguf *g_real_gg;
static sg_st   *g_real_st;

/* Loads SURGE_GATE_MODEL if set (a real model: a .gguf path or a
 * safetensors directory, same dispatch rule tests/test_gpu_prefill.c uses),
 * otherwise falls back to the mini hybrid fixture that ships in the repo --
 * so this gate is hermetic by default and only needs sg_gpu_init to succeed
 * (a machine with no Metal device SKIPs tier 2 entirely, same as
 * test_gpu_fwd.c / test_metal_ops.c). *gg_out / *st_out receive whichever
 * loader owns the returned model's backing memory (a pointer into one of
 * the file-scope statics above, so main() can close it after g_gpu is
 * freed); exactly one of them is non-NULL on success. */
static bool load_probe_model(sg_model *m, sg_gguf **gg_out, sg_st **st_out) {
    *gg_out = NULL;
    *st_out = NULL;
    const char *path = getenv("SURGE_GATE_MODEL");
    if (!path) path = MINI_DIR "/model.gguf";

    size_t plen = strlen(path);
    bool is_gguf = plen >= 5 && strcmp(path + plen - 5, ".gguf") == 0;
    sg_err e;
    if (is_gguf) {
        e = sg_gguf_open(path, gg_out);
        tt_assert(!sg_failed(e), "load_probe_model: sg_gguf_open %s: %s",
                  path, e.msg ? e.msg : "ok");
        if (sg_failed(e)) return false;
        e = sg_model_from_gguf(*gg_out, m);
        tt_assert(!sg_failed(e), "load_probe_model: sg_model_from_gguf: %s",
                  e.msg ? e.msg : "ok");
        if (sg_failed(e)) { sg_gguf_close(*gg_out); *gg_out = NULL; return false; }
    } else {
        e = sg_st_open(path, st_out);
        tt_assert(!sg_failed(e), "load_probe_model: sg_st_open %s: %s",
                  path, e.msg ? e.msg : "ok");
        if (sg_failed(e)) return false;
        e = sg_model_from_st(*st_out, m);
        tt_assert(!sg_failed(e), "load_probe_model: sg_model_from_st: %s",
                  e.msg ? e.msg : "ok");
        if (sg_failed(e)) { sg_st_close(*st_out); *st_out = NULL; return false; }
    }
    fprintf(stderr, "   probe model: %s (%u layers, vocab %u)\n",
            path, m->cfg.n_layers, m->cfg.vocab);
    return true;
}

/* Gate (2): after loading a model, sg_proc_phys_footprint() strictly
 * exceeds sg_gpu_current_alloc_bytes() -- process footprint is code, stacks
 * and every other allocation ON TOP OF whatever Metal itself is holding, so
 * it can never be smaller once anything beyond a bare Metal handle exists.
 *
 * Deliberately the MINI FIXTURE ONLY, never SURGE_GATE_MODEL: this ordering
 * holds only while the model's WRAPPED (no-copy) weights stay small next to
 * the process's other footprint. Metal's currentAllocatedSize counts a
 * newBufferWithBytesNoCopy wrap (sg_gpu_wrap, used for every matmul weight)
 * at its full DECLARED length the instant it is created, regardless of how
 * many pages are actually resident -- verified live against the real 2B
 * (SURGE_GATE_MODEL=/Users/macmini/models/qwen35-2b): immediately after
 * sg_gpu_load_model, gpu_alloc read 3,768,385,536 bytes (the ~3.5 GiB of
 * wrapped bf16 weights) while phys_footprint read only 10,683,616 bytes,
 * the opposite of this gate's assertion. See the full writeup on
 * sg_proc_phys_footprint in surge.h. The mini fixture's weights (hidden=32)
 * are a few KB, negligible next to the process's baseline footprint, so the
 * ordering holds reliably and hermetically here; gpu_alloc_vs_phys_
 * footprint_real_model_note below observes (without asserting an ordering
 * on) the real-model case instead. */
static void phys_footprint_exceeds_gpu_alloc_after_load(void) {
    sg_err e = sg_gguf_open(MINI_DIR "/model.gguf", &g_mini_gg);
    tt_assert(!sg_failed(e), "sg_gguf_open: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    e = sg_model_from_gguf(g_mini_gg, &g_mini_model);
    tt_assert(!sg_failed(e), "sg_model_from_gguf: %s", e.msg ? e.msg : "ok");
    if (sg_failed(e)) return;

    e = sg_gpu_load_model(g_gpu, &g_mini_model);
    tt_assert(!sg_failed(e), "sg_gpu_load_model: %s", e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        uint64_t alloc = sg_gpu_current_alloc_bytes(g_gpu);
        uint64_t phys  = sg_proc_phys_footprint();
        fprintf(stderr, "   gpu_alloc=%llu bytes phys_footprint=%llu bytes\n",
                (unsigned long long)alloc, (unsigned long long)phys);
        tt_assert(phys > alloc,
                  "phys_footprint (%llu) should exceed gpu_alloc (%llu) after model load",
                  (unsigned long long)phys, (unsigned long long)alloc);
        tt_assert(alloc > 0, "gpu_alloc should be nonzero after a real model load");
    }

    /* g_mini_model / g_mini_gg deliberately NOT freed here -- see the
     * file-scope comment above; main() cleans up after sg_gpu_free. */
}

/* SURGE_GATE_MODEL, env-gated (SKIPs cleanly without it, like tests/
 * test_gpu_prefill.c's gate_real_model / test_metal_ops.c's
 * q8_loads_and_decodes): loads a real model and OBSERVES both probes rather
 * than asserting the gate-2 ordering on them, per the comment above -- a
 * real model's wrapped weights routinely push gpu_alloc past
 * phys_footprint right after load. What this DOES assert: both probes
 * return real, nonzero numbers for a real model, so a probe that silently
 * broke (always-0 due to a bad task_info call, or a NULL-guard tripping
 * where it should not) would still be caught here. */
static void gpu_alloc_vs_phys_footprint_real_model_note(void) {
    const char *path = getenv("SURGE_GATE_MODEL");
    if (!path) {
        fprintf(stderr, "   SKIP: set SURGE_GATE_MODEL to a real model to see the "
                        "wrapped-weight gpu_alloc vs phys_footprint relationship\n");
        return;
    }
    if (!load_probe_model(&g_real_model, &g_real_gg, &g_real_st)) return;

    sg_err e = sg_gpu_load_model(g_gpu, &g_real_model);
    tt_assert(!sg_failed(e), "sg_gpu_load_model (real model): %s", e.msg ? e.msg : "ok");
    if (!sg_failed(e)) {
        uint64_t alloc = sg_gpu_current_alloc_bytes(g_gpu);
        uint64_t phys  = sg_proc_phys_footprint();
        fprintf(stderr,
                "   real model: gpu_alloc=%llu bytes (%.2f GiB) phys_footprint=%llu bytes (%.2f GiB)\n",
                (unsigned long long)alloc, (double)alloc / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)phys, (double)phys / (1024.0 * 1024.0 * 1024.0));
        tt_assert(alloc > 0, "gpu_alloc nonzero after a real model load");
        tt_assert(phys > 0, "phys_footprint nonzero after a real model load");
    }

    /* g_real_model / g_real_gg / g_real_st deliberately NOT freed here --
     * see the file-scope comment above; main() cleans up after
     * sg_gpu_free. */
}

/* Gate (3): sg_kv_new's growth is trackable, and reaches at least 90% of
 * the computed fp16 budget. A synthetic single-layer cfg (independent of
 * whatever gate 2 loaded), sized so a MODEST cap (76800 positions, well
 * under SG_KV_CAP_MAX) lands the budget at exactly 300 MiB -- fast and
 * light enough to allocate in CI, nowhere near the 16 GiB the real 27B
 * shape would need at full cap (see tests/test_kv.c, which asserts THAT
 * number by pure math alone, never allocating it). full_attn_interval=1
 * with one layer makes that one layer full-attention, so no DeltaNet dims
 * are needed. */
static void kv_growth_meets_90pct_of_budget(void) {
    sg_cfg c;
    memset(&c, 0, sizeof c);
    c.n_layers = 1;
    c.n_kv_heads = 8;
    c.head_dim = 128;
    c.full_attn_interval = 1;
    uint32_t cap = 76800;

    uint64_t budget = sg_kv_bytes(&c, cap, SG_T_F16);
    uint64_t want_budget = 300ULL * 1024 * 1024;   /* pin the advertised 300 MiB exactly */
    tt_assert(budget == want_budget,
              "sanity: cfg/cap should compute exactly 300 MiB (%llu), got %llu bytes -- "
              "if this drifts, the 90%% growth bar below is no longer proving what it claims",
              (unsigned long long)want_budget, (unsigned long long)budget);
    fprintf(stderr, "   kv budget: cap=%u -> %llu bytes (%.1f MiB)\n",
            cap, (unsigned long long)budget, (double)budget / (1024.0 * 1024.0));

    sg_mem_tracker before;
    sg_mem_tracker_reset(&before);
    sg_mem_tracker_sample(&before, sg_gpu_current_alloc_bytes(g_gpu), sg_proc_phys_footprint());
    uint64_t peak_before = sg_mem_tracker_peak(&before);

    sg_kv *kv = NULL;
    sg_err e = sg_kv_new(g_gpu, &c, cap, SG_T_F16, &kv);
    tt_assert(!sg_failed(e) && kv, "sg_kv_new: %s", e.msg ? e.msg : "ok");
    if (!kv) return;

    sg_mem_tracker after;
    sg_mem_tracker_reset(&after);
    sg_mem_tracker_sample(&after, sg_gpu_current_alloc_bytes(g_gpu), sg_proc_phys_footprint());
    uint64_t peak_after = sg_mem_tracker_peak(&after);

    uint64_t growth = (peak_after > peak_before) ? peak_after - peak_before : 0;
    double pct = budget > 0 ? (100.0 * (double)growth / (double)budget) : 0.0;
    fprintf(stderr, "   peak_before=%llu peak_after=%llu growth=%llu (%.1f%% of budget)\n",
            (unsigned long long)peak_before, (unsigned long long)peak_after,
            (unsigned long long)growth, pct);
    tt_assert((double)growth >= 0.9 * (double)budget,
              "tracked peak grew %llu bytes, want >= 90%% of %llu (%.1f%%)",
              (unsigned long long)growth, (unsigned long long)budget, pct);

    sg_kv_free(kv);
}
#endif /* SURGE_NO_METAL */

int main(void) {
    tt_run("tracker max() unit-exact over the gate's known sequence", tracker_max_is_unit_exact);
    tt_run("tracker sample() is order-independent", tracker_sample_order_independent);
    tt_run("tracker NULL-safe", tracker_null_is_safe);

#ifndef SURGE_NO_METAL
    {
        sg_err e = sg_gpu_init(&g_gpu);
        if (sg_failed(e)) {
            fprintf(stderr, "SKIP test_gpu_mem tier 2 (live Metal): %s\n", e.msg);
            return tt_report();
        }
        tt_run("phys_footprint > gpu_alloc after model load",
               phys_footprint_exceeds_gpu_alloc_after_load);
        tt_run("gpu_alloc vs phys_footprint on a real model (SURGE_GATE_MODEL, observational)",
               gpu_alloc_vs_phys_footprint_real_model_note);
        tt_run("kv growth >= 90% of computed fp16 budget",
               kv_growth_meets_90pct_of_budget);
        sg_gpu_free(g_gpu);

        /* Only now: g_gpu no longer references either model or its
         * checkpoint mmap (see the file-scope comment above
         * load_probe_model). Every one of these tolerates a
         * never-populated/NULL argument (static storage zero-inits them),
         * so this is safe regardless of which loads above ran or failed. */
        sg_model_free(&g_mini_model);
        if (g_mini_gg) sg_gguf_close(g_mini_gg);
        sg_model_free(&g_real_model);
        if (g_real_gg) sg_gguf_close(g_real_gg);
        if (g_real_st) sg_st_close(g_real_st);
    }
#else
    fprintf(stderr, "SKIP test_gpu_mem tier 2: built with -DSURGE_NO_METAL "
                    "(Metal and the ASan/UBSan run do not mix)\n");
#endif

    return tt_report();
}
