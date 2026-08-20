# surge architecture (C4 model)

Source of truth for surge's structure. Read before any architecture change; update on
every architecture change (containers, components, data flows, status). surge is a
from-scratch LLM inference engine in C11 + Metal for the Mac Studio M3 Ultra, no
dependencies beyond what macOS ships. Public: github.com/omar16100/surge (MIT).

## Level 1: Context

surge takes a model on disk (GGUF or safetensors) and a prompt, and generates tokens on
the Apple Silicon GPU (Metal). It is a single-process, single-user, greedy-first engine
built to run the hybrid `qwen3_5`/`qwen35` family model end to end and, per the M3+M5
plan, to run the same Qwen3.6-27B-Q8_0 GGUF the mlx-lm and llama.cpp engines are compared
on at 262,144-token context. Task P1 (loader-only) adds the plain DENSE `qwen3` GGUF
family alongside it (e.g. Qwen3-4B-Instruct-2507), motivated by the llm-rnd 256K
comparison: Qwen3-4B-Instruct-2507 is one of only two models that recall 8/8 needles at
262,144 tokens, so making it fast is the next milestone's goal; P1 only makes it loadable.
Correctness is validated against a CPU reference and against mlx-lm; speed against mlx-lm
on the same weights (that comparison lives in the llm-rnd project's `256k_comparison.md`
and `leaderboard.md`).

External dependencies: only Metal.framework, Foundation (for the one .m file), and
optionally Accelerate for the CPU reference path. No third-party libraries.

## Level 2: Containers

- **libsurge (static object set)** compiled from `src/*.c` (minus the CLI mains and
  `metal.m`), linked into every binary and every pure-C test. Metal-free, so pure-C
  tests build and run without a GPU.
- **Metal layer**: TWO Objective-C host translation units, `src/metal.m` (device,
  command buffers, kernel registration, per-op dispatch, decode) and
  `src/metal_prefill.m` (chunked prompt prefill), over the shared internal header
  `src/metal_internal.h`; plus TWO shader translation units, `src/kernels.metal` and
  `src/kernels_splitk.metal`, over the shared header `src/kernels_common.metal.h`.
  `xcrun metal` compiles each shader source to its own `.air` and `metallib` links both
  into the ONE `src/kernels.metallib`, so the host still looks every kernel up by name in
  a single library (task R1, 2026-08-18: a pure move out of a 2548-line file, gated on a
  byte-identical metallib and per-function AIR equivalence, not just green tests). The
  two `.m` files are different in kind from the two `.metal` ones: they REALLY link, so
  the split promoted nine `static` helpers to external linkage and R1's byte-identity
  gate does not transfer (task R2, 2026-08-20; gated on the exact `make check` /
  `make debug` counts instead). Both `.m` files are on every link line that names the
  Metal layer (the Makefile's `METAL_M`); neither is optional.
  Only binaries that need the GPU link these.
- **CLIs**: `surge-info` (GGUF dump), `surge-ref` (CPU-reference forward + `--logits`),
  `surge` (Metal decode), `surge-bench` (benchmark harness; B5 shipped, wires B1-B4 + the
  B2 peak-memory probe to M5 prefill + the shared greedy driver, emits one leaderboard row).
- **Tests**: `tests/test_*.c` under a tinytest harness; Metal tests skip gracefully when
  `sg_gpu_init` fails and are excluded from the ASan build via `SURGE_NO_METAL`.

## Level 3: Components (files, one responsibility each)

| file | responsibility |
|---|---|
| `src/gguf.c` | mmap GGUF v3 reader: metadata table, tensor directory, Q8_0/bf16/f16/f32 tensor views. Bounds-checked; no copies. |
| `src/st.c` | read-only safetensors bf16 loader + minimal config.json JSON scanner. Handles unaligned data sections (mlx repacks) via a copying accessor. |
| `src/tok.c` | byte-level BPE tokenizer built from `tokenizer.ggml.*` GGUF metadata; fixture-exact vs the real tokenizer. |
| `src/model_qwen.c` | config extraction + weight-name mapping for the hybrid qwen3_5/qwen35 arch (full-attention + gated-DeltaNet layers) and, since Task P1, the plain dense `qwen3` arch (uniform full-attention, no DeltaNet, single-width q_proj), from GGUF and safetensors (dense is GGUF-only so far). Carries per-tensor dtype, the source-flags `ssm_a_form` / `v_heads_tiled`, and `cfg.attn_output_gate` (P1: true for the hybrid's folded sigmoid gate, false for dense). |
| `src/ref.c` | scalar CPU reference for every op (the correctness oracle), incl. `sg_ref_matvec_q8`, attention, gated-DeltaNet, partial RoPE, the full forward pass, (Task P2.0) `sg_ref_attn_combine`, the split-K attention combine (log-sum-exp rescaling that merges per-partition partial attention results), and (Task P2.1) `sg_ref_attn_decode` / `sg_ref_attn_decode_splitk`, the direct and split-K decode-attention-core oracles built on top of it -- together the CPU-proven math for the not-yet-built split-K decode-attention Metal kernel. |
| `src/kv.c` | (M5.1) fp16 growable KV cache for full-attention layers + fixed-size DeltaNet recurrent state, over opaque GPU-buffer handles; Metal-free (allocation injected). `sg_kv_bytes` = 16 GiB K+V at 262,144. Wired into decode by M5.2 (see below). |
| `src/bench.c` | (B-series) pure-C benchmark math: decode-by-slope, leaderboard-row/JSON formatters, prompt file read, ingestion/truncation guard, NIAH recall scorer, process `phys_footprint` probe + the `sg_mem_tracker` peak-max tracker (B2). |
| `src/greedy.c` | the ONE greedy-decode argmax (`sg_argmax_f32`, lowest index wins an exact tie). Pure C in LIB_SRC; `surge` and `surge-bench` both call it so their gen_ids cannot drift (B5). |
| `src/sched.c` | (P3.0) decode pacing + GPU clamp detection. Pure C in LIB_SRC, no Metal/Foundation/GPU. `sg_decode_pace_decide` is the WHOLE policy and is a pure function of (step timings, budget, rest) with no clock read and no sleep in it, so `make debug` covers it with no GPU; `sg_decode_pace_step` is that plus the `nanosleep`. Two separable parts: a duty cycle (`work_budget_ms`/`rest_ms`, the decode analogue of B8's prefill rest, OFF unless both are > 0) and a clamp detector (per-step decode wall time against the MEDIAN of the run's first 8 steps, `> 1.5x` for 3 consecutive steps to latch, 3 to clear, hysteresis both ways) that runs unconditionally, never sleeps on its own, and by default (`clamp_div == 1`) drives no rest schedule at all. It lives OUTSIDE `metal.m` because there is no decode loop inside `metal.m` (`sg_gpu_forward` is one step); the per-token loop belongs to the caller, so the pacer is a caller-owned value. Deliberately contains no fan/power/daemon hook of any kind. |
| `src/metal.m` | (task R2, 2026-08-20: the chunked-prefill half moved to `src/metal_prefill.m`, leaving 3547 lines here.) Metal device init, weight-buffer wrapping (mmap, no-copy; bf16/f32/Q8_0 sizing), per-weight matvec kernel dispatch (`matmul_kernel_for`), one command buffer per decode token; one command buffer per prefill chunk (`sg_gpu_prefill`, M5.6, now in `src/metal_prefill.m`) unless `sg_gpu_set_prefill_max_burst` splits the layer sweep across several (2026-08-15), kernel registration (KI_ enum + SG_KERNELS table + size/param checks), `sg_gpu_current_alloc_bytes` (B2, `MTLDevice.currentAllocatedSize`). Registers itself as `sg_kv`'s allocation backend at init. Decode state: the DEFAULT full-attention KV cache is fp16, allocated through `sg_kv` as SEPARATE per-layer K/V buffers (M5.2, `SURGE_KV_DTYPE` env toggle); `SURGE_KV_DTYPE=f32` selects the original combined-buffer f32 path unchanged, kept so the M2 gate's oracle never moves. DeltaNet conv/S state stays on its pre-M5.1 ad hoc allocation for decode; prefill (M5.6) threads DeltaNet state through `sg_kv`'s conv/S carriers (so `g->kv` is now allocated for any DeltaNet model on the f16 path) and bridges the final state back into the ad hoc buffers when it finishes. `sg_gpu_prefill`'s chunk loop also carries the B8 prefill duty-cycle: `sg_gpu_set_prefill_rest` arms an optional idle sleep (`pf_sleep_ms`, `nanosleep`+EINTR-retry) between chunks once the accumulated GPU-busy time PLUS an estimate of the next chunk's would cross a work budget, so the GPU is yielded before the budget is blown rather than after. Its purpose is keeping the compositor alive, not dodging a firmware clock clamp (rationale corrected 2026-08-15, see the B8 block below). Alongside it, `sg_gpu_set_prefill_max_burst` bounds how long a SINGLE command buffer holds the GPU by splitting the layer sweep. Both are disabled by default and both are pure timing or pure submission-boundary changes (touching no buffer/state/accumulator), so neither changes computed output. |
| `src/metal_prefill.m` | (task R2, 2026-08-20) the chunked prompt prefill half of the Metal host layer, 826 lines moved verbatim out of `src/metal.m` when that file reached 4641 against the ~2000-line guideline: the two per-layer CHUNK encoders `enc_attn_prefill` (M5.4, full attention) and `enc_gdn_prefill` (M5.5, gated DeltaNet), and `sg_gpu_prefill` (M5.6) which drives them across every layer one command buffer per chunk, saves and restores the one-token decode scratch around chunk-sized scratch, and bridges the final DeltaNet state out of `sg_kv`'s carriers into `L->conv_buf`/`L->ssm`. Also the B8 duty cycle (`sg_gpu_set_prefill_rest`, `pf_sleep_ms`) and the command-buffer segmentation (`sg_gpu_set_prefill_max_burst`), both disabled by default and both pure timing or submission-boundary changes that touch no buffer, state or accumulator. THIS SEAM WAS CHOSEN BECAUSE NOTHING IN DECODE CALLS IT: `sg_gpu_forward`, `enc_attn` and `enc_gdn` stayed in `src/metal.m` and reach nothing here, so the traffic is one-way, into the shared helpers. Unlike task R1's shader split this could NOT be gated on binary identity (Objective-C translation units genuinely link), so the gate was the exact check counts: `make check` 87604/0 and `make debug` 83614/0, both unmoved from the parent. |
| `src/metal_internal.h` | (task R2) the ONLY declarations `src/metal.m` and `src/metal_prefill.m` share: `SG_TG`, the `KI_` kernel index enum, `sg_gpu_buf`, `sg_gpu_layer`, `struct sg_gpu`, `sg_enc`, the `PARAMS` macro, and the helpers that cross the seam. Two kinds of function live here and the difference is deliberate: the one-line accessors and bit casts on the per-element encode path (`bufof`, `offof`, `fbits`, `mul_ck`, `add_ck`) are `static inline` DEFINITIONS so they keep inlining in both files and gain no external symbol, while the nine larger helpers (`gpu_errf`, `scratch_ensure`, `gpu_elem_width`, `gemm_kernel_for`, `gpu_embed_row`, `gpu_alloc_f32`, `enc_op`, `enc_kv_store`, `enc_matmul`) are PROTOTYPES only: those really did lose `static`, gain external linkage and lose cross-translation-unit inlining, which is per-dispatch or per-allocation cost, never per-element. Nothing here is a copy of anything in `src/metal.m`; a helper with callers on one side only stays static in that file. Not a public header: `struct sg_gpu` holds `id<MTL...>` members, so only a `.m` built against the frameworks can include it. |
| `src/kernels.metal` | deterministic Metal kernels: decode-step ones fold a fixed reduction tree (no atomics/simd_sum) -- bf16/f32/Q8_0 matvec, attention decode (f32-KV `k_attn_decode` and fp16-KV `k_attn_decode_f16`, M5.2), a decode-step fp16 store (`k_kv_store_f16`, M5.2), gated-DeltaNet decode, RoPE, RMSNorm, SwiGLU. Prefill's tiled GEMM (M5.3, `k_matmul_bf16`/`k_matmul_f32`/`k_matmul_q8`, `Y[N,M]=X[N,K]@W[M,K]^T`) uses a different determinism mechanism: one thread per output element of a 16x16 output tile, a private serial K-loop, no cross-thread reduction at all (nothing to fold). Full-attention prefill (M5.4) adds `k_rope_chunk` (partial RoPE over a whole chunk) and `k_attn_prefill` (causal chunk attention over the fp16 KV cache, one threadgroup per (query token, query head), same fixed-tree softmax as `k_attn_decode_f16`). DeltaNet prefill (M5.5) adds `k_conv1d_chunk` / `k_delta_gates_chunk` / `k_delta_chunk` / `k_rmsnorm_gated_chunk`: a whole chunk of N tokens through one DeltaNet layer via a SEQUENTIAL within-chunk scan (per-token loop inside the kernel, threading the conv tail + S matrix), bit-identical to the matching decode kernel looped over the chunk. The SPLIT-K decode-attention kernels are NOT here: they are in `src/kernels_splitk.metal` (next row), a second translation unit of the same metallib. |
| `src/kernels_splitk.metal` | the split-K decode-attention kernels (P2.2 through P2.9), moved out of `src/kernels.metal` by task R1 when that file reached 2548 lines against the ~2000-line guideline. A SEPARATE translation unit compiled to `src/kernels_splitk.air` and linked into the SAME `src/kernels.metallib`, so no host code changed; the move was gated on the linked metallib being byte-identical to the parent commit's and on every one of the 37 emitted AIR functions being instruction-for-instruction identical. Split-K decode attention (P2.2/P2.3) adds `k_attn_decode_splitk_partial` + `k_attn_decode_splitk_combine`, and P2.4 adds `k_attn_decode_splitk_partial_gqa`, the same partial with one threadgroup per GQA GROUP instead of per query head (each K/V element read once for all `repeat` heads that share it, `repeat` separate per-thread accumulators, same output bytes and same output layout). P2.8 adds `k_attn_decode_splitk_partial_gqa_online`, the ONLINE-SOFTMAX form of that partial: one streaming pass holding a running `(m, s, acc)` per head, rescaled by `exp(m_old - m_new)` when the maximum moves, so it has no device-memory score row and SEVEN bindings instead of eight. `m` and `s` are uniform (off the new R-wide fixed trees `tg_max_group`/`tg_sum_group`) and `acc[d]` is owned by exactly one thread, which is what keeps the streaming state at `4R` registers instead of `R*head_dim`, and which is why the kernel wants `head_dim <= SG_TG` (past that its `dbase` loop re-streams the split per band, still correct, so the decline is policy not validity). Not byte-identical to the four-pass partials: streaming reorders the exponential sums, so the bar is the CPU oracle plus determinism plus byte-exact greedy tokens. P2.9 then splits the tile's KEYS across `SG_TG / kw` key groups when `head_dim < SG_TG` (`kw` = smallest power of two >= `head_dim`, floored at 32) so that every thread owns an output dim in the V phase instead of half of them idling, merging the groups' partials with the combine kernel's own log-sum-exp weights inside the same `R x SG_TG` scratch; `head_dim >= SG_TG` gives one key group and is unchanged. |
| `src/kernels_common.metal.h` | the ONLY declarations the two shader translation units share: `SG_TG` (the one dispatched threadgroup width) and the two fixed-shape fold trees `tg_sum` / `tg_max` it sizes. Defined once and included by both, because a second copy that drifts is exactly the failure the determinism mandate exists to prevent. A helper with callers in only one of the two files stays in that file: this is a shared-definition seam, not a helper collection. The determinism mandate itself is stated once and did not move (`src/kernels.metal:7-27`); `kernels_splitk.metal`'s header points at it. |
| `src/cli_*.c` | the four CLI mains (`cli_info`, `cli_ref`, `cli_metal` = `surge`, `cli_bench` = `surge-bench`). `cli_bench` (B5): raw tokenize (no chat template), `--bos`/`--no-bos` (default `tokenizer.ggml.add_bos_token`), M5 tiled prefill (`--chunk`, default 1024) + shared greedy decode, GEMM gate (`--gemm-gate-tflops F`, need F>20.5) + ingestion guard, whole-run peak-mem sampling, NIAH recall (text input only), decode-by-slope (`--warmup`, `--emit-timeseries`), md row to stdout + `--json`. Exit 0 DONE / 3 VOID / other hard error. `--prefill-work-ms`/`--prefill-rest-ms` (B8) arm the metal.m duty-cycle before prefill; JSON gains `prefill_rest_s` (slept subset of `prefill_wall_s`) and `prefill_compute_tps` (rest excluded, the fair full-clock number). `--decode-work-ms`/`--decode-rest-ms` (P3.0) arm `src/sched.c`'s pacer, fed at the per-token boundary right after `sg_gpu_forward` returns, with `--decode-clamp-div` and `--decode-baseline-ms` for the detector; JSON gains the mirror pair `decode_rest_s`/`decode_compute_tps` plus `decode_rests`, `decode_clamp_events` and `decode_baseline_ms`, the last three filled in on EVERY run so a row carries whether its own decode number was taken under a rising step time. The 8-column markdown row is unchanged. |

## Level 4: Data flows

- **Decode (M2, shipped; KV dtype refit M5.2):** token -> host embedding lookup -> per
  layer, one command buffer: RMSNorm, projections (matvec), qk-norm + partial RoPE, KV
  append (fp16 by default: q/k-norm and RoPE run in an f32 scratch, then `k_kv_store_f16`
  casts into `sg_kv`'s per-layer half buffers; `SURGE_KV_DTYPE=f32` keeps the original
  path), attention decode (full-attn layers: since Task P2.3 the SPLIT-K PAIR
  `k_attn_decode_splitk_partial` + `k_attn_decode_splitk_combine` once the sequence
  reaches 1024 keys, `k_attn_decode_f16` below that and under `SURGE_ATTN_SPLITK=0`, and
  `k_attn_decode` on the f32-KV path; Task P2.4 adds the GQA-shared
  `k_attn_decode_splitk_partial_gqa` as a same-bytes alternative to the partial, gated on
  hardware 2026-08-17 and **SHIPPED AS THE DEFAULT PARTIAL SINCE TASK P4.0 (2026-08-18)**,
  so a decode step at or above the occupancy floor (`n_splits * n_kv_heads >= 128`, task
  P2.7) now dispatches the GQA partial and one below that floor still dispatches the
  per-head one, in the SAME run; `SURGE_ATTN_SPLITK_GQA=0` pins the per-head partial
  everywhere and is the A/B control. The two write the same bytes at a fixed `n_splits`.
  Task P2.8 adds `k_attn_decode_splitk_partial_gqa_online`, the streaming form of
  that partial, reached only under `SURGE_ATTN_SPLITK_ONLINE=1`, also gated on hardware
  2026-08-17 and also off by default because it is measurably faster on the 27B shape and
  measurably slower on the 4B one) or
  conv-step + delta-step (DeltaNet layers), output gate, residuals; final norm + lm_head
  -> logits -> host argmax -> next token. Since Task P3.0 `surge-bench`'s driver also feeds
  that step's measured wall time to `src/sched.c`'s pacer at the boundary right after the
  forward returns (GPU idle at that instant), which may sleep there; pacing is OFF by
  default and is pure timing, so gen_ids are unchanged either way.
  Byte-exact GREEDY TOKENS to the CPU reference at
  temp 0 (M2 gate, oracle is the f32-KV path); fp16 KV adds ~1e-6 logit noise on the
  measured fixture, absorbed by argmax.
- **Prefill (M5.6, shipped):** `sg_gpu_prefill` chunks the prompt (default 1024 tokens);
  per chunk one command buffer runs all layers via tiled GEMM + `k_attn_prefill`
  (full-attn) + the DeltaNet chunked scan (`enc_gdn_prefill`), writing full-attn K/V into
  `g->kv` and threading DeltaNet conv/S through `sg_kv`'s carriers, advancing `g->used` and
  `sg_kv` by the chunk size; only the final chunk's last row computes out_norm + lm_head.
  It then BRIDGES each DeltaNet layer's final conv tail + S out of the `sg_kv` carriers
  into the ad hoc `L->conv_buf`/`L->ssm` decode reads, so a following `sg_gpu_forward` at
  pos == n_tokens continues from the prefilled state. Requires the fp16 KV path. Uses
  chunk-sized scratch swapped into the `g->b_*` decode fields for the run's duration.
  Correctness gate is token-level: prefill-then-decode == feed-one-at-a-time (byte-exact
  gen_ids), worst last-logit rel gap 1.2e-6 vs the serial forward.

## Architecture status (built vs planned)

- **Built (M0-M2, on main):** gguf/st/tok/model_qwen/ref, Metal decode, CPU-ref forward.
  M1 gate: 100% top-1 vs mlx-lm on Qwen3.5-2B. M2 gate: byte-exact Metal-vs-ref greedy.
  Decode ~76 tok/s on the 2B bf16; measured at 0.57x of mlx-lm (speed is M4's milestone).
- **In progress (branch `feat/m3-m5`):**
  - M3 (Q8_0 weights end-to-end): DONE. M3.1 `k_matvec_q8`; M3.2+M3.3 (merged: the
    decode encoder selects the matvec kernel from the weight dtype via
    `matmul_kernel_for`, and `sg_gpu_load_model` wraps Q8_0 tensors no-copy and dequantizes
    the Q8_0 embedding row on the host; the 27B Q8_0 GGUF now loads and decodes coherently
    on Metal). The 27B's matmul weights are uniformly Q8_0 (loader-enforced) while its norms
    and DeltaNet scalars stay F32, so per-tensor dispatch resolves to one kernel per model
    plus the existing F32 small-tensor path. M3.4 (Q8_0 forward numeric gate) DONE: on the
    27B Q8_0 GGUF, (A) surge Metal Q8_0 vs surge CPU-ref Q8_0 teacher-forced over 68
    positions is 100% top-1 (max |logit delta| 1.6e-5, no near-tie flips), and (B) surge
    greedy is byte-identical to llama.cpp (llama-simple) on 4 prompts with tokenizer parity.
    Gate is `make gate` (manual, needs the 28 GB GGUF + llama.cpp + one ~11 min CPU forward,
    never in `make check`); regression fixtures frozen under `tests/fixtures/m3q8/`.
  - M5 (fp16 KV to 262,144 + tiled prefill): M5.1 `kv.c` done. M5.2 (fp16 KV wired into
    the decode path) DONE: default full-attention KV cache is fp16 via `sg_kv` (separate
    K/V buffers), `k_kv_store_f16`/`k_attn_decode_f16` bit-identical to the f32 kernels
    fed pre-rounded inputs (100 reruns), M2 gate re-verified byte-for-bit unchanged on
    the f32 path (`git stash`-diffed, not just "still passes"). M5.3 (tiled GEMM kernels)
    DONE: `k_matmul_bf16`/`k_matmul_f32`/`k_matmul_q8` added (new `SG_K_TILES2D` grid
    class), not yet wired into `sg_gpu_forward` (that is M5.4+'s job); vs a host f64
    reference and vs the existing matvec kernels, worst measured 2.5e-6 relative
    (bf16/f32, gate 1e-5) and 8e-7 (Q8_0, gate 2e-2); 100/100 byte-identical determinism.
    M5.4 (full-attention tiled prefill kernels) DONE: `k_rope_chunk` (partial RoPE over a
    whole chunk, each token at its absolute position; bit-identical to `k_rope_heads`
    per token) and `k_attn_prefill` (one threadgroup per (query token, query head),
    causal over the fp16 KV cache: token t attends the first `base+t+1` positions, so no
    query reads a strictly-future key). The per-(token,head) structure mirrors
    `k_attn_decode_f16` statement for statement, so prefill of a chunk is BIT-EXACT to
    looped decode (measured rel 0.0, gate 1e-4) and byte-identical over 100 reruns. Host
    side: public `sg_gpu_run_attn_prefill` (three-input one-shot, like the f16 decode
    entry) and the encoder `enc_attn_prefill` (chunked twin of `enc_attn`: tiled GEMM
    Q/K/V, chunk qk-norm + `k_rope_chunk`, store-chunk-K/V-then-attend, output gate,
    o_proj). `enc_attn_prefill` is built but not yet driven (that is M5.6's whole-prompt
    orchestration, which also sizes its chunk buffers).
    M5.5 (gated-DeltaNet chunked-scan prefill kernels) DONE: `k_conv1d_chunk`,
    `k_delta_gates_chunk`, `k_delta_chunk`, `k_rmsnorm_gated_chunk` push a chunk of N
    tokens through one DeltaNet layer, threading the recurrent state (conv tail + S
    matrix) across chunks in `sg_kv`'s per-layer conv/S buffers (read at chunk entry,
    rewritten in place). The within-chunk scan is SEQUENTIAL (a literal per-token loop
    inside each kernel), so each is BIT-IDENTICAL to the matching decode kernel
    (`k_conv1d_step` / `k_delta_gates` / `k_delta_multi` / `k_rmsnorm_gated`) looped over
    the chunk with the state carried -- measured 0 differing bytes vs the GPU decode
    oracle for N in {1,2,7,64}, both v-head maps (tiled + grouped), and split-chunk ==
    whole-chunk for both conv tail and S; byte-identical over 100 reruns. `k_delta_chunk`
    needs no barrier between tokens because each thread owns rows lid, lid+256, ... of its
    head's S block privately for the whole chunk (no cross-thread reads; k_delta_multi has
    no reduction). Host side: five public one-shots (`sg_gpu_run_conv1d_chunk` /
    `_delta_gates_chunk` / `_delta_chunk` / `_rmsnorm_gated_chunk`, plus
    `sg_gpu_run_delta_multi` as the per-op oracle) and the encoder `enc_gdn_prefill`
    (chunked twin of `enc_gdn`: tiled-GEMM in_proj, `k_conv1d_chunk`+SiLU, per-token qk
    RMSNorm-heads + scale over the chunk, `k_delta_gates_chunk`, `k_delta_chunk` with S
    threaded, gated output norm, o_proj).
    M5.6 (chunked prefill orchestration) DONE: `sg_gpu_prefill` drives
    `enc_attn_prefill` + `enc_gdn_prefill` + the chunked MLP across all layers, one
    command buffer per chunk, sizing chunk scratch by swapping it into the `g->b_*` decode
    fields; it closes the three carry-forwards (state bridging sg_kv -> L->conv_buf/L->ssm,
    `g->kv` allocated for DeltaNet models, first end-to-end wiring). CLI gets `--chunk N`
    (default 1024) and `--no-prefill`, prefill being the default for -p/--ids on the Metal
    path. Gated by tests/test_gpu_prefill.c: prefill last-argmax == serial forward
    last-argmax on both mini formats for chunk {1,2,3} (worst rel 1.2e-6, bar 2e-4),
    prefill+decode gen_ids == serial+decode (16/16), byte-identical reruns, and the CLI
    with/without --no-prefill identical.
    M5.7 (long-context gate, CLOSES M5) DONE: on the real 2B bf16 model
    (`SURGE_GATE_MODEL`, driven through the C API since 262144 ids do not fit through
    argv), (A) prefill(prompt)+decode 32 == serial-forward(prompt one at a time)+decode 32
    with 0 token-id mismatches at each of 8192 / 16384 / 32768; (B) a 262144-token prefill
    (chunk 1024) leaves the used counter at 262144 with no Metal fault and non-degenerate
    final logits, and a decode of 32 tokens at ~256K depth stays non-degenerate (finite,
    in-range, not one id repeated). Because `SG_KV_CAP_MAX` == 262144 leaves no cache slot
    to append after a full-cap prefill, gate B's decode run prefills 262112 then decodes 32
    (reaching used == 262144). (C) the `surge` CLI now rejects a prompt that exceeds an
    explicit `--max-ctx` with a clear message + nonzero exit (covered by
    tests/test_cli_prefill.sh in `make check`). The 2B safetensors carries no
    surge-readable tokenizer, so the coherence bar here is non-degenerate ids; valid-UTF-8
    coherence on real text is B7's job on the 27B. Gate is `tools/prefill_longctx_gate.sh`
    (live GPU, env-gated); the C test SKIPs cleanly without `SURGE_GATE_MODEL`, so
    `make check` stays mini-only and hermetic. Public `sg_gpu_used` added so the gate reads
    the used counter without reaching into the opaque `sg_gpu`. M5 COMPLETE.
  - Bench harness: B1/B3/B4 (pure C), B2 (peak-memory probe), B5 (`surge-bench` CLI +
    shared `sg_argmax_f32`), B6 (offline decode-slope + wall-accounting verification), and B8
    (prefill duty-cycle) done; B7 (gated 256K run) COMPLETE 2026-08-14 (rc=0).

    B8 RATIONALE CORRECTED 2026-08-15. B8 was built on the premise that the B7 27B/256K
    prefill HUNG because the Mac Studio M3's firmware GPU limiter clamps to 338 MHz after
    3-4 min of sustained load and recovers only after 60-120s idle. Telemetry from the
    completed 30-hour 256K run (9199 samples, 378 bursts) does not support that premise.
    GPU clock is FLAT across a burst rather than decaying (early bursts 712/723/716/710/701
    MHz at t+0/30/60/90/120s, late bursts 573/591/590/591/588, with constant bin counts, so
    not selection bias); rest length does not predict the next burst's clock (r = +0.017
    over 376 burst pairs); and 338 MHz appears in 27 of 7724 loaded samples (0.3%). What
    clock tracks is CONTEXT LENGTH (r = -0.575), and it is set at burst start before any
    load could accumulate: long-context prefill becoming memory-bandwidth bound as the KV
    cache grows (16.0 GiB K+V at full context). GPU temp averaged 58.0 C (max 73.4) with
    fans at 2521 of 3625 RPM, and corr(temp, clock) is POSITIVE (+0.369), so it is not
    thermal throttling either. Cost of the wrong premise: 367 rests x 90 s = 9.2 hours of
    the 30-hour run spent sleeping for no measured benefit.

    What actually happened on 2026-08-14 is that WindowServer was watchdog-killed twice
    (16:43:01, 16:43:41) while surge-bench held the GPU, logging out the GUI session.
    Symbolicated: ws_main_thread blocked in SkyLight -> Metal -> IOGPU -> IOKit ->
    IOGPUFamily -> AGXG15C, missing its 80 s check-in. B7's original "HUNG" was almost
    certainly the same event. The duty cycle is therefore RETAINED but REPURPOSED: its job
    is compositor protection, not clock recovery. Full analysis in
    `docs/15082026_prefill_duty_cycle_plan.md`.

    Mechanism unchanged: `sg_gpu_set_prefill_rest(g, work_budget_ms, rest_ms)` (metal.m,
    surface in surge.h) arms an optional duty cycle; once accumulated GPU-busy wall time
    (measured strictly as `commit`..`waitUntilCompleted` per chunk) reaches
    `work_budget_ms`, `sg_gpu_prefill` sleeps `rest_ms` (via a `nanosleep`+EINTR-retry
    helper, `pf_sleep_ms`, not bare `usleep`, so a signal cannot truncate a real 90s rest)
    with no command buffer in flight, then resumes. Either argument 0 (the default) disables
    it: `sg_gpu_prefill`'s OUTPUT is then byte-identical to before this task. PURE TIMING:
    proven, not just asserted, by a `tests/test_cli_bench.sh` gate showing a forced-rest run
    (11 rests over a 12-chunk prefill) produces gen_ids byte-identical to the same run with
    the feature off; a second gate checks the exact expected total rest time (11 * 50ms,
    +/-30ms) and that `prefill_compute_tps` (rest excluded) exceeds `prefill_tps`.

    PREDICTIVE BUDGET TEST (2026-08-15). The gate now asks whether the NEXT chunk would
    carry the accumulator past the budget, estimating it as the previous chunk's GPU time
    scaled by `PF_EST_MARGIN` (1.25; a heuristic, not a guarantee). Testing after the fact
    meant a burst always ran to budget + one chunk, and chunk cost grows with context: over
    the 256K run, 367 of 367 bursts overran a 150 s budget, median 199.5 s, worst 332.9 s,
    every one of them past the 80 s watchdog window. The final chunk is now protected by the
    preceding chunk's test rather than skipped by a `!last` guard.

    COMMAND-BUFFER SEGMENTATION (2026-08-15). `sg_gpu_set_prefill_max_burst(g, max_burst_ms)`,
    CLI `--prefill-max-burst-ms`, default 0 = off. The rest above can only yield BETWEEN
    chunks, so it cannot help once ONE chunk's command buffer outlasts the watchdog window;
    at 220k context a single 256-token chunk measured ~130 s. When a submission overruns the
    ceiling, the layer sweep is split across more command buffers (halving, floor 1 layer).
    The ceiling is a TARGET, not a cap, and NOT a promise of a single overrun: the check
    is reactive, so each overrunning submission runs in full and only then halves. A
    64-layer sweep can overrun at 64, then 32, then 16, converging over up to log2(layers)
    overruns; a segment at the 1-layer floor cannot shrink further and keeps overrunning.
    Set the ceiling well under the watchdog window so the overruns along the way still
    land inside it.
    Unlike changing `chunk`, this cannot alter output: command buffer boundaries carry no
    state, the same kernels run with the same arguments in the same order, and buffers
    committed in sequence on one queue execute in order. That claim is GATED, not assumed:
    `tests/test_cli_bench.sh` checks segmented gen_ids equal unsegmented over an uneven chunk
    schedule (12 tokens at `--chunk 5`) AND that `prefill_segments` actually rose, so the
    parity check cannot pass vacuously. The segment cursor advances by layers encoded, never
    by `seg`, because `seg` shrinks mid-loop; the first draft used `l0 += seg` and silently
    reprocessed layers, producing wrong gen_ids. The gate caught it.

    `sg_gpu_prefill_rest_ms(g)` is reset to 0 as the FIRST mutation of `g` in every
    `sg_gpu_prefill` call (including a failed one), gated by a dedicated
    `tests/test_gpu_prefill.c` unit test (`prefill_rest_reset_on_error`) so a failed call
    never reports a stale value from a prior successful one. `sg_gpu_prefill_segments(g)`
    (new, also on `sg_bench_row`/JSON as `prefill_segments`) is reset the same way. Rest
    accounting cross-checked against real wall-clock (`time` on a
    `--chunk 3 --prefill-work-ms 1 --prefill-rest-ms 2000` run: 6.111s real vs
    `prefill_rest_s:6`/`prefill_wall_s:6.07584`, exact match).
  - B6 adds two additive `sg_bench_row` fields,
    `prefill_wall_s`/`decode_wall_s` (own `now_s()` spans around the prefill/decode phases in
    `cli_bench.c`, independent of `wall_s`/the per-token series), plus a `tests/test_cli_bench.sh`
    block (same Metal-guard as the rest of the file) that drives `surge-bench` on the mini fixture
    and independently refits `--emit-timeseries` offline in `python3`, checking: reported
    `decode_tps_slope` matches the offline `[warmup,n)` OLS refit within 0.5%; `|slope-avg|/avg`
    within 3%; `prefill_wall_s+decode_wall_s` closes `wall_s` within 2% (observed ~0 to ~2e-6
    relative over repeated mini runs, i.e. two genuinely independent clocks, not a tautology,
    far under the 2% gate); and the reported slope is decisively (>1%, over a best-of-5 pool) far
    from the naive `[0,n)` refit that still carries the post-prefill transient, ruling out a
    warmup-ignoring bug
    (verified by deliberately reintroducing that exact bug and confirming the gate catches it 8/8
    times). Full rationale in `.superpowers/sdd/2026-08-09-surge-m3-m5/task-B6-report.md`. B5
    resolves the B2 peak-RAM question below: it tracks the two signals SEPARATELY across
    the whole run and reports `row.peak_ram_gib` = peak `phys_footprint` (resident,
    comparable to mlx-lm/llama.cpp) and `row.gpu_alloc_gib` = peak `currentAllocatedSize`
    (allocated upper bound, ~28 GiB of no-copy weight wraps for the 27B), so the two are
    never conflated. Gated by `tests/test_cli_bench.sh` (`make check` + `make bench-check`):
    surge-bench gen_ids byte-equal to `surge` on both mini formats, VOID/exit-3 on a
    failed/missing GEMM gate or out-of-window ingestion, and (env-gated on a real tokenizer)
    a `--bos`/`--no-bos` +/-1 prompt-token delta. B2: `sg_gpu_current_alloc_bytes` (Metal-only,
    `MTLDevice.currentAllocatedSize`) + `sg_proc_phys_footprint` (pure C, mach
    `task_info`/`TASK_VM_INFO`) feed a pure-C `sg_mem_tracker` (`peak =
    max(peak, max(current_alloc, phys_footprint))`), unit-tested deterministically under
    `make debug` with no Metal/mach in that path. Live-verified: `phys_footprint >
    current_alloc` after loading the mini fixture (small wrapped weights), and the
    tracked peak grows to 100% of a computed 300 MiB fp16-KV budget after `sg_kv_new`.
    Important finding for B5: `sg_gpu_current_alloc_bytes` counts a `newBufferWithBytes
    NoCopy`-wrapped checkpoint (every matmul weight) at its full declared length the
    instant it is wrapped, regardless of page residency -- on the real 2B, gpu_alloc read
    3.5 GiB immediately after load while phys_footprint read only ~10 MiB, so B5 should
    not treat "right after load" as a representative peak-RAM sample; sampling across a
    full prefill+decode run (which pages in the weights it actually reads) is what makes
    the two probes converge on a real number. See `todo.md`'s Task B2 Results for the
    full writeup.
  - Task P1 (loader-only, `src/model_qwen.c` + conditional plumbing in `src/ref.c` and
    `src/metal.m`): surge now LOADS the plain dense `qwen3` GGUF family (e.g.
    Qwen3-4B-Instruct-2507-Q8_0.gguf) alongside the hybrid, unblocking the next milestone
    (making that specific model's long-context decode fast; it is one of only two models
    that recall 8/8 needles at 262,144 tokens in the llm-rnd 256K comparison, but decodes
    at 2.37 tok/s there today). Four real-file blockers fixed, all verified against the
    actual downloaded 4B GGUF via `surge-info`/a standalone GGUF-header parse (kept
    GPU-free while B7 held the device): (1) `general.architecture == "qwen3"` (bare, no
    "5"/"_5") is now accepted, alongside the existing `qwen35`/`qwen3_5`. (2) A new
    `sg_cfg.attn_output_gate` flag (true for the hybrid, false for dense) makes the
    q_proj double-width expectation, the sigmoid output-gate dispatch, and the q-norm/RoPE
    per-head stride CONDITIONAL in both `src/ref.c`'s `attn_layer` and `src/metal.m`'s
    `enc_attn`/`enc_attn_prefill` (a `q_stride` local replaces the hardcoded `2*head_dim`
    in both; the hybrid's literal `2 * hd` stays byte-for-bit where the gate dispatch
    itself is skipped, i.e. only reached when the gate exists). (3) A dense file carries no
    `<arch>.full_attention_interval` key at all, so the dense branch sets
    `full_attn_interval = 1` outright (every layer a full-attention layer by the same
    `(L+1)%interval==0` rule `kv_is_attn`/`check_layer_groups` already use). (4) THE TRAP
    (most important): a dense file ALSO carries no `<arch>.rope.dimension_count` key, and
    Qwen3 uses FULL rotary (rope_dim == head_dim, 128 of 128) -- copying the safetensors
    loader's existing `partial_rotary_factor`-0.25 default here would have silently
    computed rope_dim 32 (even and <= head_dim, so it passes every existing validity
    check) and loaded a model that RUNS with WRONG output and no error. The dense branch
    instead defaults `rope_dim` to `head_dim` and only lets an explicit
    `qwen3.rope.dimension_count` override it; a regression to 32 is covered by an explicit
    test. A FIFTH fix, found empirically (not in the original four, which the task brief
    had already verified) and necessary for the loader to reach any of the above: the real
    dense file's pre-MLP norm tensor is `blk.N.ffn_norm.weight`, not
    `blk.N.post_attention_norm.weight` like the hybrid GGUF -- the loader now tries the
    hybrid name first (tensor presence, not architecture, decides, consistent with this
    file's existing convention), falling back to `ffn_norm.weight`. Scope was loader/config
    only, per the task brief: no kernel touched, the hybrid path re-verified BYTE-IDENTICAL
    (real hybrid 2B safetensors frozen-digest regression, 32/32 argmax, max delta 2.2e-07,
    unchanged from before this task). GATES: `make debug` (SURGE_NO_METAL, ASan/UBSan)
    clean; new pure-C tests assert the real 4B's config (36 layers, 32 heads, 8 kv,
    head_dim 128, hidden 2560, `full_attn_interval==1`, `attn_output_gate==false`, and the
    anti-trap `rope_dim==128`), all 36 layers classify full-attention with a 0 ssm census,
    and a CPU-reference forward runs 3 positions on the real 4B producing finite in-range
    logits. `src/metal.m`'s conditional-gate changes could not be built with the real Metal
    frameworks or run (a 28-hour B7-retry benchmark held the GPU for this whole task) --
    verified instead by `clang -fsyntax-only` (clean, no warnings under
    `-Wall -Wextra -Werror`) and by the fact that `src/ref.c`'s CPU twin, which follows the
    exact same `q_stride`/conditional-gate pattern, is proven byte-identical on the hybrid
    path; the Metal-side numeric gate (byte-exact Metal-vs-ref greedy on the dense 4B, and
    top-1 vs mlx-lm) is deferred to when the GPU frees. Full report:
    `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P1-report.md`.
  - Task P2.0 (`src/ref.c` + `surge.h`, pure C, no kernel/encoder touched): the split-K
    (flash-decoding) attention COMBINE math, proven in pure C ahead of the Metal kernel
    that will use it. Why: decode attention today dispatches exactly `n_heads`
    threadgroups (`src/metal.m`'s `enc_attn`/`k_attn_decode_f16`), so on a 32-head model
    48 of this machine's 80 GPU cores sit idle while one threadgroup walks the whole KV
    sequence -- measured 1.25 tok/s at 262,144 context. The fix is to split the KV
    sequence across many threadgroups (each emitting a partial max/sum-exp/weighted-V-sum
    triple) and combine the partials by log-sum-exp rescaling; `sg_ref_attn_combine(m, s,
    acc, n_parts, head_dim, out)` is that combine step, built and gated exactly as
    `src/kv.c` (M5.1) and `src/bench.c` (B1/B3/B4) were: pure C first, Metal later.
    Partitions are folded in strictly increasing index order in every pass (no sort, no
    reassociation), matching `src/kernels.metal:7-27`'s fixed-shape determinism rule, so
    this is a byte-identical CPU oracle for the eventual kernel. Degenerate partitions
    (empty, all-but-one-empty, all-empty, `n_parts==0`) are handled without NaN via an
    explicitly documented zero-output convention rather than left to divide by zero.
    GATES (`tests/test_attn_combine.c`, new): equivalence to a direct single-pass
    softmax-then-weighted-V-sum reference for K in {1,2,3,7,64,257} against ragged
    (non-dividing) partition counts, via a combined absolute+relative tolerance
    (`atol=1e-6, rtol=1e-6`; a bare relative-error ratio is not meaningful when an output
    dimension's true value is coincidentally near zero from softmax cross-key
    cancellation -- measured worst absolute diff 5.96e-08, about one float32 ULP);
    K==1 bit-exact (a true identity: `exp(0.0)==1.0` exactly); 100 repeated calls
    byte-identical; large-magnitude (+/-80 and beyond) scores stay finite. `make debug`
    (SURGE_NO_METAL, ASan/UBSan) exits 0, 869 checks, 0 failures, no sanitizer
    diagnostics. The Metal kernel that dispatches multiple threadgroups per head and
    calls this combine is NOT built yet (a later task); nothing in the live decode path
    changed. Full report: `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.0-report.md`.
  - Task P2.1 (`src/ref.c` + `surge.h`, pure C, no kernel/encoder touched): the direct and
    split-K decode-attention-CORE oracles, isolating exactly what
    `k_attn_decode_f16` computes (`src/kernels.metal:455-509`) -- `attn_layer`
    (`src/ref.c`, static) cannot serve this role, since it is a whole hybrid-layer
    function (projections, qk-norm, RoPE, KV write, attention, gate, o_proj), not a
    standalone attention-core primitive. `sg_ref_attn_decode(q, kc, vc, n_heads,
    n_kv_heads, head_dim, seq, q_stride, scale, out)` is the single-pass reference (`q`
    `[n_heads, q_stride]`, only `q[h][0..head_dim)` is the query, matching P1's
    gate/query split; `kc`/`vc` `[seq, n_kv_heads, head_dim]` head-interleaved, the
    `sg_kv` layout; GQA `hk = h/(n_heads/n_kv_heads)` matching
    `src/kernels.metal:471-472`). `sg_ref_attn_decode_splitk(..., n_parts, out)`
    partitions `[0, seq)` into `n_parts` contiguous ranges by the fixed rule
    `t0=i*seq/n_parts`, computes each `(head, partition)`'s `(m, s, acc)` triple, and
    combines each head's triples with ONE call to `sg_ref_attn_combine` (P2.0, reused
    not reimplemented). Both share one static core in `src/ref.c`
    (`attn_decode_core`), with `sg_ref_attn_decode` calling it at `n_parts==1` --
    making the direct-vs-split-K identity at `n_parts==1` structural rather than
    coincidental. GATES (`tests/test_attn_decode.c`, new): split-K equals direct
    (max ABSOLUTE error < 1e-6) for `n_parts` in {1,2,3,7,64,257} against ragged seq
    and `n_parts > seq` (forced empty partitions), across three shapes incl. the real
    Qwen3-4B-Instruct-2507 32/8/128 GQA shape; `n_parts==1` bit-exact; GQA MAPPING
    proven (not just numerically close) via an identical-query trick at both
    repeat=4 and repeat=1 (plus, since fix round 1, the repeat==0 fallback); both
    `q_stride` variants, with the hybrid gate half NaN-poisoned to prove it is never
    read as query data; 100x determinism for both functions. `make debug`
    (SURGE_NO_METAL, ASan/UBSan) exits 0, 81616 checks, 0 failures, no sanitizer
    diagnostics, worst observed error 1.192e-07. The Metal kernel itself remains a
    later, not-yet-started task; nothing in the live decode path changed. Full report:
    `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.1-report.md`.
  - Task P2.1 fix round 1 (review: CHANGES-REQUIRED, 4 findings, all closed): the
    original gates above proved only K-INVARIANCE (split-K tiling + the
    `sg_ref_attn_combine` wiring), since both public functions share one
    `attn_decode_core`/`attn_partial` static core -- a bug consistent across every
    partition boundary would pass both gates 1 and 2 invisibly. CRITICAL fix: a new
    test-local `gold_attn_head` (`tests/test_attn_decode.c`), independently derived
    and structured the way `attn_layer` (`src/ref.c:1160-1232`, mlx-fixture-validated)
    computes attention -- materialize `scores[]`, normalize with the real
    `sg_ref_softmax` BEFORE the weighted-V pass, the OPPOSITE order from
    `attn_partial`'s defer-to-combine -- cross-checked against both
    `sg_ref_attn_decode` and `sg_ref_attn_decode_splitk`; worst observed error
    1.192e-07, the same order as gate 1's own K-invariance error. `attn_layer` itself
    was NOT touched (no refactor, no risk to its own frozen-digest/mini_fwd gates).
    IMPORTANT fix: the split-K scratch `malloc` size (`np*2 + np*hd` for
    caller-supplied, uncapped `n_parts`/`head_dim`) was an unguarded `size_t`
    multiply that could wrap and undersize the allocation; new `ref_mul_ck`/
    `ref_add_ck` (`src/ref.c`) close it, mirroring `src/metal.m`'s `mul_ck`/`add_ck`
    style in the `size_t` domain instead of GPU-buffer `uint64_t`. Two MINOR test
    gaps closed (`n_parts==0`, the GQA `repeat==0` fallback). Hand-off note (not a
    fix): `seq==0` diverges from `k_attn_decode_f16` (`src/kernels.metal:469` leaves
    `out` unwritten there; this oracle writes an explicit 0.0), now documented
    explicitly in both functions' `surge.h` contracts so the future Metal-vs-oracle
    gate does not spuriously disagree at that boundary.
  - Task P2.2 (`src/kernels.metal` + `src/metal.m` + `surge.h` + `tests/test_metal_ops.c`,
    purely additive): the METAL twin of those oracles, WRITTEN AND COMPILED ONLY. A
    28-hour benchmark owned the GPU (PID 98563, `pgrep -f "surge-bench|bench_niah"`
    non-empty before and after), so EVERY numeric gate is DEFERRED and nothing in this
    entry is a measured correctness result. Two kernels: `k_attn_decode_splitk_partial`
    (one threadgroup per (query head, split), emitting the m/s/acc triple per partition
    over its own `[t0, t1)`) and `k_attn_decode_splitk_combine` (one threadgroup per
    query head, folding that head's splits in strictly increasing index order by
    log-sum-exp rescaling). Partition rule, empty-split `-INFINITY`/0/0 encoding, GQA
    mapping, KV indexing and q_stride handling are `sg_ref_attn_decode_splitk` /
    `k_attn_decode_f16` verbatim; determinism is the file's existing fixed-tree
    `tg_max`/`tg_sum` in the partial and a per-thread strictly-increasing serial fold
    (no cross-thread reduction at all) in the combine. Registered in `src/metal.m` as
    `KI_ATTN_SPLITK_PARTIAL`/`KI_ATTN_SPLITK_COMBINE` with a NEW `SG_K_HEADS2D` grid
    class (the second kind after `SG_K_TILES2D` needing two group dimensions, since
    `SG_K_ATTN` carries a single `*groups` count), `check_params` rules shared by both,
    a `check_sizes` routing rule, and `splitk_sizes()` guarding every byte count through
    the existing `mul_ck`/`add_ck`. `seq == 0` DELIBERATELY matches the oracle here
    (every split empty, `out[d] = 0.0`) rather than inheriting `k_attn_decode_f16`'s
    unwritten-`out` divergence, so this pair agrees with `sg_ref_attn_decode_splitk` at
    every `seq`. NOT wired into decode: `enc_attn`/`enc_attn_f16` are byte-for-byte
    untouched and still dispatch `k_attn_decode_f16`. Verified: the Metal compile
    (`-fno-fast-math -Wall`), the `metallib` link (both kernels present), `metal.m`
    under `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror`, the new test compiling
    under the same flags, and `make debug` (SURGE_NO_METAL, ASan/UBSan) exit 0 at 83523
    checks / 0 failures / 0 sanitizer diagnostics, unchanged from the pre-task baseline.
    UNVERIFIED (needs the GPU): every numeric comparison against both oracles, the 100x
    determinism rerun, the empty-split encoding assertions, the argument-rejection
    assertions, and whether the kernels run at all. Gate written and registered as
    `metal_attn_splitk_matches_ref` in `tests/test_metal_ops.c`, ready to run. Full
    report: `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.2-report.md`.
  - Task P2.3a (`tests/bench_splitk.c` + `Makefile`, purely additive): the TIMING harness
    for P2.2's split-K pair, a DECISION gate (not a parameter fit) for whether P2.3 wires
    split-K into `enc_attn` at all -- if split-K does not beat the incumbent at the shapes
    that matter, it should not be wired in. A Phase 0 benchmark held the GPU for this
    entire task too (`pgrep -f "bench_niah|phase0|surge-bench"` non-empty before and
    after), so this is WRITTEN and COMPILE-CHECKED ONLY; no timing number has been
    measured. Sweeps `n_splits` over {1,2,4,8,16,32,64,128,256,512,1024}, clamped into the
    occupancy band `surge.h` documents (`4 <= n_splits <= seq/256`), across the real 27B
    decode shape (24 heads/4 kv/head_dim 256) and 4B dense shape (32/8/128) at seq
    8192/32768/131072/262144, ALWAYS timing the incumbent `k_attn_decode_f16` on the
    identical shape in the same run so every row is a direct A/B, not an isolated number.
    Reports mean/min/max over N reps (default 20, one discarded warm-up), plus achieved
    KV-read GB/s (K+V f16 bytes, once per query-head threadgroup, the same yardstick for
    both kernels; full derivation and honesty caveats in the file's own header). Kept OUT
    of `make check` structurally, not by convention: named `tests/bench_splitk.c` rather
    than `test_*.c` (the wildcard `check`'s `$(TESTS)` is built from), so it is excluded
    from that list by construction; reachable only via the new `make bench-splitk` target,
    which builds AND runs it. Verified: `xcrun clang -fsyntax-only -std=c11 -Wall -Wextra
    -Werror` clean on both the Metal path and the `-DSURGE_NO_METAL` stub path; `xcrun
    -sdk macosx metal -fno-fast-math -Wall -c src/kernels.metal` still compiles clean and
    `metallib` still links (kernels.metal is untouched by this task, purely additive);
    `make debug` exit 0 at the SAME 83523 checks / 0 failures / 0 sanitizer diagnostics as
    the P2.2 baseline. UNVERIFIED (needs the GPU): whether `make bench-splitk` builds and
    links against the real Metal frameworks at all, and every timing number, achieved-GB/s
    figure and speedup ratio the harness will print. Full report:
    `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.3a-report.md`.
  - Task P2.3a MEASURED + Task P2.3 (`src/metal.m` + `surge.h` + `src/kernels.metal`
    comments + `tests/test_gpu_fwd.c` + `tests/bench_splitk.c`): split-K is now the
    DECODE PATH's attention. The P2.3a sweep ran once the GPU freed and settled the
    decision: at seq 262144 the pair beats `k_attn_decode_f16` by 15.9x on the 27B decode
    shape and 21.9x on the 4B dense shape, and the fastest `n_splits` was exactly
    `seq / SG_TG` at every measured cell, so the wired default is the closed form
    `n_splits = clamp(seq / SG_TG, 4, 1024)` (the top of the occupancy band: every split
    gets exactly SG_TG keys). `enc_attn`'s fp16 branch calls the new `enc_attn_splitk`,
    which encodes the partial (hand-rolled 2D grid, x = split, y = head, since `gpu_grid`
    cannot carry two group dimensions) and then the combine into the SAME open command
    buffer, relying on `MTLDispatchTypeSerial`'s implicit inter-dispatch barrier instead
    of the one-shots' commit-and-wait. FALLBACK: below seq 1024 (`SG_TG * 4`, the point
    where the clamp's floor starts binding and splits fall under SG_TG keys) decode keeps
    `k_attn_decode_f16`, and a `--seqs` sweep added to `tests/bench_splitk.c` MEASURED that
    crossover rather than assuming it (at seq 256 split-K is 0.69-0.71x, i.e. slower; at
    512 it is 0.95-1.02x; at 1024 it is 1.27-1.88x and rising). `SURGE_ATTN_SPLITK=0` pins
    the incumbent, which is what makes the A/B measurable and why that kernel stays
    reachable; the f32-KV path always uses it (the split-K kernels read half-typed K/V).
    THE SCRATCH HAZARD IS STRUCTURALLY CLOSED: the partial binds the DEDICATED
    `sg_gpu.splitk_scratch` (P2.2 review finding 1), never the shared `g->scratch` that
    the same encoder binds for other kernels, and the m/s/acc partial buffers plus that
    scratch are sized ONCE in `sg_gpu_state_new` from `max_ctx` (using
    `n_splits * ceil(seq/n_splits) <= seq + n_splits - 1`), so nothing allocates or grows
    mid-encode. Prefill is untouched (`k_attn_prefill`, `enc_attn_prefill`,
    `enc_gdn_prefill`, `sg_gpu_prefill` have zero diff hits). Full report and every
    measured gate: `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.3-report.md`, gate
    doc `docs/16082026_splitk_decode_gate.md`.
  - Task P2.4 (`src/kernels.metal` + `src/metal.m` + `surge.h` + `tests/test_metal_ops.c`
    + `tests/bench_splitk.c`, purely additive): GQA-SHARED SPLIT-K THREADGROUPS, WRITTEN
    AND COMPILED ONLY. A 27B MLX NIAH benchmark owned the GPU for the whole task
    (`pgrep -f "bench_niah|mlx_raw_niah|llama-server|surge-bench"` non-empty before the
    first edit and after the last), so EVERY numeric and timing gate is DEFERRED and
    nothing in this entry is a measured result. New kernel
    `k_attn_decode_splitk_partial_gqa`: the P2.2 partial's grid is
    `(n_splits, n_heads)` and GQA maps `repeat = n_heads/n_kv_heads` query heads onto one
    kv head, so those `repeat` threadgroups each stream the SAME K/V slices (4x the unique
    bytes on the 4B shape, 6x on the 27B one, which is also why `bench_splitk`'s
    issued-bytes GB/s column reads 4.0x the unique-bytes figure there). The new kernel's
    grid is `(n_splits, n_kv_heads)`: one threadgroup serves the whole GQA group, reads
    each K/V element ONCE and applies it to all `repeat` query vectors, holding `repeat`
    separate per-thread accumulators (no shared accumulator, no atomics, no `simd_sum`, so
    the determinism rule is intact). OUTPUT LAYOUT UNCHANGED (`m`/`s` at
    `h*n_splits+part`, `acc` at `part_idx*hd`), so `k_attn_decode_splitk_combine`, the
    m/s/acc buffers and the `splitk_scratch` sizing are untouched, and the DEDICATED
    `sg_gpu.splitk_scratch` separation from P2.2 is preserved. THE BAR IS BYTE-IDENTICAL,
    not a tolerance: per head the dot products, the fixed-shape `tg_max`/`tg_sum` folds and
    the acc sums keep their operands and their order exactly. The group size is a
    COMPILE-TIME template parameter (`SG_SPLITK_GQA_MAX == 8`, mirrored in `metal.m`)
    because `repeat` accumulators indexed at runtime would be a device-memory stack array;
    a larger group falls into a correct one-head-at-a-time arm with no reuse. Host side:
    `KI_ATTN_SPLITK_PARTIAL_GQA` on the existing `SG_K_HEADS2D` class, the two partial
    one-shots refactored onto ONE shared body (`splitk_partial_run`, so they cannot drift
    on a validation rule), public `sg_gpu_run_attn_splitk_partial_gqa`, and
    `enc_attn_splitk` selecting pipeline + grid height. SWITCHABLE, AND SINCE TASK P4.0
    (2026-08-18) ON BY DEFAULT: `SURGE_ATTN_SPLITK_GQA=0` (read in `sg_gpu_state_new` like
    `SURGE_ATTN_SPLITK`) is what pins the per-head kernel now, and `splitk_gqa_use` still
    declines groups outside [2, 8] and grids under P2.7's 128-threadgroup floor. When this
    text was written nothing here had been run and the default was OFF; what moved it is
    P2.4's byte-identity, P2.6's real-model greedy gate where the split policies diverge,
    and P2.7's floor. No threadgroup-count floor was
    invented: the GQA grid is `repeat`x smaller, so a short-sequence crossover probably
    exists and `tests/bench_splitk.bin --gqa` (new flag) is the instrument for it. Verified:
    the Metal compile (`-fno-fast-math -Wall`, 0 warnings) and `metallib` link with the new
    kernel present, `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror` on `metal.m` +
    both test files (Metal and `-DSURGE_NO_METAL` paths), `make debug` rc 0 with 0
    sanitizer diagnostics and a check count identical to the same-environment baseline.
    UNVERIFIED (needs the GPU): the byte-identity comparison itself, the 100x determinism
    rerun, whether the kernel runs at all, whether its pipeline keeps a 256-thread
    threadgroup width, and every speed number. Gate written and registered as
    `metal_attn_splitk_gqa_bit_identical` in `tests/test_metal_ops.c`, ready to run. Gate
    doc `docs/17082026_splitk_gqa_threadgroups.md`; full report
    `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.4-report.md`.
    FIX ROUND 1 (review findings, all in the GATE rather than the kernel; kernel arithmetic,
    output layout, combine and scratch separation untouched): the end-to-end A/B was VACUOUS
    because the two partials are contracted to produce the same bytes, so byte-identical
    logits is also what a never-selected GQA kernel gives. `sg_gpu_forward` now COUNTS which
    partial `enc_attn_splitk` encoded, exposed read-only as
    `sg_gpu_splitk_dispatch_counts`, and the group-size policy is queryable through
    `sg_gpu_splitk_gqa_selected` (which calls the same internal predicate the encoder
    consults, so the [2, 8] band, the repeat-1 decline, the repeat-9 decline, the
    non-multiple decline and the switch-off case are tested rather than commented). Both are
    diagnostics: no kernel reads them and no dispatch shape depends on them. New decode-path
    subtest `mini_f16_splitk_gqa_dispatches_and_matches` (`tests/test_gpu_fwd.c`) asserts
    the selection, the policy and only then end-to-end byte-identity over 1600 positions on
    the mini fixture (4 heads over 2 kv, i.e. repeat 2, in band). The per-op gate now
    dispatches ALL NINE `switch(repeat)` arms (repeat 1..8 plus the past-the-bound default
    arm at 16; 7 and 8 are common real ratios and were previously never run), gives EVERY
    shape its own `> seq` split count so the empty-split -INFINITY/0/0 encoding is exercised
    at every group size rather than only at one 200-token shape, asserts both compared
    buffers actually left their 0xA5 poison, and runs the GQA partial FIRST so the kernel
    under test cannot inherit the reference kernel's score-scratch bytes.
  - **P2.5 + P2.6 (`src/metal.m`, `surge.h`, `tests/test_gpu_fwd.c`): the GQA arm's own
    split policy, and the gate for the regime it creates.** P2.5 gave the GQA partial
    `splitk_gqa_n_splits(seq) = clamp(min(seq/SG_TG, 256), 4, 1024)`, measured (mean regret
    0.43%, worst 2.73% against the per-point optimum); `enc_attn_splitk` overrides `p[6]` in
    a local `pd[8]` copy so the partial and the combine it is paired with agree on how many
    splits were written. The cap can only LOWER the count relative to `splitk_n_splits`, so
    no buffer sizing changed. CONSEQUENCE: from seq 65792 == `SG_TG * (cap + 1)` on, the two
    decode arms partition the same keys differently and agree only to float rounding, which
    narrows P2.4's byte-identity claim to below that seq. P2.6 GATES that instead of arguing
    it. The cap became a per-state value (`SURGE_SPLITK_GQA_CAP`, read in `sg_gpu_state_new`,
    REJECTED rather than ignored outside `[4, 1024]` because a silently dropped value makes
    the gate vacuous), so `=4` reproduces the identical mechanism at seq 1280.
    `splitk_gqa_n_splits` is now a pure function of `(seq, cap)` with one resolver,
    `splitk_gqa_cap_of`, shared by the encoder and by the new read-only diagnostics
    `sg_gpu_splitk_gqa_n_splits_at`, `sg_gpu_splitk_gqa_cap` and `sg_gpu_splitk_n_splits`.
    New decode-path subtest `mini_f16_splitk_gqa_cap_override_greedy_matches` requires
    byte-identity BELOW the divergence seq and a bit difference AT OR ABOVE it while every
    greedy argmax agrees; proved non-vacuous by three mutations that were applied, built and
    run. Defaults unchanged: cap 256, `attn_splitk_gqa` off.
  - **P2.7 (`src/metal.m`, `tests/test_gpu_fwd.c`): a MEASURED occupancy floor on the GQA
    arm.** Collapsing `repeat` per-head threadgroups into one divides the grid by `repeat`, so
    at short context the GQA kernel trades redundant traffic for an idle GPU and measurably
    loses (0.822x on the 27B at seq 2048). `splitk_gqa_use` now also requires
    `splitk_gqa_n_splits(seq, cap) * n_kv_heads >= SG_SPLITK_GQA_MIN_TG == 128`, a floor on the
    THREADGROUP COUNT rather than on seq, because the two real shapes' crossovers are 1.7x
    apart in seq and in the opposite order. Both existing GQA gates had to move above the floor
    (seq 16896) or they would have run the per-head kernel in both arms.
  - **P2.8 (`src/kernels.metal`, `src/metal.m`, `surge.h`, `tests/test_metal_ops.c`,
    `tests/test_gpu_fwd.c`, `tests/bench_splitk.c`): ONLINE-SOFTMAX GQA partial,
    `k_attn_decode_splitk_partial_gqa_online`, `SURGE_ATTN_SPLITK_ONLINE=1`, default OFF.**
    One streaming pass instead of four over a device-memory score row: no `scores` binding
    (seven, not eight), and `splitk_scratch` is neither grown nor bound on that path (about
    25 MB at the 27B's 24 heads and 262144 context if it ever became the only partial; the
    buffer and `splitk_sizes` are untouched here because the four-pass kernels still need it).
    The design turns on WHERE the running accumulator lives: `m`/`s` are uniform and every
    thread keeps a copy, while `acc[d]` has exactly one owner thread, so the state is `4R`
    registers rather than `R*head_dim`, at the cost of transposing the `R x SG_TG` score block
    through 8 KB of threadgroup memory that doubles as the fold scratch for the new R-wide
    trees. Policy: the shared `splitk_gqa_shape_ok` (group band + P2.7 floor) plus
    `head_dim <= SG_TG`; when both kernel switches are on the online arm takes the dispatch and
    `splitk_gqa_use` yields, so the counters cannot double-count. NOT byte-identical (streaming
    reorders the sums), so the bar is the CPU oracle (worst rel 2.109e-06), 100x determinism,
    and byte-exact greedy tokens on the 4B (`gen_ids` identical, 62 of 64 margins differ).
    MEASURED TWO-SIDED and that is why it ships off: 1.013x-1.114x on the 27B at every
    decode-policy point, 0.690x-0.984x on the 4B at depth. `SURGE_ATTN_SPLITK_ONLINE` is
    REJECTED rather than ignored on a bad value, P2.6's rule.
  - **P2.9 (`src/kernels.metal`, `tests/test_metal_ops.c` only; NO host change, no new
    switch): KEY GROUPS in the online partial's V phase.** One output dim per thread left the
    threads with `lid >= head_dim` idle, half the threadgroup at `head_dim` 128. The tile's
    KEYS are now split across `n_kgroups = SG_TG / kw` groups (`kw` = smallest power of two
    `>= head_dim`, capped at `SG_TG`, floored at `SG_SPLITK_ONLINE_KW_MIN` 32), thread `lid`
    joining group `lid / kw` and owning dim `lid % kw`, so every thread owns a dim; the folds
    run per group over `kw` lanes and the V phase walks only its group's keys, so its serial
    depth drops by `n_kgroups`. The tile is still `SG_TG` keys and thread `lid` still scores
    key `base + lid`, so the score phase is untouched. The per-group `(m, s, acc)` are merged
    by the SAME log-sum-exp weights `k_attn_decode_splitk_combine` uses (shared
    `attn_combine_weight`), in fixed group order, and since `kw * n_kgroups == SG_TG` the
    group fold regions and the acc exchange tile the SAME `R x SG_TG` scratch, so the 8 KB
    allocation does not grow. `n_kgroups == 1` (`head_dim >= SG_TG`, the 27B) skips the merge
    and is P2.8's code exactly. MEASURED: the 4B's deep loss is gone (0.690x to 1.015x at seq
    262144, 0.763x to 0.954x at 131072, 1.25x-1.57x faster than P2.8's kernel wherever it lost)
    with the 27B path unmoved; the 8 KB allocation, P2.8's other suspect, was then measured at
    only 0.4%-1.8%. Both switches STILL OFF by default, and the four-pass kernels are
    deliberately untouched (their byte-identity gate couples them to the default decode path).
- **Not built:** the rest of M4 (kernel excellence / beat mlx-lm; split-K decode attention
  is shipped and gated, including P2.4's GQA-shared threadgroup kernel, P2.6's
  greedy-token gate for it and P2.8's online-softmax variant of it, though both kernels are
  still off by default; P2.9 closed the phase-4 thread waste for `head_dim < SG_TG` in the
  ONLINE partial only, so the four-pass partials still have it, and the prefill
  kernels' own optimization is not started), the dense-qwen3
  GPU forward's own numeric gate (P1 is loader-only; deferred until the GPU is free),
  server mode, non-Metal platforms, MoE, continuous batching, sampling beyond
  greedy/temp/top-p/top-k.

## Design constraints

C11, snake_case, `-Wall -Wextra -Werror`, no dependencies beyond macOS frameworks. Metal
reductions are fixed-tree and deterministic (no atomics, no simd_sum) so greedy decode is
byte-identical run-to-run. Correctness is byte-exact greedy TOKENS, not byte-exact logits
(reduction-order and fp16-KV differences are absorbed by argmax). No em dashes in any
file. Full milestone plan: `docs/superpowers/plans/2026-08-09-surge-m3-m5.md`; original
design: `docs/superpowers/specs/2026-08-08-surge-design.md`.
