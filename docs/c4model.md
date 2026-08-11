# surge architecture (C4 model)

Source of truth for surge's structure. Read before any architecture change; update on
every architecture change (containers, components, data flows, status). surge is a
from-scratch LLM inference engine in C11 + Metal for the Mac Studio M3 Ultra, no
dependencies beyond what macOS ships. Public: github.com/omar16100/surge (MIT).

## Level 1: Context

surge takes a model on disk (GGUF or safetensors) and a prompt, and generates tokens on
the Apple Silicon GPU (Metal). It is a single-process, single-user, greedy-first engine
built to run one hybrid `qwen3_5`/`qwen35` family model end to end and, per the M3+M5
plan, to run the same Qwen3.6-27B-Q8_0 GGUF the mlx-lm and llama.cpp engines are compared
on at 262,144-token context. Correctness is validated against a CPU reference and against
mlx-lm; speed against mlx-lm on the same weights (that comparison lives in the llm-rnd
project's `256k_comparison.md` and `leaderboard.md`).

External dependencies: only Metal.framework, Foundation (for the one .m file), and
optionally Accelerate for the CPU reference path. No third-party libraries.

## Level 2: Containers

- **libsurge (static object set)** compiled from `src/*.c` (minus the CLI mains and
  `metal.m`), linked into every binary and every pure-C test. Metal-free, so pure-C
  tests build and run without a GPU.
- **Metal layer**: `src/metal.m` (Objective-C, device + command buffers + kernel
  dispatch) plus `src/kernels.metal` (compiled to `kernels.metallib` by `xcrun metal`).
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
| `src/model_qwen.c` | config extraction + weight-name mapping for the hybrid qwen3_5/qwen35 arch (full-attention + gated-DeltaNet layers), from GGUF and safetensors. Carries per-tensor dtype and the source-flags `ssm_a_form` / `v_heads_tiled`. |
| `src/ref.c` | scalar CPU reference for every op (the correctness oracle), incl. `sg_ref_matvec_q8`, attention, gated-DeltaNet, partial RoPE, and the full forward pass. |
| `src/kv.c` | (M5.1) fp16 growable KV cache for full-attention layers + fixed-size DeltaNet recurrent state, over opaque GPU-buffer handles; Metal-free (allocation injected). `sg_kv_bytes` = 16 GiB K+V at 262,144. Wired into decode by M5.2 (see below). |
| `src/bench.c` | (B-series) pure-C benchmark math: decode-by-slope, leaderboard-row/JSON formatters, prompt file read, ingestion/truncation guard, NIAH recall scorer, process `phys_footprint` probe + the `sg_mem_tracker` peak-max tracker (B2). |
| `src/greedy.c` | the ONE greedy-decode argmax (`sg_argmax_f32`, lowest index wins an exact tie). Pure C in LIB_SRC; `surge` and `surge-bench` both call it so their gen_ids cannot drift (B5). |
| `src/metal.m` | Metal device init, weight-buffer wrapping (mmap, no-copy; bf16/f32/Q8_0 sizing), per-weight matvec kernel dispatch (`matmul_kernel_for`), one command buffer per decode token, one command buffer per prefill chunk (`sg_gpu_prefill`, M5.6), kernel registration (KI_ enum + SG_KERNELS table + size/param checks), `sg_gpu_current_alloc_bytes` (B2, `MTLDevice.currentAllocatedSize`). Registers itself as `sg_kv`'s allocation backend at init. Decode state: the DEFAULT full-attention KV cache is fp16, allocated through `sg_kv` as SEPARATE per-layer K/V buffers (M5.2, `SURGE_KV_DTYPE` env toggle); `SURGE_KV_DTYPE=f32` selects the original combined-buffer f32 path unchanged, kept so the M2 gate's oracle never moves. DeltaNet conv/S state stays on its pre-M5.1 ad hoc allocation for decode; prefill (M5.6) threads DeltaNet state through `sg_kv`'s conv/S carriers (so `g->kv` is now allocated for any DeltaNet model on the f16 path) and bridges the final state back into the ad hoc buffers when it finishes. |
| `src/kernels.metal` | deterministic Metal kernels: decode-step ones fold a fixed reduction tree (no atomics/simd_sum) -- bf16/f32/Q8_0 matvec, attention decode (f32-KV `k_attn_decode` and fp16-KV `k_attn_decode_f16`, M5.2), a decode-step fp16 store (`k_kv_store_f16`, M5.2), gated-DeltaNet decode, RoPE, RMSNorm, SwiGLU. Prefill's tiled GEMM (M5.3, `k_matmul_bf16`/`k_matmul_f32`/`k_matmul_q8`, `Y[N,M]=X[N,K]@W[M,K]^T`) uses a different determinism mechanism: one thread per output element of a 16x16 output tile, a private serial K-loop, no cross-thread reduction at all (nothing to fold). Full-attention prefill (M5.4) adds `k_rope_chunk` (partial RoPE over a whole chunk) and `k_attn_prefill` (causal chunk attention over the fp16 KV cache, one threadgroup per (query token, query head), same fixed-tree softmax as `k_attn_decode_f16`). DeltaNet prefill (M5.5) adds `k_conv1d_chunk` / `k_delta_gates_chunk` / `k_delta_chunk` / `k_rmsnorm_gated_chunk`: a whole chunk of N tokens through one DeltaNet layer via a SEQUENTIAL within-chunk scan (per-token loop inside the kernel, threading the conv tail + S matrix), bit-identical to the matching decode kernel looped over the chunk. |
| `src/cli_*.c` | the four CLI mains (`cli_info`, `cli_ref`, `cli_metal` = `surge`, `cli_bench` = `surge-bench`). `cli_bench` (B5): raw tokenize (no chat template), `--bos`/`--no-bos` (default `tokenizer.ggml.add_bos_token`), M5 tiled prefill (`--chunk`, default 1024) + shared greedy decode, GEMM gate (`--gemm-gate-tflops F`, need F>20.5) + ingestion guard, whole-run peak-mem sampling, NIAH recall (text input only), decode-by-slope (`--warmup`, `--emit-timeseries`), md row to stdout + `--json`. Exit 0 DONE / 3 VOID / other hard error. |

## Level 4: Data flows

- **Decode (M2, shipped; KV dtype refit M5.2):** token -> host embedding lookup -> per
  layer, one command buffer: RMSNorm, projections (matvec), qk-norm + partial RoPE, KV
  append (fp16 by default: q/k-norm and RoPE run in an f32 scratch, then `k_kv_store_f16`
  casts into `sg_kv`'s per-layer half buffers; `SURGE_KV_DTYPE=f32` keeps the original
  path), attention decode (full-attn layers, `k_attn_decode_f16` or `k_attn_decode`) or
  conv-step + delta-step (DeltaNet layers), output gate, residuals; final norm + lm_head
  -> logits -> host argmax -> next token. Byte-exact GREEDY TOKENS to the CPU reference at
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
    shared `sg_argmax_f32`), and B6 (offline decode-slope + wall-accounting verification) done;
    B7 (gated 256K run) pending. B6 adds two additive `sg_bench_row` fields,
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
- **Not built:** M4 (kernel excellence / beat mlx-lm), server mode, non-Metal platforms,
  MoE, continuous batching, sampling beyond greedy/temp/top-p/top-k.

## Design constraints

C11, snake_case, `-Wall -Wextra -Werror`, no dependencies beyond macOS frameworks. Metal
reductions are fixed-tree and deterministic (no atomics, no simd_sum) so greedy decode is
byte-identical run-to-run. Correctness is byte-exact greedy TOKENS, not byte-exact logits
(reduction-order and fp16-KV differences are absorbed by argmax). No em dashes in any
file. Full milestone plan: `docs/superpowers/plans/2026-08-09-surge-m3-m5.md`; original
design: `docs/superpowers/specs/2026-08-08-surge-design.md`.
