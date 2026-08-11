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
  `surge` (Metal decode), `surge-bench` (benchmark harness; being built in the bench
  tasks).
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
| `src/bench.c` | (B-series) pure-C benchmark math: decode-by-slope, leaderboard-row/JSON formatters, prompt file read, ingestion/truncation guard, NIAH recall scorer. |
| `src/metal.m` | Metal device init, weight-buffer wrapping (mmap, no-copy; bf16/f32/Q8_0 sizing), per-weight matvec kernel dispatch (`matmul_kernel_for`), one command buffer per decode token, kernel registration (KI_ enum + SG_KERNELS table + size/param checks). Registers itself as `sg_kv`'s allocation backend at init. Decode state: the DEFAULT full-attention KV cache is fp16, allocated through `sg_kv` as SEPARATE per-layer K/V buffers (M5.2, `SURGE_KV_DTYPE` env toggle); `SURGE_KV_DTYPE=f32` selects the original combined-buffer f32 path unchanged, kept so the M2 gate's oracle never moves. DeltaNet conv/S state stays on its pre-M5.1 ad hoc allocation either way (DeltaNet decode is not yet refit onto `sg_kv`). |
| `src/kernels.metal` | deterministic Metal kernels: decode-step ones fold a fixed reduction tree (no atomics/simd_sum) -- bf16/f32/Q8_0 matvec, attention decode (f32-KV `k_attn_decode` and fp16-KV `k_attn_decode_f16`, M5.2), a decode-step fp16 store (`k_kv_store_f16`, M5.2), gated-DeltaNet decode, RoPE, RMSNorm, SwiGLU. Prefill's tiled GEMM (M5.3, `k_matmul_bf16`/`k_matmul_f32`/`k_matmul_q8`, `Y[N,M]=X[N,K]@W[M,K]^T`) uses a different determinism mechanism: one thread per output element of a 16x16 output tile, a private serial K-loop, no cross-thread reduction at all (nothing to fold). Full-attention prefill (M5.4) adds `k_rope_chunk` (partial RoPE over a whole chunk) and `k_attn_prefill` (causal chunk attention over the fp16 KV cache, one threadgroup per (query token, query head), same fixed-tree softmax as `k_attn_decode_f16`). DeltaNet prefill (M5.5) adds `k_conv1d_chunk` / `k_delta_gates_chunk` / `k_delta_chunk` / `k_rmsnorm_gated_chunk`: a whole chunk of N tokens through one DeltaNet layer via a SEQUENTIAL within-chunk scan (per-token loop inside the kernel, threading the conv tail + S matrix), bit-identical to the matching decode kernel looped over the chunk. |
| `src/cli_*.c` | the four CLI mains. |

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
- **Prefill (M5, in progress):** chunk the prompt (default 1024 tokens); per chunk one
  command buffer runs all layers via tiled GEMM + tiled/flash attention + the DeltaNet
  chunked scan, advancing `sg_kv` by the chunk size; only the final chunk's last row
  computes lm_head. Correctness gate is token-level: prefill-then-decode == feed-one-at-
  a-time.

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
    threaded, gated output norm, o_proj). `enc_gdn_prefill` is built but not yet driven
    (M5.6's orchestration sizes its chunk buffers and gates the whole prefill numerically
    prefill-then-decode == feed-one-at-a-time). M5.6-M5.7 pending.
  - Bench harness: B1/B3/B4 (pure C) done; B2/B5/B6 (Metal/CLI) and B7 (gated 256K run)
    pending.
- **Not built:** M4 (kernel excellence / beat mlx-lm), server mode, non-Metal platforms,
  MoE, continuous batching, sampling beyond greedy/temp/top-p/top-k.

## Design constraints

C11, snake_case, `-Wall -Wextra -Werror`, no dependencies beyond macOS frameworks. Metal
reductions are fixed-tree and deterministic (no atomics, no simd_sum) so greedy decode is
byte-identical run-to-run. Correctness is byte-exact greedy TOKENS, not byte-exact logits
(reduction-order and fp16-KV differences are absorbed by argmax). No em dashes in any
file. Full milestone plan: `docs/superpowers/plans/2026-08-09-surge-m3-m5.md`; original
design: `docs/superpowers/specs/2026-08-08-surge-design.md`.
