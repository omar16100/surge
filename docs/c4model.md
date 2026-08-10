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
| `src/kv.c` | (M5.1) fp16 growable KV cache for full-attention layers + fixed-size DeltaNet recurrent state, over opaque GPU-buffer handles; Metal-free (allocation injected). `sg_kv_bytes` = 16 GiB K+V at 262,144. |
| `src/bench.c` | (B-series) pure-C benchmark math: decode-by-slope, leaderboard-row/JSON formatters, prompt file read, ingestion/truncation guard, NIAH recall scorer. |
| `src/metal.m` | Metal device init, weight-buffer wrapping (mmap, no-copy), one command buffer per decode token, kernel registration (KI_ enum + SG_KERNELS table + size/param checks), state via `sg_kv`. |
| `src/kernels.metal` | deterministic Metal kernels (fixed-tree reductions, no atomics/simd_sum): bf16/f32/Q8_0 matvec, attention decode, gated-DeltaNet decode, RoPE, RMSNorm, SwiGLU. Tiled prefill kernels arrive in M5. |
| `src/cli_*.c` | the four CLI mains. |

## Level 4: Data flows

- **Decode (M2, shipped):** token -> host embedding lookup -> per layer, one command
  buffer: RMSNorm, projections (matvec), qk-norm + partial RoPE, KV append, attention
  decode (full-attn layers) or conv-step + delta-step (DeltaNet layers), output gate,
  residuals; final norm + lm_head -> logits -> host argmax -> next token. Byte-exact to
  the CPU reference at temp 0 (M2 gate).
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
  - M3 (Q8_0 weights end-to-end): M3.1 `k_matvec_q8` done; M3.2 kernel twins, M3.3 loader
    dispatch, M3.4 Q8_0 forward gate pending.
  - M5 (fp16 KV to 262,144 + tiled prefill): M5.1 `kv.c` done; M5.2-M5.7 pending.
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
