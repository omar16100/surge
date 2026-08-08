# surge: a production C inference engine for the M3 Ultra (design)

Date: 2026-08-08. Status: approved design, pre-implementation. Name `surge` confirmed
by the owner; public MIT repo at github.com/omar16100/surge.

## Context

The owner runs local LLMs daily on a Mac Studio M3 Ultra (512 GB, macOS 26.3) and has
unusually precise knowledge of this machine from the llm-rnd benchmarking campaign:
measured 630 GB/s streaming bandwidth and 21.9 TFLOPS GEMM ceilings, a firmware GPU
power limiter that clamps to 338 MHz after ~3 minutes at ~141-151 W and releases after
60-120 s of idle, fan control via the owner's fanpro tool (max fans roughly double the
sustained budget), and a proven byte-identical prompt-lookup speculation win (2.14x on
edit/agentic workloads, measured on ds4). Existing engines (ds4, llama.cpp, MLX) know
none of this. surge is a production daily driver built to exploit it.

Decomposition: sub-project 1 (this spec) is the production core targeting a dense model.
Sub-project 2 (DeepSeek-V4-Flash MoE support: MLA, compressed attention, lightning
indexer) is designed only after sub-project 1 ships its success bar.

## Goals and success bar (sub-project 1)

Target model: Qwen3.6-27B (dense, `qwen3_5` architecture family), 8-bit first.
Plan-time addendum: exact-correctness validation runs on `Qwen/Qwen3.5-2B` bf16 (same
architecture; disk cannot hold a 56 GB bf16 27B next to the quants), and the 27B is
validated at Q8_0 against mlx-lm with the top-1/threshold gate.

1. Correctness: teacher-forced logit agreement against a bf16 reference at the
   thresholds established in llm-rnd (top-1 agreement, mean KL, margin distribution),
   plus the sequence-level first-divergence protocol. Greedy Metal bf16 output must
   match the CPU reference exactly at temp 0.
2. Speed, the owner's explicit bar: **single-stream decode strictly faster than mlx-lm
   0.31.3 on the same 8-bit quant at 2k, 8k, and 32k context**, measured with the
   llm-rnd paired protocol (fresh GEMM gate, interleaved arms, decode by slope).
   Expected margin 5-12 percent from MLX overhead reclamation; honest floor is 2-3
   percent, and the paired protocol decides.
3. Production surface: OpenAI-compatible HTTP server (chat + completions, SSE
   streaming), CLI, 128k context, stable unattended operation.
4. Machine-aware features (the differentiators): limiter-aware pacing scheduler,
   clamp detection, optional fanpro pre-spin hook, prompt-lookup speculation
   (byte-identical gate), per-kernel achieved-GB/s instrumentation.

Non-goals for sub-project 1: MoE, CUDA/other platforms, multi-user continuous batching
(server queues requests FIFO, one active generation), training, quant conversion
tooling beyond load-time repacking, sampling beyond temp/top-p/top-k/repetition.

## Architecture

Plain C11 + Metal, fanpro conventions: Makefile, compile_flags.txt, tinytest harness,
no dependencies beyond macOS frameworks (Metal, Foundation for the .m file, Accelerate
optionally for the reference path). snake_case throughout. Target size: ~8-15k lines C,
~2-3k lines Metal.

Components (one file each unless noted, interfaces in `surge.h`):

- `gguf.c` - mmap GGUF v3 reader: metadata table, tensor directory, Q8_0 and bf16
  tensor views. No copies; weights stay mapped. Q4_K added in a later milestone.
- `tok.c` - byte-level BPE tokenizer from GGUF metadata (plan-time addendum: the
  original component list omitted the tokenizer).
- `st.c` - safetensors bf16 loader for the reference path only.
- `model_qwen.c` - the dense graph: embedding, RMSNorm, RoPE (family variant read from
  GGUF metadata), GQA attention, SwiGLU MLP, final norm + lm_head. Dimensions
  (layers, heads, kv heads, head_dim, vocab) come from GGUF metadata at load; nothing
  is hardcoded. KV memory budget is computed at load from those dims and printed.
- `ref.c` - scalar CPU implementation of every op. Slow by design; correctness oracle.
- `metal.m` - device init, MTLResidencySet for all weight views, one command buffer per
  decode token (all layer kernels pre-encoded, fused chains), tiled prefill encoder,
  pipeline cache, autotune cache (per-shape threadgroup choices persisted to
  `~/.surge/autotune.json`).
- `kernels.metal` - fused dequant(Q8_0)+matvec, bf16 matvec, RMSNorm+matvec fusion,
  RoPE, flash-style single-pass decode attention over fp16 KV, tiled prefill matmul +
  attention, SwiGLU, argmax/logit readback.
- `kv.c` - preallocated fp16 KV to a configured max (default 131072 tokens),
  head-interleaved layout chosen for coalesced GQA decode reads; contiguous per layer.
- `sched.c` - pacing scheduler. Maintains an energy budget model calibrated from
  llm-rnd measurements (trip after ~3 min at ~141-151 W, recovery 60-120 s; fans-max
  roughly doubles the budget). Modes: `burst` (default, no pacing), `sustain` (inject
  idle gaps to stay under the trip line, like ds4 --power but budget-model driven),
  `auto` (burst until the model predicts a trip, then sustain). Clamp detector: a
  2 ms timing probe kernel whose latency identifies the clamped state; on detection,
  log and optionally back off. fanpro hook: optional configured command executed at
  session start/end (off by default; documented as requiring the user's fanpro setup).
- `spec.c` - prompt-lookup speculation: ngram-4 match over session history, draft depth
  default 7, batched verify (<= 8 rows) through the same fused kernels, exact rollback.
  Byte-identity with speculation off is a release gate, as proven achievable on ds4.
- `server.c` - plain-socket HTTP server, OpenAI-compatible `/v1/chat/completions` and
  `/v1/completions`, SSE streaming, chat template read from GGUF metadata or a
  sidecar jinja file (subset renderer; the Qwen template's used features only).
- `cli.c` - one-shot and interactive modes, flags mirroring server options.
- `bench.c` - built-in ladder bench (2k/8k/32k/128k), decode by slope, emits the
  llm-rnd leaderboard row format plus per-kernel achieved GB/s vs the 630 GB/s roofline.

Data flow (decode): tokens -> embedding lookup -> per-layer fused chains encoded in one
command buffer -> logits -> sampler (CPU) -> next token. Prefill: chunked tiled matmul
path with its own encoder, chunk size autotuned.

## Correctness ladder

1. `ref.c` on bf16 safetensors is ground truth. Validated once against mlx-lm
   teacher-forced on the same weights; cross-framework bf16 is not bit-exact, so the
   acceptance is 100 percent top-1 agreement over the fixture set and max absolute
   logit delta consistent with reduction-order noise (< 1e-2), not byte equality.
2. Metal bf16 must match `ref.c` greedy byte-exactly at temp 0 (dense kernels, no
   atomics, deterministic reduction order is a design requirement).
3. Q8_0 fused path gates against bf16 with the llm-rnd teacher-forced thresholds and
   the first-divergence distribution protocol.
4. Speculation on vs off: byte-identical, every release.
5. Golden logit fixtures (16 prompts, first 64 positions) checked into the repo;
   `make check` runs per-op fuzz vs ref + fixtures; `make gate` runs the teacher-forced
   comparator against a configured model dir.

## Performance plan (the kernel-excellence phase)

Physics: 8-bit 27.8B dense decode moves ~29.5 GB of weights per token; at 630 GB/s the
ceiling is ~21 t/s, and mlx-lm runs near it. The bar is beating mlx-lm outright, so the
margin comes from overhead MLX pays and surge does not:

1. One pre-encoded command buffer per token; fused op chains; target < 40 total
   dispatches per token (mlx-lm pays per-op graph eval + launch overhead).
2. Load-time weight repacking into a matvec-native Q8_0 layout (interleaved scales,
   aligned rows, head-blocked for the attention projections).
3. First-run autotuner over threadgroup sizes/occupancy per shape on this 80-core GPU,
   cached to disk.
4. Fused residual+RMSNorm+matvec chains to remove activation round-trips.
5. Head-interleaved KV so 32k-depth GQA reads stay coalesced (KV reads are the
   second-order bandwidth term at depth).

Budget: 2-3 weeks of profiling iteration in M4. Instrumentation first: every kernel
reports bytes-moved/elapsed so the gap to roofline is always attributable. Risk stated
honestly: if mlx-lm's overhead is smaller than measured elsewhere, the margin may land
at 2-3 percent; the paired protocol is the referee and the result is published either
way.

## Testing

- tinytest (fanpro convention), `make check` green required for every commit.
- Per-op fuzz: random shapes/values, Metal vs ref, tolerance per dtype.
- Golden fixtures + teacher-forced gate as make targets.
- `make bench` runs the standard llm-rnd protocol (fresh gate, fans state recorded,
  slope decode) and appends to a local results log; leaderboard rows are copied to
  llm-rnd's docs/leaderboard.md manually with source paths.
- Server: a black-box pytest-free shell test (curl scripts) for API conformance + SSE.

## Milestones

- M0: repo skeleton, Makefile, tinytest, gguf.c reads the Qwen3.6-27B-8bit GGUF
  (metadata + tensor table printed, mmap verified).
- M1: ref.c full forward; teacher-forced validation vs mlx-lm on bf16; fixtures frozen.
- M2: Metal bf16 decode path byte-exact vs ref at temp 0.
- M3: fused Q8_0 decode at >= 90 percent of roofline; correctness gate passes.
- M4: kernel excellence: autotune + repack + fusion until paired-protocol decode beats
  mlx-lm at 2k/8k/32k. The bar of the whole sub-project.
- M5: kv.c to 128k + tiled prefill path + long-context gate run.
- M6: server.c + cli.c; API conformance tests; chat template.
- M7: sched.c pacing + clamp detection + fanpro hook; sustained-session A/B published.
- M8: spec.c prompt-lookup; byte-identity gate; agentic-workload A/B.
- M9: bench.c leaderboard entry, docs, llm-rnd cross-links. Ship.

Review cadence per llm-rnd convention: external review (codex/kimi when quota returns,
internal adversarial agent otherwise) at M1, M4, and M9 minimum.

## Risks

- The M4 margin may be small (2-3 percent); mitigations are speculation and sustained
  mode, which win regardless, but the stated bar is M4's and a miss is reported, not
  spun.
- GGUF Q8_0 for Qwen3.6-27B must exist or be produced (llama.cpp quantize can make it
  from the bf16 safetensors; conversion is a documented step, not engine code).
- mlx-lm is a moving target; the comparison pins 0.31.3 (the installed daily driver).
- Single-maintainer C + Metal: memory-safety discipline via ASan/TSan debug builds and
  the fuzz suite; no dependencies means no supply-chain surface but also no free
  fixes.
- macOS updates can change Metal behavior and the limiter itself (26.5 reportedly
  overhauls AGX kexts); the machine stays on 26.3 for the project duration per the
  llm-rnd freeze.

## Resolved owner decisions (2026-08-08)

- Name: `surge`, confirmed.
- Visibility: public open-source from day one (github.com/omar16100/surge), MIT license
  (matching fanpro).
- Attribution: the README credits the llm-rnd experimentation campaign on this machine
  (the firmware-limiter discovery, roofline measurements, and benchmarking methodology
  that this engine's design exploits), linking the published write-up at
  https://omarshabab.com/mac-studio-firmware-gpu-limiter/ and the public
  github.com/omar16100/llm-benchmark harness.
