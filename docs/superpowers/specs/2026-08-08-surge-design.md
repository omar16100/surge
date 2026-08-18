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

Target model: Qwen3.6-27B (`qwen3_5` architecture family), 8-bit first. Post-Task-6
correction: the architecture is HYBRID (1-in-4 full-attention layers, 3-in-4 gated
DeltaNet linear-attention layers, attention output gate); the engine implements both
layer types. The word "dense" in this spec previously reflected a wrong assumption.
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
6. Added 2026-08-18, after measurement: decode-attention PARALLELISM. The five items
   above are bandwidth and overhead levers, and at depth the dominant term was neither.
   See "M4 amendment (2026-08-18)" below for what was measured and shipped.

Budget: 2-3 weeks of profiling iteration in M4. Instrumentation first: every kernel
reports bytes-moved/elapsed so the gap to roofline is always attributable. Risk stated
honestly: if mlx-lm's overhead is smaller than measured elsewhere, the margin may land
at 2-3 percent; the paired protocol is the referee and the result is published either
way.

Measurement discipline for M4, added 2026-08-18 because the decode work (tasks P2.3 to
P4.0) needed all of it. M4 is a performance milestone on a machine whose firmware power
limiter makes naive performance numbers unreliable, so an M4 measured badly will reach
confident wrong conclusions. Full protocol, not restated here:
`/Users/macmini/projects/llm-rnd/docs/benchmarking_methodology.md`.

- Decode tok/s from `surge-bench` CANNOT rank two kernels on this machine. Three
  IDENTICAL arms measured 38.47 / 25.40 / 16.71 tok/s, and after 150 s of idle the same
  three reversed to 39.72 / 41.16 / 34.66 (`docs/18082026_decode_pacing.md:11-15`). That
  is power state, not kernels.
- Prefer a same-run A/B harness where one exists. `tests/bench_splitk.bin` times both
  arms in one process on identical shapes, so its output is a ratio rather than an
  isolated number, which is where every kernel ratio in the decode work came from
  (`docs/18082026_decode_optimization_summary.md:133-142`).
- Use `--reps 20` or more for anything you intend to conclude from. At default
  repetitions two crossover findings in that work reversed, because the effects are a
  few percent and at or below this machine's noise floor
  (`docs/18082026_decode_optimization_summary.md:144-146`).
- Run the fresh GEMM gate per arm, not once per session. On 2026-08-17 a four-arm surge
  A/B measured it once up front, read 0.90 TFLOPS off a still-clamped GPU, and all four
  arms were correctly voided; the accept bar is 20.5 TFLOPS and recovery from a clamped
  state takes 60 to 120 s of genuine idle
  (`/Users/macmini/projects/llm-rnd/docs/benchmarking_methodology.md:26-43`). The gate is
  itself GPU work.

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
  mlx-lm at 2k/8k/32k. The bar of the whole sub-project. AMENDED 2026-08-18: tasks P2.3
  to P4.0 measured and rebuilt the decode path, and the framing on this line missed its
  dominant term. See "M4 amendment (2026-08-18)" below, which records what decode
  actually needed and raises a prefill RECOMMENDATION for the owner to accept or reject.
- M5: kv.c to 128k + tiled prefill path + long-context gate run.
- M6: server.c + cli.c; API conformance tests; chat template.
- M7: sched.c pacing + clamp detection + fanpro hook; sustained-session A/B published.
- M8: spec.c prompt-lookup; byte-identity gate; agentic-workload A/B.
- M9: bench.c leaderboard entry, docs, llm-rnd cross-links. Ship.

Review cadence per llm-rnd convention: external review (codex/kimi when quota returns,
internal adversarial agent otherwise) at M1, M4, and M9 minimum. Added 2026-08-18: M4's
review should check that the measurement discipline above was actually followed, because
end-to-end decode tok/s on this machine reversed the order of three identical arms
(`docs/18082026_decode_pacing.md:11-15`), so a kernel A/B run the wrong way returns a
confident answer rather than an obviously noisy one.

## M4 amendment (2026-08-18): measured decode, and the prefill question

M4 as written above treats decode as a bandwidth-and-fusion problem. Tasks P2.3 through
P4.0 measured it, and the dominant term was neither bandwidth nor fusion. This section
records what decode attention actually needed, what now ships, the two results that would
be re-derived wrongly if the spec stayed silent, and one recommendation the owner has not
yet ruled on. Consolidated write-up with every source:
`docs/18082026_decode_optimization_summary.md`. Per-task gate docs:
`docs/16082026_splitk_decode_gate.md` (P2.3),
`docs/17082026_splitk_gqa_threadgroups.md` (P2.4 to P2.9 and P4.0),
`docs/18082026_decode_pacing.md` (P3.0).

### What decode attention actually needed, and that it is DONE

The missing term was OCCUPANCY. The incumbent kernel `k_attn_decode_f16`
(`src/kernels.metal:455`) is dispatched one threadgroup per query head
(`src/metal.m:1743`), which is what the decode path did at every depth before P2.3. On
the 27B's 24-head shape that gave only 24 of this machine's 80 GPU cores work, with one
threadgroup walking the entire KV sequence
(`docs/18082026_decode_optimization_summary.md:13-15`). No amount of autotune, repack or
fusion addresses that. Splitting the KV sequence across threadgroups does, and nothing on
the M4 line above proposed it.

| task | change | measured effect |
|---|---|---|
| P2.3 | split-K (flash decoding) wired into `enc_attn` | 9.8x on real 4B decode at 32825 tokens (1.74 to 17.12 tok/s) |
| P2.4 | one threadgroup per GQA GROUP instead of per query head | a further 1.74x at the 27B 262144 shape |
| P2.5 | GQA-specific split policy, `clamp(min(seq / SG_TG, 256), 4, 1024)` (`src/metal.m:1491`) | mean regret 0.43 percent against the per-point optimum, versus 3.1 percent for the inherited policy |
| P2.6 | greedy-token gate where the two split policies diverge | closed the last correctness gap |
| P2.7 | occupancy floor, `n_splits * n_kv_heads >= 128` (`src/metal.m:1305`) | removed a measured 0.822x regression at short context |
| P2.8 | online softmax, one streaming pass, no device score row | 27B 1.013x to 1.114x; 4B 0.690x to 0.984x, so shipped OFF |
| P2.9 | key groups in the online V phase | 4B 262144 went 0.690x to 1.015x; 27B unmoved |
| P3.0 | decode pacing with clamp detection (`src/sched.c`) | mechanism correct and gated; effect NOT demonstrated |
| P4.0 | flip `attn_splitk_gqa` ON by default (`src/metal.m:3762`) | user-approved 2026-08-18 |

Cumulative on the kernel that matters: 28.3x over the original decode attention at the
27B `24h/4kv/256d` shape at 262144, and this SHIPS BY DEFAULT as of P4.0
(`docs/18082026_decode_optimization_summary.md:35-36`, and the 4217.15 us at 256 splits
against the incumbent in `docs/17082026_splitk_gqa_threadgroups.md:210`). Correctness
held throughout at the ladder above: `make check` 87600 checks and `make debug` 83614
checks, both 0 failures, with byte-identity claimed only where it applies and positive
controls that count dispatches rather than assert equality
(`docs/18082026_decode_optimization_summary.md:97-108`).

Online softmax stays OFF (`SURGE_ATTN_SPLITK_ONLINE`): a 9 to 12 percent win on the 27B,
still 1.9 to 5 percent behind best-vs-best on the 4B
(`docs/18082026_decode_optimization_summary.md:119-120`).

### Two counter-intuitive results, because they are the reusable part

Both are cases where the obvious rule was measured to be wrong, and both would be
re-derived wrongly by the next person planning M4.

**The split policy is a CAP, not a rescaling.** "GQA wants half the per-head splits" fits
the two deepest data points and is the WORST of four candidates at 13.4 percent
worst-case regret, because the optimum saturates with depth rather than scaling with it
(`docs/18082026_decode_optimization_summary.md:39-52`):

| policy | mean regret | worst |
|---|---|---|
| `seq/SG_TG` (inherited) | 3.1% | 7.3% |
| `min(seq/SG_TG, 256)` (shipped) | 0.43% | 2.73% |
| `min(seq/SG_TG, 512)` | 1.5% | 4.2% |
| `seq/(2*SG_TG)` (the intuition) | 3.8% | 13.4% |

A two-point extrapolation would have shipped a 13 percent regression.

**The occupancy guard is a threadgroup count, not a sequence length.** The GQA kernel
collapses `repeat` threadgroups into one, dividing the grid by `repeat`, so at seq 2048
the 27B gets 8 * 4 = 32 threadgroups against 80 cores and measures 0.822x, slower than
the kernel it replaces. A `seq` threshold cannot express the fix: the per-shape
crossovers are 1.7x apart in `seq` (5120-6144 on the 27B, 3072-4096 on the 4B) AND in the
OPPOSITE order from the threadgroup view, because the 4B has twice the kv heads. One
threadgroup threshold separates every measured point; one `seq` threshold must either
admit a 27B loss or reject 4B wins
(`docs/18082026_decode_optimization_summary.md:54-63`).

### RECOMMENDATION to the owner: M4's remaining target is prefill

This is a recommendation, not a decision taken here. The M4 bar above is unchanged and is
not deleted: "paired-protocol decode beats mlx-lm at 2k/8k/32k" stands until the owner
rules otherwise.

Two things about that bar are worth putting in front of the owner. First, its work is
substantially done while the bar itself is unmeasured: P2.3 to P4.0 rebuilt decode
attention, but no gate doc or ledger entry records the paired-protocol comparison against
mlx-lm at 2k/8k/32k ever having been run, so the milestone cannot be called met.

Second, the two gaps now point in opposite directions. On the identical 234,158-token raw
prompt at 262144, surge prefills at 2.99 tok/s compute (2.08 wall, the difference being
B8's deliberate duty-cycle idle) against llama.cpp's 95.6 and mlx-lm's 101.9, roughly 32x
behind (`/Users/macmini/projects/llm-rnd/docs/256k_comparison.md:118-127` and `:221-222`).
And prefill is where surge's wall time goes: the completed 256K run took 31.4 h of which
112359 s was prefill, roughly 99 percent of the run
(`/Users/macmini/projects/llm-rnd/docs/256k_comparison.md:365`,
`docs/18082026_decode_optimization_summary.md:112-113`).

Recommendation: move M4's emphasis to the prefill path and keep the decode bar as a
non-regression check rather than as the milestone's work. The owner accepts or rejects
this; no milestone above has been changed on the strength of it.

Two honesty notes that belong with the recommendation. surge's published 256K row is
STALE: it records 0.537 tok/s decode and predates every task in the table above.
Refreshing it costs about 31 hours that are roughly 99 percent prefill, which none of
this work improved. And the decode gain is not yet an end-to-end claim: the
attention-only projection is about 14.8 tok/s, the honest band is 8 to 15, it has never
been measured end to end at that depth, and at 21120 tokens the measured end-to-end gain
was only 1.07x because attention is not the dominant cost there
(`docs/18082026_decode_optimization_summary.md:112-116`).

## Risks

- The M4 margin may be small (2-3 percent); mitigations are speculation and sustained
  mode, which win regardless, but the stated bar is M4's and a miss is reported, not
  spun. REVISITED 2026-08-18: this risk anticipated the wrong axis. On decode the
  headroom was not a few percent of MLX overhead, it was 28.3x of surge's own occupancy
  loss on the attention kernel at the 27B 262144 shape
  (`docs/18082026_decode_optimization_summary.md:35-36`). That is a kernel-level ratio
  against surge's previous kernel, NOT a margin over mlx-lm, and the paired-protocol
  comparison at 2k/8k/32k that the bar names has still not been run. The risk that
  remains is on the other axis: surge is roughly 32x BEHIND llama.cpp on prefill on the
  same prompt (`/Users/macmini/projects/llm-rnd/docs/256k_comparison.md:221-222`). See
  the M4 amendment above.
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
