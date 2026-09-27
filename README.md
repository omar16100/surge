# surge

An experimental LLM inference engine in C11 and Metal, built for one machine: the
Mac Studio M3 Ultra. No dependencies beyond what macOS ships. It is a work in
progress: single process, single user, greedy decoding only, and slower than
mlx-lm and llama.cpp on the workloads it has been measured on (see "Where it
stands").

Architecture and built-vs-planned status live in `docs/c4model.md`, which is the
source of truth; `docs/index.md` lists every design and gate doc.

## Status

Built and gated:

- **M0-M2**: GGUF and safetensors loading for the hybrid qwen3_5 family (full
  attention plus gated DeltaNet), a scalar CPU reference forward (M1 gate: 100%
  top-1 vs mlx-lm on Qwen3.5-2B), and Metal decode whose greedy tokens are
  byte-exact to the CPU reference (M2 gate).
- **M3**: Q8_0 weights end to end. On the Qwen3.6-27B Q8_0 GGUF, Metal vs the CPU
  reference is 100% top-1 teacher-forced, and greedy output is byte-identical to
  llama.cpp on 4 prompts (`docs/11082026_m34_q8_gate.md`).
- **M5**: fp16 KV cache up to 262,144 tokens and chunked tiled prefill, gated
  prefill-then-decode == serial-then-decode (`docs/11082026_m57_longctx_gate.md`).
- **Benchmark harness** `surge-bench` (tasks B1 to B8).
- **Decode attention** (tasks P2.3 to P4.0): split-K attention and GQA-shared
  threadgroups, 28.3x faster on the decode-attention kernel at the 27B shape and
  262,144 context (`docs/18082026_decode_optimization_summary.md`).
- **Dense qwen3 GGUF loading** (task P1, loader only; its GPU numeric gate is
  still open).

Not built: the rest of M4 (kernel work toward the mlx-lm bar; prefill
optimization has not started), sampling, server mode, MoE, continuous batching,
and non-Metal platforms. The design spec's optional fanpro fan pre-spin hook is
deliberately not built, and prompt-lookup speculation is not implemented.

## Where it stands

The stated goal is to beat mlx-lm single-stream decode on the same quant on this
hardware. It has not been met. A 256K-context run of Qwen3.6-27B-Q8_0 (task B7,
2026-08-13, before the decode attention work above) decoded at 0.537 tok/s,
against 7.58 for mlx-lm and 5.10 for llama.cpp on the identical prompt, and that
row has not been re-measured since. Prefill is now the larger gap: 2.99 tok/s of
compute against llama.cpp's 95.6 on the same prompt, about 32x behind. Both
figures are recorded in `docs/18082026_decode_optimization_summary.md`.

## The GPU limiter premise, retracted

surge was originally designed around a premise from earlier measurement of this
machine: that a firmware GPU limiter clamps the clock to 338 MHz after about 3
minutes of sustained load and releases it after 60 to 120 seconds idle. The B8
prefill duty cycle was built on it. On 2026-08-15, telemetry from surge's own
30-hour 256K run did not support that premise
(`docs/15082026_prefill_duty_cycle_plan.md`):

- GPU clock was flat within each burst rather than decaying.
- Idle time before a burst did not predict its clock (r = +0.017 over 376 burst
  pairs).
- 338 MHz appeared in 27 of 7,724 loaded samples (0.3%).
- Clock tracked context length (r = -0.575), consistent with long-context prefill
  becoming memory-bandwidth bound. It was not thermal: temperature and clock
  correlated positively.

This does not prove that no limiter exists; the discriminating experiment has not
been run. The prefill duty cycle was kept, off by default, and repurposed: it
periodically yields the GPU to reduce the risk of the macOS compositor being
killed by its watchdog during long prefills, as happened on 2026-08-14 (the
mitigation is not proven sufficient). `src/sched.c` (task P3.0) adds a decode
duty cycle, also off by default, and a clamp detector that by default only
reports (the opt-in `--decode-clamp-div` shortens the work budget while a clamp
is confirmed). No throughput benefit from pacing has been demonstrated
(`docs/18082026_decode_pacing.md`).

## Build and test

- `make check`: builds and runs the unit tests plus the Metal and CLI checks.
  Needs Xcode's Metal toolchain; the Metal tests skip when no GPU is available.
- `make debug`: the pure-C tests under ASan and UBSan, with Metal excluded.
  This is what CI runs (`.github/workflows/ci.yml`); `make check` is a local
  gate, since its full coverage needs the real machine.
- `make surge`, `make surge-bench`: the decode CLI and the benchmark harness.

Background reading: the write-up the premise came from,
[macOS clamps my M3 Ultra's GPU to 338 MHz before the fans even try](https://omarshabab.com/mac-studio-firmware-gpu-limiter/),
and the benchmark harness at
[llm-benchmark](https://github.com/omar16100/llm-benchmark).

License: MIT.
