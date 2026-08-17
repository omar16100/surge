# surge documentation index

Read this first. surge is a from-scratch C11 + Metal LLM inference engine for the Mac
Studio M3 Ultra (github.com/omar16100/surge, MIT). All doc paths are under
`/Users/macmini/projects/surge/`.

## Conventions

- No em dashes in prose. Factual claims only, each traceable to code, a test, or a log.
- Code identifiers snake_case. C11, `-Wall -Wextra -Werror`, no dependencies beyond macOS.
- Dated docs: `DDMMYYYY_topic.md`. Evergreen docs (architecture, this index): `topic.md`.
- SDD (subagent-driven-development) artifacts live under `.superpowers/sdd/` (gitignored);
  the plan and spec under `docs/superpowers/`.

## Index

| Doc | Category | Path | Hook |
|---|---|---|---|
| Architecture (C4 model) | Architecture (evergreen) | `docs/c4model.md` | Source of truth for containers, components (per-file responsibilities), decode/prefill data flows, and built-vs-planned status (M0-M3 + M5 shipped; bench harness B1-B6 + B8 shipped, B7 COMPLETE 2026-08-14; B8 rationale corrected 2026-08-15). |
| surge design spec | Design (evergreen) | `docs/superpowers/specs/2026-08-08-surge-design.md` | The approved design: goals, the M0-M9 milestones, correctness ladder, the beat-mlx-lm bar, and the hybrid-arch correction. |
| M3 + M5 + bench plan | Plan (dated) | `docs/superpowers/plans/2026-08-09-surge-m3-m5.md` | The ~18-task implementation plan to let surge run the 27B-Q8_0 GGUF at 262,144 context: M3 (Q8_0), M5 (fp16 KV + tiled prefill), bench harness (B1-B7). Execution order, gates, risks. |
| M3.4 Q8_0 forward gate | Gate (dated) | `docs/11082026_m34_q8_gate.md` | How to run and re-freeze the manual Q8_0 correctness gate on the 27B GGUF: (A) surge Metal Q8_0 vs surge CPU-ref (100% top-1, teacher-forced), (B) surge vs llama.cpp greedy (byte-identical). `make gate`; frozen fixtures in `tests/fixtures/m3q8/`. |
| B8 duty-cycle correction | Plan (dated) | `docs/15082026_prefill_duty_cycle_plan.md` | Why B8's firmware-clamp rationale is not supported by the 256K run's telemetry, the WindowServer watchdog kill that actually motivates yielding the GPU, the predictive budget test, and command-buffer segmentation (`--prefill-max-burst-ms`). |
| Split-K decode gate | Gate (dated) | `docs/16082026_splitk_decode_gate.md` | Task P2.3: how the decode path picks `n_splits = clamp(seq / SG_TG, 4, 1024)` and why it keeps `k_attn_decode_f16` below seq 1024 (both measured, with the crossover table), how the shared-scratch hazard is closed structurally, and the command + result for every gate (make check/debug, the mutation-proved wiring subtest, 100x determinism, kernel A/B, real-model byte-identical greedy). |
| Split-K GQA threadgroups | Gate (dated) | `docs/17082026_splitk_gqa_threadgroups.md` | Tasks P2.4 through P2.8 on `k_attn_decode_splitk_partial_gqa`: one threadgroup per GQA GROUP instead of per query head, its measured split policy (cap 256), the greedy-token gate where the two policies diverge, and P2.7's OCCUPANCY FLOOR of 128 threadgroups. The floor exists because collapsing `repeat` threadgroups into one divides the grid by `repeat`, so at short context the GQA kernel was measurably SLOWER than the per-head one it replaces (0.822x on the 27B at seq 2048). A threadgroup floor fits both real shapes where a seq threshold cannot, since the per-shape crossovers are 1.7x apart in seq and in the opposite order. `attn_splitk_gqa` is still OFF by default. P2.8 adds the ONLINE-SOFTMAX variant `k_attn_decode_splitk_partial_gqa_online` (`SURGE_ATTN_SPLITK_ONLINE`, also OFF by default): one streaming pass with a running `(m, s, acc)` instead of four passes over a device-memory score row, so it needs no split-K scratch at all. Where the accumulator lives is the whole design (one register per head per thread, which requires `head_dim <= SG_TG`), and the measurement is two-sided: 1.013x-1.114x on the 27B at every decode-policy point, but 0.690x-0.984x on the 4B at depth, with an 8 KB threadgroup allocation and the `head_dim < SG_TG` thread waste as the leading suspects. P2.9 fixes the second of those by splitting the tile's KEYS across `SG_TG / kw` key groups (`kw` = smallest power of two >= `head_dim`) so every thread owns an output dim in the V phase, merging the groups with the split-K combine's own log-sum-exp weights and no extra threadgroup memory. The 4B's deep regression is gone (0.690x to 1.015x at seq 262144, 0.763x to 0.954x at 131072, and 1.25x-1.57x faster than P2.8's kernel wherever it used to lose) while the 27B `n_kgroups == 1` path is unmoved. The 8 KB allocation was then MEASURED at only ~1%, so P2.8's leading suspect was the smaller one. Both switches still OFF by default. |
| M5.7 long-context gate | Gate (dated) | `docs/11082026_m57_longctx_gate.md` | How to run the M5.7 gate that closes M5 (`tools/prefill_longctx_gate.sh`, env-gated on `SURGE_GATE_MODEL`): (A) prefill+decode == serial+decode at 8192/16384/32768 (0 token-id mismatches), (B) 262144 ingest (262112 prefill + 32 decode) reaches used==262144 with non-degenerate output, (C) CLI `--max-ctx` cap rejection. |

## Related (not in docs/)

- `todo.md` (repo root): running status log per task.
- The 256K comparison surge is being built to join lives in the llm-rnd project:
  `/Users/macmini/projects/llm-rnd/docs/256k_comparison.md` and `leaderboard.md`.
- Benchmark logs: `~/bench_logs/`.
