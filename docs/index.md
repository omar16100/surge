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
| Architecture (C4 model) | Architecture (evergreen) | `docs/c4model.md` | Source of truth for containers, components (per-file responsibilities), decode/prefill data flows, and built-vs-planned status (M0-M3 + M5 shipped; bench harness B1-B5 shipped, B6/B7 pending). |
| surge design spec | Design (evergreen) | `docs/superpowers/specs/2026-08-08-surge-design.md` | The approved design: goals, the M0-M9 milestones, correctness ladder, the beat-mlx-lm bar, and the hybrid-arch correction. |
| M3 + M5 + bench plan | Plan (dated) | `docs/superpowers/plans/2026-08-09-surge-m3-m5.md` | The ~18-task implementation plan to let surge run the 27B-Q8_0 GGUF at 262,144 context: M3 (Q8_0), M5 (fp16 KV + tiled prefill), bench harness (B1-B7). Execution order, gates, risks. |
| M3.4 Q8_0 forward gate | Gate (dated) | `docs/11082026_m34_q8_gate.md` | How to run and re-freeze the manual Q8_0 correctness gate on the 27B GGUF: (A) surge Metal Q8_0 vs surge CPU-ref (100% top-1, teacher-forced), (B) surge vs llama.cpp greedy (byte-identical). `make gate`; frozen fixtures in `tests/fixtures/m3q8/`. |
| M5.7 long-context gate | Gate (dated) | `docs/11082026_m57_longctx_gate.md` | How to run the M5.7 gate that closes M5 (`tools/prefill_longctx_gate.sh`, env-gated on `SURGE_GATE_MODEL`): (A) prefill+decode == serial+decode at 8192/16384/32768 (0 token-id mismatches), (B) 262144 ingest (262112 prefill + 32 decode) reaches used==262144 with non-degenerate output, (C) CLI `--max-ctx` cap rejection. |

## Related (not in docs/)

- `todo.md` (repo root): running status log per task.
- The 256K comparison surge is being built to join lives in the llm-rnd project:
  `/Users/macmini/projects/llm-rnd/docs/256k_comparison.md` and `leaderboard.md`.
- Benchmark logs: `~/bench_logs/`.
