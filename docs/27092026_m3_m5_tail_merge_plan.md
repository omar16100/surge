# Merge the feat/m3-m5 tail, and make the README match the code (plan)

Category: Plan (dated). Date: 2026-09-27. Branch: `chore/m3-m5-tail`. Status: in review.

## Why

PR #1 merged `feat/m3-m5` at `f11637c` into main (merge `ab05345`, 2026-08-20). The branch
kept moving after that merge: tasks R2 and R3 (split `src/metal.m` into three translation
units), R4 (`sg_` prefix for the twelve promoted host globals), the P2.3 claim corrections,
and two unpushed commits hardening `tests/test_cli_bench.sh` (`c7d3726`, `1173c82`). None
of it reached main. Separately, the README described surge as "production", listed
features that are not in the source, stated the retracted 338 MHz limiter premise as fact,
and quoted bandwidth and TFLOPS figures from a private repo with no data file here.

## Scope

1. Merge the 13-commit tail (`origin/main..1173c82`) into main with a MERGE COMMIT, not a
   squash, so branches cut from the tail (the ANE backend) keep their ancestry.
2. Scrub personal absolute paths from `todo.md` and `docs/index.md` in a new commit on top.
   Published commits from `origin/feat/m3-m5` are not rewritten.
3. Rewrite `README.md` against `docs/c4model.md`:
   - "production" replaced with "experimental, work in progress".
   - Status updated from "M0-M2 complete" to M0-M3 + M5 + bench harness + decode work.
   - The 338 MHz limiter premise described as retracted, with the 2026-08-15 telemetry
     (`docs/15082026_prefill_duty_cycle_plan.md`).
   - The fanpro pre-spin hook and prompt-lookup speculation marked as not built (neither
     appears in `src/`, `surge.h`, `tools/` or the Makefile).
   - 630 GB/s and 21.9 TFLOPS removed (they come from the private llm-rnd repo, and no
     committed file here measures them).
   - Every remaining figure cites the committed doc it comes from.
4. `docs/c4model.md`: the M3/M5 status block said "In progress (branch `feat/m3-m5`)";
   it now says built and merged.
5. GitHub description: drop "limiter-aware pacing scheduler", which restates the premise.

## Verification

- `make check` on a clean worktree at `1173c82` (the code under merge; this PR adds only
  docs on top): exit 0, 19 test binaries, 87604 checks, 0 failures, matching the count
  `docs/c4model.md` records for R2 and R3. `tools/check_metal_globals.sh` passed,
  `tests/test_cli_prefill.sh` 11 cases, `tests/test_cli_bench.sh` 18 cases. Skipped: the
  env-gated real-model gates (`SURGE_GGUF`, `SURGE_GGUF_QWEN3`, `SURGE_ST`,
  `SURGE_GGUF_TWIN`, `SURGE_BENCH_TOK_MODEL`, `SURGE_PACE_MODEL`), which need model files.
- Personal-data scan over `git log -p origin/main..HEAD`, plus gitleaks over the range and
  the tree.
- After merge: `git fetch` and confirm origin/main contains `1173c82`.

## Decisions and deviations

- Merge commit rather than squash: an explicit exception for this repo, so the ANE branch
  (cut from the new main) does not carry the old tail as a duplicate history.
- Pre-existing personal paths elsewhere in the tree (Makefile gate defaults, test SKIP
  messages, older dated docs, `surge.h` comments) are out of scope for this PR and are
  listed in the sweep report.
