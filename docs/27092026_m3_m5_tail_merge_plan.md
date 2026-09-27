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
6. From review: a sounder p30 clamp-escalation discriminator in `tests/test_cli_bench.sh`,
   backed by a new additive `decode_step_ms` field in `surge-bench`'s JSON (see Review).
7. CI: the repo had none. `.github/workflows/ci.yml` runs `make debug` (pure-C tests under
   ASan and UBSan, `-DSURGE_NO_METAL`) on `macos-15`, which needs no GPU and no Metal
   toolchain. `make check` stays a local gate: it needs the Metal toolchain, its GPU and
   CLI cases skip without a device (so full coverage needs the real machine), and its
   timing-based cases are not meaningful on a shared runner.

## Verification

- `make check` on a clean worktree at `1173c82` (the tail as it arrived, before this PR's
  own commits): exit 0, 19 test binaries, 87604 checks, 0 failures, matching the count
  `docs/c4model.md` records for R2 and R3. `tools/check_metal_globals.sh` passed,
  `tests/test_cli_prefill.sh` 11 cases, `tests/test_cli_bench.sh` 18 cases. Skipped: the
  env-gated real-model gates (`SURGE_GGUF`, `SURGE_GGUF_QWEN3`, `SURGE_ST`,
  `SURGE_GGUF_TWIN`, `SURGE_BENCH_TOK_MODEL`, `SURGE_PACE_MODEL`), which need model files.
- After the review fixes below: `make check` exit 0, 19 test binaries, 87605 checks (one
  new round-trip check for `decode_step_ms`), 0 failures, `tests/test_cli_bench.sh` 19
  cases (one new). `make debug` exit 0, 83615 checks, 0 failures, 0 sanitizer
  diagnostics. Mutation checks on copies of the script: forcing the div-1 arm to rest by
  default (clamp div 4 in its slot) now FAILS with steps 30.6 ms under a 46 ms budget
  while its wall was 84.9 ms (the wall rule would have called it stale); a 0.05x budget
  multiplier gives three recalibrations then the loud SKIP; an unreadable first probe
  reaches its FAIL message. One earlier full run hit the known B6 `check2` timing flake
  (3 percent bar) while another job held the CPU at load average 39; it passed on every
  rerun once the load dropped. The flake is pre-existing and recorded in
  `docs/18082026_decode_optimization_summary.md`.
- Personal-data scan over `git log -p origin/main..HEAD`, plus gitleaks over the range and
  the tree.
- After merge: `git fetch` and confirm origin/main contains `1173c82`.

## Review (codex, 2026-09-27)

- MAJOR, fixed (round 1): the p30 clamp-escalation case compared the div-1 arm's decode
  WALL with the budget, and the wall includes the very rests under test. On a fast
  machine two erroneous 20 ms rests pushed a 35 ms phase past a 52 ms budget, so a
  detector bug was retried as a stale calibration and ended as a SKIP.
- MAJOR, fixed (round 2): the first fix subtracted `decode_rest_s`, but that is the
  CONFIGURED rest (rests x rest_ms) and a real sleep can overrun it, so the masking could
  come back on a loaded machine. `surge-bench` now reports `decode_step_ms`, the sum of
  the valid per-step times the pacer's budget accumulates (additive JSON field,
  `surge.h`, `src/bench.c`, `src/cli_bench.c`, round-trip checked in `tests/test_bench.c`).
  With clamp div 1, a rest while that sum is below the budget is proof of a fault; at or
  above it the verdict is inconclusive and the case recalibrates (`p30_esc_classify`).
  A new injected-value case (4a) pins it.
- Minor, fixed (round 2): an unreadable first probe expanded unset variables under
  `set -u` before its FAIL message; they are now initialized.
- Minors, fixed (round 3): the field was first written as seconds with `%.6g`, which
  could flip a verdict at the budget boundary; it is now milliseconds with `%.17g`, and
  (4a) pins three boundary rows. `cli_bench.c` now drops non-finite and non-positive steps
  exactly as `sg_decode_pace_decide` does. The wording no longer calls the verdict
  "exact" in both directions, and the `decode_rest_s` comment in `surge.h` now says it is
  the configured rest, not the measured sleep.
- Minor, fixed: README no longer states a date range for the blog's runs (no committed
  source), says the compositor mitigation reduces risk rather than prevents it, and
  notes the opt-in `--decode-clamp-div` escalation.

## Decisions and deviations

- Merge commit rather than squash: an explicit exception for this repo, so the ANE branch
  (cut from the new main) does not carry the old tail as a duplicate history.
- Pre-existing personal paths elsewhere in the tree (Makefile gate defaults, test SKIP
  messages, older dated docs, `surge.h` comments) are out of scope for this PR and are
  listed in the sweep report.
