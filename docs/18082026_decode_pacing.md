# Decode pacing and clamp detection (Task P3.0)

`src/sched.c`, declared in `surge.h`, wired into `surge-bench` only. Off by default.

## Why

Decode throughput on this Mac Studio M3 Ultra is partly a function of how recently the GPU
was busy, and that is measured rather than inferred. P2.9 (2026-08-18) ran three IDENTICAL
decode arms on the same binary and prompt:

| arm | first pass (tok/s) | after 150 s idle (tok/s) |
|---|---|---|
| 1 | 38.47 | 39.72 |
| 2 | 25.40 | 41.16 |
| 3 | 16.71 | 34.66 |

Nothing about the kernels changed between the two passes. That is a 2.4x spread on identical
work, and it is why P2.9 declared surge-bench decode tok/s unusable for A/B ranking. Prefill
already had a mitigation (B8's duty cycle, `sg_gpu_set_prefill_rest`); decode had none.

## What was built

Two separable things, both in one pure-C file with no Metal, no Foundation and no GPU:

1. **A decode duty cycle**, the direct analogue of B8's. After `work_budget_ms` of accumulated
   decode step time, rest `rest_ms` between tokens with nothing in flight. Armed by
   `--decode-work-ms W --decode-rest-ms R` on `surge-bench`; either 0 (the default) leaves it
   off, exactly as B8's rule.
2. **A clamp detector**, which needs no configuration, never sleeps on its own, and reports on
   every run whether that run's decode number is trustworthy.

`src/metal.m` is not touched. B8's rest had to live there because the loop it paces is inside
`sg_gpu_prefill`; there is no decode loop inside `metal.m` at all (`sg_gpu_forward` is one
step), so the pacer is a caller-owned value fed at the per-token boundary in `src/cli_bench.c`.

Deliberately NOT built, per the task brief: no fan-daemon hook of any kind. surge is a public
engine and driving a machine's cooling is not its business.

## The clamp signal, and what it cannot see

surge cannot read GPU frequency. The signal is the only thing it can observe from inside a
decode loop: **per-step wall time rising persistently against a baseline**.

- The step time is a `clock_gettime` pair around `sg_gpu_forward` ALONE, not the whole loop
  iteration. Including the argmax, the periodic memory sample and the progress print would feed
  the detector work whose cost varies for unrelated reasons.
- The baseline is the **median** of the first `baseline_n = 8` steps, then fixed. Median, not
  mean, because the first decode steps after a prefill carry first-touch and warm-up costs; and
  because a baseline biased HIGH only desensitises the detector, while one biased LOW would
  insert rests that buy nothing.
- A step is "over" if `step_ms > baseline_ms * 1.5`.
- The detector **latches** only after `confirm_n = 3` CONSECUTIVE over steps and **clears**
  after 3 consecutive not-over steps. One slow step cannot latch it; one fast step cannot clear
  it.

### False positives

Any SUSTAINED slowdown fires it, and surge cannot tell the causes apart: another process taking
a share of the GPU, thermal throttling, or legitimate drift as decode extends the KV cache the
attention kernels read. The detector's claim is only "step time has persistently risen against
this run's own baseline". It is not a claim about a cause.

Measured on real decode series from this machine, replayed offline through the exact policy
(`surge-bench --emit-timeseries`, Qwen3-4B-Instruct-2507-Q8_0):

| series | steps | baseline | max step | max/baseline | steps over 1.5x | clamp events |
|---|---|---|---|---|---|---|
| 300-token decode, 9-token prompt | 299 | 19.75 ms | 24.39 ms | 1.235 | 0 | 0 |
| 1024-token decode, same prompt | 1023 | 19.21 ms | 52.10 ms | 2.712 | 375 | 1 |
| 512-token decode at 16k context (machine contended) | 511 | 308.99 ms | 355.59 ms | 1.151 | 0 | 0 |

The 300-token run is the clean case: not one step out of 299 reaches even 1.5x the baseline, so
the ratio has roughly 20 percentage points of headroom over the worst ordinary variance
observed. The 1024-token run is a genuine positive on unpaced, uncontended work: step time
climbed from a 19.2 ms baseline to 52.1 ms and stayed there for 375 of 1023 steps, and the
detector recorded exactly one transition. Reported decode throughput fell from 45.86 tok/s
(300 tokens) to 36.65 tok/s (1024 tokens) on the same prompt and binary. The third row was
collected while other work held the GPU, so it is not a clean sample; it is listed because it is
the blind spot in miniature (uniformly 16x slower than row 1, perfectly flat, zero events).

### The blind spot, which matters more than the false positives

The detector sees a RISE. It cannot see a run that was **already slow at its first token**,
because a self-measured baseline defines that run's opening steps as normal by construction.
That is precisely the P2.9 case: arms 2 and 3 were slow from the start, and a self-measured
baseline would have reported zero clamp events for both. `tests/test_sched.c` pins this
limitation with an assertion rather than only describing it.

Two things cover it, and both are needed:

- `--decode-baseline-ms B` (`sg_decode_pace_set_baseline`) seeds the baseline from outside the
  run, e.g. from a known-idle measurement, so a uniformly slow run latches immediately.
- `decode_baseline_ms` is reported in every JSON row. It is comparable ACROSS runs, so a
  clamped-from-start run is visible in the log even when nothing was seeded.

## What a confirmed clamp does

By default, **nothing**: `clamp_div` is 1, so the detector counts and reports and changes no
rest schedule. `--decode-clamp-div D` (D > 1) makes a confirmed clamp drop the rest threshold to
`work_budget_ms / D` while the signal persists.

That default is deliberate. The obvious rationale for escalating (rest harder and the clock
comes back) is the one this project's own telemetry does not support: over 376 burst pairs in
the 2026-08-14 256K run, rest length did not predict the next burst's clock (r = +0.017), which
is why B8's rationale in `surge.h` was corrected away from clock recovery toward compositor
protection. The rationale that survives is contention: if the slowdown is another process
holding the GPU, yielding more is the right response. An operator who knows that is the case
can set `D`; surge will not guess it. Nothing here is claimed to un-clamp anything.

The divisor form bounds the cost: the densest possible schedule is one `rest_ms` per
`work_budget_ms / D` of work, computable before the run.

## Reporting

`sg_bench_row` and the JSON gain five fields, mirroring B8's prefill pair:

| field | meaning |
|---|---|
| `decode_rest_s` | total time the duty cycle slept, a SUBSET of `decode_wall_s`. 0 when pacing is off. |
| `decode_compute_tps` | `n_gen / (decode_wall_s - decode_rest_s)`, the fair full-clock decode rate. `decode_tps_slope` / `decode_tps_avg` stay wall-clock and so fall when pacing is active. |
| `decode_rests` | rests emitted. Next to `decode_rest_s` so a test can check the accounting identity without depending on how long a decode step takes on the machine running it. |
| `decode_clamp_events` | detector latches. > 0 means that row's decode tok/s mixes two GPU clock states. |
| `decode_baseline_ms` | the baseline those events were measured against, comparable across runs. |

The 8-column markdown row is unchanged.

## Gates

1. `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror` clean on `src/sched.c`,
   `src/cli_bench.c` and `src/bench.c`, with and without `-DSURGE_NO_METAL`.
2. `make debug` (SURGE_NO_METAL + ASan/UBSan): `tests/test_sched.c`, 86 checks. This is the main
   correctness gate, since the policy is pure. Cases: disabled-is-inert (asserted as `== 0`, for
   both knobs and each alone), an EXACT rest count for a known budget, a single 10x spike
   repeated 50 times does NOT latch, pairs of spikes do not latch at `confirm_n = 3`, a
   sustained 1.4x rise under a 1.5x threshold does not latch, 3 consecutive over steps latch
   ONCE, hysteresis, the escalation off by default and measurably 4x denser when set, the median
   baseline surviving an outlier a mean would not, the clamped-from-start blind spot and its
   seeded-baseline fix, NaN/inf/non-positive steps dropped, NULL-safety and argument clamping,
   and `sg_decode_pace_step` proven against a real clock to actually sleep.
3. `make check` green, with every P2.3-P2.9 gate unmoved (pacing is off by default, so none of
   them should see anything).
4. Determinism: `tests/test_cli_bench.sh` asserts paced gen_ids byte-equal unpaced gen_ids, on
   the mini fixture in every `make check` and on a real GGUF when `SURGE_BENCH_TOK_MODEL` is
   set.
5. Not silently inert, five ways in `make check`: the accounting IDENTITY
   `decode_rest_s == decode_rests * rest_ms` with the unpaced arm exactly 0 and `decode_rests`
   bounded by the run's 15 per-token pacing points; the paced run's `decode_wall_s` at least its
   own `decode_rest_s`, so the rest was spent and not merely counted (mutation-verified);
   a two-sided detector-wired check (a 0.001 ms seeded baseline must produce clamp events, a
   1000000 ms one must not); a calibrated escalation check (`--decode-clamp-div 1` rests zero
   times against a budget larger than the whole decode phase, `--decode-clamp-div 4` rests); and
   on a real model, the EXACT count, one rest per pacing point.

## Throughput effect: attempted twice, not established

Both experiments and their raw rows are in
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P3.0-throughput.txt` and
`task-P3.0-throughput-long.txt`.

**Experiment 1**, 8 runs, counterbalanced A B B A B A A B, 512 decode tokens on the 4B, A =
unpaced, B = `--decode-work-ms 2000 --decode-rest-ms 200` (a 9.1% duty cost by construction):

| arm | mean wall tok/s | mean compute tok/s |
|---|---|---|
| A (unpaced) | 40.27 | 40.48 |
| B (paced) | 36.20 | 39.65 |

Pacing cost 10.1% of wall-clock throughput, which is the configured duty overhead, and did NOT
raise compute throughput: B is 2.0% lower on the mean and lower in all four adjacent A/B pairs.
So no benefit was visible. That is not the same as "there is no benefit": the machine drifted
inside the experiment (the baseline fell from 22.5 ms to 19.1 ms and wall throughput rose from
37.7 to 43.2 tok/s between the first and last unpaced runs, a step change between runs 5 and 6),
which is the same order as any effect being looked for. 512 tokens may also be too short to reach
the regime pacing would help: in the clean series above, the slowdown only appeared past step 600.

**Experiment 2**, 4 runs, A B B A, 1536 decode tokens, same arms, run in exactly that regime:

| order | arm | wall tok/s | compute tok/s | rest s | clamp events | baseline ms |
|---|---|---|---|---|---|---|
| 1 | A | 8.19 | 8.33 | 0 | 1 | 21.4 |
| 2 | B | 10.15 | 10.96 | 13.4 | 0 | 164.5 |
| 3 | B | 19.10 | 20.80 | 7.2 | 2 | 41.2 |
| 4 | A | 22.07 | 22.24 | 0 | 8 | 24.0 |

This one is uninterpretable as an A/B, and instructively so. Throughput rose MONOTONICALLY across
the sequence, 8.19 to 22.07 tok/s, a 2.7x drift that no 4-run counterbalancing can remove; the arm
means (A 15.13, B 14.63) are noise on top of it. The machine had been under heavy load for the
preceding hour and recovered across the experiment.

**No throughput claim is made.** The mechanism is correct, gated, and off by default.

The same table is worth reading for what it says about the DETECTOR, because both of its modes
appear live in four runs. Run 2 started at the deeply slow state, self-measured a 164.5 ms
baseline, and reported ZERO clamp events while running about 4x slower than run 4: the
clamped-from-start blind spot, exactly as described above, and the reason `decode_baseline_ms` is
reported per row. Run 4 was the FASTEST run and reported EIGHT clamp events, because it started
fast and had transient slowdowns to rise against. Clamp events count rises, not slowness, and the
two are not the same thing.

## Where the gates could still be wrong

Two weaknesses were found by review and closed before commit, both of the "passes while inert"
shape this project has been bitten by:

1. The accounting lives in `sg_decode_pace_decide`, so a decode loop that took the accounting and
   skipped the sleep (calling `_decide` rather than `_step`) would have satisfied the accounting
   identity AND kept `decode_compute_tps > decode_tps_avg`, since `decode_rest_s` is subtracted
   from the denominator either way. Two wall-clock assertions now exist: `tests/test_sched.c`
   case (k) holds a real clock against `_step`, and the shell gate requires the paced run's
   `decode_wall_s` to be at least its own `decode_rest_s`. Mutation-verified: swapping `_step` for
   `_decide` in `src/cli_bench.c` fails the shell gate with
   `decode_wall_s=0.010504` against `decode_rest_s=0.4`.
2. The seeded-baseline test asserted the seed survives the baseline window but fed only 3 steps
   against a window of 8, so it would have passed against the overwrite bug it names. It now feeds
   more than the window.
