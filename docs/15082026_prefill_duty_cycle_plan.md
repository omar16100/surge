# B8 prefill duty-cycle: correct the rationale, fix the overshoot

Status: IMPLEMENTED on `fix/prefill-duty-cycle-overshoot`. Codex-reviewed before coding;
review changed the design substantially and the record of that is below.
Date: 2026-08-15.

## What review changed

Worth recording, because the first draft of Part B would have shipped a silent
correctness bug:

1. **Adaptive chunk sizing was the wrong mechanism.** Mutating `chunk` inside
   `for (...; base += chunk)` desynchronises `base` from the tokens actually processed.
   Replaced with COMMAND-BUFFER SEGMENTATION: split the layer sweep across more command
   buffers, leaving `chunk` alone. Strictly safer, because command buffer boundaries carry
   no state at all, whereas chunk boundaries carry KV positions.
2. **I then reintroduced the same bug one level down**, as `l0 += seg` while `seg` shrinks
   mid-loop, which silently reprocessed layers. The segmentation gate caught it: gen_ids
   diverged immediately. The cursor now advances by layers encoded (`l0 = l1`). Two
   independent chances to make the identical mistake is the argument for the gate existing.
3. **"Unboundedly" was overstated.** Overshoot is bounded by one chunk's wall time. The
   accurate claim is that the bound is one chunk, which is too large at long context.
4. **Part A's `!last` guard was wrong for the new rationale.** Skipping the budget test
   before the final chunk meant the longest chunk of the run was the one most likely to be
   submitted onto an exhausted budget. The predictive test now covers it.
5. **Claim 1 is weaker than I first wrote it.** See the revised wording below: the run's
   data does not SUPPORT the clamp premise, which is not the same as refuting a clamp. A
   proper discriminating experiment is specified as follow-up rather than claimed as done.
6. **The parity gate had to prove segmentation engaged**, or it would pass vacuously if
   segmentation silently never triggered. Hence `prefill_segments` on the JSON row.

## What triggered this

WindowServer was killed twice by the userspace watchdog on 2026-08-14 (16:43:01 and
16:43:41), logging out the GUI session. The machine did not panic or reboot; uptime is
continuous since 2026-08-04.

Symbolicated main-thread stack from
`/Library/Logs/DiagnosticReports/WindowServer_2026-08-14-164345_*.spin`:

    ws_main_thread  DispatchQueue "com.apple.SkyLight.mtl_submit"  [TH_WAIT | TH_UNINT]
      WindowServer -> SkyLight -> Metal -> IOGPU -> IOKit
       *-> kernel -> IOGPUFamily -> AGXG15C (345.20.4) -> kernel wait

WindowServer blocked in a Metal submit inside the Apple GPU driver, missed its 80 s
watchdog check-in, and was killed. It was `surge-bench` holding the GPU.

## Finding 1: B8's stated rationale is not supported by this run's data

`docs/c4model.md:163-168` records the premise for B8:

> the Mac Studio M3's firmware GPU limiter (clamps to 338 MHz after ~3-4 min of
> sustained load, recovers only after 60-120s idle)

Tested against 9,199 telemetry samples from the 30-hour
`ctx256k_qwen27b_surge_20260813_151809` run (378 detected GPU bursts). Three independent
checks, all negative.

Stated precisely: this shows the observed run is not dominated by a repeatable 338 MHz
clamp that recovers after 60-120 s idle. It does not prove no limiter exists. A limiter
could be masked if the memory-stalled workload already requests a low core clock, or if it
acts on memory/fabric/power rather than the reported core clock. The discriminating
experiment is specified under "Follow-up" below and has NOT been run.

**1. No within-burst decay.** Mean GPU clock binned by seconds-into-burst, split by run
half (a context-length proxy). Sample counts per bin shown in parentheses:

| | t+0s | t+30s | t+60s | t+90s | t+120s |
|---|---|---|---|---|---|
| early bursts (low ctx) | 712 (567) | 723 (567) | 716 (567) | 710 (567) | 701 (563) |
| late bursts (high ctx) | 573 (568) | 591 (566) | 590 (567) | 591 (564) | 588 (560) |

Flat. A load-accumulating clamp would decay monotonically within each burst. The apparent
falloff past t+150s is selection bias: only the slowest (highest-context) bursts last
that long, and the bin counts collapse (331, 138, 24).

**2. Rest does not restore clock.** `corr(preceding_idle_seconds,
next_burst_mean_MHz) = +0.017` over 376 burst pairs. The 90 s rest buys nothing
measurable.

**3. The 338 MHz floor is essentially never reached.** 27 of 7,724 loaded samples
(0.3%); at or below 400 MHz is 1.7%.

What actually sets the clock is context length, and it is set at burst start, before any
load could accumulate: `corr(ctx_len, gpu_freq) = -0.575`, while
`corr(gpu_temp, gpu_freq) = +0.369` (positive, so not thermal throttling either; GPU temp
averaged 58.0 C, max 73.4 C, fans peaked at 2521 of 3625 RPM). Long-context prefill
becomes memory-bandwidth bound as the KV cache grows (16.0 GiB K+V at full context, per
the run's own `summary.txt`), the GPU stalls on memory, and DVFS drops the clock because
there is no compute pressure holding it up. Prefill throughput tracks this exactly:
12 tok/s at 256 tokens, 2 tok/s at 221k.

**Cost of the misdiagnosis:** 367 rests x 90 s = 33,030 s = **9.2 hours of the 30-hour
run spent sleeping for no measured benefit.**

**Corollary:** B7's original "HUNG", which is why B8 was inserted, was almost certainly
this same WindowServer watchdog kill rather than a GPU clamp. The session logging out
would present as a hang.

The duty cycle should therefore be *repurposed*, not deleted. Yielding the GPU
periodically is genuinely necessary, just for a different reason than documented: so the
compositor can render and the watchdog does not fire.

## Finding 2: the duty-cycle gate overshoots by up to one chunk

`src/metal.m:3477` checks the work budget only *after* a chunk's command buffer
completes. So a burst runs to `budget + one_chunk_duration`. That is bounded, not
unbounded, but the bound is one chunk and one chunk's duration grows with context until it
alone exceeds the watchdog window.

Measured across 367 bursts in that run (`--prefill-work-ms 150000`):

- min 150.0 s, median 199.5 s, **max 332.9 s**
- **367 of 367 (100%)** exceeded the 80 s watchdog window
- 183 exceeded 200 s
- overshoot grows with context: first bursts 163-174 s, final bursts 272-281 s

At ~220k context a single 256-token chunk takes roughly 130 s. That matters
independently of the duty cycle: **one chunk alone exceeds the 80 s watchdog**, so no
amount of resting *between* chunks can protect the compositor at high context.

## Fix

Two parts, because the two failure modes are independent.

### Part A: predictive gate (bounds accumulated burst)

Rest when the *predicted* next chunk would exceed the budget, rather than after it
already has:

    if (!last && pf_work_acc_ms + pf_est_next_ms >= budget) -> rest

`pf_est_next_ms` from the previous chunk's measured duration. Chunk durations grow
monotonically with context, so the previous chunk is a slight underestimate; apply a
small margin (proposal: 1.25x, to be justified or replaced during review).

Small, contained, no numerical impact. Keeps the existing byte-identical-when-disabled
guarantee: the estimate is a loop-local double feeding only the sleep decision.

### Part B: bound single-command-buffer GPU time (new, opt-in)

Part A alone cannot help once one chunk exceeds the watchdog. `--prefill-max-burst-ms M`
(default 0 = off): when a single command buffer overruns M, split the layer sweep across
more command buffers, halving down to a floor of one layer.

The ceiling is a target, not a hard cap, and not a promise of a single overrun. The check
is reactive, so each overrunning submission runs in full and only then halves: a 64-layer
sweep can overrun at 64, then 32, then 16, converging over up to log2(layers) overruns. A
segment already at the 1-layer floor cannot shrink further and will keep overrunning. Set
the ceiling well under the watchdog window so the overruns along the way still land
inside it.

Segmentation rather than chunk resizing, per review. This is the safer mechanism by a
wide margin: command buffer boundaries carry no state, so the same kernels run with the
same arguments in the same order and buffers committed in sequence on one queue execute
in order. Chunk boundaries, by contrast, carry KV positions and the logits special case.

Gated, not assumed. `tests/test_cli_bench.sh` checks segmented gen_ids equal unsegmented
over an uneven chunk schedule (12 tokens at `--chunk 5`, so the final short chunk and the
logits-carrying final segment are both exercised), AND that `prefill_segments` rose, so
the parity check cannot pass vacuously. This gate immediately caught the `l0 += seg`
cursor bug described above.

### Part C: correct the documentation

`docs/c4model.md` is the stated source of truth and currently records a hypothesis the
data refutes. Rewrite the B8 rationale block with the measurements above, keeping the
implementation description accurate, and note the real purpose (compositor protection).
This is not optional cleanup; leaving it would mean the next reader re-derives a wrong
model of the hardware.

## Alternatives considered

- **Delete the duty cycle.** Tempting given 9.2 hours of waste, but wrong: the GPU
  yielding is what keeps the session alive. Repurpose instead.
- **Adaptive chunk sizing always-on.** Rejected as a default: changes the shape of every
  existing benchmark run and would invalidate comparability with the frozen gates.
- **Raise the watchdog threshold.** Not user-configurable, and would be the wrong fix
  even if it were.
- **Run headless.** Genuinely the most robust mitigation and worth documenting
  regardless, but it is an operational workaround, not a fix to the overshoot bug.

## Tests

Implemented:

- `tests/test_cli_bench.sh`: the segmentation parity gate (segmented gen_ids == unsegmented
  over an uneven chunk schedule, AND `prefill_segments` rose so it cannot pass vacuously).
- `tests/test_gpu_prefill.c`: `prefill_segments` obeys the same reset-as-first-mutation
  contract as `prefill_rest_total_ms`, so an early-failing call cannot report a prior
  call's count.
- The existing B8 gates (byte-identical disabled path, exact rest accounting) stay green
  unchanged.

PLANNED, NOT IMPLEMENTED. Recorded as a gap rather than quietly dropped:

- A no-GPU unit test of the predictive gate arithmetic: given a synthetic sequence of chunk
  durations, assert the rest points land before the budget is crossed rather than after.
  The predictive test is currently three inline lines in `sg_gpu_prefill`, so this needs a
  small extraction to be testable.
- A `tests/test_cli_bench.sh` assertion that per-burst worked time never exceeds
  `budget * margin`. This is the property that was silently violated for 367 consecutive
  bursts, so it is the single most valuable missing test.

Consequence: the predictive gate is covered only by the existing rest-accounting gate,
which pins the NUMBER of rests, not the worked time between them.

## Docs to update in the same commit

- `docs/c4model.md`: Part C rewrite, plus the new flag if Part B ships.
- `docs/index.md`: register this doc.
- `todo.md`: task entry.

## Result

- `make check`: 14 cases pass, including the pre-existing B8 gates unchanged and the new
  segmentation gate (`command buffers 3 -> 8`, gen_ids identical).
- `tools/prefill_longctx_gate.sh` on the real 2B model: **PASS**, 295 checks / 0 failures,
  exit 0, wall time 28,741 s. Phase A depth-equivalence 0 mismatches; phase B 262,112-token
  prefill reached `used==262144` with non-degenerate decode. Worst prefill-vs-serial
  last-logit relative gap **1.222e-06, identical to the hermetic mini run**, so the
  segmentation edit did not perturb numerics at real scale.
- 8 hours is normal for this gate, not a regression: the 262k prefill alone accounts for
  21,971 s of the 28,741 s.

### Unplanned corroboration of Finding 1

The gate arms neither `--prefill-work-ms` nor `--prefill-max-burst-ms`, so it held the GPU
saturated for ~8 hours with NO rests at all, on a 2B (a different model from the 27B run
the analysis came from). Prefill throughput decayed smoothly and monotonically with
context:

    ctx 173056  17 tok/s      ctx 222208  14 tok/s
    ctx 189440  16 tok/s      ctx 238592  13 tok/s
    ctx 205824  15 tok/s      ctx 262112  12 tok/s

This is WEAK corroboration and should not be read as more. There is no clock or power
telemetry for this run: tok/s is all I have. Token throughput can decay with context even
under a fixed clamp, because the work per token grows, so a flat clamp is not excluded by
this trace. What the trace does rule out is a dramatic clamp-like throughput cliff followed
by a plateau, which is what a hard 338 MHz pin arriving after 3-4 minutes would most
plausibly look like.

So: consistent with the memory-bound explanation, on a second workload and a second model,
and not consistent with the most obvious shape of the documented premise. Not proof. The
discriminating experiment below, which holds work fixed and varies idle, is still the thing
that would settle it, and it has not been run.

### And a limit worth stating

WindowServer was NOT killed during those 8 saturated hours, and no new watchdog reports
were written. That is not evidence the risk is gone: the 2026-08-14 kill was one event
across 367 bursts, so a single clean run proves little either way. It does mean the
failure is probabilistic rather than deterministic, and I cannot currently say what made
that particular burst fatal. Worth keeping in mind before treating the mitigation as
sufficient.
- New surface: `sg_gpu_set_prefill_max_burst`, `sg_gpu_prefill_segments`, CLI
  `--prefill-max-burst-ms`, JSON `prefill_segments`.

## Follow-up (specified, NOT done)

**The discriminating experiment for Claim 1.** Prefill to a fixed high context, then run
the same fixed-shape microbench repeatedly after 0/30/90/180 s of idle. If rest restores
clock or throughput at identical context and identical work, a clamp exists and this
analysis is wrong. If not, the memory-bound explanation stands. Separately, run a pure
compute-bound kernel for over 5 minutes: if a general sustained-load clamp exists, it
should appear there, where memory-boundness cannot explain it away.

Until that runs, the honest position is that the rests are not measurably buying clock
recovery on this workload, so their justification is compositor protection alone.

## Explicitly unverified

`PF_EST_MARGIN` (1.25) and the halving policy are heuristics, not measured optima. The
predictive gate has not been exercised against a real long-context run: doing so means
another multi-hour 256K prefill. The segmentation ceiling has only been exercised at the
mini fixture's 4 layers, not at the 27B's 64.
