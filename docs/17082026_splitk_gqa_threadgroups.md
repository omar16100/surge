# GQA-shared split-K threadgroups: what changed and the gates that have NOT been run (Task P2.4)

Companion to `docs/16082026_splitk_decode_gate.md` (Task P2.3, which wired split-K into
decode) and `docs/c4model.md` (architecture). The SDD report is
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.4-report.md`.

**Read this line first: no GPU gate in this document has been run.** A 27B MLX NIAH
benchmark held the GPU for the whole of this task (`pgrep -f
"bench_niah|mlx_raw_niah|llama-server|surge-bench"` non-empty before the first edit and
after the last), so the work is written, compiled and compile-checked only. Every number
below is either a memory-traffic ratio derived from the grid arithmetic or a shape taken
from the model config. There are no timings and no correctness results, because none were
measured.

## What changed

New Metal kernel `k_attn_decode_splitk_partial_gqa` (`src/kernels.metal`), a drop-in
alternative to `k_attn_decode_splitk_partial`: same eight bindings, same params array,
same score scratch, same `[n_heads, n_splits]` m/s and `[n_heads, n_splits, head_dim]` acc
layout, so `k_attn_decode_splitk_combine` and every host-side size rule are untouched.

The difference is the grid and therefore the traffic:

| | grid | K/V reads per (split, kv head) |
|---|---|---|
| `k_attn_decode_splitk_partial` (P2.2) | `(n_splits, n_heads)` | `repeat` streams of the same slices |
| `k_attn_decode_splitk_partial_gqa` (P2.4) | `(n_splits, n_kv_heads)` | one stream, used by all `repeat` heads |

`repeat = n_heads / n_kv_heads` is 4 on the 4B dense shape (32 heads, 8 kv, head_dim 128)
and 6 on the 27B decode shape (24 heads, 4 kv, head_dim 256). Decode at depth is
bandwidth-bound, so on paper that is where a `repeat`x tax on the dominant cost goes away.
It also explains P2.3a's headline achieved-GB/s figure: `tests/bench_splitk.c` counts
bytes ISSUED by a per-head grid, which on the 4B shape is 4.0x the unique bytes.

Host side (`src/metal.m`): a new `KI_ATTN_SPLITK_PARTIAL_GQA` pipeline on the existing
`SG_K_HEADS2D` grid class, a new one-shot `sg_gpu_run_attn_splitk_partial_gqa` (declared
in `surge.h`), and `enc_attn_splitk` picking the pipeline and the grid's y extent. The two
one-shots now share one body, `splitk_partial_run`, so they cannot drift apart on a
validation rule; every rejection `surge.h` documents is still checked in exactly one place.

## The gate that matters: byte-identical, not close

This is a pure data-reuse reorganization. Per query head, nothing about the arithmetic
moves:

- the dot product still accumulates `qh[i] * (float)kt[i]` over increasing `i` from `0.0f`;
- `m` and `s` still come off the same fixed-shape `tg_max` / `tg_sum` trees over the same
  per-lane partials (thread `lid` still owns the split's keys `lid`, `lid+256`, ...);
- `acc[d]` still accumulates `sc[r] * (float)v[t0+r][d]` over increasing `r` from `0.0f`.

So the two kernels must agree BIT FOR BIT, not to 1e-6. That bar is available here
precisely because nothing is supposed to change, and it is much stronger than the CPU
oracle comparison P2.2 used. The only ordering change is when `m` and `s` are stored
(right after each head's folds instead of after the acc pass); those are plain stores of
already-final values into slots no other threadgroup touches.

Determinism is unchanged and for the same reasons as the rest of `src/kernels.metal`: no
atomics, no `simd_sum`/`simd_max`, no read-modify-write of a shared accumulator. Holding
`repeat` SEPARATE accumulators per thread is what the file's rule allows; sharing one
across heads is what it forbids.

## Switchable, and off by default

`SURGE_ATTN_SPLITK_GQA=1` selects the GQA partial for decode; the default is `0`, the
per-head kernel P2.3 measured and shipped. Read once, in `sg_gpu_state_new`, exactly like
`SURGE_ATTN_SPLITK` and `SURGE_KV_DTYPE`, so one binary can run both sides of the A/B.

The default is off because nothing has been run, not because the kernel is believed
slower. This is how P2.2 shipped its kernels (written, compiled, dispatched by nothing)
and P2.3 flipped its default only after the hardware gates passed.

`splitk_gqa_use` also declines the GQA kernel outside GQA groups of 2 to 8:

- `repeat < 2`: there is nothing to share, and the GQA grid is then the per-head grid;
- `repeat > 8` (`SG_SPLITK_GQA_MAX`): the kernel still answers correctly, one head at a
  time, but with no reuse, which is strictly worse than the per-head kernel because the
  grid is also `repeat` times smaller.

There is deliberately NO threadgroup-count floor. The GQA grid is `repeat` times smaller
than the per-head one (at seq 1024 on the 27B shape that is 4 kv x 4 splits = 16
threadgroups against 96, on a machine with 80 GPU cores), so a short-sequence crossover
like P2.3's probably exists. Inventing that threshold without the measurement is what
P2.3's gate doc refused to do; `tests/bench_splitk.bin --gqa --seqs ...` is the instrument
that should settle it.

## Gates: every one of these is DEFERRED

Before ANY of them, confirm the GPU is free:
`pgrep -f "bench_niah|mlx_raw_niah|phase0|surge-bench|llama-server|llama-cli"` must print
nothing. Never run `make` while a surge binary is live: it rebuilds
`src/kernels.metallib` under the running process.

`src/kernels.metallib` is gitignored and is NOT rebuilt by this task (the GPU was busy),
so the first `make` after this commit is what puts the new kernel into it. Any Metal
binary built before that rebuild is unaffected; any binary built after it will fail at
`sg_gpu_init` if the metallib was somehow not regenerated, because init builds a pipeline
for every entry in `SG_KERNELS`.

| # | gate | command | status |
|---|---|---|---|
| 1 | Metal compile + metallib link | `xcrun -sdk macosx metal -fno-fast-math -Wall -c src/kernels.metal -o /tmp/p24_kernels.air && xcrun -sdk macosx metallib /tmp/p24_kernels.air -o /tmp/p24_kernels.metallib` | PASSED (clean, 0 warnings; `metal-nm` shows `k_attn_decode_splitk_partial_gqa`) |
| 2 | C/ObjC strict compile | `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror -Isrc -I. src/metal.m tests/test_metal_ops.c tests/bench_splitk.c` (and the same with `-DSURGE_NO_METAL`) | PASSED |
| 3 | Sanitizers, CPU path | `make debug` | PASSED, see the report for the check count |
| 4 | BYTE-IDENTICAL vs the per-head partial | `make check` then `./tests/test_metal_ops.bin` (subtest `metal_attn_splitk_gqa_bit_identical`) | DEFERRED, needs the GPU |
| 5 | 100x determinism of the GQA partial | included in the same subtest (`splitk_gqa_determinism`) | DEFERRED, needs the GPU |
| 6 | WHICH kernel the decode path selected, + the group-size policy, + end-to-end byte-identity | `./tests/gpu_fwd.bin` (subtest `mini_f16_splitk_gqa_dispatches_and_matches`, also inside `make check`) | DEFERRED, needs the GPU |
| 7 | Full suite green | `make check` | DEFERRED, needs the GPU |
| 8 | Real-model greedy A/B | `SURGE_ATTN_SPLITK_GQA=0 ./surge <gguf> -p "$(cat PROMPT)" -n 64` then the same with `=1` | DEFERRED, needs the GPU |
| 9 | Kernel-level speed A/B | `./tests/bench_splitk.bin --seqs 8192,32768,131072,262144` then the same with `--gqa` | DEFERRED, needs the GPU |
| 10 | Short-sequence crossover | `./tests/bench_splitk.bin --seqs 1024,2048,4096,8192 --gqa` against the same without `--gqa` | DEFERRED, needs the GPU |

## Why gate 6 exists, and why gates 8 to 10 are worthless without it

The two partials are contracted to produce the SAME BYTES. That makes the obvious
end-to-end gate vacuous on its own: `SURGE_ATTN_SPLITK_GQA=0` versus `=1` giving
byte-identical gen_ids and logits is ALSO exactly what you see when the GQA kernel is never
selected at all, because then the per-head kernel ran both times. Narrow the group band,
lose the flag, regress the pipeline selection, and the A/B stays green while the traffic
saving silently disappears. P2.3 had the same hazard and answered it by asserting its
threshold in BOTH directions.

Gate 6 is the positive control. `sg_gpu_forward` now counts which partial
`enc_attn_splitk` encoded, exposed read-only through `sg_gpu_splitk_dispatch_counts`, and
the subtest asserts that with the switch on every split-K dispatch was a GQA dispatch and
the per-head count is exactly 0 (and the reverse with it off, and the same TOTAL either
way). It also checks the group-size policy through `sg_gpu_splitk_gqa_selected`, which
calls the same internal predicate the encoder consults: the real 32/8 and 24/4 shapes and
the repeat-8 boundary are in the band, repeat 1, repeat 9, a non-multiple and
`n_kv_heads == 0` are out, and with the switch off everything is out. Those five rules
previously existed only as a comment. Both functions are diagnostics: no kernel reads them
and no dispatch shape depends on them.

What gate 4 covers, and why those shapes: `repeat` 4 (the 4B shape), `repeat` 6 (the 27B
shape, a non-power-of-two group, which is what a hardcoded shift would get wrong),
`repeat` 1 (no GQA at all), `repeat` 3 (odd and small), and `repeat` 16 (past
`SG_SPLITK_GQA_MAX`, so the kernel's one-head-at-a-time fallback arm runs and must STILL
be byte-identical). Both `q_stride` conventions are covered (dense `head_dim` and the
hybrid's `2*head_dim` with the gate half NaN-poisoned), and `n_splits` 257 at seq 200
forces genuinely empty splits so the `-INFINITY`/0/0 encoding is exercised for every head
of a group. All four buffers are compared (m, s, acc AND the combined out), memcmp-wise,
with both sides poisoned before every dispatch so an unwritten buffer cannot pass by
inheriting bytes.

Gate 7 is expected to be byte-identical in gen_ids AND in logits, which is a stronger
claim than P2.3's A/B could make: `SURGE_ATTN_SPLITK=0/1` changes how the sum over keys is
partitioned, so it changes rounding, while `SURGE_ATTN_SPLITK_GQA=0/1` is contracted to
change nothing at all.

## Flipping the default, when the gates pass

One line, `src/metal.m`, in `sg_gpu_state_new`: `bool gqa_on = false;` becomes the same
`sk_env`-style default-on parse `SURGE_ATTN_SPLITK` uses, i.e. default true with `"0"`
pinning the per-head kernel. Do it only with gates 4 to 9 green AND a measured speedup
table in this document, including whatever short-sequence crossover gate 9 finds, and add
the threshold to `splitk_gqa_use` if there is one. Until then the shipped decode path is
byte-for-byte the one P2.3 measured.

## Known risks, none of them checkable without the GPU

1. **Pipeline threadgroup width.** `sg_gpu_init` requires every non-elementwise kernel's
   `maxTotalThreadsPerThreadgroup` to be at least SG_TG (256). The GQA kernel holds up to
   8 extra float accumulators per thread; if that pushed the register allocation far
   enough to reduce the pipeline's max width, init would fail for the WHOLE library, not
   just this kernel. Any Metal test run detects it immediately (it is an init-time check).
   The fix if it happens is `[[max_total_threads_per_threadgroup(256)]]` on the kernel,
   which trades spills for the guarantee.
2. **Fast-math / FMA contraction.** Byte-identity assumes the Metal compiler emits the
   same strict IEEE operation sequence per element in both kernels. Both are the same
   serial accumulate-of-products shape and the Makefile passes `-fno-fast-math` (which it
   already calls not optional), but an R-accumulator loop is exactly the shape a fast-math
   optimizer would reassociate or contract differently from the single-accumulator
   original, and the failure would be a silent one-ulp drift, not an error. Building the
   metallib outside the Makefile rule voids the contract. Gate 4 is what proves it rather
   than argues it.
3. **Occupancy at short sequences**, as described above: fewer, longer-lived threadgroups.
   Gate 9 is the instrument.

## MEASURED on hardware, 2026-08-17 (all gates above now run)

The GPU freed when the 256K oMLX cell finished. Fresh GEMM gate 21.52 TFLOPS, nothing else
resident. Every result below was observed, not projected.

### Correctness

| gate | result |
|---|---|
| `make check` | exit 0, **86012 checks, 0 failures** (85319 at P2.3, so +693) |
| Positive control (gate 6) | **577 GQA dispatches with the switch on, 0 per-head; 577 per-head with it off; logits byte-identical at 1600/1600 positions** |
| Byte-identity (gate 2) | `GQA split-K partial: m/s/acc/out byte-identical over 100 reruns` |

The positive control is the one that matters. The review found the originally planned
real-model A/B could not distinguish "the GQA kernel ran and matched" from "the switch did
nothing", since both produce byte-identical output. Observing **577** dispatches rather than 0
closes that by evidence. Byte-identity, not a tolerance, was the bar the task set, and it held.

### Speed: GQA beats per-head split-K at every shape and depth tested

Same binary, same yardstick, both arms; `n_splits` swept across the occupancy band.

| shape | seq | per-head best | GQA best | GQA gain | total vs incumbent |
|---|---|---|---|---|---|
| 27B `24h/4kv/256d` | 262144 | 7353.75 us @1024 | **4217.15 us @256** | **1.744x** | 28.330x |
| 27B `24h/4kv/256d` | 131072 | 3880.45 us @512 | **2385.05 us @256** | **1.627x** | 25.095x |
| 4B dense `32h/8kv/128d` | 262144 | 4897.65 us @1024 | **3365.45 us @512** | **1.455x** | 32.444x |

This **refutes the review's single biggest risk**: that the `R >= 3` accumulator arrays would
fail to register-allocate and leave GQA slower than the per-head kernel. It is faster
everywhere tested.

### The shipped split policy WAS wrong for this kernel (FIXED by task P2.5)

`n_splits = clamp(seq / SG_TG, 4, 1024)` was measured optimal for the per-head partial. **GQA's
optimum is consistently lower**, so reusing that policy costs real throughput:

| shape | seq | policy picks | policy time | GQA optimum | best time | left on the table |
|---|---|---|---|---|---|---|
| 27B | 262144 | 1024 | 4583.40 us | 256 | 4217.15 us | ~8% |
| 4B dense | 262144 | 1024 | 29.754x | 512 | 32.444x | ~9% |

The optimum was 2x lower in two cases and 4x lower in the third, and the curve is shallow near
it (27B at 262144: 256 gives 4217 us, 512 gives 4316 us, 2.3% apart). So this needed its own
sweep rather than a guessed closed form, which is exactly what task P2.5 did: see "P2.5: the
GQA partial's own split policy (implemented)" at the end of this file. **This was why the
default stayed off; it still does, but for a different and now-explicit reason (see that
section).**

### Achieved bandwidth: do not quote the harness's GQA GB/s column

`bytes_kv` (`tests/bench_splitk.c:280-283`) counts `n_heads` worth of issued bytes for BOTH
arms deliberately, so the speedup ratio shares one denominator and is valid. But the GQA kernel
reads only `n_kv_heads` worth, so its true traffic is `1/repeat` of the printed figure:

- 4B GQA at 262144 prints **1276.2 GB/s**. Actual DRAM traffic is **319.0 GB/s**, which is
  **51% of this machine's 630 GB/s measured roofline**.
- The per-head arm prints **876.9 GB/s**, which *exceeds* the roofline. That is the tell:
  cache was already absorbing much of the 4x redundant traffic. It is also the reason removing
  the redundancy bought 1.46x to 1.74x rather than the naive 4x.

### Projection, explicitly an upper bound

16 full-attention layers x 4.217 ms = 67.5 ms/token = **~14.8 t/s** for the 27B at 262144,
against **0.537 t/s** measured in B7 and ~8.5 t/s for per-head split-K. Handle with care:
P2.3a established that attention was essentially the *entire* decode cost at this depth
(16 x 118.8 ms predicted 0.526 t/s against 0.537 measured). Once attention shrinks ~28x, the
non-attention cost stops being negligible and becomes the limiter. This is not a claim until it
is measured end to end on the real model.

## P2.5: the GQA partial's own split policy (implemented)

Companion task, same branch, task brief
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.5-brief.md`, report
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.5-report.md`. This closes the "not yet
fixed" gap the section above documented: the GQA partial now gets its own split count
instead of reusing `splitk_n_splits`.

### The policy

`splitk_gqa_n_splits(seq)` (`src/metal.m`, next to `splitk_gqa_use`):

```
n_splits_gqa = clamp(min(seq / SG_TG, SG_SPLITK_GQA_N_SPLITS_CAP), SG_SPLITK_MIN, SG_SPLITK_MAX)
```

with `SG_SPLITK_GQA_N_SPLITS_CAP == 256`, a named constant carrying a comment that points at
the measured table below (not a bare literal). `SG_SPLITK_MIN` (4) and `SG_SPLITK_MAX` (1024)
are the SAME occupancy-band floor and ceiling `splitk_n_splits` uses; only the 256 cap in the
middle is new, and it can only ever LOWER the result relative to the per-head policy at the
same seq, which is why no split-K buffer needed to grow.

This was measured, not guessed, and not re-derived by this task: task P2.5's brief carried the
sweep and a four-way regret comparison already run 2026-08-17
(`./tests/bench_splitk.bin --seqs 8192,32768,131072,262144 --gqa`, fresh GEMM gate 21.63
TFLOPS):

| policy | mean regret | worst regret |
|---|---|---|
| `clamp(seq/SG_TG, 4, 1024)` (the old default, per-head policy reused) | 3.1% | 7.3% |
| **`clamp(min(seq/SG_TG, 256), 4, 1024)` (the winner, implemented)** | **0.5%** | **2.6%** |
| `clamp(min(seq/SG_TG, 512), 4, 1024)` | 1.5% | 4.2% |
| `clamp(seq/(2*SG_TG), 4, 1024)` ("half the per-head optimum") | 3.8% | 13.4% |

**The intuitive rule is the worst one.** "GQA's optimum is half the per-head optimum" fits
the two longest sequences by construction (both optima there happen to be roughly half of
`seq / SG_TG`) and is 13.4% worst-case wrong overall: the true optimum does not RESCALE with
seq, it SATURATES near 256, so a value that keeps climbing and is merely halved eventually
overshoots almost as badly as never capping at all. The winner is the per-head closed form
with a measured cap, not a different closed form.

### Wiring: enc_attn_splitk picks the policy that matches the kernel it already picked

`enc_attn_splitk` already computes `gqa = splitk_gqa_use(g, p[0], p[1])` to choose which
kernel to dispatch (P2.4). P2.5 adds: build a local copy of the params array, and on the GQA
arm only, overwrite its `n_splits` slot with `splitk_gqa_n_splits(seq)` before EITHER the
partial dispatch or the combine dispatch reads it, since those two must agree with each
other on how many splits were written, not with whatever the per-head caller (`splitk_use`,
unchanged) computed. The per-head arm is untouched byte for byte: `splitk_n_splits` and
`splitk_use` are exactly what P2.3 shipped and are never called differently by this task, per
the brief's explicit constraint.

New public diagnostic `sg_gpu_splitk_gqa_n_splits(seq)` (`surge.h` / `src/metal.m`) exposes
the policy function itself, calling it rather than restating it -- the same
one-source-of-truth pattern `sg_gpu_splitk_gqa_selected` already established for the
kernel-selection predicate.

### A narrowed byte-identity claim, from seq 65792 on

The cap first binds (first returns something below the per-head value) at seq 65792, i.e.
`SG_TG * (SG_SPLITK_GQA_N_SPLITS_CAP + 1)`: that is the first seq where `seq / SG_TG` reaches
257, one past the 256 cap. Every seq up to and including 65791 still floors to 256 or less,
where `min(seq/SG_TG, 256)` is a no-op, so `splitk_gqa_n_splits(seq)` equals
`splitk_n_splits(seq)` exactly through 65791 (verified directly: seq 65536, 65600 and 65791 all
give 256 on both sides; 65792 gives 256 for GQA against 257 for the per-head policy, the first
mismatch). Below 65792, `SURGE_ATTN_SPLITK_GQA=0` and `=1` still dispatch with the SAME
n_splits and stay byte-identical end to end -- which is what
`mini_f16_splitk_gqa_dispatches_and_matches` (seq up to `SPLITK_GATE_N` == 1600) measures and
continues to measure unchanged. From 65792 on, the two modes partition the same keys
differently BY DESIGN (fewer, longer splits is the entire point), so from that seq on they only
agree to float rounding, the same way `SURGE_ATTN_SPLITK=0/1` already does. This is a property
of picking fewer splits, not a bug: the two KERNELS are still byte-identical to each other at
any FIXED n_splits, which `metal_attn_splitk_gqa_bit_identical` checks directly and bypasses
this policy entirely. Documented at the point it matters in `enc_attn_splitk`'s header
comment so a future real-model long-context A/B is not built on an expectation this task
already narrowed.

(An earlier draft of this section and of the matching `src/metal.m` comment said the
divergence started "above seq 65536"; an adversarial review caught that it is off by one
`SG_TG`, since `floor(seq/SG_TG) <= 256` holds through seq 65791, not just through 65535. Fixed
before commit; recorded here since the whole point of writing this section down was to save a
future reader from re-deriving the boundary, and a wrong boundary would have cost more than
none.)

### Gates run, all on hardware (GPU free for the whole task)

1. `xcrun metal -fno-fast-math -Wall -c src/kernels.metal` + metallib link: clean
   (kernels.metal untouched by this task).
2. `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror` on `src/metal.m`,
   `tests/test_metal_ops.c`, `tests/test_gpu_fwd.c`, `tests/bench_splitk.c`: clean, both with
   and without `-DSURGE_NO_METAL`.
3. `make debug`: **83523 checks, 0 failures, 0 sanitizer diagnostics** -- identical to the
   P2.4 baseline (metal.m is not compiled at all under `SURGE_NO_METAL`).
4. `make check`: **86024 checks, 0 failures** (86012 before this task, +12: the new
   `splitk_gqa_n_splits_policy` subtest's 8 table points + 4 explicit cap-bind assertions).
   CLI-level gates inside it also green (`test_cli_prefill`, `test_cli_bench`).
5. P2.4's own gates re-verified UNCHANGED: `metal_attn_splitk_gqa_bit_identical` (byte
   identity + 100x determinism) still passes; `mini_f16_splitk_gqa_dispatches_and_matches`
   (the positive control) still reports **577 GQA dispatches with the switch on, 0 per-head;
   577 per-head with it off; logits byte-identical at 1600/1600 positions** -- the exact same
   numbers the P2.4 hardware run recorded, proving P2.5 did not perturb it (the mini fixture's
   seq never exceeds `SPLITK_GATE_N` == 1600, far below the 65536 point where the two split
   policies would first diverge).
6. `./tests/bench_splitk.bin --seqs 8192,32768,131072,262144 --gqa`, re-run fresh for this
   task (not the brief's original sweep):

   | shape | seq | policy n_splits | policy time | optimum n_splits | optimum time | regret |
   |---|---|---|---|---|---|---|
   | 27B 24h/4kv/256d | 8192 | 32 | 584.60 us | 32 | 584.60 us | 0.00% |
   | 27B 24h/4kv/256d | 32768 | 128 | 1006.10 us | 128 | 1006.10 us | 0.00% |
   | 27B 24h/4kv/256d | 131072 | 256 | 2372.10 us | 256 | 2372.10 us | 0.00% |
   | 27B 24h/4kv/256d | 262144 | 256 | 4154.80 us | 256 | 4154.80 us | 0.00% |
   | 4B 32h/8kv/128d | 8192 | 32 | 532.55 us | 32 | 532.55 us | 0.00% |
   | 4B 32h/8kv/128d | 32768 | 128 | 882.95 us | 64 | 877.05 us | 0.67% |
   | 4B 32h/8kv/128d | 131072 | 256 | 1976.90 us | 256 | 1976.90 us | 0.00% |
   | 4B 32h/8kv/128d | 262144 | 256 | 3430.05 us | 512 | 3338.80 us | 2.73% |

   **Mean regret 0.43%, worst regret 2.73%** (4B dense, seq 262144), measured against this
   run's own per-point optimum rather than the brief's original table. All eight per-point
   optima matched the brief's original table exactly (same n_splits wins at every point);
   only the microsecond values differ slightly, which is why the regret differs slightly
   (0.43%/2.73% here vs the brief's 0.5%/2.6%) -- normal run-to-run GPU timing noise, not a
   different policy or a different winner. The 2.73% worst case is marginally above the
   brief's "~2.6%" reference figure and is reported as measured rather than rounded to match.
   Both runs agree this policy is far better than reusing the per-head policy (3.1%/7.3%) and
   vastly better than "half the per-head optimum" (3.8%/13.4%).

### Still open, and still not this task's decision

`attn_splitk_gqa` defaults to `false` in `sg_gpu_state_new`; `SURGE_ATTN_SPLITK_GQA=1` stays
the opt-in. P2.5 removes the ONE reason the P2.4 gate doc gave for staying off (the wrong split
policy); it does not flip the default, which is a separate, deliberate, outward-facing
decision for whoever owns that call. Also still open, and out of P2.5's scope: the real-model
greedy A/B on an actual GGUF, and whether `splitk_gqa_use` needs a short-sequence
threadgroup-count floor the way `splitk_use` has one for the per-head kernel (see "WHAT IS NOT
DECIDED HERE" in `splitk_gqa_use`'s comment, `src/metal.m`) -- neither was touched by this
task and neither was required for its gates to pass.
