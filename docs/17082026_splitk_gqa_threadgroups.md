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
   seq never exceeds `SPLITK_GATE_N` == 1600, far below the 65792 point where the two split
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

### Residual gap before the default can be flipped: CLOSED by task P2.6, 2026-08-17

**What the gap was.** surge's correctness standard is byte-exact greedy TOKENS, not identical
logits. From seq 65792 on, the GQA policy deliberately picks a different `n_splits` than the
per-head path, so the two agree only to float rounding, and nothing verified that greedy tokens
still match where the cap actually binds: the positive control runs at seq <= 1600 (both
policies return 6 there) and `metal_attn_splitk_gqa_bit_identical` pins `n_splits` identically
on both sides, so both bypass exactly the regime P2.5 introduced. The margin argument (M2's
top1-to-top2 gap around 8360x the observed perturbation; P2.4's worst logit delta 9.537e-07
against a smallest margin 1.385e-03) was an argument, not a gate.

**Why the obvious gate was the wrong one.** The obvious gate is a real-model A/B above 65791
tokens. Task P2.6 did not use that as the primary gate, and the reason is measured, not
guessed: see "the one gate still not run" below. The property under test is only *the two arms
pick different `n_splits`, do greedy tokens still match*, which needs DIVERGENCE, not DEPTH.

**The mechanism P2.6 added.** The divergence point is `SG_TG * (cap + 1)`, so the cap became a
per-state value: `SURGE_SPLITK_GQA_CAP`, read once in `sg_gpu_state_new` beside
`SURGE_ATTN_SPLITK`, `SURGE_ATTN_SPLITK_GQA` and `SURGE_KV_DTYPE`. `=4` moves the identical
257-vs-256 mechanism down to seq 1280. The default is unchanged at the measured 256.

Two deliberate differences from the other three env vars:

- an unusable value is **REJECTED** (`sg_gpu_state_new` returns an error), not warned about and
  ignored. A silently dropped value is precisely how a gate that sets this var would go
  vacuous, with both arms picking the same `n_splits` and nothing under test. Accepted values
  are plain integers in `[SG_SPLITK_MIN, SG_SPLITK_MAX]` == `[4, 1024]`, parsed with `strtol`
  plus a full-string end check so `4x` and `""` are errors rather than 4 and 0. Outside that
  band the value is meaningless anyway, because the policy's own floor or ceiling clamp eats
  it.
- it is a **gate and retuning knob, not a run option**. Nothing about a lowered cap is a
  performance configuration; cap 4 is deliberately pessimal occupancy.

`splitk_gqa_n_splits` became a pure function of `(seq, cap)` rather than reading the macro, so
the P2.5 policy assertions above stay meaningful, and `splitk_gqa_cap_of(g)` is the ONE cap
resolver that both `enc_attn_splitk` and the new diagnostics call. An unset field resolves to
the compiled default, never to "cap 0" (which the clamp would silently turn into
`SG_SPLITK_MIN` for every seq). The `splitk_gqa_n_splits(seq, cap) <= splitk_n_splits(seq)`
invariant holds at ANY cap (`min(x, cap) <= x`, then the identical monotone clamp), so no
override can make the GQA arm exceed the m/s/acc buffers sized from
`splitk_n_splits(max_ctx)`; no buffer changed size for P2.6 either. New read-only diagnostics:
`sg_gpu_splitk_gqa_n_splits_at(g, seq)` (what THIS state's GQA arm will dispatch),
`sg_gpu_splitk_gqa_cap(g)` (the resolved cap, i.e. whether an override was parsed) and
`sg_gpu_splitk_n_splits(seq)` (the per-head arm's count, so a gate asserts the DIVERGENCE
rather than one side of it). `sg_gpu_splitk_gqa_n_splits(seq)` keeps its P2.5 signature and
meaning, the shipped table at the compiled cap.

#### Gate 1: the `make check` subtest, and what it observed

`mini_f16_splitk_gqa_cap_override_greedy_matches` (`tests/test_gpu_fwd.c`), cap 4 on both arms,
1600 positions on the mini hybrid fixture (4 heads over 2 kv, repeat 2). Observed:

> cap 4 so the policies split 6 (per-head) vs 4 (GQA) at seq 1600 and diverge from seq 1280;
> 577 GQA dispatches with the switch on (0 per-head), 577 per-head with it off; 321/321
> positions differ above the divergence seq, 0 of 1279 below; worst |delta| 7.153e-07 (scaled
> 2.902e-07) vs min top1-top2 margin 1.385e-03; greedy argmax 1600/1600 agree

Five assertions, and the first three exist so it cannot pass vacuously: the override was
PARSED (resolved cap read off the live state); the policies really diverge BY SPECIFIC VALUE at
and only at the documented seq (4 vs 4 at seq 1279, 5 vs 4 at 1280, 6 vs 4 at 1600, read
through the same functions the encoder dispatches with, plus an assertion that at the SHIPPED
cap the two would still be EQUAL at seq 1600, which is the explicit statement that this subtest
has content only because of the override); WHICH KERNEL RAN, from
`sg_gpu_splitk_dispatch_counts`. Only then the two numeric halves: the bits must be IDENTICAL
at every position below seq 1280 and DIFFERENT at every position at or above it, INCLUDING the
one at exactly 1280, and every greedy argmax must agree.

The env parse is gated in both directions too, since a silently ignored value is the whole
hazard: 10 unusable values (`0`, `3`, `1025`, `-4`, `4x`, `x`, `" "`, `" 4"`, `"+4"`, and a
`strtol` range overflow) must each make `sg_gpu_state_new` FAIL, the three in-band edges (`4`,
`256`, `1024`) must each be accepted and reported back through `sg_gpu_splitk_gqa_cap`, and
unset must resolve to 256. That last check doubles as proof that a rejected state leaves
`sg_gpu_state_new` usable.

Note the cap-4 divergence is a HARSHER test than the natural case, not a softer one: a 1.5x
split-count ratio at seq 1600 (6 vs 4) against 1.004x at 65792 (257 vs 256).

#### Gate 2: mutation-proved, three mutations actually applied and run

Not argued. Each was patched into `src/metal.m`, built, run, and reverted:

| mutation | observed result |
|---|---|
| `sg_gpu_state_new` parses `SURGE_SPLITK_GQA_CAP` and drops the value | **7 failures**: "SURGE_SPLITK_GQA_CAP=4 was not applied (resolved cap 256 ...)", then "at the divergence seq 1280 the policies must split 5 vs 4, got per-head 5 and GQA 5", then "0 of 321 positions differ" |
| `enc_attn_splitk` dispatches with the compiled `SG_SPLITK_GQA_N_SPLITS_CAP` instead of the state's cap | **1 failure**: "at seq >= 1280 the capped GQA policy must actually change the bits (0 of 321 positions differ, so the cap override never reached the dispatch and this gate is vacuous)", worst \|delta\| 0.000e+00 |
| the range/format validation accepts everything (warn-and-ignore like the other env vars) | **10 failures**, one per rejected value: "must be rejected, not ignored" |
| `enc_attn_splitk` dispatches with `splitk_gqa_cap_of(g) + 1`, so the ACCESSORS still report cap 4 while the DISPATCH uses 5 | **2 failures**: "the position at exactly the divergence seq 1280 must differ" and "at seq >= 1280 the capped GQA policy must change the bits at EVERY position (65 of 321 differ)". The greedy argmax was still 1600/1600 under this mutation, so the token comparison ALONE would not have caught it |

The fourth mutation came from an external adversarial review of the commit, and it is the reason
the bit-level check is TWO-SIDED at the boundary. The first version asserted only
`above_diff > 0`, a one-sided existence test over a 321-position window, which a mutation that
shifts the real divergence LATER passes: at dispatched cap 5 the divergence moves to seq 1536
and positions 1536-1600 still differ. The gate now also requires the position at EXACTLY the
divergence seq to differ, and requires EVERY position at or above it to differ. Both are stated
as equalities rather than inequalities because each mode is deterministic (P2.3's subtest reruns
split-K 100x byte-identically), so 321/321 is a fixed count for this fixture and cap, not a
sampled one.

#### Gate 3: the real-model greedy A/B, which is the one that closes the gap

`Qwen3-4B-Instruct-2507-Q8_0.gguf` (32 query heads over 8 kv, repeat 4, 36 full-attention
layers), 5483-token prompt (surge's own three gate docs concatenated, sha256 `8844a188...`),
`-n 64 --margins`, chunked prefill, three arms, only the two env vars changed between them:

| arm | `SURGE_ATTN_SPLITK_GQA` | cap | GQA `n_splits` | per-head `n_splits` | decode |
|---|---|---|---|---|---|
| A0 | 0 | 4 | n/a | 21 | 39.25 tok/s |
| A1 | 1 | 4 | 4 | n/a | 23.79 tok/s |
| A2 | 1 | 256 (default) | 21 | n/a | 38.51 tok/s |

- **A0 vs A1, the divergent pair (21 splits against 4): `gen_ids` BYTE-IDENTICAL**, sha256
  `7972849e3c0f0d18d6b638cf3e7ab24c8b6c3662a3c9d0e9fc58a148b680bb2d` on both arms. **63 of the
  64 top1-top2 margins DIFFER**, which is what makes the token match non-vacuous: the two arms
  demonstrably computed different bits and still chose the same tokens. (`margin[0]` is
  identical by construction, since it comes from the prefill's last-position logits and prefill
  does not use the decode split-K kernel; that it is the ONLY identical one is itself a
  consistency check.) The smallest decision margin observed was 4.760456e-02 and the
  perturbation on that same position was 2.661e-04, **179x headroom**; the largest absolute
  margin perturbation anywhere was 3.860e-03, on a margin of 1.038e+01.
- **A0 vs A2, the control (same `n_splits` 21 on both sides): margins AND `gen_ids`
  byte-identical.** This is P2.4's byte-identity contract holding on a real model at real
  depth, and it proves the divergence seen in A1 comes from the cap and from nothing else.
- A1 being slower is expected and is further evidence the cap took effect: 4 splits x 8 kv rows
  is 32 threadgroups.

#### Everything else that was re-run

| gate | result |
|---|---|
| `make check` | **86067 checks, 0 failures** (86024 at the parent commit, +43, all in the new subtest); `test_cli_prefill` 11 cases and `test_cli_bench` 14 cases green |
| `make debug` (SURGE_NO_METAL, ASan/UBSan) | rc 0, **83523 checks, 0 failures, 0 sanitizer diagnostics**, identical to the P2.4/P2.5 baseline |
| P2.4 positive control | **577 GQA dispatches with the switch on (0 per-head), 577 per-head with it off, logits byte-identical at 1600/1600 positions**, the same numbers P2.4 and P2.5 recorded |
| P2.5 policy assertions + byte identity + 100x determinism | unchanged, 0 failures |

A pre-existing flake was identified along the way and is NOT caused by P2.6: `test_cli_bench`'s
B6 `check2` (reported decode-tps slope within 3% of avg) failed once at rel_diff 0.0336 during
back-to-back `make check` runs, and reproduced AT THE PARENT COMMIT with P2.6 stashed (3rd
consecutive run, rel_diff 0.0376, decode throughput down to 1214/1170 tok/s, i.e. the M3
firmware clock clamp). It is a machine-timing tolerance on a live 1024-token decode and touches
no split-K code; it passes on a cooled machine.

### The one gate still not run, with its measured cost

The natural-regime confirmation, i.e. the SAME A/B at a prompt just over 65791 tokens with the
cap at its real 256 (per-head 257 splits against GQA 256), **was started and abandoned on cost,
not skipped silently.** Measured, on the 4B with a 66219-token prompt (9.10 GiB KV cache):

| tokens prefilled | elapsed | rate |
|---|---|---|
| 1024 | 14.8 s | 69 tok/s |
| 9216 | 267.6 s | 34 tok/s |

Fitting `t = c n^2 + d n` to those two in-run points gives c = 1.78e-6 s/token^2, d = 1.26e-2
s/token, hence **about 144 min of prefill per arm and 4.8 h for the two-arm A/B**. (Two shorter
runs on the same machine and binary, 5483 tokens in 72.9 s and 10966 tokens in 171.7 s, fit
c = 4.32e-7 and predict only 44 min per arm; the in-run curve is roughly 2x worse, consistent
with the B8 duty-cycle rests plus the documented M3 firmware clock clamp under sustained load,
and it is the in-run number that should be believed.) That was judged not affordable for a
confirmation whose perturbation is STRICTLY SMALLER than what gate 3 above already measured: a
1.004x split-count ratio at 65792 against the 5.25x ratio (21 vs 4) that produced
byte-identical `gen_ids` there.

**What was run instead, as the closest affordable approximation of the natural regime**: the
same real-model A/B at `SURGE_SPLITK_GQA_CAP=63` on a 16449-token prompt (3 copies of the gate
docs, sha256 `ac283bb6...` is the 66k one, this one is 3x `8844a188...`), which makes the two
arms split **64 (per-head) against 63 (GQA)** at seq 16450. That is an OFF-BY-ONE divergence at
large split counts and real depth, i.e. the same shape of perturbation as the natural 257
against 256, at a 1.016x split-count ratio instead of 1.004x, for 9.5 min per arm instead of
144:

| arm | `SURGE_ATTN_SPLITK_GQA` | cap | `n_splits` | prefill | decode |
|---|---|---|---|---|---|
| M0 | 0 | 63 | 64 (per-head) | 16449 tok in 554.751 s (29.65 tok/s) | 64 tok in 5.007 s (12.58 tok/s) |
| M1 | 1 | 63 | 63 (GQA) | 16449 tok in 573.699 s (28.67 tok/s) | 64 tok in 4.174 s (15.09 tok/s) |

- **`gen_ids` BYTE-IDENTICAL**, sha256
  `7f1de3c66f355f1b91d783ca504e84f9d3def54c34ef76765e9abd809342346a` on both arms.
- **61 of 64 margins differ**, so the arms genuinely diverged. The 3 that match are `margin[0]`
  (prefill-derived, identical by construction) and two positions whose gaps are 1.504e+01 and
  1.198e+01, where the perturbation fell below what `%.6e` resolves.
- Tightest decision anywhere: gap **1.152420e-02** at `margin[52]`, perturbed by **2.022e-04**
  on that same position, **57x headroom**. Largest absolute perturbation anywhere 1.660e-03, on
  a gap of 1.544e+01.

So the greedy-token property now holds on a real model at three separate split-count ratios
(5.25x, 1.016x and the 1.0x control), and the untested case is only the specific pair 257/256
at seq >= 65792.

Anyone who wants the true >65791 confirmation should budget about 5 h of exclusive GPU and
9.1 GiB of KV cache per arm, and should use `--margins` so the result is falsifiable: if the
two arms' margins come back identical, the arms did not actually diverge and the run proves
nothing.

### Flipping the default

With the above, the correctness case for `SURGE_ATTN_SPLITK_GQA=1` is: the two kernels are
byte-identical at any fixed `n_splits` (P2.4, per-op and end to end), the GQA split policy is
measured (P2.5, mean regret 0.43%), and where the policies diverge the greedy tokens still
match, gated in `make check` and on a real model (P2.6). `attn_splitk_gqa` nevertheless still
defaults to `false`: flipping it is an outward-facing change to a public engine and is the
user's decision, which none of P2.4, P2.5, P2.6 or P2.7 makes.

## P2.7: the occupancy floor, so flipping the default cannot regress short context

Task brief `.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.7-brief.md`, report
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.7-report.md`. This closes the item every
section above left open by name ("whether `splitk_gqa_use` needs a short-sequence
threadgroup-count floor"). It does NOT flip the default.

**The problem, measured.** The GQA kernel's whole trick is collapsing `repeat` query-head
threadgroups into one, which divides the grid by `repeat`. At short context the collapsed grid
no longer fills this machine's 80 GPU cores, and the kernel is SLOWER than the per-head one it
replaces: at seq 2048 on the 27B shape the GQA grid is 8 splits x 4 kv = 32 threadgroups, and
the measured speedup there is 0.918x to 0.938x. That is the same shape of crossover P2.3 found
for split-K itself (slower than `k_attn_decode_f16` below seq 1024) and it was never guarded.

### The rule

```c
#define SG_SPLITK_GQA_MIN_TG 128u
/* in splitk_gqa_use, after the [2, 8] group-band checks: */
uint64_t tgs = (uint64_t)splitk_gqa_n_splits(seq, splitk_gqa_cap_of(g)) * n_kv;
return tgs >= SG_SPLITK_GQA_MIN_TG;
```

The floor is on the THREADGROUP COUNT the dispatch will actually have, not on `seq`, and it is
computed by CALLING the same `splitk_gqa_n_splits` / `splitk_gqa_cap_of` pair
`enc_attn_splitk` dispatches with, so the guard cannot guard a grid the encoder does not use
(there is a `make check` assertion for exactly that mutation, below). `splitk_gqa_use` and its
public diagnostic `sg_gpu_splitk_gqa_selected` both gained a `seq` parameter; the accessor
still CALLS the predicate rather than restating it, as P2.4 and P2.5 established.

### The measurement the 128 comes from

`./tests/bench_splitk.bin --reps 20 --seqs 2048,2560,3072,3584,4096,5120,6144,7168,8192,16384`,
run once per arm (with and without `--gqa`) per round, rounds ALTERNATED between arms so
thermal drift cannot land on one arm, GPU idle-checked with `pgrep` before and after. Speedup =
per-head time / GQA time at the split count the decode policy would dispatch
(`n_splits = seq / SG_TG` here, where the 256 cap does not bind), so below 1.000 the GQA kernel
is the wrong choice. Three studies:

- **A**: `--reps 20`, 3 rounds per arm, before the code change.
- **B**: `--reps 50`, 6 rounds per arm, on the 6 seqs around the crossover.
- **C**: `--reps 20`, 3 rounds per arm, after the code change (the harness calls the one-shot
  entry points, not `splitk_gqa_use`, so C is a reproduction, not a re-decision).

Medians per study, by threadgroup count (`n_splits * n_kv_heads`):

| threadgroups | 27B `24h/4kv` seq | A | B | C | pooled median (n) |
|---|---|---|---|---|---|
| 32 | 2048 | 0.938 | - | 0.918 | 0.928 (6) |
| 40 | 2560 | 0.867 | - | 0.884 | 0.884 (6) |
| 48 | 3072 | 0.980 | 0.992 | 0.991 | 0.987 (12) |
| 56 | 3584 | 0.971 | 0.949 | 0.987 | 0.966 (12) |
| 64 | 4096 | 0.990 | 0.967 | 0.985 | 0.980 (12) |
| 80 | 5120 | 0.988 | 0.956 | 0.958 | 0.961 (12) |
| 96 | 6144 | 1.025 | 1.040 | 1.033 | **1.032** (12) |
| 112 | 7168 | 1.110 | 1.104 | 1.103 | **1.108** (12) |
| 128 | 8192 | 1.148 | - | 1.162 | **1.155** (6) |
| 256 | 16384 | 1.248 | - | 1.187 | **1.213** (6) |

| threadgroups | 4B dense `32h/8kv` seq | A | B | C | pooled median (n) |
|---|---|---|---|---|---|
| 64 | 2048 | 0.975 | - | 0.922 | 0.965 (6) |
| 80 | 2560 | 0.925 | - | 0.923 | 0.923 (6) |
| 96 | 3072 | 0.975 | 0.974 | 0.975 | 0.974 (12) |
| 112 | 3584 | 1.008 | 0.991 | 0.989 | 0.997 (12) |
| 128 | 4096 | 1.004 | 1.004 | 0.994 | **1.003** (12) |
| 160 | 5120 | 1.029 | 1.018 | 1.053 | **1.027** (12) |
| 192 | 6144 | 1.019 | 1.014 | 1.049 | **1.026** (12) |
| 224 | 7168 | 1.065 | 1.021 | 1.055 | **1.043** (12) |
| 256 | 8192 | 1.056 | - | 1.074 | **1.065** (6) |
| 512 | 16384 | 1.182 | - | 1.149 | **1.166** (6) |

Raw harness output for all 24 runs is committed next to the report as
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.7-{bench,bound,confirm}-{perhead,gqa}-r*.txt`.

**Why 128, and what it costs.** The 4B shape is the binding one. Its pooled ratio crosses 1.000
between 112 threadgroups (0.997) and 128 (1.003), and at 96 it is a clear 0.974 loss in all
three studies. The 27B crosses lower, between 80 (0.961) and 96 (1.032), because repeat 6 and
head_dim 256 save more traffic per threadgroup given up. 128 is the smallest single constant
that admits no measured loss on EITHER shape. The cost, stated rather than hidden: the 27B's
pooled +3.2% at 96 threadgroups and +10.8% at 112 (seq 6144 to 8191) are left unclaimed. That
is the right direction to err in, since a wrong-way error is a slowdown on a switch whose whole
purpose is to be flippable without one.

**A `seq` threshold does not fit, and this is the fit that was asked for.** The same crossovers
in `seq` are 5120..6144 (27B) and 3072..4096 (4B): 1.7x apart and in the OPPOSITE order from
the threadgroup view, because the 4B has twice the kv heads. Any single `seq` threshold must
either admit the 27B's 0.961 at seq 5120 or reject the 4B's wins from 4096 up. One threadgroup
threshold separates all 20 measured points; the only admitted point that is not a clear win is
the 4B at exactly 128 threadgroups, which is LEVEL (pooled 1.003 over 12 rounds, individual
rounds 0.963 to 1.049, i.e. inside this machine's noise).

**The better-fitting rule that was NOT shipped**, recorded so it is not re-derived: the
per-head grid `n_splits * n_heads >= 512` (equivalently `tgs >= 512 / repeat`) also classifies
all 20 points correctly AND keeps the 27B's 96- and 112-threadgroup wins, because it lets a
bigger `repeat` buy a smaller grid. It was rejected because it is a two-parameter rule fit from
exactly two `repeat` values, and at repeat 8 it extrapolates to a floor of 64 threadgroups,
BELOW this machine's 80 cores, which is precisely the starved region the measurement says to
avoid. A third shape (repeat 8, e.g. 64 heads over 8 kv) measured across the same threadgroup
sweep would settle it; until then the shipped floor is the shape-independent one.

**REPETITIONS ARE NOT OPTIONAL HERE.** The brief recorded that a single-rep sweep of these
points reported the 4B losing at 8192 and the 27B losing at 4096, both of which reversed under
20 reps. This task saw the same instability at the margin from the other side: the 4B at 128
threadgroups came out 1.004, 1.004 and 0.994 in the three studies. Any conclusion drawn from
one rep per point at this scale is a conclusion drawn from noise.

**Not measured, and not claimed:** whether the end-to-end decode gain moves. The brief's
corroborating real-27B run (decode 6.57 to 7.04 t/s at 21120 tokens, 1.07x) sits at
`n_splits = 82`, i.e. 328 threadgroups, far above the floor, so this guard cannot have changed
it; that was not re-run for this task.

### What P2.7 changed in the two existing GQA gates, and why it had to

Both GQA subtests ran at `SPLITK_GATE_N == 1600`. The mini fixture has 2 kv heads, so at seq
1600 its GQA grid is 6 splits x 2 kv = 12 threadgroups, far under the new floor: the guard
declines, both arms run the PER-HEAD kernel, and both gates would have gone green while
comparing a kernel with itself. That is the worst possible outcome for a positive control, so
the fixtures moved rather than the floor:

| | before | after | why |
|---|---|---|---|
| GQA gates' run length | `SPLITK_GATE_N` 1600 | `SPLITK_GQA_GATE_N` 16896 (66 x SG_TG) | 2 kv heads need 64 splits to reach 128 threadgroups, i.e. seq >= 16384; decode is sequential, so the gate has to walk there. Adds about 45 s to `make check`. |
| P2.6 cap override | `SURGE_SPLITK_GQA_CAP=4` | `=64` | The cap CEILINGS the GQA grid at `cap * n_kv` threadgroups. At cap 4 that is 8, under the floor at every seq, so the GQA kernel would never be selected and the cap gate would be vacuous. 64 is the smallest cap that can reach the floor on this fixture. |
| P2.6 divergence seq | 1280 (`SG_TG * 5`) | 16640 (`SG_TG * 65`) | Same closed form `SG_TG * (cap + 1)`, moved by the cap. |
| P2.6 split-count ratio under test | 6 vs 4 (1.5x) | 66 vs 64 (1.031x), 65 vs 64 at the boundary | Consequence of the bigger cap. SMALLER perturbation than P2.6 tested, and much closer to the natural 257-vs-256 case; still detected at every position (257 of 257). |
| P2.3's split-K subtest | 1600 positions | unchanged | Its threshold is the split-K one (1024), untouched by this task. |

The P2.3 subtest's output is byte-for-byte the same as the P2.4/P2.5/P2.6 runs recorded:
`577/577 positions differ above seq 1024, 0 below; worst |delta| 9.537e-07 (scaled 4.608e-07)
vs min top1-top2 margin 1.385e-03; argmax 1600/1600 agree`.

The P2.4 positive control no longer reports 577 GQA dispatches, and it CANNOT: with the floor
in place, a run that starts at seq 1 dispatches the per-head partial until seq 16384 and the GQA
partial after it. The count is now asserted EXACTLY, on both sides, with the layer multiplier
derived from the run rather than hardcoded:

> split-K GQA decode: 513 GQA dispatches with the switch on (seq >= 16384, the P2.7 floor) +
> 15360 per-head below it, 15873 per-head with it off, logits byte-identical at 16896/16896
> positions

513 = positions 16384..16896, 15360 = the split-K positions below the floor, 15873 = the total
in both arms. That is a strictly STRONGER control than `gqa > 0`: it pins WHERE the floor is
with the dispatch counters, so a floor at the wrong seq fails even though the GQA kernel ran.

The P2.6 cap gate reports:

> split-K GQA cap override: cap 64 so the policies split 66 (per-head) vs 64 (GQA) at seq 16896
> and diverge from seq 16640; 513 GQA dispatches with the switch on (+15360 per-head below the
> P2.7 floor seq 16384), 15873 per-head with it off; 257/257 positions differ above the
> divergence seq, 0 of 16639 below; worst |delta| 7.153e-07 (scaled 4.206e-07) vs min top1-top2
> margin 2.400e-04; greedy argmax 16896/16896 agree

Its "identical below the divergence" half now also covers positions 16384 to 16639, where the
GQA KERNEL runs at the SAME `n_splits` as the per-head one: P2.4's byte-identity contract,
gated inside the real decode path at a grid the guard admits, which no previous gate did.

### New assertions (gate 3 of the brief)

`splitk_gqa_floor_policy` (`tests/test_gpu_fwd.c`) asserts the predicate against the measured
table itself: FALSE at every shape/seq where a loss was measured (27B at 2048/2560/4096/5120,
4B at 2048/2560/3072/3584), TRUE where a win was (27B at 8192/16384/262144, 4B at
4096/8192/16384), the fixture's own boundary from both sides (63 splits x 2 kv = 126
threadgroups declined, 64 x 2 = 128 selected), monotonicity in `seq` for a fixed shape over 300
steps, and the state's resolved cap, since the whole table shifts at a different cap. Each case
carries its measured speedup in the failure message, so a future retune has to argue with the
measurement rather than with a constant.

`splitk_gqa_floor_uses_state_cap` closes a mutation the two long runs cannot see: at cap 64 on
a 2-kv fixture the capped and uncapped split counts both clear the floor, so a guard reading
`splitk_n_splits(seq)` instead of the capped count would pass. At cap 4 they differ by 16x, and
one call settles it: the 4B shape at seq 262144 must be DECLINED (4 splits x 8 kv = 32
threadgroups) while the same shape with the default cap must be SELECTED.

### Gates run, all on hardware, GPU idle-checked (`pgrep` empty before each)

1. `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror` on `src/metal.m` and
   `tests/test_gpu_fwd.c`, with and without `-DSURGE_NO_METAL`: clean.
2. `make check`: **86099 checks, 0 failures** (86067 at the parent commit, +32, all in
   `tests/test_gpu_fwd.bin`, which went 165 -> 197). `test_cli_prefill` 11 cases and
   `test_cli_bench` 14 cases green. 1 min 15 s wall.
3. `make debug` (`-DSURGE_NO_METAL`, ASan + UBSan): rc 0, **83523 checks, 0 failures, 0
   sanitizer diagnostics** -- identical to the P2.4/P2.5/P2.6 baseline.
4. P2.4's byte-identity and 100x determinism subtest (`metal_attn_splitk_gqa_bit_identical`)
   and P2.5's `splitk_gqa_n_splits_policy`: unchanged, 0 failures (both are per-op and never
   reach the selection predicate).
5. Bench re-run after the change, study C above: the admitted region (>= 128 threadgroups)
   contains no measured loss. Weakest admitted point 4B at 128 threadgroups, pooled 1.003 over
   12 rounds; every other admitted point 1.03x to 1.21x.

### Mutation-proved (each patched into `src/metal.m`, built, run, reverted)

| mutation | observed |
|---|---|
| `SG_SPLITK_GQA_MIN_TG 0` (i.e. the pre-P2.7 behaviour, no floor) | **16 failures**: 10 in the floor table (every measured-loss shape reported SELECTED, including the 126-threadgroup boundary case), both dispatch-count gates ("expected 513 dispatches, got 15873" and "expected 15360 per-head, got 0"), and both cap-awareness assertions |
| `SG_SPLITK_GQA_MIN_TG 64` (a floor that admits measured losses) | **12 failures**: the 4B's 64/80/96/112-threadgroup losses and the 27B's 64/80 reported SELECTED, plus "expected 513, got 8705" and "expected 15360, got 7168" from the two dispatch gates |

### Still not this task's decision

`attn_splitk_gqa` still defaults to `false`. P2.7 removes the last measured reason not to flip
it (a short-context regression) and makes the flip non-regressive on both real shapes; the flip
itself remains the user's call.

## P2.8: online (streaming) softmax in the GQA partial

New kernel `k_attn_decode_splitk_partial_gqa_online` (`src/kernels.metal`), selected by
`SURGE_ATTN_SPLITK_ONLINE=1`, **default OFF**. The four-pass GQA partial stays intact and
reachable; this is an A/B, not a replacement.

The four-pass kernel writes every score of a split into `splitk_scratch` and walks that row
three more times (maximum, exponentiate, accumulate). The online kernel keeps a running
`(m, s, acc)` per head and updates it as each 256-key tile is folded in, rescaling by
`exp(m_old - m_new)` when the running maximum moves. It has **seven bindings, not eight**: no
`scores` argument, and neither the one-shot nor the encoder grows or binds `splitk_scratch` on
that path.

### Where the running accumulator lives, which is the whole design problem

A per-thread `acc[R][head_dim]` does not exist: at `R = 8` and `head_dim = 256` that is 2048
floats per thread, 2 MB per threadgroup. The two pieces of state are therefore attached to
different roles, which is the same split the four-pass kernel already makes between its phases:

| state | who holds it | cost |
|---|---|---|
| `m`, `s` | UNIFORM across the threadgroup (they come off a fold tree), so every thread keeps its own copy | `2R` registers |
| `acc[d]` | exactly ONE thread, the one that owns output dim `d` (`d = dbase + lid`) | `R` registers, not `R*head_dim` |
| this thread's own key's score | private | `R` registers |

Total streaming state is `4R` floats per thread, 32 at `R = 8`. That works because
`head_dim <= SG_TG`, which holds for every shape surge targets (27B 256, exactly `SG_TG`; 4B
dense 128). For `head_dim > SG_TG` the kernel's `dbase` loop runs more than once and re-streams
the split per 256-wide band of output dims: still exactly correct (gated below at `head_dim`
320, which makes the second band deliberately partial), but it re-reads K, so
`splitk_online_use` declines those shapes. Same kind of policy decline as a GQA group wider than
`SG_SPLITK_GQA_MAX`, which this kernel also still answers correctly one head at a time.

The price is a transpose through threadgroup memory, and it is unavoidable: the score of key `t`
is computed by the thread that owns key `t` and consumed by every thread that owns an output
dim, so an `R x SG_TG` block has to change hands. Device memory is what the four-pass kernel
uses for exactly that and is what this kernel exists to avoid. The buffer is
`SG_SPLITK_GQA_MAX * SG_TG` floats (**8 KB**, sized for the worst-case group so nothing on the
host has to agree with the kernel about an allocation length) and it does double duty as the
fold-tree scratch. Two new helpers `tg_max_group<R>` / `tg_sum_group<R>` fold all R heads in one
pass of the same fixed stride schedule, which is bit-identical to R separate `tg_max`/`tg_sum`
calls (no row reads another row's data) but keeps the barrier count independent of `R`; that
matters because the online kernel folds once per TILE rather than once per split.

Determinism is unchanged: each thread streams the SAME fixed subset of keys the four-pass kernel
gives it (`lid`, `lid+256`, ...), cross-thread combination is only ever a fixed-shape tree, every
`acc` slot has exactly one writer, and the rescale is a multiply by a value every thread
recomputes identically from the tree's output.

### Byte-identity is NOT the bar, and three things are still exact

Streaming reorders the exponential sums, so `s` and `acc` differ from the four-pass kernels in
the last bits whenever a split spans more than one tile. Still exact, and asserted:

1. **`m`**, bit-identical to the four-pass GQA kernel. A maximum is order-independent and the
   key subsets match, so a difference here would mean the streaming loop visited different keys.
2. **The empty-split encoding** `m = -INFINITY / s = 0 / acc = 0`, checked structurally.
3. **Determinism**, 100 reruns byte-identical.

Measured bonus: when a split fits ONE tile the whole triple comes out bit-identical (one key per
lane, same trees, same order). Observed at every shape, e.g. `32x8x128 seq200` at every split
count and `24x4x256 seq1000` at K=2: `0/N floats differ, worst |delta| 0.000e+00`.

### Gates run, all on hardware (the 256K benchmark holding the GPU cleared at 22:42)

1. `xcrun -sdk macosx metal -fno-fast-math -Wall -c src/kernels.metal` clean; `metallib` links;
   `k_attn_decode_splitk_partial_gqa_online` present in the library.
2. `clang -fsyntax-only -std=c11 -Wall -Wextra -Werror -Isrc -I.` on `src/metal.m`,
   `tests/test_metal_ops.c`, `tests/test_gpu_fwd.c`, `tests/bench_splitk.c`, with and without
   `-DSURGE_NO_METAL`: clean. Also compiled under `make debug`'s ASan/UBSan flag set.
3. `make debug`: rc 0, **83523 checks, 0 failures, 0 sanitizer diagnostics**, identical to the
   P2.4/P2.5/P2.6/P2.7 baseline (the Metal test files collapse to skip stubs there, so the new
   subtests add no checks to this count).
4. `make check`: **87257 checks, 0 failures** (86099 at the parent commit, +1158: +1122 in
   `test_metal_ops.bin`, +36 in `test_gpu_fwd.bin`). `test_cli_prefill` 11 cases and
   `test_cli_bench` 14 cases green. Every earlier gate unchanged: the P2.4/P2.7 positive control
   still reports **513 GQA dispatches + 15360 per-head with the switch on, 15873 per-head with
   it off, logits byte-identical at 16896/16896**, and the P2.6 cap gate still reports
   **257/257 positions differ above the divergence seq, 0 of 16639 below, argmax 16896/16896**.
5. **Accuracy vs the CPU oracles** (`metal_attn_splitk_online_matches_ref`, 12 shapes x 7 split
   counts x 2 oracles): worst **rel 2.109e-06, abs 4.172e-07**, at `24x4x256 seq1000 dense r6`
   with K=1251 (a split count above seq, i.e. mostly empty splits, which the four-pass sweep
   never tested). Apples to apples at P2.2's own headline point, `32x8x128 seq1000 gated`:

   | n_splits | four-pass per-head rel | online rel |
   |---|---|---|
   | 1 | 1.027e-06 | 1.118e-06 |
   | 2 | 5.479e-07 | 6.707e-07 |
   | 3 | 4.880e-07 | 4.564e-07 |
   | 7 | 3.082e-07 | 3.353e-07 |
   | 64 | 3.424e-07 | 2.981e-07 |
   | 257 | 4.794e-07 | 5.589e-07 |

   So not materially worse, and better at two of the six points. `worst relative error across all
   ops` for the whole per-op suite is still 2.524e-06 and still belongs to `matmul_f32`, not to
   this kernel.
6. **100x determinism**: `online split-K partial: m/s/acc/out byte-identical over 100 reruns`.
7. **Real-model greedy A/B**, `Qwen3-4B-Instruct-2507-Q8_0.gguf`, 5267-token prompt (the first
   17000 bytes of the three gate docs concatenated, sha256 `03ce9a25...`), `-n 64 --margins`,
   three arms, only the env vars changed. All three dispatch the same `n_splits` (20), so the
   kernel is the only variable:

   | arm | `SURGE_ATTN_SPLITK_GQA` | `SURGE_ATTN_SPLITK_ONLINE` | decode | `gen_ids` |
   |---|---|---|---|---|
   | B0 | 0 | 0 | 26.85 tok/s | sha256 `70f22515...` |
   | B1 | 0 | 1 | 22.98 tok/s | sha256 `70f22515...` |
   | B2 | 1 | 0 | 24.84 tok/s | sha256 `70f22515...` |

   - **B0 vs B1: `gen_ids` BYTE-IDENTICAL, 0 of 64 token mismatches, and 62 of 64 margins
     DIFFER**, which is what makes the token match non-vacuous. Largest absolute margin
     perturbation 3.310e-03. The smallest decision margin was 1.068974e-02 (position 56) and the
     perturbation on that same position was 5.302e-04, **20x headroom**.
   - **B0 vs B2, the control: margins AND `gen_ids` byte-identical** (0 of 64 margins differ).
     P2.4's byte-identity contract holding at real depth, and the proof that B1's divergence
     comes from the online kernel and from nothing else.
8. **Timing A/B**, `./tests/bench_splitk.bin --reps 20 --seqs 8192,32768,131072,262144`, `--gqa`
   against `--gqa --online`, 3 alternating rounds, median of the 3. At the split count the decode
   policy would dispatch:

   | shape | seq | n_splits | four-pass us | online us | online speedup | per round |
   |---|---|---|---|---|---|---|
   | 27B 24h/4kv/256d | 8192 | 32 | 591.45 | 530.75 | **1.114x** | 1.092/1.114/1.116 |
   | 27B 24h/4kv/256d | 32768 | 128 | 997.50 | 984.95 | **1.013x** | 0.992/1.013/0.954 |
   | 27B 24h/4kv/256d | 131072 | 256 | 2368.00 | 2250.00 | **1.052x** | 1.083/1.052/1.071 |
   | 27B 24h/4kv/256d | 262144 | 256 | 4195.05 | 3850.35 | **1.090x** | 1.092/1.090/1.083 |
   | 4B 32h/8kv/128d | 8192 | 32 | 527.55 | 520.45 | 1.014x | 1.011/1.014/0.989 |
   | 4B 32h/8kv/128d | 32768 | 128 | 922.75 | 938.10 | 0.984x | 0.959/0.984/1.003 |
   | 4B 32h/8kv/128d | 131072 | 256 | 2024.00 | 2603.05 | **0.778x** | 0.791/0.778/0.783 |
   | 4B 32h/8kv/128d | 262144 | 256 | 3472.15 | 5032.85 | **0.690x** | 0.685/0.690/0.701 |

### What that measurement means, and why the switch stays off

**The 27B shape wins at every decode-policy point, 1.013x to 1.114x, and the 4B shape loses at
depth, down to 0.690x.** The losses are large, tightly reproducible across the three rounds
(spread under 0.02) and concentrated: on the 4B they appear from 256 threadgroups up
(`n_splits >= 32`), the worst single point being 0.605x at seq 262144 with `n_splits` 64, while
the same shape at 128 threadgroups (`n_splits` 16) is 1.043x. The end-to-end 4B run above is
consistent (B1 slowest of the three arms despite running second, so it is not thermal drift).

The best-supported explanation, and it is a HYPOTHESIS the numbers only bound rather than prove:
the 8 KB threadgroup allocation limits how many threadgroups a core can hold, which costs
nothing while the grid is smaller than a couple of threadgroups per core and costs a lot once it
is not. Two observations point at it and one narrows it:

- the 27B at 256 threadgroups (`n_splits` 64) also loses, 0.966x, so the effect is not unique to
  the 4B;
- the 4B at the SAME 256 threadgroups loses much harder (0.644x), and the difference between the
  two shapes there is `head_dim` 128 against 256. At `head_dim` 128 only half the 256 threads
  own an output dim, so the online kernel's per-tile V phase runs at half occupancy while its
  threadgroup-memory footprint is unchanged. That is exactly the phase-4 thread waste step 4 of
  the decode plan addresses, which makes step 4 a PREREQUISITE for this kernel on that shape
  rather than an independent nicety;
- barrier count does not explain it: the 4B at seq 262144 with `n_splits` 16 folds 64 tiles
  (about 1400 barriers) and WINS 1.043x, while `n_splits` 256 folds 4 tiles (about 88 barriers)
  and loses 0.690x.

Two follow-ups, in order of expected value:

1. **Size the threadgroup buffer to the actual group**, `R * SG_TG * 4` bytes via a host-provided
   `[[threadgroup(0)]]` length instead of the static worst-case 8 KB. That is 4 KB for the 4B's
   `R = 4` and 6 KB for the 27B's `R = 6`. It was deliberately NOT done in this task: the failure
   mode of a wrong length is threadgroup memory corruption, and the task was written with the GPU
   held, so the safe form was the one that cannot be under-allocated.
2. **Step 4 of the decode plan** (phase-4 thread waste), which is where the `head_dim < SG_TG`
   half of the loss lives.

`SURGE_ATTN_SPLITK_ONLINE` therefore stays **OFF by default**, and so does `attn_splitk_gqa`. On
the evidence above the online kernel is a win on the shape the project is measured on and a loss
on the other real shape, so turning it on would need a policy that admits only the first, and
that policy would be fitted to two shapes at four depths. Not this task's call.

### The env var is REJECTED, not ignored, on a bad value

`SURGE_ATTN_SPLITK_ONLINE` accepts exactly `0` and `1`; anything else makes
`sg_gpu_state_new` return an error. This follows P2.6's rule rather than the warn-and-default
rule `SURGE_ATTN_SPLITK` and `SURGE_ATTN_SPLITK_GQA` use, and for the same reason: the gate for
this switch is an A/B, and an A/B whose on-arm was silently never turned on passes perfectly.
`mini_f16_splitk_online_policy` (`tests/test_gpu_fwd.c`, no forward passes, so it costs nothing)
gates that plus the `head_dim <= SG_TG` bound, the inherited group band, the inherited P2.7
floor, and the mutual exclusion with the four-pass GQA arm.

### What the online path would save if it became the only partial

`splitk_scratch` is sized `n_heads * (max_ctx + n_splits)` floats, which at the 27B's 24 heads
and 262144 context is about **25 MB**, the largest scratch surge allocates for decode. This task
does not delete it or change `splitk_sizes`: the four-pass kernels still need it, and the online
path simply never mentions it.

### Not measured

- Whether the GQA split policy (`splitk_gqa_n_splits`, cap 256) is still the right one for the
  streaming kernel. The online arm dispatches with it because it has the same grid and the same
  per-threadgroup reuse; the sweep above shows the online kernel's best `n_splits` on the 27B at
  seq 262144 is 512 or 1024 (1.124x, 1.142x) rather than 256, so a re-fit is open.
- The 27B end to end. The real-model A/B above is the 4B, the checkpoint the brief names; the
  27B-Q8_0 run is a separate manual gate.
- Occupancy directly (no Metal capture was taken), which is why the explanation above is labelled
  a hypothesis.
