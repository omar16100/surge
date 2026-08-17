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
