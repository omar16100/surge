# Split-K decode attention: wiring, policy and gates (Task P2.3)

What changed, why the numbers are what they are, and how to re-run every gate. Companion
to `docs/c4model.md` (architecture) and the SDD report
`.superpowers/sdd/2026-08-09-surge-m3-m5/task-P2.3-report.md`.

## What changed

`sg_gpu_forward`'s per-layer attention (the fp16 KV path, which is the default) now
dispatches the split-K pair `k_attn_decode_splitk_partial` + `k_attn_decode_splitk_combine`
instead of the single-threadgroup-per-head `k_attn_decode_f16`, inside the SAME open
command buffer as the rest of the token's work. The two kernels themselves are unchanged
since Task P2.2; this task is the wiring, the split-count policy, and the gates.

New encoder helper `enc_attn_splitk` (`src/metal.m`). It has to encode the partial's grid
by hand: that grid is 2D (x = split, y = query head) and `gpu_grid`'s (groups, elems) pair
(`sg_gpu_grid` in `src/metal_validate.m` since tasks R3 and R4; `gpu_grid` in
`src/metal.m` when this was written) deliberately cannot express two group dimensions. There is no wait between the two
dispatches; the default `MTLDispatchTypeSerial` gives an implicit barrier, which is
exactly the property the one-shot entry points had to buy with a commit-and-wait.

## The split count

    n_splits = clamp(seq / SG_TG, 4, 1024)          SG_TG == 256

MEASURED, not chosen. `make bench-splitk` (Task P2.3a's harness) swept
`n_splits` over {1..1024} at seq 8192 / 32768 / 131072 / 262144 on the real 27B decode
shape (24 heads, 4 kv, head_dim 256) and the real 4B dense shape (32 heads, 8 kv, head_dim
128); the fastest value was exactly `seq / SG_TG` in seven of those eight cells, which is
the top of the occupancy band documented on the kernel: give every split exactly SG_TG keys
so no lane of the threadgroup idles. At 262144 the pair beat the incumbent by 15.9x (27B
shape) and 21.9x (4B shape).

A re-measurement on 2026-08-16 at seq 32768 confirms both the ratio and the optimum:

| shape | incumbent | split-K @ n_splits 128 | speedup |
|---|---|---|---|
| 27b_decode_24h_4kv_256d | 15193.45 us | 1393.00 us | 10.907x |
| 4b_dense_32h_8kv_128d | 13783.10 us | 1035.20 us | 13.314x |

128 is `32768 / 256`, i.e. the closed form, and it was the best of {4, 8, 16, 32, 64, 128}
for both shapes in that run.

The eighth cell is a real counterexample, now reproduced twice. In a fresh P2.3 sweep at seq
8192 the 27B shape's best was n_splits 16 (5.226x) against the closed form's 32 (4.965x),
and the P2.3 reviewer's independent sweep (`--reps 20`, fans on firmware auto) read 16 at
5.447x against 32 at 5.180x. Two runs agreeing on a roughly 5 percent gap in the same
direction is not run-to-run noise, so P2.3a's "optimum at every measured cell" is wrong as
written. The same reviewer saw the 27B curve go non-monotonic at 32768 as well (16 at
9.699x above 32 at 9.129x and 64 at 9.123x, before 128 at 10.783x wins the cell); the 4B
dense shape at 8192 does peak at the closed form.

The closed form remains the policy, and that is a decision rather than an oversight. It is
the TOP OF THE OCCUPANCY BAND the kernel header derives, not a constant fitted to this
table, so it extrapolates to shapes and depths nobody swept. The curve is shallow near the
optimum, so being one power of two off costs single-digit percent. And a single 5 percent
outlier at one shape at one depth, where attention is not the decode bottleneck, does not
pay for a shape-specific special case that would need its own sweep and its own regression
gate to stay true. Anyone extending the policy re-measures rather than assuming.

## The fallback, and why the threshold is 1024

Below `seq == SG_TG * 4 == 1024` the clamp's FLOOR is what binds rather than `seq / SG_TG`,
so splits come out shorter than SG_TG keys and lanes start idling. Decode keeps
`k_attn_decode_f16` there. The crossover was measured (`--seqs`, added to
`tests/bench_splitk.c` for exactly this question) rather than argued:

| seq | 27B shape speedup | 4B dense shape speedup |
|---|---|---|
| 256 | 0.710x | 0.689x |
| 512 | 1.021x | 0.948x |
| 1024 | 1.880x | 1.272x |
| 2048 | 2.007x | 2.067x |
| 4096 | 3.434x | 3.226x |
| 8192 | 5.226x | 5.222x |

Split-K is a REGRESSION at 256 keys (about 1.4x slower), roughly break-even to slightly
slower at 512, and a clear win from 1024 up. The P2.3 reviewer re-measured the 512 row at
0.924x / 0.948x, i.e. a small regression on BOTH shapes rather than the wash this doc first
called it; that run had fans on firmware auto, so limiter shape versus a real measurement
disagreement is unresolved, and either way it justifies the 1024 threshold slightly better
rather than worse. The threshold therefore sits exactly where the policy's own clamp stops
binding, and no configuration is used outside the range where it was measured.

`SURGE_ATTN_SPLITK=0` pins the incumbent for every step. That is what makes an A/B
possible, and it is why `k_attn_decode_f16` is kept reachable rather than deleted. The f32
KV path (`SURGE_KV_DTYPE=f32`) always uses the incumbent: the split-K kernels read
half-typed separate K and V buffers, which only the fp16 cache has.

## The scratch hazard, closed structurally

`sg_gpu.scratch` is one process-wide allocation that `k_attn_decode`, `k_attn_decode_f16`,
`k_attn_prefill` and the decode encoder all bind at offset 0, and it is only ever grown,
never partitioned. Dispatching the split-K partial from inside the same open command
buffer is precisely the situation where sharing it would matter. Two structural
properties, not comments, keep that from being a question:

1. the partial binds the DEDICATED `sg_gpu.splitk_scratch` (added by P2.2's fix round),
   and `enc_attn_splitk` contains no reference to `g->scratch` at all;
2. `splitk_scratch` and the m/s/acc partial buffers are sized ONCE, in
   `sg_gpu_state_new`, for the worst case at `max_ctx`, so nothing can allocate, grow or
   release a buffer mid-encode. The bound used is
   `n_splits * ceil(seq/n_splits) <= seq + n_splits - 1`, with `n_splits` nondecreasing in
   `seq`, so `n_heads * (max_ctx + max_splits)` floats covers every step.

The m/s/acc buffers are shared by every full-attention layer and every step, which is safe
for the same serial-dispatch reason the existing `g->b_ctx` and `g->scratch` sharing is.

## Gates and how to re-run them

Before ANY of these, confirm the GPU is free:
`pgrep -f "bench_niah|phase0|surge-bench|llama-server|llama-cli"` must print nothing.
Never run `make` while a surge binary is live: it rebuilds `src/kernels.metallib` under
the running process.

| gate | command | result on 2026-08-16 |
|---|---|---|
| Full suite | `make check` | 85319 checks, 0 failures (baseline at the parent commit: 85306; the 13 new checks are the split-K decode subtest) |
| Sanitizers | `make debug` | rc 0, 83953 checks, 0 failures, 0 sanitizer diagnostics; identical count to a clean worktree at the parent commit |
| Wiring, in `check` | `./tests/test_gpu_fwd.bin` | 577/577 positions above seq 1024 differ from the incumbent, 0 below; worst abs logit delta 9.537e-07 vs smallest top1-top2 margin 1.385e-03; argmax 1600/1600 agree |
| 100x determinism | `SURGE_SPLITK_DET_REPS=100 ./tests/test_gpu_fwd.bin` | 160000/160000 (position, rerun) pairs byte-identical |
| Kernel A/B | `./tests/bench_splitk.bin --seqs 32768` | see the table above |
| Short-seq crossover | `./tests/bench_splitk.bin --seqs 256,512,1024,2048,4096,8192` | see the table above |
| Real-model greedy | `SURGE_ATTN_SPLITK={0,1} ./surge <4B gguf> -p "$(cat PROMPT)" -n 64` | gen_ids byte-identical between the two modes at 3199 and 32825 prompt tokens |

The wiring subtest is non-vacuous by construction and was mutation-proved: forcing
`splitk_use` to return 0 makes it fail with "0 of 577 positions differ, so the split-K
path never ran", and swapping the partial's two grid axes makes it fail with a worst
logit delta of 4.035e-01 and 27 of 1600 argmaxes wrong. The subtest deliberately runs
1600 positions so that `n_splits` (6) differs from the fixture's `n_heads` (4): a swapped
axis is invisible whenever those two are equal.

## Measured effect on real decode

Real 4B dense Q8_0 (`Qwen3-4B-Instruct-2507-Q8_0.gguf`, 36 full-attention layers), greedy,
identical prompt and binary, only `SURGE_ATTN_SPLITK` changed:

| prompt tokens | incumbent decode | split-K decode |
|---|---|---|
| 3199 | 14.80 tok/s | 41.08 tok/s |
| 32825 | 1.74 tok/s | 17.12 tok/s (7.40 tok/s on a later, more thermally clamped repeat) |

Generated ids were byte-identical in every one of those runs. The wide spread on the
repeat is the M3 firmware clock clamp, not the kernels: prefill throughput on the same
machine fell from 28.86 to 13.92 tok/s across the three consecutive 32825-token runs even
though prefill code is untouched. The thermally matched, same-run comparison is the
kernel-level table above.
