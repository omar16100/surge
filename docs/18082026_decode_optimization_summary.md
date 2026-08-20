# surge decode optimization: P2.3 through P4.0

Category: Reference (dated summary). Authored 2026-08-18. Per-task gate docs remain the detailed
record; this is the end-to-end story, the measured numbers, and the decisions.

## Where this started

surge completed a 262,144-token prefill of Qwen3.6-27B-Q8_0 on 2026-08-13 (task B7): 31.4 hours,
recall 8/8, only 2 percent of GPU samples clamped. It was the first C engine to do so on this
machine. It was also **slow**: decode 0.537 tok/s, against mlx-lm's 7.58 and llama.cpp's 5.10 on
the identical prompt.

The cause was identified in the decode plan and confirmed in code. `k_attn_decode_f16` dispatched
exactly `n_heads` threadgroups, so on a 24-head model only 24 of this machine's 80 GPU cores had
work, with one threadgroup walking the entire KV sequence. Three further defects compounded it:
GQA re-read the same K/V slice once per query head (4x to 6x redundant traffic), scores were
materialized to device memory and walked three more times, and half the threadgroup idled in the
V accumulation phase whenever `head_dim < SG_TG`.

## What was built

| task | change | measured effect |
|---|---|---|
| P2.3 | split-K (flash decoding) wired into `enc_attn` | 9.8x on real 4B decode at 32825 tokens (1.74 to 17.12 tok/s) |
| P2.4 | one threadgroup per GQA GROUP instead of per query head | a further **1.74x** at the 27B 262144 shape |
| P2.5 | GQA-specific split policy, `clamp(min(seq/SG_TG, 256), 4, 1024)` | mean regret 0.43 percent against per-point optimum, versus 3.1 percent for the inherited policy |
| P2.6 | greedy-token gate where the two split policies diverge | closed the last correctness gap |
| P2.7 | occupancy floor, `n_splits * n_kv_heads >= 128` | removed a measured 0.822x regression at short context |
| P2.8 | online softmax, one streaming pass, no device score row | 27B 1.013x to 1.114x; **4B 0.690x to 0.984x**, so shipped OFF |
| P2.9 | key groups in the online V phase | 4B 262144 went 0.690x to **1.015x**; 27B unmoved |
| P3.0 | decode pacing with clamp detection (`src/sched.c`) | mechanism correct and gated; effect NOT demonstrated |
| P4.0 | flip `attn_splitk_gqa` ON by default | user-approved 2026-08-18 |

Cumulative on the kernel that matters: **28.3x** over the original decode attention at the 27B
`24h/4kv/256d` shape at 262144 (119472 us to 4217 us).

## The measurements that changed decisions

### The split policy is a cap, not a rescaling

The intuitive rule, "GQA wants half the per-head splits", fits the two deepest data points and is
the **worst** of four candidates at 13.4 percent worst-case regret, because the optimum saturates
with depth rather than scaling with it. Measured across both shapes at four depths:

| policy | mean regret | worst |
|---|---|---|
| `seq/SG_TG` (inherited) | 3.1% | 7.3% |
| **`min(seq/SG_TG, 256)`** | **0.43%** | **2.73%** |
| `min(seq/SG_TG, 512)` | 1.5% | 4.2% |
| `seq/(2*SG_TG)` (the intuition) | 3.8% | 13.4% |

A two-point extrapolation would have shipped a 13 percent regression.

### The occupancy floor is a threadgroup count, not a sequence length

The GQA kernel collapses `repeat` threadgroups into one, dividing the grid by `repeat`, so at
short context it starves the GPU: at seq 2048 the 27B gets 8 * 4 = 32 threadgroups against 80
cores and measures **0.822x**, i.e. slower than the kernel it replaces.

A `seq` threshold cannot express the fix. The per-shape crossovers are 1.7x apart in seq
(5120-6144 on the 27B, 3072-4096 on the 4B) and in the OPPOSITE order from the threadgroup view,
because the 4B has twice the kv heads. One threadgroup threshold separates every measured point;
one seq threshold must either admit a 27B loss or reject 4B wins.

### Online softmax needed the V-phase fix to be viable

P2.8's storage problem: a per-thread `acc[R][head_dim]` is 2048 floats at R=8 and head_dim=256 and
simply does not exist. Solved by attaching each piece of state to the role that makes it small:
`m` and `s` come off a fold tree so they are uniform (2R registers), and `acc[d]` has exactly one
owner thread (R registers, not R*head_dim). Streaming state is 4R floats.

It shipped 27B-positive and 4B-negative. P2.9 diagnosed why (at head_dim 128 only half the 256
threads own an output dim in the V phase) and fixed it with key groups, taking the 4B at 262144
from 0.690x to 1.015x while leaving the 27B untouched. P2.9 also **overturned P2.8's own leading
hypothesis**: the 8 KB threadgroup allocation that P2.8 blamed was measured with a diagnostic 4 KB
metallib and is worth only 0.4 to 1.8 percent.

### Decode pacing: built, correct, and unsupported by evidence

Phase 3 assumed decode needs "the treatment prefill got in B8", meaning rest harder and the clock
returns. Against 376 burst pairs in existing telemetry that correlation is **r = +0.017**, i.e.
absent. So a confirmed clamp drives no rest schedule by default and escalation is opt-in.

Throughput was attempted twice and no claim is made. A counterbalanced 8-run experiment showed
pacing costs 10.1 percent of wall, exactly its configured duty, and did not raise compute
throughput. A second experiment was uninterpretable: throughput rose monotonically 8.19, 10.15,
19.10, 22.07 tok/s, a 2.7x drift no four-run counterbalancing removes.

The mechanism is the instrument that lets the question be asked. Building it was right; expecting
a win from it is not currently supported.

## Correctness, and why it is trusted

Every step held byte-exact greedy tokens, which is surge's standard, and the gates are
mutation-proved rather than merely green.

- `make check` grew from 85319 (P2.3) to **87600 checks, 0 failures**; `make debug` 83614, 0
  failures, 0 sanitizer diagnostics.
- **Byte-identity where it is claimable**: the GQA partial is byte-identical to the per-head
  partial at any fixed `n_splits`, over 100 reruns.
- **Byte-identity is correctly NOT claimed where it does not apply**: online softmax reorders sums,
  and above seq 65791 the GQA and per-head split policies pick different `n_splits`. Those are
  gated on CPU-oracle accuracy (worst rel 2.109e-06 over 15 shapes x 7 splits x 2 oracles),
  determinism, and real-model greedy tokens.
- **Positive controls, not equality assertions.** The recurring failure in this project is a gate
  that passes while the code under test never runs, caught four separate times. The controls now
  count dispatches (`577 GQA dispatches with the switch on, 0 per-head`), pin divergence
  boundaries from both sides, and assert wall clock actually elapsed.

## What is still open

- **surge's published 256K row is stale.** It records 0.537 tok/s decode and predates every task
  above. Refreshing it costs about 31 hours that are roughly 99 percent prefill, which none of this
  work improved. Attention-only projection is about 14.8 tok/s; the honest band is 8 to 15, and it
  has never been measured end to end at that depth. At 21120 tokens the measured end-to-end gain
  was only 1.07x, because attention is not the dominant cost there.
- **Prefill is now the gap.** surge prefills at 2.99 tok/s compute against llama.cpp's 95.6 on the
  same prompt, a 32x difference that no decode work touches. That is milestone M4 territory.
- **`attn_splitk_online` stays OFF.** It is a 9 to 12 percent win on the 27B but still 1.9 to 5
  percent behind best-vs-best on the 4B.
- **File size.** `src/kernels.metal` is 2548 lines and `src/metal.m` 4641, both over the ~2000
  house guideline. The recommended cut is `src/kernels_splitk.metal` along the existing seam,
  after further split-K work and before the next task touching a non-split-K kernel there.
  (Both recommended cuts have since been done: task R1 on 2026-08-18 split
  `src/kernels_splitk.metal` out, leaving `src/kernels.metal` at 1295, and task R2 on
  2026-08-20 split `src/metal_prefill.m` out, leaving `src/metal.m` at 3547, still over the
  guideline. The `4616` figure this bullet originally carried was already wrong when written:
  `src/metal.m` was 4641 at the time. Corrected in R2 fix round 1, 2026-08-20, because this
  was the last place in the tree still repeating the stale 4616 that R2 removed from
  `src/sched.c`. Dated summary docs are otherwise left as historical records; this one is
  annotated rather than rewritten.)
- **A real defect found in passing**: B5's `bos-toggle` case hard-fails when
  `SURGE_BENCH_TOK_MODEL` points at the 4B GGUF, which carries no `tokenizer.ggml.bos_token_id`.
  Deserves its own task.
- **A known flake**: `test_cli_bench` b6 `check2`, a 3 percent bar on a timing ratio, fires when
  run immediately after heavy GPU work and passes after about 45 seconds idle. Reproduced at a
  parent commit with changes stashed, so it is pre-existing.

## Reproduction

```sh
cd /Users/macmini/projects/surge
make check                                   # full suite
make bench-splitk                            # build the kernel A/B harness
./tests/bench_splitk.bin --reps 20 --seqs 8192,32768,131072,262144        # per-head arm
./tests/bench_splitk.bin --reps 20 --seqs 8192,32768,131072,262144 --gqa  # GQA arm
# real-model greedy A/B (the gate that protects users)
SURGE_ATTN_SPLITK_GQA=0 ./surge <gguf> -p "$(cat PROMPT)" -n 64
SURGE_ATTN_SPLITK_GQA=1 ./surge <gguf> -p "$(cat PROMPT)" -n 64   # gen_ids must match
```

Use `--reps 20` or more for anything you intend to conclude from. At default repetitions two
crossover findings in this work reversed under repetition, because the effects are a few percent
and at or below this machine's noise floor.

## Related

- `/Users/macmini/projects/surge/docs/16082026_splitk_decode_gate.md` (P2.3)
- `/Users/macmini/projects/surge/docs/17082026_splitk_gqa_threadgroups.md` (P2.4 to P2.7, and P2.9)
- `/Users/macmini/projects/surge/docs/18082026_decode_pacing.md` (P3.0)
- `/Users/macmini/projects/llm-rnd/docs/benchmarking_methodology.md` (how these numbers are made trustworthy)
- `/Users/macmini/projects/llm-rnd/docs/18082026_256k_project_report.md` (the comparison surge participates in)
