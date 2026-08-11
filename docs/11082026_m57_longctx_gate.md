# M5.7 long-context gate (closes M5)

How to run and interpret the M5.7 gate: it validates the chunked prefill path
(M5.2-M5.6) on a REAL model at depth, proves surge can ingest 262144 tokens, and
enforces the context cap. This closes milestone M5.

## What it checks

Three gates, on the real 2B bf16 `qwen3_5` interim model
(`/Users/macmini/models/qwen35-2b`, the M1/M2 gate model), driven through the C
API because the id sequences (up to 262144) do not fit through the CLI's argv.
Token-id sequences are generated in the test by a deterministic 64-bit LCG in
`[0, vocab)`, so the prefill and serial paths ingest identical ids.

- **(A) DEPTH EQUIVALENCE** at 8192 / 16384 / 32768: `sg_gpu_prefill(prompt)` then
  greedily decode 32 tokens must produce the SAME 32 token ids as feeding the
  prompt one token at a time through `sg_gpu_forward` (the serial oracle) then
  decoding 32. 0 token-id mismatches at each depth, same `argmax_f32` both sides,
  state reset between the two runs. This is the strongest form: byte-exact greedy
  tokens.
- **(B) 262144 INGEST**: fill the whole 262144-position cache and decode at ~256K
  depth. `SG_KV_CAP_MAX` == 262144 leaves NO cache slot to append after a
  full-cap prefill, so the context is filled as **262112 prefill + 32 decode**,
  which reaches the used counter == 262144 while still exercising decode at depth.
  Asserted: `sg_gpu_used(g) == 262144`, no Metal fault, non-degenerate final
  prefill-position logits (finite, not all-equal, argmax in range), and 32
  non-degenerate decoded tokens (finite each step, ids in range, not one id
  repeated 32x). The 2B safetensors carries no surge-readable tokenizer, so the
  interim coherence bar here is NON-DEGENERATE token ids; valid-UTF-8 coherence
  on real text is B7's job on the 27B (which has a tokenizer).
- **(C) CAP ENFORCEMENT**: the `surge` CLI rejects a prompt whose token count
  exceeds an explicit `--max-ctx` with a clear message and a nonzero exit, rather
  than silently enlarging the cache or overflowing it. Covered cheaply (no big
  model) by `tests/test_cli_prefill.sh` inside `make check`.

## How to run

    # (A) + (B), full gate:
    SURGE_GATE_MODEL=/Users/macmini/models/qwen35-2b bash tools/prefill_longctx_gate.sh

    # (B) only (skip the already-proven 32k serial reference, ~40 min under the
    # box's power limiter):
    SURGE_GATE_MODEL=/Users/macmini/models/qwen35-2b SURGE_GATE_SKIP_A=1 \
        bash tools/prefill_longctx_gate.sh

The gate is LIVE GPU and env-gated: without `SURGE_GATE_MODEL` the C test
(`tests/test_gpu_prefill.c`) SKIPs cleanly, so `make check` stays mini-only and
hermetic. The script refuses to run while `bench_niah_mlx.py` / `llama-cli` /
`llama-bench` hold the GPU (the 256K comparison cron), to avoid corrupting both.

M5 is a FUNCTIONAL gate, not a speed gate: the serial reference at 32k is ~33k
forwards and the 262144 prefill is a large quadratic-attention run, so slow
(power-limiter-shaped) wall times are expected and fine. Let it finish.

## Results (2026-08-11, M3 Ultra, 2B bf16)

Gate (A) depth equivalence, 0 token-id mismatches at every depth:

| depth | tokens match | prefill+decode | serial+decode |
|---|---|---|---|
| 8192  | 32/32 | 29.95 s  | 205.09 s   |
| 16384 | 32/32 | 62.26 s  | 1002.17 s  |
| 32768 | 32/32 | 278.31 s | 2495.97 s  |

Gate (B) 262144 ingest: prefill 262112 (chunk 1024) then decode 32, `used ==
262144`, no fault, non-degenerate final prefill logits (argmax 197) and 32
non-degenerate decoded tokens (2 distinct). Wall time: 262112 prefill 13063.3s
(~3.63 h), 32-token decode 25.5 s; total gate B 13089 s. The 32 decoded tokens
are all fed back (positions 262112..262143), filling the whole 262144-position
cache so `used` reaches exactly 262144.

Prefill throughput starts around 300 tok/s and falls with depth to ~20 tok/s
near 262K, because per-token attention cost grows linearly with the context
length (the whole prefill is O(n^2)) and the GPU is drawing only ~6-20 W (the
prefill kernel is memory-latency-bound at depth, so the 170 W firmware power
limiter is not even the binding constraint here). Speed is out of scope for M5
(a FUNCTIONAL gate); kernel efficiency at depth is M4's problem.

Gate (C): `surge <mini.gguf> --ids 1,2,3,4,5,6,7,8 --max-ctx 4` exits nonzero
with "the prompt exceeds the context cap" (asserted in `tests/test_cli_prefill.sh`).

The serial+decode times grow super-linearly (not 2x per doubling) because the
M3 firmware power limiter clamps the GPU after a few minutes of sustained load;
this is a known property of this box and does not affect the FUNCTIONAL result
(0 mismatches, used == 262144, non-degenerate output).
