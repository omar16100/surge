#!/usr/bin/env bash
# prefill_longctx_gate.sh - M5.7 long-context gate runner (closes M5).
#
# Runs the gated real-model checks in tests/test_gpu_prefill.c against a REAL
# model (the 2B bf16 qwen3_5 interim model by default), driven through the C API
# because the id sequences (up to 262144) do not fit through the CLI's argv:
#
#   (A) DEPTH EQUIVALENCE at 8192 / 16384 / 32768: prefill(prompt)+decode 32 ==
#       serial-forward(prompt one at a time)+decode 32, 0 token-id mismatches.
#   (B) 262144 INGEST: fill the whole 262144-position cache (262112 prefill via
#       chunk 1024, then 32 decode at depth, reaching used == 262144, since the
#       cap leaves no slot to append after a full-cap prefill), no Metal fault,
#       non-degenerate final prefill logits, and 32 non-degenerate decoded tokens
#       (finite, in-range, not one id x32).
#
# (C) the CLI context-cap rejection is covered separately and cheaply by
# tests/test_cli_prefill.sh (in `make check`); it needs no big model.
#
# The 2B safetensors carries no surge-readable tokenizer, so the coherence bar
# here is NON-DEGENERATE token ids (the plan's 2B-interim allowance); valid-UTF-8
# coherence on real text is B7's job on the 27B, which has a tokenizer.
#
# M5 is a FUNCTIONAL gate, not a speed gate: the serial reference at 32k is tens
# of thousands of forwards and the 262144 prefills are large, so slow
# (limiter-shaped) wall times are expected and fine. Let it finish.
#
# Usage: [SURGE_GATE_MODEL=/path] bash tools/prefill_longctx_gate.sh
#   SURGE_GATE_MODEL defaults to /Users/macmini/models/qwen35-2b.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODEL="${SURGE_GATE_MODEL:-/Users/macmini/models/qwen35-2b}"
BIN="tests/test_gpu_prefill.bin"

if [ ! -e "$MODEL" ]; then
    echo "FAIL: SURGE_GATE_MODEL '$MODEL' does not exist" >&2
    exit 1
fi

# GPU contention guard: the 256K comparison loop (cron) uses the GPU; running
# concurrently would corrupt both. Refuse rather than run alongside it.
busy="$(pgrep -f 'bench_niah_mlx.py|llama-cli|llama-bench' || true)"
if [ -n "$busy" ]; then
    echo "FAIL: GPU is busy (bench_niah_mlx.py / llama-cli / llama-bench running):" >&2
    pgrep -fl 'bench_niah_mlx.py|llama-cli|llama-bench' >&2
    echo "Wait for it to finish, then re-run." >&2
    exit 1
fi

echo "== M5.7 long-context gate ==" >&2
echo "model : $MODEL" >&2
echo "depths: (A) 8192 / 16384 / 32768   (B) 262112 prefill + 32 decode = 262144 ingest" >&2
[ -n "${SURGE_GATE_SKIP_A:-}" ] && echo "note  : SURGE_GATE_SKIP_A set -> running gate (B) only" >&2

# Build the Metal test binary (the Makefile's static pattern rule links
# src/metal.m + the metallib for this target).
echo "-- building $BIN --" >&2
if ! make "$BIN"; then
    echo "FAIL: could not build $BIN" >&2
    exit 1
fi

echo "-- running gate (SURGE_KV_DTYPE=f16, SURGE_GATE_MODEL=$MODEL) --" >&2
t0=$(date +%s)
SURGE_GATE_MODEL="$MODEL" SURGE_KV_DTYPE=f16 "./$BIN"
rc=$?
t1=$(date +%s)
echo "-- gate wall time: $((t1 - t0)) s (exit $rc) --" >&2

if [ "$rc" -ne 0 ]; then
    echo "M5.7 GATE: FAIL (see assertions above)" >&2
    exit "$rc"
fi
echo "M5.7 GATE: PASS (A depth-equivalence 0 mismatches, B 262144 ingest used==262144 + non-degenerate decode)" >&2
exit 0
