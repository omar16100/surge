#!/usr/bin/env bash
# M3.4 Q8_0 forward correctness gate (manual; NOT part of `make check`).
#
#   (A) surge Metal Q8_0 vs surge CPU-ref Q8_0, teacher-forced: 100% top-1.
#   (B) surge greedy vs llama.cpp (llama-simple) greedy on the same GGUF:
#       no early systematic divergence.
#
# Slow: gate A runs one full scalar-C 27B CPU forward (~10 min). Needs the
# 28 GB GGUF and llama.cpp (llama-simple, llama-tokenize). Re-runnable:
#   make gate            (regression: compares fresh Metal digest vs frozen)
#   make gate FREEZE=1   (re-freeze the committed Metal digest fixtures)
set -uo pipefail

GGUF="${1:-/Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf}"
PY="${2:-/Users/macmini/models/dsv4-venv/bin/python}"
FREEZE="${FREEZE:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ ! -f "$GGUF" ]; then echo "gate: model $GGUF not found" >&2; exit 2; fi
# The Metal gates must not fight a running GPU benchmark for the device.
if pgrep -f "bench_niah|llama-bench" >/dev/null; then
  echo "gate: a GPU benchmark is running; wait for it to finish" >&2; exit 2
fi

freeze_flag=""
[ "$FREEZE" = "1" ] && freeze_flag="--freeze"

echo "=== M3.4 gate A: surge Metal Q8_0 vs surge CPU-ref Q8_0 (teacher-forced) ==="
"$PY" tools/tf_compare_q8.py --gguf "$GGUF" $freeze_flag; A=$?
echo
echo "=== M3.4 gate B: surge greedy vs llama.cpp greedy ==="
"$PY" tools/xcheck_llama_q8.py --gguf "$GGUF"; B=$?
echo
if [ "$A" -eq 0 ] && [ "$B" -eq 0 ]; then
  echo "M3.4 GATE: PASS (A top-1 100%, B no early systematic divergence)"
  exit 0
fi
echo "M3.4 GATE: FAIL (A exit $A, B exit $B)"
exit 1
