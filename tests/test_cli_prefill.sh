#!/usr/bin/env bash
# test_cli_prefill.sh - Gate 4 regression (Task M5.6).
#
# `surge -p/--ids ... -n N` must produce BYTE-IDENTICAL gen_ids whether the
# prompt is ingested by the chunked Metal prefill (the default) or the serial
# one-token path (--no-prefill). test_gpu_prefill.c already proves that parity at
# the sg_gpu_prefill / sg_gpu_forward API level; this checks the *CLI* branching
# in cli_metal.c (the flag parse, the do_prefill gate, the shared argmax_f32),
# which a C test does not exercise -- so a change there that broke the two paths
# apart would still leave `make check` green without this test.
#
# It drives the actual `surge` binary on the mini hybrid fixture, both formats
# (gguf + safetensors twin, reached via --ids so no tokenizer is needed), across
# chunk sizes {1, 2, 1024} plus chunk 5 (which splits the 12-token prompt into
# 5,5,2 -- a partial final chunk) and a single-token prompt. For every case it
# compares the `gen_ids:` line from the default (prefill) run against the
# --no-prefill run and FAILS on any difference (or on an empty/missing line).
#
# Wired into `make check` only when Metal is enabled; under -DSURGE_NO_METAL
# (how `make debug` runs) the Makefile omits it, matching the Metal C tests.
# Skips cleanly (exit 0) if surge is missing or the machine has no Metal device.
#
# Usage: bash tests/test_cli_prefill.sh [path-to-surge]   (default ./surge)
set -u

SURGE="${1:-./surge}"
GGUF="tests/fixtures/mini_fwd/model.gguf"
ST="tests/fixtures/mini_fwd"

if [ ! -x "$SURGE" ]; then
    echo "SKIP test_cli_prefill: $SURGE not built" >&2
    exit 0
fi

# Probe: skip if this machine has no Metal device (the Metal C tests skip the
# same way). Any other nonzero exit is a real failure, surfaced below.
probe_err="$("$SURGE" "$GGUF" --ids 1 -n 0 --quiet 2>&1 >/dev/null)"
probe_rc=$?
if [ "$probe_rc" -ne 0 ]; then
    case "$probe_err" in
        *"no Metal device"*)
            echo "SKIP test_cli_prefill: no Metal device" >&2
            exit 0 ;;
        *)
            echo "FAIL test_cli_prefill: probe run failed (rc=$probe_rc): $probe_err" >&2
            exit 1 ;;
    esac
fi

# The `gen_ids:` payload from one surge run.
gen_ids() { "$SURGE" "$@" --quiet 2>/dev/null | sed -n 's/^gen_ids: //p'; }

fail=0
ncase=0

# 12-token mini prompt (chunk 5 -> 5,5,2 forces a partial final chunk); a
# single-token prompt as the smallest case. --ids works for both formats.
IDS12="33,6,21,4,13,11,19,38,11,6,6,21"
IDS1="33"

# $1=label $2=model $3=n $4=chunk $5=ids
check_pair() {
    local label="$1" model="$2" n="$3" chunk="$4" ids="$5"
    ncase=$((ncase + 1))
    local pf np
    pf="$(gen_ids "$model" --ids "$ids" -n "$n" --chunk "$chunk")"
    np="$(gen_ids "$model" --ids "$ids" -n "$n" --no-prefill)"
    if [ -z "$pf" ]; then
        echo "FAIL $label: prefill produced no gen_ids line" >&2
        fail=1
    elif [ "$pf" != "$np" ]; then
        echo "FAIL $label: prefill gen_ids != --no-prefill" >&2
        echo "  prefill    (chunk $chunk): $pf" >&2
        echo "  no-prefill           : $np" >&2
        fail=1
    else
        echo "  ok $label: gen_ids=$pf" >&2
    fi
}

for model in "$GGUF" "$ST"; do
    tag="gguf"; [ "$model" = "$ST" ] && tag="st"
    check_pair "$tag chunk1  12tok"          "$model" 16 1    "$IDS12"
    check_pair "$tag chunk2  12tok"          "$model" 16 2    "$IDS12"
    check_pair "$tag chunk5  12tok(partial)" "$model" 16 5    "$IDS12"
    check_pair "$tag chunk1024 12tok"        "$model" 16 1024 "$IDS12"
    check_pair "$tag chunk1  1tok"           "$model" 8  1    "$IDS1"
done

# (C) M5.7 context-cap guard: an explicit --max-ctx smaller than the prompt must
# be REJECTED (nonzero exit + a clear message), never silently enlarged. The
# guard fires before any GPU work, so this needs only the tiny mini gguf and a
# short --ids -- no big model. 8 prompt tokens against --max-ctx 4.
ncase=$((ncase + 1))
cap_out="$("$SURGE" "$GGUF" --ids 1,2,3,4,5,6,7,8 --max-ctx 4 --quiet 2>&1)"
cap_rc=$?
if [ "$cap_rc" -eq 0 ]; then
    echo "FAIL cap-guard: surge accepted an 8-token prompt with --max-ctx 4 (rc=0)" >&2
    fail=1
elif ! printf '%s' "$cap_out" | grep -qiE "max-ctx|context cap|exceeds"; then
    echo "FAIL cap-guard: nonzero exit but no clear cap message: $cap_out" >&2
    fail=1
else
    echo "  ok cap-guard: over-cap prompt rejected (rc=$cap_rc)" >&2
fi

if [ "$fail" -ne 0 ]; then
    echo "test_cli_prefill: FAILED ($ncase cases)" >&2
    exit 1
fi
echo "test_cli_prefill: $ncase cases, prefill gen_ids == --no-prefill (byte-identical)" >&2
exit 0
