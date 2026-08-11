#!/usr/bin/env bash
# test_cli_bench.sh - Task B5 gate for the `surge-bench` CLI.
#
# Three checks, all on the mini hybrid fixture (fast, hermetic) except the
# optional BOS toggle, which needs a real tokenizer and is env-gated:
#
#   (1) gen_ids PARITY: `surge-bench <model> --ids ... -n N` decodes
#       BYTE-IDENTICAL token ids to `surge <model> --ids ... -n N`. This is
#       B5's primary gate: both binaries share sg_argmax_f32 (src/greedy.c) and
#       sg_gpu_prefill, so a change that let their drivers drift would break
#       here. Run on both fixture formats (gguf + safetensors twin, via --ids).
#
#   (2) VOID / exit 3: a missing or below-threshold GEMM gate, and an
#       out-of-window ingestion count, each make surge-bench emit a VOID row and
#       exit 3 (never 0) -- before any GPU load.
#
#   (3) BOS toggle (env-gated on SURGE_BENCH_TOK_MODEL, a GGUF with a real
#       tokenizer): `--bos` vs `--no-bos` change n_prompt_tokens by exactly 1.
#       Skipped cleanly when the env var is unset (make check stays hermetic).
#
# Wired into `make check` (Metal only) and `make bench-check`. Skips cleanly
# (exit 0) if the binaries are missing or the machine has no Metal device.
#
# Usage: bash tests/test_cli_bench.sh [surge] [surge-bench]
set -u

SURGE="${1:-./surge}"
BENCH="${2:-./surge-bench}"
GGUF="tests/fixtures/mini_fwd/model.gguf"
ST="tests/fixtures/mini_fwd"

if [ ! -x "$SURGE" ] || [ ! -x "$BENCH" ]; then
    echo "SKIP test_cli_bench: $SURGE or $BENCH not built" >&2
    exit 0
fi

# Probe: an admitted -n 0 run loads the GPU + prefills one token. A "no Metal
# device" error means skip; any other nonzero exit is a real failure.
probe_err="$("$BENCH" "$GGUF" --ids 1 -n 0 --gemm-gate-tflops 100 --max-ctx 8 --quiet 2>&1 >/dev/null)"
probe_rc=$?
if [ "$probe_rc" -ne 0 ]; then
    case "$probe_err" in
        *"no Metal device"*)
            echo "SKIP test_cli_bench: no Metal device" >&2
            exit 0 ;;
        *)
            echo "FAIL test_cli_bench: probe run failed (rc=$probe_rc): $probe_err" >&2
            exit 1 ;;
    esac
fi

fail=0
ncase=0

IDS12="33,6,21,4,13,11,19,38,11,6,6,21"
IDS1="33"

# ------------------------------------------------------------------
# (1) gen_ids parity: surge-bench == surge.
# ------------------------------------------------------------------
surge_gen()  { "$SURGE" "$@" --quiet 2>/dev/null | sed -n 's/^gen_ids: //p'; }
bench_gen()  { "$BENCH" "$@" --gemm-gate-tflops 100 --quiet 2>/dev/null | sed -n 's/^gen_ids: //p'; }

# $1=label $2=model $3=n $4=ids
check_parity() {
    local label="$1" model="$2" n="$3" ids="$4"
    ncase=$((ncase + 1))
    local s b
    s="$(surge_gen "$model" --ids "$ids" -n "$n" --max-ctx 512)"
    b="$(bench_gen "$model" --ids "$ids" -n "$n" --max-ctx 512)"
    if [ -z "$b" ]; then
        echo "FAIL $label: surge-bench produced no gen_ids line" >&2
        fail=1
    elif [ "$s" != "$b" ]; then
        echo "FAIL $label: surge-bench gen_ids != surge" >&2
        echo "  surge      : $s" >&2
        echo "  surge-bench: $b" >&2
        fail=1
    else
        echo "  ok $label: gen_ids=$b" >&2
    fi
}

for model in "$GGUF" "$ST"; do
    tag="gguf"; [ "$model" = "$ST" ] && tag="st"
    check_parity "$tag parity 12tok n16" "$model" 16 "$IDS12"
    check_parity "$tag parity 1tok  n8"  "$model" 8  "$IDS1"
done

# ------------------------------------------------------------------
# (2) VOID / exit 3 on a failed GEMM gate, missing gate, or bad ingestion.
# ------------------------------------------------------------------
# $1=label ; remaining args passed to surge-bench. Expects exit 3 + VOID row.
check_void() {
    local label="$1"; shift
    ncase=$((ncase + 1))
    local out rc
    out="$("$BENCH" "$GGUF" --ids "$IDS12" -n 4 --max-ctx 512 --quiet "$@" 2>/dev/null)"
    rc=$?
    if [ "$rc" -ne 3 ]; then
        echo "FAIL $label: expected exit 3, got $rc" >&2
        fail=1
    elif ! printf '%s' "$out" | grep -q "VOID"; then
        echo "FAIL $label: exit 3 but no VOID row: $out" >&2
        fail=1
    else
        echo "  ok $label: exit 3, VOID row" >&2
    fi
}

check_void "void: below-threshold gemm gate" --gemm-gate-tflops 10
check_void "void: missing gemm gate"
check_void "void: ingestion out of window" --gemm-gate-tflops 100 --expect-min 999 --expect-max 1000

# A passing gate + in-window ingestion must NOT VOID (exit 0, DONE row).
ncase=$((ncase + 1))
done_out="$("$BENCH" "$GGUF" --ids "$IDS12" -n 4 --max-ctx 512 --gemm-gate-tflops 100 --quiet 2>/dev/null)"
done_rc=$?
if [ "$done_rc" -ne 0 ]; then
    echo "FAIL admit: expected exit 0 for a passing gate, got $done_rc" >&2
    fail=1
elif ! printf '%s' "$done_out" | grep -q "DONE"; then
    echo "FAIL admit: exit 0 but no DONE row: $done_out" >&2
    fail=1
else
    echo "  ok admit: passing gate -> exit 0, DONE row" >&2
fi

# (finding 1) row.n_gen reports the ACTUAL produced count, consistent with the
# gen_ids line: a DONE run of -n 4 must report n_gen==4 in JSON and print 4 ids.
ncase=$((ncase + 1))
tmpjson="$(mktemp)"
ng_out="$("$BENCH" "$GGUF" --ids "$IDS12" -n 4 --max-ctx 512 --gemm-gate-tflops 100 --json "$tmpjson" --quiet 2>/dev/null)"
gen_count="$(printf '%s' "$ng_out" | sed -n 's/^gen_ids: //p' | awk -F, 'NF{print NF}')"
json_ng="$(sed -n 's/.*"n_gen":\([0-9][0-9]*\).*/\1/p' "$tmpjson")"
rm -f "$tmpjson"
if [ "$json_ng" != "4" ] || [ "$gen_count" != "4" ]; then
    echo "FAIL n_gen: json n_gen='$json_ng' gen_ids count='$gen_count' (want 4/4)" >&2
    fail=1
else
    echo "  ok n_gen: json n_gen=$json_ng == gen_ids count=$gen_count" >&2
fi

# (finding 2) prompt + n_gen exceeding --max-ctx must NOT silently DONE: it is
# hard-rejected (parity with surge), never truncated then reported complete.
# IDS12 is 12 tokens; -n 10 --max-ctx 12 => 22 > 12.
ncase=$((ncase + 1))
of_out="$("$BENCH" "$GGUF" --ids "$IDS12" -n 10 --max-ctx 12 --gemm-gate-tflops 100 --quiet 2>/dev/null)"
of_rc=$?
if [ "$of_rc" -eq 0 ]; then
    echo "FAIL overflow: prompt+gen > max-ctx returned exit 0 (silent DONE)" >&2
    fail=1
elif printf '%s' "$of_out" | grep -q "DONE"; then
    echo "FAIL overflow: emitted a DONE row despite exceeding max-ctx: $of_out" >&2
    fail=1
else
    echo "  ok overflow: prompt+gen > max-ctx refused (rc=$of_rc, no DONE)" >&2
fi

# ------------------------------------------------------------------
# (3) BOS toggle (env-gated): --bos vs --no-bos changes n_prompt_tokens by 1.
# ------------------------------------------------------------------
TOK_MODEL="${SURGE_BENCH_TOK_MODEL:-}"
if [ -n "$TOK_MODEL" ] && [ -e "$TOK_MODEL" ]; then
    ncase=$((ncase + 1))
    # A below-threshold gate keeps this to a tokenize + VOID exit (no GPU load),
    # but n_prompt_tokens is logged to stderr first, which is all we need.
    bench_nprompt() {
        "$BENCH" "$TOK_MODEL" -p "The quick brown fox jumps over the lazy dog." \
            "$1" --gemm-gate-tflops 10 --quiet 2>&1 >/dev/null \
            | sed -n 's/.*n_prompt_tokens=\([0-9][0-9]*\).*/\1/p' | head -1
    }
    nb="$(bench_nprompt --no-bos)"
    wb="$(bench_nprompt --bos)"
    if [ -z "$nb" ] || [ -z "$wb" ]; then
        echo "FAIL bos-toggle: could not read n_prompt_tokens (no-bos='$nb' bos='$wb')" >&2
        fail=1
    elif [ "$wb" -ne "$((nb + 1))" ]; then
        echo "FAIL bos-toggle: --bos ($wb) != --no-bos ($nb) + 1" >&2
        fail=1
    else
        echo "  ok bos-toggle: no-bos=$nb bos=$wb (+1)" >&2
    fi
else
    echo "  skip bos-toggle: set SURGE_BENCH_TOK_MODEL to a tokenizer GGUF to run it" >&2
fi

if [ "$fail" -ne 0 ]; then
    echo "test_cli_bench: FAILED ($ncase cases)" >&2
    exit 1
fi
echo "test_cli_bench: $ncase cases passed (gen_ids parity, VOID/exit-3, admit)" >&2
exit 0
