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
# Later tasks appended their own blocks below: B6's decode-slope verification,
# B8's prefill duty-cycle, command-buffer segmentation, and P3.0's decode
# duty-cycle + clamp detector.
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

# ------------------------------------------------------------------
# B6: decode-by-slope + wall-accounting verification (offline, hermetic).
#
# Drives surge-bench on the mini fixture with a fixed --ids list and
# --warmup 3, --emit-timeseries, and --json, then independently refits the
# per-token cumulative-wall-time series in python3 (stdlib only) mirroring
# sg_bench_slope's mean-centered OLS and sg_bench_avg_tps exactly (see
# src/bench.c), and checks:
#   1. reported decode_tps_slope == offline [warmup, n) refit within 0.5%.
#   2. |decode_tps_slope - decode_tps_avg| / decode_tps_avg < 3%.
#   3. prefill_wall_s + decode_wall_s closes wall_s within 2%.
#   4. the reported slope matches the warmup-EXCLUDED refit (not the naive
#      [0, n) refit that still carries token 0's post-prefill transient).
#
# Two run shapes, because the mini fixture's per-token decode time is only
# ~0.6 ms and the two checks pull N in opposite directions:
#   - checks 1-3 want a STEADY, low-relative-noise fit: a bigger -n averages
#     out per-token scheduling jitter so slope and the endpoint avg agree
#     comfortably inside 3%. This matters more here than a quick standalone
#     measurement suggests: this B6 block runs straight after the parity /
#     VOID / admit checks above, which already did ~10 GPU model loads +
#     decodes, and empirically that leaves the NEXT few decodes measurably
#     noisier than a fresh process's decode (e.g. -n 192 alone in a tight
#     loop: max 2.7% over 40 repeats; -n 192 run right after the same churn
#     this script itself generates: max 5.7% over 10 repeats, well above the
#     3% gate). -n 1024 keeps that same post-churn max under 1% over 15
#     repeats (see task-B6-report.md for the measurements). One run is
#     enough at that N; do NOT lower it back towards 192 without re-checking
#     post-churn, not standalone, noise.
#   - check 4 wants the excluded warmup tokens to carry real LEVERAGE on the
#     fit: a bigger -n dilutes a fixed warmup=3's share of the window and
#     shrinks the naive-vs-warmup-excluded gap towards the noise floor, so
#     check 4 uses a much smaller -n 12 (warmup stays the brief-suggested
#     "small" value, 3; N shrinks instead of warmup growing, so this still
#     exercises a realistic --warmup). Even there, the mini fixture's
#     first-decode-dispatch transient is real but not huge, so a SINGLE run
#     can occasionally (~4% of individual runs, empirically) land the
#     naive-vs-warmup gap small by chance -- real hardware jitter, not
#     fabricated.
#     Critically, the gate does NOT just check that the two OFFLINE refits
#     (naive vs warmup-excluded) differ: a run where surge-bench ignored
#     --warmup entirely and always reported the naive [0,n) fit would still
#     pass a "the two refits differ" check on plenty of runs, and could
#     ALSO slip past check 1's OWN 0.5% tolerance whenever the natural
#     naive-vs-warmup gap happens to be under 0.5% (commonly the case here).
#     So check 4 instead measures reported's OWN distance from the naive
#     refit (reported_vs_naive_rel_err below) and requires it to clear
#     DECISIVE_MARGIN (1%), a full 2x above check 1's 0.5% tolerance: if
#     --warmup were ignored, reported would equal the naive refit to
#     ~1e-6 precision (the same floor check 1 measures), not sit >=1% away
#     from it. 5 independent real bench runs are taken and the gate passes
#     if ANY shows reported diverging from the naive refit by more than
#     that margin (individual-run miss rate ~4% at -n 12, so all 5 missing
#     is ~1e-7) -- while requiring on EVERY run that reported still matches
#     ITS OWN warmup-excluded refit within check 1's 0.5% (so a passing
#     check 4 run has reported close to refit[warmup,n) and far from
#     refit[0,n), not just "far from something").
# ------------------------------------------------------------------
PYTHON3="/opt/homebrew/bin/python3"
if [ ! -x "$PYTHON3" ]; then
    PYTHON3="$(command -v python3 2>/dev/null || true)"
fi
b6_label="B6 decode-slope/wall"
if [ -z "$PYTHON3" ] || [ ! -x "$PYTHON3" ]; then
    echo "SKIP test_cli_bench B6: no python3 found (tried /opt/homebrew/bin/python3 and PATH)" >&2
    b6_label="B6 SKIPPED (no python3)"
else
    ncase=$((ncase + 1))
    N_STEADY=1024
    N_TRANSIENT=12
    K_TRANSIENT=5
    WARMUP_B6=3
    b6_tmpdir="$(mktemp -d)"

    run_b6() {
        local n="$1" json="$2" ts="$3" max_ctx
        max_ctx=$((n > 512 ? n + 512 : 1024))
        "$BENCH" "$GGUF" --ids "$IDS12" -n "$n" --max-ctx "$max_ctx" \
            --gemm-gate-tflops 100 --json "$json" --emit-timeseries "$ts" \
            --warmup "$WARMUP_B6" --quiet >/dev/null 2>/dev/null
    }

    b6_ok=1

    steady_json="$b6_tmpdir/steady.json"
    steady_ts="$b6_tmpdir/steady.ts"
    if ! run_b6 "$N_STEADY" "$steady_json" "$steady_ts"; then
        echo "FAIL b6: surge-bench steady run failed" >&2
        b6_ok=0
    fi

    b6_args=("$WARMUP_B6" "$N_STEADY" "$steady_json" "$steady_ts" "$N_TRANSIENT")
    for k in $(seq 1 "$K_TRANSIENT"); do
        j="$b6_tmpdir/trans_$k.json"
        t="$b6_tmpdir/trans_$k.ts"
        if ! run_b6 "$N_TRANSIENT" "$j" "$t"; then
            echo "FAIL b6: surge-bench transient run $k failed" >&2
            b6_ok=0
        fi
        b6_args+=("$j" "$t")
    done

    if [ "$b6_ok" -ne 1 ]; then
        fail=1
    else
        cat > "$b6_tmpdir/verify.py" <<'PYEOF'
# B6 offline verification: mirrors sg_bench_slope (mean-centered OLS) and
# sg_bench_avg_tps (src/bench.c) exactly so the refit agrees with the C
# implementation to floating-point precision, then checks the B6 gates.
import json
import sys


def slope(xs, n, warmup):
    if warmup >= n:
        return 0.0
    idxs = list(range(warmup, n))
    count = len(idxs)
    if count < 2:
        return 0.0
    mean_x = sum(xs[i] for i in idxs) / count
    mean_y = sum(idxs) / count
    sum_dxdy = 0.0
    sum_dxdx = 0.0
    for i in idxs:
        dx = xs[i] - mean_x
        dy = i - mean_y
        sum_dxdy += dx * dy
        sum_dxdx += dx * dx
    if sum_dxdx == 0.0:
        return 0.0
    return sum_dxdy / sum_dxdx


def avg_tps(xs, n, warmup):
    count = (n - 1) - warmup
    if count < 1:
        return 0.0
    dt = xs[n - 1] - xs[warmup]
    if dt <= 0.0:
        return 0.0
    return count / dt


def load_ts(path):
    """Returns (indices, times); caller checks indices == range(len(xs))."""
    idxs = []
    xs = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            i, t = line.split()
            idxs.append(int(i))
            xs.append(float(t))
    return idxs, xs


def check_run_shape(row, idxs, n, expected_n, label):
    """(finding 3) The refit must actually cover [0, expected_n): a wrong
    row count (EOS stopped early, a timeseries write bug) would silently
    make the fit run over [warmup, produced) instead of the requested
    [warmup, N), invalidating the whole comparison without ever raising an
    error. status must be DONE (an admitted run), n_gen and the timeseries
    row count must both equal expected_n exactly, and the index column
    must be exactly 0..expected_n-1 (guards a corrupted/reordered file)."""
    ok = True
    if row.get("status") != "DONE":
        print(f"FAIL {label}: status={row.get('status')!r}, expected DONE")
        ok = False
    if row.get("n_gen") != expected_n:
        print(f"FAIL {label}: n_gen={row.get('n_gen')!r} != expected {expected_n} "
              f"(EOS stopped early, or -n was not honored)")
        ok = False
    if n != expected_n:
        print(f"FAIL {label}: timeseries has {n} rows != expected {expected_n}")
        ok = False
    elif idxs != list(range(expected_n)):
        print(f"FAIL {label}: timeseries index column is not exactly "
              f"0..{expected_n - 1} (got {idxs[:5]}...)")
        ok = False
    return ok


# check 1's own tolerance (0.5%): reported decode_tps_slope must match the
# offline [warmup, n) refit this closely, or closer.
CHECK1_TOL = 0.005

# check 4's decisive margin (finding 1 in the B6 code review): 2x CHECK1_TOL,
# applied to reported's OWN distance from the naive [0,n) refit (not to the
# distance between the two offline refits, which does not involve `reported`
# and so cannot rule out a warmup-ignoring bug on its own -- see the shell
# comment above this heredoc for the full reasoning).
DECISIVE_MARGIN = 0.01


def check1(row, xs, n, warmup, label):
    """reported decode_tps_slope == offline [warmup, n) refit within
    CHECK1_TOL. Returns (ok, refit_w)."""
    refit_w = slope(xs, n, warmup)
    if refit_w <= 0.0:
        print(f"FAIL {label}: non-positive warmup-excluded refit ({refit_w:.6g})")
        return False, refit_w
    rel_err = abs(row["decode_tps_slope"] - refit_w) / refit_w
    print(f"{label}: n={n} reported_slope={row['decode_tps_slope']:.6g} "
          f"refit[warmup,n)={refit_w:.6g} rel_err={rel_err:.6g}")
    if rel_err > CHECK1_TOL:
        print(f"FAIL {label} check1: reported vs refit[warmup,n) "
              f"rel_err {rel_err:.6g} > {CHECK1_TOL:.6g}")
        return False, refit_w
    return True, refit_w


argv = sys.argv[1:]
warmup = int(argv[0])
n_steady = int(argv[1])
steady_json, steady_ts = argv[2], argv[3]
n_transient = int(argv[4])
transient = []
for i in range(5, len(argv), 2):
    transient.append((argv[i], argv[i + 1]))

ok = True

# ---- checks 1-3: the steady (-n 1024) run. ----
with open(steady_json) as f:
    srow = json.load(f)
sidxs, sxs = load_ts(steady_ts)
sn = len(sxs)

ok = check_run_shape(srow, sidxs, sn, n_steady, "steady") and ok

c1_ok, _ = check1(srow, sxs, sn, warmup, "steady")
ok = ok and c1_ok

avg = srow["decode_tps_avg"]
if avg <= 0.0:
    print(f"FAIL check2: decode_tps_avg is non-positive ({avg:.6g})")
    ok = False
else:
    # Sanity: decode_tps_avg itself must be the honest sg_bench_avg_tps
    # refit (not just "some number that happens to be close to slope"),
    # so the check2 comparison below is between two independently-honest
    # numbers, not a coincidence.
    offline_avg = avg_tps(sxs, sn, warmup)
    rel_err_avg = abs(avg - offline_avg) / offline_avg if offline_avg > 0.0 else 1.0
    print(f"check2 (avg sanity): reported_avg={avg:.6g} "
          f"offline_avg_tps={offline_avg:.6g} rel_err={rel_err_avg:.6g}")
    if rel_err_avg > CHECK1_TOL:
        print(f"FAIL check2 (avg sanity): reported decode_tps_avg vs offline "
              f"sg_bench_avg_tps refit rel_err {rel_err_avg:.6g} > {CHECK1_TOL:.6g}")
        ok = False

    rel_diff = abs(srow["decode_tps_slope"] - avg) / avg
    print(f"check2: slope={srow['decode_tps_slope']:.6g} avg={avg:.6g} "
          f"rel_diff={rel_diff:.6g}")
    if rel_diff >= 0.03:
        print(f"FAIL check2: rel_diff {rel_diff:.6g} >= 0.03")
        ok = False

wall_s = srow["wall_s"]
summed = srow["prefill_wall_s"] + srow["decode_wall_s"]
if wall_s <= 0.0:
    print("FAIL check3: wall_s <= 0.0")
    ok = False
else:
    rel_gap = abs(summed - wall_s) / wall_s
    print(f"check3: prefill_wall_s={srow['prefill_wall_s']:.6g} "
          f"decode_wall_s={srow['decode_wall_s']:.6g} sum={summed:.6g} "
          f"wall_s={wall_s:.6g} rel_gap={rel_gap:.6g}")
    if rel_gap >= 0.02:
        print(f"FAIL check3: rel_gap {rel_gap:.6g} >= 0.02")
        ok = False

# ---- check 4: K independent transient (-n 12) runs, best-of-K. ----
max_d_naive = 0.0
best_run = None
for run_idx, (json_path, ts_path) in enumerate(transient, start=1):
    with open(json_path) as f:
        row = json.load(f)
    idxs, xs = load_ts(ts_path)
    n = len(xs)

    if not check_run_shape(row, idxs, n, n_transient, f"transient{run_idx}"):
        ok = False
        continue

    c1_ok, refit_w = check1(row, xs, n, warmup, f"transient{run_idx}")
    ok = ok and c1_ok
    if refit_w <= 0.0:
        continue

    refit_naive = slope(xs, n, 0)
    if refit_naive <= 0.0:
        print(f"FAIL transient{run_idx}: non-positive naive refit ({refit_naive:.6g})")
        ok = False
        continue

    # The decisive quantity: reported's OWN distance from the naive refit,
    # not just how far apart the two offline refits are from each other.
    d_naive = abs(row["decode_tps_slope"] - refit_naive) / refit_naive
    naive_diff = abs(refit_naive - refit_w) / refit_w
    print(f"transient{run_idx}: refit[0,n)={refit_naive:.6g} "
          f"naive_vs_warmup_diff={naive_diff:.6g} "
          f"reported_vs_naive_rel_err={d_naive:.6g}")
    if d_naive > max_d_naive:
        max_d_naive = d_naive
        best_run = run_idx

print(f"max reported_vs_naive_rel_err across {len(transient)} transient runs: "
      f"{max_d_naive:.6g} (best run: {best_run})")
if max_d_naive <= DECISIVE_MARGIN:
    print(f"FAIL check4: reported slope never diverged from the naive [0,n) "
          f"refit by more than {DECISIVE_MARGIN:.6g} across {len(transient)} "
          f"runs (max {max_d_naive:.6g}) -- cannot rule out that surge-bench "
          f"ignored --warmup and reported the naive fit instead (check 1's "
          f"own {CHECK1_TOL:.6g} tolerance would not have caught that)")
    ok = False
else:
    print(f"check4: reported slope is {max_d_naive:.6g} away from the naive "
          f"[0,n) refit (run {best_run}, > the {DECISIVE_MARGIN:.6g} decisive "
          f"margin, {DECISIVE_MARGIN / CHECK1_TOL:.6g}x check 1's own "
          f"{CHECK1_TOL:.6g} tolerance) while matching its own "
          f"warmup-excluded refit within {CHECK1_TOL:.6g} (check 1) -- this "
          f"rules out a warmup-ignoring bug: if --warmup had been ignored, "
          f"reported would equal the naive refit to ~1e-6 precision (check "
          f"1's own noise floor), not sit {max_d_naive:.6g} away from it")

sys.exit(0 if ok else 1)
PYEOF
        b6_out="$("$PYTHON3" "$b6_tmpdir/verify.py" "${b6_args[@]}" 2>&1)"
        b6_rc=$?
        echo "$b6_out" | sed 's/^/  b6: /' >&2
        if [ "$b6_rc" -ne 0 ]; then
            echo "FAIL b6: offline verification failed (see above)" >&2
            fail=1
        else
            echo "  ok b6: decode-slope + wall-accounting verification passed" >&2
        fi
    fi
    rm -rf "$b6_tmpdir"
fi

# ------------------------------------------------------------------
# B8: prefill duty-cycle (firmware GPU-clamp mitigation).
#
#   (1) RESTS DON'T CHANGE OUTPUT: a forced-rest run (--prefill-work-ms 1
#       --prefill-rest-ms 50, so the tiny 1ms budget forces a rest at nearly
#       every chunk boundary) must produce gen_ids BYTE-IDENTICAL to the same
#       run with the feature off. --chunk 1 on the 12-token IDS12 gives 11
#       inter-chunk boundaries, so this exercises many rests, not zero or
#       one. rest_ms stays tiny (50ms) per the task brief so this stays fast
#       even with 11 rests (~0.55s of sleep, negligible next to the rest of
#       this script).
#   (2) REST ACCOUNTING: with IDS12 (12 tokens) and --chunk 1, the prefill
#       runs exactly 12 chunks, 11 of which have a chunk after them -- so a
#       correct work_budget_ms=1/rest_ms=50 run rests at EVERY one of those
#       11 boundaries (each chunk's own GPU-busy time already exceeds the 1ms
#       budget on this hardware) and NEVER at the 12th (last chunk, no rest
#       after it). That makes the expected total exact, not just "some rest
#       happened": prefill_rest_s must land within B8_REST_TOL_S of
#       B8_EXPECT_RESTS * 0.050s. This is decisive against a buggy
#       accumulator/threshold/reset (which would rest a different number of
#       times, landing off by a whole 50ms multiple, well outside the
#       tolerance) in a way a bare ">0" check is not. prefill_compute_tps
#       (rest excluded from the denominator) must also exceed prefill_tps
#       (plain wall-clock, rest included) for the same run.
# ------------------------------------------------------------------
B8_CHUNK=1
B8_EXPECT_RESTS=11     # 12 chunks (12 tokens / chunk 1) - 1 (no rest after the last)
B8_REST_MS=50
B8_REST_TOL_S=0.03

ncase=$((ncase + 1))
b8_off="$(bench_gen "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 --chunk "$B8_CHUNK")"
b8_rest="$(bench_gen "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 --chunk "$B8_CHUNK" \
    --prefill-work-ms 1 --prefill-rest-ms "$B8_REST_MS")"
if [ -z "$b8_off" ] || [ -z "$b8_rest" ]; then
    echo "FAIL b8 rests-dont-change-output: empty gen_ids (off='$b8_off' rest='$b8_rest')" >&2
    fail=1
elif [ "$b8_off" != "$b8_rest" ]; then
    echo "FAIL b8 rests-dont-change-output: forced-rest gen_ids != feature-off gen_ids" >&2
    echo "  off : $b8_off" >&2
    echo "  rest: $b8_rest" >&2
    fail=1
else
    echo "  ok b8 rests-dont-change-output: gen_ids=$b8_rest" >&2
fi

ncase=$((ncase + 1))
b8_json="$(mktemp)"
"$BENCH" "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 --chunk "$B8_CHUNK" \
    --prefill-work-ms 1 --prefill-rest-ms "$B8_REST_MS" --gemm-gate-tflops 100 \
    --json "$b8_json" --quiet >/dev/null 2>/dev/null
b8_rest_s="$(sed -n 's/.*"prefill_rest_s":\([0-9.eE+-]*\).*/\1/p' "$b8_json")"
b8_wall_tps="$(sed -n 's/.*"prefill_tps":\([0-9.eE+-]*\).*/\1/p' "$b8_json")"
b8_compute_tps="$(sed -n 's/.*"prefill_compute_tps":\([0-9.eE+-]*\).*/\1/p' "$b8_json")"
rm -f "$b8_json"
if [ -z "$b8_rest_s" ] || [ -z "$b8_wall_tps" ] || [ -z "$b8_compute_tps" ]; then
    echo "FAIL b8 rest-accounting: could not read JSON fields (rest_s='$b8_rest_s' " \
         "wall_tps='$b8_wall_tps' compute_tps='$b8_compute_tps')" >&2
    fail=1
else
    b8_expect_s="$(awk -v n="$B8_EXPECT_RESTS" -v r="$B8_REST_MS" 'BEGIN { print n * r / 1000.0 }')"
    b8_ok="$(awk -v got="$b8_rest_s" -v want="$b8_expect_s" -v tol="$B8_REST_TOL_S" \
        -v w="$b8_wall_tps" -v c="$b8_compute_tps" \
        'BEGIN { d = got - want; if (d < 0) d = -d;
                 print (d <= tol && c > w) ? "1" : "0" }')"
    if [ "$b8_ok" != "1" ]; then
        echo "FAIL b8 rest-accounting: prefill_rest_s=$b8_rest_s, want $b8_expect_s " \
             "+/- $B8_REST_TOL_S ($B8_EXPECT_RESTS rests * ${B8_REST_MS}ms); " \
             "prefill_compute_tps=$b8_compute_tps vs prefill_tps=$b8_wall_tps (want compute > wall)" >&2
        fail=1
    else
        echo "  ok b8 rest-accounting: prefill_rest_s=$b8_rest_s (want $b8_expect_s +/- " \
             "$B8_REST_TOL_S), prefill_tps=$b8_wall_tps prefill_compute_tps=$b8_compute_tps" >&2
    fi
fi

# ------------------------------------------------------------------
# Command-buffer segmentation (--prefill-max-burst-ms).
#
# The duty-cycle rest can only yield the GPU BETWEEN chunks. It cannot help
# once ONE chunk's command buffer outlasts the macOS watchdog window on its
# own, which is what killed WindowServer twice on 2026-08-14 (~130 s for a
# single 256-token chunk at 220k context). Segmentation splits the layer
# sweep across several command buffers so no single submission holds the GPU
# that long.
#
# The safety claim is that this CANNOT change output: command buffer
# boundaries carry no state, so the same kernels run with the same arguments
# in the same order. Two things are checked, because either alone is weak:
#
#   (1) gen_ids with segmentation on must equal gen_ids with it off, over an
#       uneven chunk schedule (12 tokens at --chunk 5 gives 5,5,2) so the
#       final short chunk and the logits-carrying final segment are both
#       exercised.
#   (2) prefill_segments must actually GO UP. Without this the parity check
#       would pass vacuously if segmentation silently never engaged, which is
#       the failure mode that makes a "cannot change output" test worthless.
# ------------------------------------------------------------------
SEG_CHUNK=5            # 12 tokens -> 5,5,2, so the last chunk is short
SEG_BURST_MS=1         # low enough that every command buffer overruns it

ncase=$((ncase + 1))
seg_off_json="$(mktemp)"; seg_on_json="$(mktemp)"
seg_off="$("$BENCH" "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 --chunk "$SEG_CHUNK" \
    --gemm-gate-tflops 100 --quiet --json "$seg_off_json" 2>/dev/null | sed -n 's/^gen_ids: //p')"
seg_on="$("$BENCH" "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 --chunk "$SEG_CHUNK" \
    --prefill-max-burst-ms "$SEG_BURST_MS" \
    --gemm-gate-tflops 100 --quiet --json "$seg_on_json" 2>/dev/null | sed -n 's/^gen_ids: //p')"
seg_n_off="$(sed -n 's/.*"prefill_segments":\([0-9]*\).*/\1/p' "$seg_off_json")"
seg_n_on="$(sed -n 's/.*"prefill_segments":\([0-9]*\).*/\1/p' "$seg_on_json")"
rm -f "$seg_off_json" "$seg_on_json"

if [ -z "$seg_off" ] || [ -z "$seg_on" ]; then
    echo "FAIL seg segmentation-doesnt-change-output: empty gen_ids (off='$seg_off' on='$seg_on')" >&2
    fail=1
elif [ "$seg_off" != "$seg_on" ]; then
    echo "FAIL seg segmentation-doesnt-change-output: segmented gen_ids != unsegmented" >&2
    echo "  off: $seg_off" >&2
    echo "  on : $seg_on" >&2
    fail=1
elif [ -z "$seg_n_off" ] || [ -z "$seg_n_on" ] || [ "$seg_n_on" -le "$seg_n_off" ]; then
    echo "FAIL seg segmentation-engaged: prefill_segments did not rise " \
         "(off='$seg_n_off' on='$seg_n_on'); the parity check above was vacuous" >&2
    fail=1
else
    echo "  ok seg segmentation-doesnt-change-output: gen_ids=$seg_on " \
         "(command buffers $seg_n_off -> $seg_n_on)" >&2
fi

# ------------------------------------------------------------------
# P3.0: decode duty-cycle + clamp detection (src/sched.c).
#
# The decode analogue of the B8 block above, and it exists for a measured
# reason: P2.9 ran three IDENTICAL decode arms and got 38.47 / 25.40 / 16.71
# tok/s, then after 150 s of idle re-ran the same three and got 39.72 /
# 41.16 / 34.66. Decode throughput here is partly a function of how recently
# the GPU was busy, and decode had no mitigation.
#
# Five checks. The first two mirror B8's exactly; the last three exist
# because a pacing switch that is enabled and silently does nothing is this
# project's established failure mode (P2.4's GQA gate).
#
#   (1) RESTS DON'T CHANGE OUTPUT: a forced-rest run (--decode-work-ms 1
#       --decode-rest-ms 50, so the 1 ms budget is exhausted by every single
#       decode step) must emit gen_ids BYTE-IDENTICAL to the same run with
#       pacing off. Resting is timing, never arithmetic.
#   (2) REST ACCOUNTING, in the parts that do not depend on this machine's
#       clock. -n 16 with no EOS produces 16 tokens and therefore 15
#       sg_gpu_forward calls (the last token needs no logits), so there are
#       exactly 15 pacing points. Four assertions, none of them timing-
#       dependent: decode_rest_s == decode_rests * rest_ms (the accounting
#       IDENTITY -- the reported seconds cannot drift from the reported
#       count); 1 <= decode_rests <= 15 (more rests than pacing points would
#       mean the pacer is being called from somewhere other than the
#       per-token boundary); the UNPACED run has decode_rest_s and
#       decode_rests both exactly 0; decode_compute_tps (rest excluded from
#       the denominator) exceeds decode_tps_avg (wall-clock); and, the one
#       assertion here that is not bookkeeping, the paced run's decode_wall_s
#       is at least its own decode_rest_s, so the reported rest was actually
#       spent and not merely counted.
#
#       An exact expected COUNT is deliberately NOT asserted here, and the
#       reason is measured: a decode step on this 8-layer fixture costs about
#       1 ms, which is the smallest budget the flag can express, so over 8
#       repeats one run in eight rested 14 times and the rest 15. The exact
#       accumulator semantics (200 steps x 10 ms against a 100 ms budget =
#       exactly 20 rests; an oversized step rests once, not repeatedly) are
#       pinned deterministically instead in tests/test_sched.c, which runs
#       under `make check` AND `make debug`, and an exact count IS asserted
#       in the real-model arm below, where a step is 20x the budget.
#   (3) THE DETECTOR IS WIRED IN, two-sided and deterministic. Seeding an
#       absurdly small baseline (--decode-baseline-ms 0.001) makes every
#       step "over" it, so decode_clamp_events MUST rise; seeding an absurdly
#       large one makes no step over it, so it MUST stay 0. Without this the
#       detector could report 0 forever because it was never fed a step, and
#       nothing else here would notice.
#   (4) THE ESCALATION IS NOT INERT. Same seeded-clamp run, a budget
#       calibrated to be larger than the whole decode phase:
#       --decode-clamp-div 1 must rest 0 times (the detector alone changes no
#       schedule, which is the default), and --decode-clamp-div 4 must rest at
#       least once (a confirmed clamp quarters the budget). One run proves the
#       default is passive, the other proves the knob does something.
#   (5) REAL MODEL (env-gated on SURGE_PACE_MODEL): the same
#       gen_ids-unchanged proof on a real GGUF rather than the 8-layer
#       fixture, PLUS the exact rest count that the fixture cannot give: a
#       real decode step is tens of ms against the 1 ms budget, so every
#       pacing point must rest and decode_rests must equal n_gen - 1
#       exactly. Skipped cleanly when the env var is unset, so `make check`
#       stays hermetic.
# ------------------------------------------------------------------
P30_REST_MS=50
P30_PACE_POINTS=15     # 16 tokens produced -> 15 forwards -> 15 pacing points
P30_REAL_N=24          # the real-model arm decodes 24 -> 23 pacing points
P30_REAL_REST_MS=20

p30_off="$(bench_gen "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512)"
p30_on="$(bench_gen "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 \
    --decode-work-ms 1 --decode-rest-ms "$P30_REST_MS")"
ncase=$((ncase + 1))
if [ -z "$p30_off" ] || [ -z "$p30_on" ]; then
    echo "FAIL p30 decode-rests-dont-change-output: empty gen_ids (off='$p30_off' on='$p30_on')" >&2
    fail=1
elif [ "$p30_off" != "$p30_on" ]; then
    echo "FAIL p30 decode-rests-dont-change-output: paced gen_ids != unpaced gen_ids" >&2
    echo "  off: $p30_off" >&2
    echo "  on : $p30_on" >&2
    fail=1
else
    echo "  ok p30 decode-rests-dont-change-output: gen_ids=$p30_on" >&2
fi

# $1 = output json path; remaining args appended to a standard 16-token run.
p30_run() {
    local out="$1"; shift
    "$BENCH" "$GGUF" --ids "$IDS12" -n 16 --max-ctx 512 --gemm-gate-tflops 100 \
        --json "$out" --quiet "$@" >/dev/null 2>/dev/null
}
p30_get() { sed -n "s/.*\"$2\":\([0-9.eE+-]*\).*/\1/p" "$1"; }
# Same, with -n overridable: the escalation case below wants a longer decode
# phase so its budget arithmetic is not dominated by per-step variance and
# integer-millisecond truncation.
p30_run_n() {
    local out="$1" n="$2"; shift 2
    "$BENCH" "$GGUF" --ids "$IDS12" -n "$n" --max-ctx 512 --gemm-gate-tflops 100 \
        --json "$out" --quiet "$@" >/dev/null 2>/dev/null
}

ncase=$((ncase + 1))
p30_on_json="$(mktemp)"; p30_off_json="$(mktemp)"
p30_run "$p30_on_json" --decode-work-ms 1 --decode-rest-ms "$P30_REST_MS"
p30_run "$p30_off_json"
p30_rest_on="$(p30_get "$p30_on_json" decode_rest_s)"
p30_rest_off="$(p30_get "$p30_off_json" decode_rest_s)"
p30_n_on="$(p30_get "$p30_on_json" decode_rests)"
p30_n_off="$(p30_get "$p30_off_json" decode_rests)"
p30_ctps="$(p30_get "$p30_on_json" decode_compute_tps)"
p30_wtps="$(p30_get "$p30_on_json" decode_tps_avg)"
p30_base_off="$(p30_get "$p30_off_json" decode_baseline_ms)"
p30_wall_on="$(p30_get "$p30_on_json" decode_wall_s)"
rm -f "$p30_on_json" "$p30_off_json"
if [ -z "$p30_rest_on" ] || [ -z "$p30_rest_off" ] || [ -z "$p30_n_on" ] || \
   [ -z "$p30_n_off" ] || [ -z "$p30_ctps" ] || [ -z "$p30_wtps" ] || [ -z "$p30_wall_on" ]; then
    echo "FAIL p30 decode-rest-accounting: could not read JSON fields " \
         "(rest_on='$p30_rest_on' rest_off='$p30_rest_off' n_on='$p30_n_on' " \
         "n_off='$p30_n_off' ctps='$p30_ctps' wtps='$p30_wtps')" >&2
    fail=1
elif [ "$p30_rest_off" != "0" ] || [ "$p30_n_off" != "0" ]; then
    echo "FAIL p30 decode-rest-accounting: pacing off, yet decode_rest_s=$p30_rest_off " \
         "and decode_rests=$p30_n_off (both must be exactly 0)" >&2
    fail=1
elif [ "$p30_n_on" -lt 1 ]; then
    echo "FAIL p30 decode-rest-accounting: armed with a 1ms budget over $P30_PACE_POINTS " \
         "decode steps, yet decode_rests=$p30_n_on -- the pacer never rested" >&2
    fail=1
elif [ "$p30_n_on" -gt "$P30_PACE_POINTS" ]; then
    echo "FAIL p30 decode-rest-accounting: decode_rests=$p30_n_on exceeds the " \
         "$P30_PACE_POINTS per-token pacing points in this run -- the pacer is being " \
         "called from somewhere other than the per-token boundary" >&2
    fail=1
else
    # The wall-clock arm. Everything above is bookkeeping, and bookkeeping
    # alone cannot tell a real sleep from a reported one: the accounting lives
    # in sg_decode_pace_decide, so a decode loop that took the accounting and
    # skipped the sleep (calling _decide instead of _step, say) would satisfy
    # the identity above AND still make decode_compute_tps exceed
    # decode_tps_avg, because decode_rest_s is subtracted from the denominator
    # either way. That is exactly the P2.4 silently-inert shape. So: the paced
    # run's decode_wall_s must be at least its own decode_rest_s. On this
    # fixture the unpaced decode phase is ~15 ms against 750 ms of prescribed
    # rest, so a non-sleeping implementation misses this by ~50x.
    p30_ok="$(awk -v got="$p30_rest_on" -v n="$p30_n_on" -v r="$P30_REST_MS" \
        -v c="$p30_ctps" -v w="$p30_wtps" -v wall="$p30_wall_on" \
        'BEGIN { want = n * r / 1000.0; d = got - want; if (d < 0) d = -d;
                 print (d <= 1e-6 && c > w && wall >= got) ? "1" : "0" }')"
    if [ "$p30_ok" != "1" ]; then
        echo "FAIL p30 decode-rest-accounting: decode_rest_s=$p30_rest_on does not equal " \
             "decode_rests=$p30_n_on x ${P30_REST_MS}ms, or decode_compute_tps=$p30_ctps " \
             "is not > decode_tps_avg=$p30_wtps, or decode_wall_s=$p30_wall_on is less than " \
             "the ${p30_rest_on}s it claims to have slept (which would mean it never slept)" >&2
        fail=1
    else
        echo "  ok p30 decode-rest-accounting: decode_rests=$p30_n_on/$P30_PACE_POINTS, " \
             "decode_rest_s=$p30_rest_on (= rests x ${P30_REST_MS}ms), unpaced=0, " \
             "decode_wall_s=$p30_wall_on (>= the rest, so it really slept), " \
             "decode_compute_tps=$p30_ctps > decode_tps_avg=$p30_wtps" >&2
    fi
fi

# (3) The detector, two-sided. Both arms are unpaced, so this isolates the
#     detector from the duty cycle entirely.
ncase=$((ncase + 1))
p30_lo_json="$(mktemp)"; p30_hi_json="$(mktemp)"
p30_run "$p30_lo_json" --decode-baseline-ms 0.001
p30_run "$p30_hi_json" --decode-baseline-ms 1000000
p30_ev_lo="$(p30_get "$p30_lo_json" decode_clamp_events)"
p30_ev_hi="$(p30_get "$p30_hi_json" decode_clamp_events)"
rm -f "$p30_lo_json" "$p30_hi_json"
if [ -z "$p30_ev_lo" ] || [ -z "$p30_ev_hi" ]; then
    echo "FAIL p30 clamp-detector-wired: could not read decode_clamp_events " \
         "(lo='$p30_ev_lo' hi='$p30_ev_hi')" >&2
    fail=1
elif [ "$p30_ev_lo" -lt 1 ]; then
    echo "FAIL p30 clamp-detector-wired: every decode step is over a 0.001 ms baseline, " \
         "yet decode_clamp_events=$p30_ev_lo -- the detector is never fed a step" >&2
    fail=1
elif [ "$p30_ev_hi" -ne 0 ]; then
    echo "FAIL p30 clamp-detector-wired: no decode step can exceed a 1000000 ms baseline, " \
         "yet decode_clamp_events=$p30_ev_hi -- the detector fires unconditionally" >&2
    fail=1
elif [ -z "$p30_base_off" ] || [ "$p30_base_off" = "0" ]; then
    echo "FAIL p30 clamp-detector-wired: decode_baseline_ms=$p30_base_off on an unseeded run, " \
         "so no baseline was ever measured from real steps" >&2
    fail=1
else
    echo "  ok p30 clamp-detector-wired: events lo=$p30_ev_lo hi=$p30_ev_hi, " \
         "self-measured baseline=${p30_base_off}ms" >&2
fi

# (4) The escalation. The budget is deliberately larger than the whole decode
#     phase, so ONLY a clamp-shortened threshold can produce a rest at all:
#     --decode-clamp-div 1 must rest zero times (the default, where the
#     detector changes no schedule) and --decode-clamp-div 4 must rest.
#
#     The budget is CALIBRATED from a probe run rather than hard-coded,
#     because "larger than the decode phase" is a property of the machine and
#     not of the fixture: a decode step here measured 0.7 ms on an idle
#     machine and 101 ms while a long prefill held the GPU.
#
#     budget = 1.5x the probed decode wall. The accumulated STEP time is
#     roughly 0.85x the decode wall (the wall also covers the argmax, the
#     periodic memory sample and the progress print), so the div-1 arm has
#     about 1.75x of headroom before it would rest and the div-4 arm (whose
#     threshold is budget/4) has about 2.3x before it would not. A first
#     attempt used 2.5x, which left the div-4 side only ~16% of margin once
#     the integer-millisecond truncation was counted, and it flaked 1 run in
#     3. -n 48 rather than 16 for the same reason: a longer decode phase
#     makes both the probe and the paced arms less sensitive to a single
#     slow step.
ncase=$((ncase + 1))
P30_ESC_N=48
p30_probe_json="$(mktemp)"
p30_run_n "$p30_probe_json" "$P30_ESC_N"
p30_probe_w="$(p30_get "$p30_probe_json" decode_wall_s)"
rm -f "$p30_probe_json"
p30_budget="$(awk -v w="$p30_probe_w" 'BEGIN { b = int(w * 1000 * 1.5); if (b < 8) b = 8; print b }')"
p30_d1_json="$(mktemp)"; p30_d4_json="$(mktemp)"
p30_run_n "$p30_d1_json" "$P30_ESC_N" --decode-work-ms "$p30_budget" --decode-rest-ms 20 \
    --decode-baseline-ms 0.001 --decode-clamp-div 1
p30_run_n "$p30_d4_json" "$P30_ESC_N" --decode-work-ms "$p30_budget" --decode-rest-ms 20 \
    --decode-baseline-ms 0.001 --decode-clamp-div 4
p30_n_d1="$(p30_get "$p30_d1_json" decode_rests)"
p30_n_d4="$(p30_get "$p30_d4_json" decode_rests)"
p30_ev_d4="$(p30_get "$p30_d4_json" decode_clamp_events)"
rm -f "$p30_d1_json" "$p30_d4_json"
if [ -z "$p30_n_d1" ] || [ -z "$p30_n_d4" ] || [ -z "$p30_probe_w" ] || [ -z "$p30_ev_d4" ]; then
    echo "FAIL p30 clamp-escalation: could not read JSON (probe_wall='$p30_probe_w' " \
         "div1='$p30_n_d1' div4='$p30_n_d4' events='$p30_ev_d4')" >&2
    fail=1
elif [ "$p30_n_d1" -ne 0 ]; then
    echo "FAIL p30 clamp-escalation: --decode-clamp-div 1 rested $p30_n_d1 time(s) against a " \
         "${p30_budget}ms budget (1.5x the ${p30_probe_w}s probed decode phase) -- the " \
         "detector must not drive rests by default" >&2
    fail=1
elif [ "$p30_ev_d4" -lt 1 ]; then
    echo "FAIL p30 clamp-escalation: the div-4 arm never latched the detector " \
         "(decode_clamp_events=$p30_ev_d4), so it proves nothing about escalation" >&2
    fail=1
elif [ "$p30_n_d4" -lt 1 ]; then
    echo "FAIL p30 clamp-escalation: --decode-clamp-div 4 with a confirmed clamp rested " \
         "$p30_n_d4 times (want >= 1) against the same ${p30_budget}ms budget the div-1 arm " \
         "never reached -- the escalation is inert" >&2
    fail=1
else
    echo "  ok p30 clamp-escalation: budget=${p30_budget}ms (1.5x probe ${p30_probe_w}s over " \
         "$P30_ESC_N tokens), div1 rests=$p30_n_d1, div4 rests=$p30_n_d4 over " \
         "$p30_ev_d4 clamp event(s)" >&2
fi

# (5) Same determinism proof on a real model, env-gated so `make check` stays
#     hermetic. This is the arm the P3.0 brief names (the 4B dense Q8_0).
#
#     Its OWN variable, not SURGE_BENCH_TOK_MODEL, because the two want
#     different things from a model: the BOS-toggle case above needs a GGUF
#     that carries tokenizer.ggml.bos_token_id (Qwen3-4B-Instruct-2507-Q8_0
#     does not, and --bos is a hard error there), while this case needs only
#     a real model with real decode-step costs. Sharing one variable would
#     make pointing it at the P3.0 model fail an unrelated B5 gate.
PACE_MODEL="${SURGE_PACE_MODEL:-}"
if [ -n "$PACE_MODEL" ] && [ -e "$PACE_MODEL" ]; then
    ncase=$((ncase + 1))
    p30_real_json="$(mktemp)"
    p30_real_off="$("$BENCH" "$PACE_MODEL" --ids "$IDS12" -n "$P30_REAL_N" --max-ctx 512 \
        --gemm-gate-tflops 100 --quiet 2>/dev/null | sed -n 's/^gen_ids: //p')"
    p30_real_on="$("$BENCH" "$PACE_MODEL" --ids "$IDS12" -n "$P30_REAL_N" --max-ctx 512 \
        --gemm-gate-tflops 100 --quiet --json "$p30_real_json" \
        --decode-work-ms 1 --decode-rest-ms "$P30_REAL_REST_MS" \
        2>/dev/null | sed -n 's/^gen_ids: //p')"
    p30_real_n="$(p30_get "$p30_real_json" decode_rests)"
    p30_real_ngen="$(p30_get "$p30_real_json" n_gen)"
    rm -f "$p30_real_json"
    if [ -z "$p30_real_off" ] || [ -z "$p30_real_on" ] || [ -z "$p30_real_n" ]; then
        echo "FAIL p30 real-model-rests-dont-change-output: empty gen_ids or JSON " \
             "(rests='$p30_real_n')" >&2
        fail=1
    elif [ "$p30_real_off" != "$p30_real_on" ]; then
        echo "FAIL p30 real-model-rests-dont-change-output: paced gen_ids != unpaced" >&2
        echo "  off: $p30_real_off" >&2
        echo "  on : $p30_real_on" >&2
        fail=1
    # A real model's decode step is tens of ms against a 1 ms budget, so
    # unlike the fixture EVERY pacing point must rest and the count is exact.
    elif [ "$p30_real_n" -ne "$((p30_real_ngen - 1))" ]; then
        echo "FAIL p30 real-model rest count: decode_rests=$p30_real_n, want " \
             "$((p30_real_ngen - 1)) (one per pacing point: every decode step on a real " \
             "model exceeds the 1 ms budget many times over)" >&2
        fail=1
    else
        echo "  ok p30 real-model-rests-dont-change-output: gen_ids=$p30_real_on " \
             "(decode_rests=$p30_real_n/$((p30_real_ngen - 1)), exact)" >&2
    fi
else
    echo "  skip p30 real-model parity: set SURGE_PACE_MODEL to a real GGUF to run it" >&2
fi

if [ "$fail" -ne 0 ]; then
    echo "test_cli_bench: FAILED ($ncase cases)" >&2
    exit 1
fi
echo "test_cli_bench: $ncase cases passed (gen_ids parity, VOID/exit-3, admit, $b6_label, B8 duty-cycle, segmentation, P3.0 decode pacing)" >&2
exit 0
