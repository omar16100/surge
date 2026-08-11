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

if [ "$fail" -ne 0 ]; then
    echo "test_cli_bench: FAILED ($ncase cases)" >&2
    exit 1
fi
echo "test_cli_bench: $ncase cases passed (gen_ids parity, VOID/exit-3, admit, $b6_label)" >&2
exit 0
