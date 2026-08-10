#!/usr/bin/env python3
"""M3.4 gate (A): surge Metal Q8_0 vs surge CPU-ref Q8_0, teacher-forced.

    /Users/macmini/models/dsv4-venv/bin/python tools/tf_compare_q8.py \
        --gguf /Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf --freeze

Both paths read the SAME Qwen3.6-27B-Q8_0 GGUF. This runs ONE teacher-forced
forward over a fixed ~64-token prompt on each path (Metal via `surge`, the
scalar CPU reference via `surge-ref`, which drives sg_ref_matvec_q8), then
compares every prompt position's logits.

TEACHER-FORCED, never free-running: both sides consume the identical id
sequence at every position and are never conditioned on their own output. A
single forward covers all ~64 positions, so this is ONE slow CPU forward
(~10 s/position on the 27B), not 64 chained decodes.

Gate: 100% top-1 argmax agreement across all positions. Also reports
max/mean |logit delta| and, for any disagreement, the top1-top2 gap on each
side (a near-tie flip vs a systematic error).

Because the two paths read the same Q8_0 bytes, near-total agreement is
expected; a systematic dequant/layout/wiring error surfaces as many
disagreements. This gate CANNOT catch a bug shared by both surge paths --
that is what the llama.cpp cross-check (tools/xcheck_llama_q8.py) is for.

FROZEN FIXTURES: the Metal path is deterministic run-to-run (fixed-tree
reductions, no atomics), so its per-position digest is committed as the
regression fixture under tests/fixtures/m3q8/. A digest is
argmax + max + mean + rms + a strided sample of the vocab per position (the
full [positions, vocab] dumps are ~64 MB each and are gitignored). ids.txt
makes the fixture self-describing. On a non-freeze run this script also
re-compares the fresh Metal digest against the frozen one and flags drift.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GGUF = "/Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf"

# One fixed prompt, ~64 tokens of factual English. No chat template, no BOS
# (the GGUF sets add_bos_token=false), tokenized by surge's own GGUF BPE.
DEFAULT_PROMPT = (
    "The history of computing begins with mechanical calculators and the "
    "abstract work of Charles Babbage and Ada Lovelace. In the twentieth "
    "century, vacuum tubes gave way to transistors, then to integrated "
    "circuits, and finally to the microprocessor. Each step made machines "
    "smaller, faster, and far more reliable than anything that came before "
    "them."
)

# Must match the M1 digest layout so a reader can reuse the same tooling.
DIGEST_SAMPLES = 64
DIGEST_HEADER = 4


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p.stdout, p.stderr


def parse_prompt_ids(stdout):
    for line in stdout.splitlines():
        if line.startswith("prompt_ids:"):
            body = line.split(":", 1)[1].strip()
            return [int(x) for x in body.split(",") if x != ""]
    raise SystemExit("no prompt_ids line in surge output")


def load_logits(path, positions):
    a = np.fromfile(path, dtype="<f4")
    if a.size % positions != 0:
        raise SystemExit(f"{path}: {a.size} floats not divisible by {positions} positions")
    vocab = a.size // positions
    return a.reshape(positions, vocab)


def digest(logits):
    """[positions, vocab] -> [positions, DIGEST_HEADER + DIGEST_SAMPLES]."""
    pos, vocab = logits.shape
    stride = vocab // DIGEST_SAMPLES
    d = np.zeros((pos, DIGEST_HEADER + DIGEST_SAMPLES), dtype=np.float32)
    d[:, 0] = logits.argmax(1).astype(np.float32)
    d[:, 1] = logits.max(1)
    d[:, 2] = logits.astype(np.float64).mean(1).astype(np.float32)
    d[:, 3] = np.sqrt((logits.astype(np.float64) ** 2).mean(1)).astype(np.float32)
    d[:, DIGEST_HEADER:] = logits[:, : stride * DIGEST_SAMPLES : stride]
    return d


def top2_gap(row, arg):
    """top-1 minus top-2 logit at one position, given the top-1 index."""
    v = row.copy()
    top1 = v[arg]
    v[arg] = -np.inf
    return float(top1 - v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--surge", default=str(REPO / "surge"))
    ap.add_argument("--surge-ref", default=str(REPO / "surge-ref"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", default=str(REPO / "tests" / "fixtures" / "m3q8"))
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="informational max |logit delta| tolerance (the GATE is top-1)")
    ap.add_argument("--freeze", action="store_true",
                    help="rewrite the committed Metal digest + ids.txt + result.json")
    args = ap.parse_args()

    gguf = pathlib.Path(args.gguf)
    surge = pathlib.Path(args.surge)
    surge_ref = pathlib.Path(args.surge_ref)
    for b in (surge, surge_ref):
        if not b.exists():
            raise SystemExit(f"{b} not found; run `make {b.name}` first")
    if not gguf.exists():
        raise SystemExit(f"{gguf} not found")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metal_full = out / "metal.full.f32"
    ref_full = out / "ref.full.f32"

    # 1. Metal path: tokenize the prompt with surge's own GGUF BPE AND dump the
    #    teacher-forced logits in one run (-n 0 => the n_prompt positions only).
    print("gate A: surge Metal Q8_0 forward (tokenize + logit dump) ...", flush=True)
    t0 = time.time()
    stdout, _ = run([str(surge), str(gguf), "-p", args.prompt, "-n", "0",
                     "--logits", str(metal_full), "--quiet"])
    ids = parse_prompt_ids(stdout)
    positions = len(ids)
    t_metal = time.time() - t0
    print(f"  {positions} prompt tokens, metal forward {t_metal:.1f} s", flush=True)

    # 2. CPU-ref path: same ids, same GGUF, teacher-forced. This is the slow one.
    print("gate A: surge-ref CPU Q8_0 forward (scalar, minutes) ...", flush=True)
    t0 = time.time()
    ids_csv = ",".join(str(x) for x in ids)
    run([str(surge_ref), str(gguf), "--ids", ids_csv, "-n", "0",
         "--logits", str(ref_full), "--quiet"])
    t_ref = time.time() - t0
    print(f"  cpu-ref forward {t_ref:.1f} s ({t_ref / positions:.1f} s/position)", flush=True)

    metal = load_logits(metal_full, positions)
    ref = load_logits(ref_full, positions)
    if metal.shape != ref.shape:
        raise SystemExit(f"shape mismatch metal {metal.shape} vs ref {ref.shape}")
    vocab = metal.shape[1]

    mi = metal.argmax(1)
    ri = ref.argmax(1)
    agree = int((mi == ri).sum())
    delta = np.abs(metal.astype(np.float64) - ref.astype(np.float64))
    max_delta = float(delta.max())
    mean_delta = float(delta.mean())

    disagree = np.where(mi != ri)[0]
    print()
    print(f"{positions} positions x {vocab} logits, teacher-forced, same Q8_0 GGUF")
    print(f"top-1 agreement : {agree}/{positions} ({100.0 * agree / positions:.2f}%)")
    print(f"max |logit delta|: {max_delta:.4e}")
    print(f"mean|logit delta|: {mean_delta:.4e}")
    if disagree.size:
        print()
        print("disagreements (position: metal_top1 vs ref_top1, gaps):")
        for pos in disagree:
            print(f"  pos {int(pos):3d}: metal {int(mi[pos]):6d} (gap {top2_gap(metal[pos], mi[pos]):.4e}) "
                  f"!= ref {int(ri[pos]):6d} (gap {top2_gap(ref[pos], ri[pos]):.4e})")
    print()

    # Regression compare against the frozen Metal digest, if present.
    frozen_path = out / "metal_digest.f32"
    fresh = digest(metal)
    drift = None
    if frozen_path.exists() and not args.freeze:
        frozen = np.fromfile(frozen_path, dtype="<f4").reshape(fresh.shape)
        argmax_drift = int((fresh[:, 0] != frozen[:, 0]).sum())
        stat_drift = float(np.abs(fresh[:, 1:4] - frozen[:, 1:4]).max())
        drift = {"argmax_drift_positions": argmax_drift, "stat_max_abs_delta": stat_drift}
        print(f"regression vs frozen Metal digest: argmax drift {argmax_drift} positions, "
              f"stat max |delta| {stat_drift:.4e}")

    ok = (agree == positions)
    verdict = "PASS" if ok else "FAIL"
    print(f"M3.4 GATE A ({verdict}): surge Metal Q8_0 vs surge CPU-ref Q8_0, "
          f"top-1 {100.0 * agree / positions:.2f}% (need 100.00%), "
          f"max |delta| {max_delta:.4e}")

    result = {
        "gguf": str(gguf),
        "positions": positions,
        "vocab": vocab,
        "ids": ids,
        "top1_agree": agree,
        "top1_pct": 100.0 * agree / positions,
        "max_delta": max_delta,
        "mean_delta": mean_delta,
        "disagreements": [
            {"pos": int(p), "metal_top1": int(mi[p]), "ref_top1": int(ri[p]),
             "metal_gap": top2_gap(metal[p], mi[p]), "ref_gap": top2_gap(ref[p], ri[p])}
            for p in disagree],
        "t_metal_s": t_metal,
        "t_ref_s": t_ref,
        "cpu_path": "pure-C sg_ref_matvec_q8 (double accumulate)",
        "regression_drift": drift,
        "pass": bool(ok),
    }

    if args.freeze:
        if not ok:
            print("NOT freezing: --freeze requires the gate to pass")
        else:
            fresh.astype("<f4").tofile(frozen_path)
            (out / "ids.txt").write_text(ids_csv + "\n", encoding="utf-8")
            print(f"froze Metal digest -> {frozen_path} and ids.txt")
        (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {out}/result.json")
    else:
        print(f"non-freeze run: {out}/metal_digest.f32 / ids.txt / result.json left as-is")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
