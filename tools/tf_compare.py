#!/usr/bin/env python3
"""The M1 gate: surge's reference forward vs mlx-lm, teacher-forced.

    /Users/macmini/models/dsv4-venv/bin/python tools/tf_compare.py \
        --model /Users/macmini/models/qwen35-2b

Runs 16 prompts through both implementations on the SAME fixed token ids and
compares every prompt position's logits. Gate: 100 percent top-1 agreement
and max |logit delta| < 1e-2.

TEACHER-FORCED, never free-running. Both sides are fed the identical id
sequence at every position; neither is ever conditioned on its own output.
A free-running comparison would prove nothing here, since this model class
diverges across runs at depth.

------------------------------------------------------------------------
TWO ORACLES, AND WHY
------------------------------------------------------------------------

mlx-lm's own loader adds the +1.0 norm-weight shift BEFORE the dtype cast,
i.e. in bf16:

    mlx_lm/utils.py           weights = model.sanitize(weights)   # bf16 arrays
    mlx_lm/models/qwen3_5.py  weights[k] = v + 1.0                # stays bf16

so the effective norm weight mlx computes with is bf16(bf16(w) + 1.0). Around
1.1 that quantizes to about 0.004 absolute, which is roughly 0.4 percent of
the weight, and it propagates: measured over 8 positions of Qwen3.5-2B it
moves the logits by up to 7.0e-2.

The model's own reference implementation does NOT do that.
transformers/models/qwen3_5/modeling_qwen3_5.py, Qwen3_5RMSNorm.forward:

    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())      # <- the shift is f32

surge follows transformers: it widens the stored bf16 weight to f32 and adds
1.0 in f32. That is a deliberate accuracy choice of the same kind Task 7 made
for RoPE ("ref.c defines correct; pin the gap rather than import someone
else's rounding"), and here it is not even a judgement call, because the
upstream reference states the semantics outright.

So this script measures BOTH and prints both:

  * "stock"   -- mlx-lm exactly as shipped, +1.0 applied in bf16.
  * "hf-norm" -- the same mlx-lm model with the five shifted norm tensors
                 recomputed as f32(w) + 1.0, i.e. transformers' semantics.
                 THIS IS THE GATE ORACLE.

Everything else about the two oracles is identical: same model object, same
weights, same f32 compute dtype (mx bf16 arithmetic would put a 4e-3 relative
floor on the logits all by itself, which is why both oracles run in f32 --
widening bf16 to f32 is exact, so this compares the same weights, not
different ones).

------------------------------------------------------------------------
PROMPTS
------------------------------------------------------------------------

The 16 prompts are built from the 23 non-empty tok fixture texts in
tests/fixtures/tok_cases.jsonl (ASCII, code, unicode, emoji, whitespace
runs, URLs). Those texts are 1 to 25 tokens each, so prompt i is the rotation
of the list starting at text i, newline-joined and truncated to --positions
tokens; that gives 16 different 64-position sequences rather than 16 sequences
of a dozen positions, which is what actually exercises RoPE at nonzero
offsets, the growing KV cache and 64 chained DeltaNet state updates.

The ids are produced by the MODEL DIRECTORY's own HF tokenizer and handed to
both sides verbatim (the fixture's recorded ids came from a different
checkpoint's tokenizer and are not reused).

------------------------------------------------------------------------
WHAT GETS FROZEN, AND WHY IT IS A DIGEST
------------------------------------------------------------------------

The full dumps are 16 x 64 x 248320 x 4 bytes = 1017 MB, which is not a
committable regression fixture by any reading. They are written as
`pNN.full.f32` and gitignored.

What IS committed is `pNN.f32`, a fixed reduction of the same data: per
position, 4 + DIGEST_SAMPLES floats,

    [0] argmax vocab index, exactly representable in f32 (vocab < 2^24)
    [1] the max logit
    [2] the mean over the whole vocab
    [3] the root mean square over the whole vocab
    [4...] the logits at vocab indices 0, stride, 2*stride, ... with
           stride = vocab // DIGEST_SAMPLES

16 x 64 x 68 x 4 = 278 KB total. Every one of those numbers depends on the
entire forward pass, so any change to ref.c moves them; the mean and RMS in
particular cannot be preserved by a local error. `ids.txt` carries the exact
token ids, one comma-separated line per prompt, so the digest is
self-describing and tests/test_ref_fwd.c can recompute it.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
TOK_CASES = REPO / "tests" / "fixtures" / "tok_cases.jsonl"

NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

# Must match DIGEST_SAMPLES / DIGEST_STRIDE_OF in tests/test_ref_fwd.c.
DIGEST_SAMPLES = 64
DIGEST_HEADER = 4


def write_digest(path, logits, np):
    """Reduce [positions, vocab] logits to the committed regression fixture."""
    pos, vocab = logits.shape
    stride = vocab // DIGEST_SAMPLES
    d = np.zeros((pos, DIGEST_HEADER + DIGEST_SAMPLES), dtype=np.float32)
    d[:, 0] = logits.argmax(1).astype(np.float32)
    d[:, 1] = logits.max(1)
    d[:, 2] = logits.astype(np.float64).mean(1).astype(np.float32)
    d[:, 3] = np.sqrt((logits.astype(np.float64) ** 2).mean(1)).astype(np.float32)
    d[:, DIGEST_HEADER:] = logits[:, : stride * DIGEST_SAMPLES: stride]
    path.write_bytes(d.astype("<f4").tobytes())


def build_prompts(n_prompts, positions, tokenizer):
    texts = []
    with TOK_CASES.open() as f:
        for line in f:
            t = json.loads(line)["text"]
            if t:
                texts.append(t)
    if not texts:
        raise SystemExit("tok_cases.jsonl has no non-empty texts")

    prompts = []
    for i in range(n_prompts):
        rotated = texts[i % len(texts):] + texts[: i % len(texts)]
        joined = "\n".join(rotated)
        ids = tokenizer.encode(joined, add_special_tokens=False)
        if len(ids) < positions:
            raise SystemExit(
                f"prompt {i} tokenized to {len(ids)} ids, fewer than the "
                f"{positions} positions requested"
            )
        prompts.append((joined[:60], ids[:positions]))
    return prompts


def run_surge(surge_bin, model_dir, prompts, out_dir, jobs):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    t0 = time.time()
    for i, (_, ids) in enumerate(prompts):
        paths.append(out_dir / f"p{i:02d}.full.f32")
    pending = list(range(len(prompts)))
    running = {}
    failures = []
    while pending or running:
        while pending and len(running) < jobs:
            i = pending.pop(0)
            cmd = [str(surge_bin), str(model_dir),
                   "--ids", ",".join(str(x) for x in prompts[i][1]),
                   "--logits", str(paths[i]), "--quiet"]
            running[i] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE)
        done = []
        for i, p in running.items():
            rc = p.poll()
            if rc is not None:
                # communicate(), not stderr.read(): a child that filled the
                # 64 KB pipe before exiting would deadlock a bare read.
                _, err_b = p.communicate()
                err = (err_b or b"").decode("utf-8", "replace").strip()
                if rc != 0:
                    failures.append(f"prompt {i}: surge-ref exit {rc}: {err}")
                done.append(i)
        for i in done:
            del running[i]
            print(f"  surge-ref: {len(prompts) - len(pending) - len(running)}"
                  f"/{len(prompts)} done ({time.time() - t0:.0f} s)", flush=True)
        if running:
            time.sleep(0.25)
    if failures:
        raise SystemExit("\n".join(failures))
    return paths


def mlx_logits(model, mx, np, prompts, positions):
    out = []
    for _, ids in prompts:
        lg = model(mx.array(ids)[None])
        mx.eval(lg)
        out.append(np.array(lg, dtype=np.float32).reshape(positions, -1))
    return out


def apply_hf_norm_semantics(model, model_dir, mx):
    """Recompute the five shifted norm tensors as f32(w) + 1.0.

    mlx-lm already added 1.0 in bf16 at load; this replaces those parameters
    with the f32 shift transformers' Qwen3_5RMSNorm.forward performs. Reads
    the residual weights straight back out of the checkpoint shards so the
    bf16-rounded sum is never used as an input.
    """
    shards = sorted(pathlib.Path(model_dir).glob("*.safetensors"))
    fixed = []
    for shard in shards:
        raw = mx.load(str(shard))
        for k, v in raw.items():
            if "mtp." in k or v.ndim != 1:
                continue
            if not any(k.endswith(s) for s in NORM_SUFFIXES):
                continue
            live = k.replace("model.language_model.", "language_model.model.", 1)
            if not live.startswith("language_model."):
                live = "language_model." + live
            fixed.append((live, v.astype(mx.float32) + 1.0))
    if not fixed:
        raise SystemExit("no shifted norm tensors found; is this a qwen3_5 checkpoint?")
    # load_weights(strict=False) SILENTLY DROPS a key that does not exist in
    # the module tree, and the remap above is a guess about the checkpoint's
    # nesting. If it were wrong, every replacement would be dropped, the
    # "hf-norm" oracle would silently BE the stock oracle, and this script
    # would then attribute the whole difference to mlx's bf16 shift while
    # measuring nothing of the sort. Resolve every name first.
    from mlx.utils import tree_flatten

    known = {k for k, _ in tree_flatten(model.parameters())}
    missing = [k for k, _ in fixed if k not in known]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(fixed)} norm tensors do not resolve against the "
            f"model's parameter tree (first: {missing[0]}); the checkpoint's name "
            f"nesting is not what this script assumes, so the hf-norm oracle would "
            f"be indistinguishable from the stock one")
    model.load_weights(fixed, strict=False)
    mx.eval(model.parameters())
    return len(fixed)


def compare(surge, ref, np):
    """Per-prompt (top1_ok, positions, max_abs_delta)."""
    rows = []
    for s, r in zip(surge, ref):
        top1 = int((s.argmax(1) == r.argmax(1)).sum())
        rows.append((top1, s.shape[0], float(np.abs(s - r).max())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/macmini/models/qwen35-2b")
    ap.add_argument("--surge", default=str(REPO / "surge-ref"))
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--out", default=str(REPO / "tests" / "fixtures" / "m1"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--reuse", action="store_true",
                    help="skip the surge-ref runs and read the existing dumps "
                         "(only valid when the sidecar manifest matches)")
    ap.add_argument("--freeze", action="store_true",
                    help="rewrite the committed pNN.f32 regression digests. "
                         "Refused unless the gate passes and the dumps are fresh.")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.freeze and args.reuse:
        raise SystemExit("--freeze needs dumps this run produced; drop --reuse")

    import numpy as np
    import mlx.core as mx
    from mlx_lm.utils import load

    surge_bin = pathlib.Path(args.surge)
    if not surge_bin.exists():
        raise SystemExit(f"{surge_bin} not found; run `make surge-ref` first")

    print(f"loading {args.model} with mlx-lm ...", flush=True)
    model, tokenizer, config = load(args.model, return_config=True)
    model.set_dtype(mx.float32)
    model.eval()
    # From config.json, not from the model object: this is the number every
    # dump's length is checked against, so it must not come from anything
    # that could have been derived from the dumps.
    vocab = config.get("vocab_size") or config.get("text_config", {}).get("vocab_size")
    if not vocab:
        raise SystemExit("config.json has no vocab_size")

    prompts = build_prompts(args.prompts, args.positions, tokenizer)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.json").write_text(
        json.dumps([{"index": i, "preview": p[0], "ids": p[1]}
                    for i, p in enumerate(prompts)], indent=1),
        encoding="utf-8")

    (out_dir / "ids.txt").write_text(
        "".join(",".join(str(x) for x in p[1]) + "\n" for p in prompts),
        encoding="utf-8")

    # Sidecar manifest: what the dumps on disk actually are. Without it
    # --reuse silently accepts a dump from a different prompt set, a
    # different position count, or a stale build of surge-ref, and prints a
    # PASS about numbers nothing produced this run.
    manifest_path = out_dir / "dumps.json"
    manifest = {
        "positions": args.positions,
        "vocab": int(vocab),
        "ids": [p[1] for p in prompts],
        "surge_bin": str(surge_bin),
        "surge_size": surge_bin.stat().st_size,
        "surge_mtime": surge_bin.stat().st_mtime,
    }
    paths = [out_dir / f"p{i:02d}.full.f32" for i in range(len(prompts))]
    if args.reuse:
        if not manifest_path.exists():
            raise SystemExit(f"--reuse but {manifest_path} is missing")
        old = json.loads(manifest_path.read_text())
        for key in ("positions", "vocab", "ids"):
            if old.get(key) != manifest[key]:
                raise SystemExit(f"--reuse but the dumps on disk were made with a "
                                 f"different {key}")
        if (old.get("surge_size"), old.get("surge_mtime")) != \
                (manifest["surge_size"], manifest["surge_mtime"]):
            raise SystemExit("--reuse but surge-ref has been rebuilt since the dumps "
                             "were made; rerun without --reuse")
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"--reuse but {missing[0]} is missing")
    else:
        print(f"running {len(prompts)} surge-ref processes ({args.jobs} at a time) ...",
              flush=True)
        paths = run_surge(surge_bin, args.model, prompts, out_dir, args.jobs)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    surge = []
    want_floats = args.positions * int(vocab)
    for i, p in enumerate(paths):
        a = np.fromfile(p, dtype="<f4")
        # Exact, not "a multiple of positions": with vocab 248320 and 64
        # positions the modulo test can never fire, so it proved nothing.
        if a.size != want_floats:
            raise SystemExit(f"{p} has {a.size} floats, want {args.positions} x "
                             f"{vocab} = {want_floats}")
        surge.append(a.reshape(args.positions, int(vocab)))
    V = surge[0].shape[1]

    print("mlx-lm oracle 1/2: stock (norm shift applied in bf16) ...", flush=True)
    stock = mlx_logits(model, mx, np, prompts, args.positions)
    n_fixed = apply_hf_norm_semantics(model, args.model, mx)
    print(f"mlx-lm oracle 2/2: hf-norm ({n_fixed} norm tensors re-shifted in f32) ...",
          flush=True)
    hfnorm = mlx_logits(model, mx, np, prompts, args.positions)

    rows_stock = compare(surge, stock, np)
    rows_hf = compare(surge, hfnorm, np)
    between = float(max(np.abs(a - b).max() for a, b in zip(stock, hfnorm)))
    # The two oracles' own top-1 disagreement. If surge loses a top-1 against
    # stock mlx-lm at a position where the two oracles ALSO disagree, that
    # position was decided by mlx's bf16 +1.0 and not by surge.
    between_top1 = sum(int((a.argmax(1) == b.argmax(1)).sum())
                       for a, b in zip(stock, hfnorm))


    print()
    print(f"{len(prompts)} prompts x {args.positions} positions x {V} logits, "
          f"teacher-forced")
    print()
    print("        |------ vs mlx-lm stock ------|  |----- vs mlx-lm hf-norm -----|")
    print("prompt  |  top-1        max |delta|   |  |  top-1        max |delta|   |  preview")
    tot_s = tot_h = 0
    worst_s = worst_h = 0.0
    for i, ((ts, n, ds), (th, _, dh)) in enumerate(zip(rows_stock, rows_hf)):
        tot_s += ts
        tot_h += th
        worst_s = max(worst_s, ds)
        worst_h = max(worst_h, dh)
        preview = prompts[i][0].replace("\n", "\\n")[:32]
        print(f"  {i:2d}    |  {ts:3d}/{n:<3d}      {ds:.3e}      |  "
              f"{th:3d}/{n:<3d}      {dh:.3e}      |  {preview}")
    n_all = len(prompts) * args.positions
    print()
    print(f"TOTAL   |  {tot_s}/{n_all} ({100.0 * tot_s / n_all:.2f}%)  "
          f"max {worst_s:.4e}  |  {tot_h}/{n_all} ({100.0 * tot_h / n_all:.2f}%)  "
          f"max {worst_h:.4e}")
    print()
    print(f"mlx stock vs mlx hf-norm (the cost of mlx's bf16 +1.0): max |delta| "
          f"{between:.4e}, top-1 {between_top1}/{n_all} "
          f"({100.0 * between_top1 / n_all:.2f}%)")
    print()

    ok = (tot_h == n_all) and (worst_h < args.tol)
    verdict = "PASS" if ok else "FAIL"
    print(f"M1 GATE ({verdict}): oracle = mlx-lm with transformers' f32 norm shift, "
          f"top-1 {100.0 * tot_h / n_all:.2f}%, max |delta| {worst_h:.4e} "
          f"(need 100.00% and < {args.tol:g})")
    if tot_s != n_all or worst_s >= args.tol:
        print(f"       against stock mlx-lm: top-1 {100.0 * tot_s / n_all:.2f}%, "
              f"max |delta| {worst_s:.4e} -- the difference is mlx's bf16 +1.0, "
              f"not surge (see this file's header)")

    # THE DIGESTS ARE THE COMMITTED REGRESSION FIXTURE, so they are never
    # rewritten as a side effect of running the gate. Overwriting them on a
    # FAIL (or from stale --reuse bytes) would bake the regression in and
    # make tests/test_ref_fwd.c's frozen check pass forever afterwards.
    if args.freeze:
        if not ok:
            print("NOT freezing: --freeze requires the gate to pass")
        else:
            for i in range(len(paths)):
                write_digest(out_dir / f"p{i:02d}.f32", surge[i], np)
            print(f"froze {len(paths)} regression digests in {out_dir}")

    (out_dir / "result.json").write_text(json.dumps({
        "model": args.model,
        "prompts": len(prompts),
        "positions": args.positions,
        "vocab": V,
        "tolerance": args.tol,
        "stock": {"top1": tot_s, "of": n_all, "max_delta": worst_s},
        "hf_norm": {"top1": tot_h, "of": n_all, "max_delta": worst_h},
        "stock_vs_hf_norm_max_delta": between,
        "stock_vs_hf_norm_top1": between_top1,
        "per_prompt": [
            {"index": i,
             "stock": {"top1": rows_stock[i][0], "max_delta": rows_stock[i][2]},
             "hf_norm": {"top1": rows_hf[i][0], "max_delta": rows_hf[i][2]}}
            for i in range(len(prompts))],
        "pass": bool(ok),
    }, indent=1), encoding="utf-8")
    print(f"wrote {out_dir}/result.json and {len(paths)} logit dumps")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
