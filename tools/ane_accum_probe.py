#!/usr/bin/env python3
"""Is the ANE's reduction accumulator narrow (fp16) or wide (fp32 class)? Settle it by scaling.

WHY. tests/test_ane.c measured 1.101e-2 relative error on a k=1024 dot product and this project
attributed it to "the ANE accumulates in fp16". Bryngelson (arXiv:2606.22283) contradicts that
attribution directly, with probe evidence: "The running sum between those two rounding points is
not fp16... the reduction in between is held in a wide register of fp32 class on every ANE
device", and blames "the per-product fp16 rounding of the inputs and weights under heavy
cancellation, not the accumulator". That paper measured M1 and M5 only. Never M2, M3, or any
Ultra. So it does not actually cover this machine, and the attribution here has to be tested
rather than conceded or defended.

THE DISCRIMINATOR IS HOW ERROR SCALES WITH k, NOT ITS VALUE AT ONE k. Both stories predict error
at a single k; they disagree about the slope.

  NARROW (fp16 running sum): the partial sum is re-rounded k times, errors random-walk, so
  RELATIVE error grows like sqrt(k) * 2^-11.
  WIDE (fp32 accumulator, fp16 operands): each product carries a fixed ~2^-11 relative rounding
  and nothing re-rounds the sum, so relative error is roughly FLAT in k.

A sweep over k = 256, 1024, 4096, 16384 separates a factor of 8 in slope. Fitting log(err)
against log(k) gives an exponent near 0.5 for narrow and near 0.0 for wide.

INPUTS ARE BUILT TO AVOID CANCELLATION ON PURPOSE. Bryngelson's mechanism is specifically about
"heavy cancellation", so a random-sign input would exercise exactly the confound and could not
separate the two stories. All-positive operands keep the true sum large and well-conditioned, so
whatever error remains is about the reduction rather than about catastrophic subtraction. A
random-sign arm is measured alongside for contrast, since the gap between them IS the
cancellation effect the paper describes.

CAVEAT THIS PROBE CANNOT REMOVE: a TREE reduction performed in fp16 scales like
sqrt(log k), which is much flatter than sqrt(k) and could be mistaken for a wide accumulator at
this k range. So a flat result means "not a sequential fp16 running sum", which is weaker than
"fp32 accumulator". Stated here rather than discovered later.
"""
import json
import sys

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ane_gemm_probe import device_assignment, _ran_on_ane  # noqa: E402


def log(msg):
    print(f"[ane_accum] {msg}", flush=True)


def dot_model(k, m, w):
    """m output rows, each a length-k dot product, in the ANE-native conv form."""
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, k, 1, 1), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        return mb.conv(x=x, weight=w.reshape(m, k, 1, 1), strides=[1, 1],
                       pad_type="valid", name="dot")

    return prog


# m=4096, not a smaller number, because CoreML places by workload size: the first run of this
# probe used m=64 and every point came back on_ane=False, i.e. it measured the CPU. n=1 with
# k=m=4096 is the smallest shape this project has confirmed lands on the ANE.
def measure(k, signed, m=4096, seed=0):
    rng = np.random.default_rng(seed)
    if signed:
        w = (rng.standard_normal((m, k)) * 0.05).astype(np.float16)
        x = (rng.standard_normal((k,)) * 0.05).astype(np.float16)
    else:
        # All-positive, narrow dynamic range: no cancellation, well-conditioned sum.
        w = (rng.random((m, k)) * 0.02 + 0.01).astype(np.float16)
        x = (rng.random((k,)) * 0.02 + 0.01).astype(np.float16)

    model = ct.convert(dot_model(k, m, w), convert_to="mlprogram",
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    assign = device_assignment(model)
    key = list(model.get_spec().description.input)[0].name
    out = model.predict({key: x.reshape(1, k, 1, 1)})
    got = np.array(list(out.values())[0]).reshape(m).astype(np.float64)

    # Reference from the SAME fp16 bytes the model holds, in float64. So any gap is the
    # reduction, never a difference in the operands.
    want = w.astype(np.float64) @ x.astype(np.float64)
    rel = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
    return {"k": k, "signed": signed, "rel_err": rel,
            "on_ane": _ran_on_ane(assign),
            "mean_want": float(np.mean(np.abs(want)))}


def main():
    ks = [256, 1024, 4096, 16384]
    results = []
    for signed in (False, True):
        label = "signed (cancellation)" if signed else "positive (no cancellation)"
        log(f"--- {label} ---")
        for k in ks:
            r = measure(k, signed)
            results.append(r)
            log(f"k={k:6d}  rel_err={r['rel_err']:.4e}  on_ane={r['on_ane']}")

    log("")
    log("SLOPE FIT: exponent p in rel_err ~ k^p. Narrow fp16 running sum predicts p ~ 0.5; "
        "a wide fp32-class accumulator predicts p ~ 0.0")
    for signed in (False, True):
        rows = [r for r in results if r["signed"] == signed and r["on_ane"] and r["rel_err"] > 0]
        if len(rows) < 2:
            log(f"signed={signed}: too few ANE points to fit")
            continue
        p = np.polyfit(np.log([r["k"] for r in rows]),
                       np.log([r["rel_err"] for r in rows]), 1)[0]
        verdict = ("NARROW, consistent with an fp16 running sum" if p > 0.35 else
                   "WIDE, NOT a sequential fp16 running sum" if p < 0.2 else
                   "AMBIGUOUS")
        log(f"signed={str(signed):5s}: p = {p:+.3f}  ->  {verdict}")

    with open("/tmp/ane_accum.json", "w") as fh:
        json.dump(results, fh, indent=2)
    log("wrote /tmp/ane_accum.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
