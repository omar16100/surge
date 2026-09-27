#!/usr/bin/env python3
"""Emit the compiled CoreML program that src/ane.m loads, plus its weights and a manifest.

WHY A GENERATOR AND NOT C. surge's rule is no third-party libraries, and this breaks it at BUILD
time (not at run time): CoreML programs are protobuf, and there is no Apple-shipped command that
authors one the way `xcrun metal` authors a metallib from source. So this is the ANE analogue of
the `$(METALLIB)` rule, with one honest difference recorded here rather than glossed: `xcrun
metal` ships with Xcode, and coremltools is a pip package in a venv (the Makefile's ANE_PY).
Emitting the protobuf directly from C would remove that dependency and is a later task; it is
not on the path to a first measurable result.

The compiled `.mlmodelc` is a build artifact exactly like `src/kernels.metallib`: generated,
not committed, loaded by path at runtime.

WHAT IT EMITS, into --out-dir:
  program.mlmodelc/   the compiled CoreML program, what src/ane.m opens
  weights.f16         W as raw fp16, [m][k] row-major, so the test can build its own
                      independent reference instead of trusting the model to check itself
  manifest.json       n, k, m, dtype, and the compute-device assignment CoreML reported

THE LAYOUT IS (1, K, 1, N) AND THAT IS NOT ARBITRARY. tools/ane_gemm_probe.py measured that at
n=1, the decode shape, a rank-2 `matmul` op is assigned to MLCPUComputeDevice while the
equivalent 1x1 convolution over a 4D (B, C, 1, S) tensor is assigned to the Neural Engine. Every
op here is therefore a conv, and activations are CHANNEL-MAJOR: element (c, s) sits at c*n + s.
surge's own GEMM is token-major, so a caller crossing this boundary transposes. That cost is
real and is the caller's to pay; hiding it inside this file would hide it from the measurement.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def log(msg):
    print(f"[ane_build] {msg}", flush=True)


def build_program(n, k, m, weight):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, k, 1, n), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        return mb.conv(x=x, weight=weight, strides=[1, 1], pad_type="valid", name="gemm")

    return prog


def device_report(model):
    """Record which processor CoreML assigned the op to, at BUILD time, into the manifest.

    A backend that cannot say where its work ran is not measurable, and the probe that preceded
    this found the assignment is size-dependent: at 512x512x512 the same op goes to the CPU. So
    a program built for a small shape may silently not be an ANE program at all, and the
    manifest is where that becomes visible instead of being discovered as a mysteriously slow
    benchmark later.
    """
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(),
                                            compute_units=ct.ComputeUnit.CPU_AND_NE)
        main = plan.model_structure.program.functions["main"]
        counts = {}
        for op in main.block.operations:
            info = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if info is None:
                continue
            dev = type(info.preferred_compute_device).__name__
            counts[dev] = counts.get(dev, 0) + 1
        return {"available": True, "op_counts": counts,
                "on_ane": any("Neural" in d for d in counts)}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}", "on_ane": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True, help="tokens (sequence positions)")
    ap.add_argument("--k", type=int, required=True, help="input features")
    ap.add_argument("--m", type=int, required=True, help="output features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--int8", action="store_true",
                    help="linearly quantize the weights. Measured 1.97x on this machine, and "
                         "the reference weights emitted alongside stay fp16, so a test using "
                         "them must widen its tolerance rather than expect fp16 parity.")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    # W is [m][k] row-major, the layout surge's own weights use. The conv wants [m][k][1][1],
    # which is the same bytes with two unit axes appended, so the reshape is free and the file
    # the test reads is byte-for-byte the weights the model holds.
    w = (rng.standard_normal((a.m, a.k)) * 0.02).astype(np.float16)
    log(f"weights {w.shape} fp16, {w.nbytes / 1e6:.1f} MB")

    prog = build_program(a.n, a.k, a.m, w.reshape(a.m, a.k, 1, 1))
    model = ct.convert(prog, convert_to="mlprogram",
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    if a.int8:
        from coremltools.optimize.coreml import (OpLinearQuantizerConfig, OptimizationConfig,
                                                 linear_quantize_weights)
        model = linear_quantize_weights(model, config=OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")))
        log("weights linearly quantized to int8")

    report = device_report(model)
    log(f"device assignment: {report}")

    # Copy the compiled directory out of coremltools' temp dir into the build output, so the
    # artifact outlives this process. get_compiled_model_path() is a temp path that is removed
    # when the model object is collected.
    src = model.get_compiled_model_path()
    dst = os.path.join(a.out_dir, "program.mlmodelc")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    log(f"compiled program -> {dst}")

    w.tofile(os.path.join(a.out_dir, "weights.f16"))
    manifest = {"n": a.n, "k": a.k, "m": a.m,
                "weight_dtype": "int8" if a.int8 else "fp16",
                "reference_weights": "weights.f16",
                "reference_weight_layout": "[m][k] row-major fp16",
                "activation_layout": "(1, k, 1, n) channel-major: element (c, s) at c*n + s",
                "device_assignment": report,
                "coremltools": ct.__version__}
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"manifest -> {a.out_dir}/manifest.json")

    if report.get("on_ane") is False:
        log("WARNING: CoreML assigned this program to the CPU, not the Neural Engine. "
            "That is expected below roughly 1024x1024x1024 and means any timing taken "
            "against it measures the CPU. Build a larger shape to exercise the ANE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
