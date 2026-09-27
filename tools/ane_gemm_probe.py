#!/usr/bin/env python3
"""Measure the Apple Neural Engine's fp16 GEMM ceiling, and prove where the work actually ran.

WHY THIS EXISTS. surge is C11 + Metal. Before any ANE backend is worth building, one number
decides it: can the ANE do fp16 matmul faster than the GPU already does? The llm-rnd GEMM gate
answers the GPU half at 4096x4096 but in FP32 (`mx.random.normal` defaults to float32), while
every published ANE figure is fp16 or int8. Those two numbers have never been comparable, so the
ANE question has been argued rather than measured. This measures both halves in fp16 on the same
shape.

THE HARD PART IS NOT TIMING, IT IS PROVING WHERE IT RAN. CoreML silently falls back: ask for
`CPU_AND_NE` and an unsupported op quietly runs on the CPU, giving a plausible number from the
wrong processor. Two independent witnesses are required here and BOTH are recorded:

  1. `MLComputePlan` (macOS 15+) reports the per-operation device assignment from CoreML itself.
     This is the authoritative one.
  2. `ane_power` sampled from macmon across the run. This is the physical one, and it is also
     the control llm-rnd Finding 41 never had: that finding read 0.00 W as "the ANE did no
     work", but 0.00 W is equally what a blind counter reads. If witness 1 says ANE and witness
     2 still says 0.00 W, the counter is blind on M3 Ultra and Finding 41's evidence has to be
     downgraded.

FRAMEWORK OVERHEAD IS SUBTRACTED, NOT IGNORED. A `predict()` call carries Python and CoreML
marshalling cost that has nothing to do with the matmul. A null model of the same input shape is
timed the same way and its median is subtracted, so the reported TFLOPS is compute, not
round-trip. The raw round-trip number is reported alongside it so the subtraction is auditable.

Requires: a venv with coremltools 9.0 and numpy (the Makefile's ANE_PY). No torch, no third-party
converter: the model is built directly with the MIL builder.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

import machine_census
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


# The interpreter the GPU (mlx) comparison arm runs under. Env-overridable so no machine's
# home directory is baked into the source; shared with ane_envelope.py and ane_vs_all.py.
DEFAULT_GPU_PY = os.environ.get("SURGE_GPU_PY",
                                os.path.expanduser("~/models/dsv4-venv/bin/python"))


def log(msg):
    print(f"[ane_gemm] {msg}", flush=True)


class ane_power_sampler:
    """Sample macmon's ane_power in a background thread for the duration of a run.

    Kept deliberately dumb: it records every sample it sees and never interprets them. The
    interpretation (median, max, whether the counter ever moved) happens once at the end, so a
    sampling bug cannot quietly become a conclusion.
    """

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []
        self.gpu_samples = []
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["macmon", "pipe", "-s", "0", "-i", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            log("macmon not found: ane_power witness UNAVAILABLE for this run")
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ane_power" in d:
                self.samples.append(float(d["ane_power"]))
            if "gpu_power" in d:
                self.gpu_samples.append(float(d["gpu_power"]))

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait for a real first sample rather than a fixed guess. macmon's startup was slower
        # than an assumed 0.6s on the first run of this probe and every arm reported
        # n_samples 0, which would have been indistinguishable from a genuinely idle ANE.
        deadline = time.time() + 5.0
        while not self.samples and time.time() < deadline:
            time.sleep(0.1)
        if not self.samples:
            log("macmon produced no sample within 5s: ane_power witness DEGRADED for this arm")
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=3)
        return False

    def summary(self):
        if not self.samples:
            return {"n_samples": 0, "ane_power_median_w": None, "ane_power_max_w": None,
                    "ane_counter_moved": None, "gpu_power_median_w": None}
        return {
            "n_samples": len(self.samples),
            "ane_power_median_w": round(statistics.median(self.samples), 4),
            "ane_power_max_w": round(max(self.samples), 4),
            "ane_counter_moved": bool(max(self.samples) > 0.0),
            "gpu_power_median_w": (round(statistics.median(self.gpu_samples), 3)
                                   if self.gpu_samples else None),
        }


# The two ways to express one linear layer, and the whole reason this probe has variants.
#
# `matmul_2d` is the obvious form, Y[n,m] = X[n,k] @ W[k,m]. It is what any engine would write
# first and it is NOT ANE-eligible: the first run of this probe put all 3 of its ops on
# MLCPUComputeDevice. Kept as a measured arm rather than deleted, because "the obvious form
# silently runs on the CPU" is the single most important fact for anyone wiring an ANE backend,
# and a claim of that shape should carry its own evidence.
#
# `conv_1x1` is the ANE-native form and is the same arithmetic: a linear layer written as a 1x1
# convolution over a 4D (B, C, 1, S) tensor, channels-as-features and sequence-as-width. This is
# the layout Apple's own ml-ane-transformers uses throughout, and it exists because the ANE's
# datapath is built around convolution, not around rank-2 matmul.
VARIANTS = ("matmul_2d", "conv_1x1")


def build_gemm_model(variant, n, k, m, seed=0):
    """One linear layer, n tokens, k input features, m output features, fp16.

    THE WEIGHT IS A BAKED CONSTANT ON PURPOSE. That is not a shortcut, it is the realistic
    shape: in an inference engine the weight matrix is constant and the activation is the input.
    It is also the only shape the ANE is built for, since its weight path is designed around
    pre-laid-out parameters. A two-input matmul would measure a configuration no LLM backend
    would ever use.
    """
    rng = np.random.default_rng(seed)

    if variant == "matmul_2d":
        w = (rng.standard_normal((k, m)) * 0.02).astype(np.float16)

        @mb.program(input_specs=[mb.TensorSpec(shape=(n, k), dtype=types.fp16)],
                    opset_version=ct.target.macOS15)
        def prog(x):
            return mb.matmul(x=x, y=w, name="gemm")

        return prog

    # conv_1x1: input (1, k, 1, n), weight (m, k, 1, 1), output (1, m, 1, n).
    w = (rng.standard_normal((m, k, 1, 1)) * 0.02).astype(np.float16)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, k, 1, n), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        return mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", name="gemm")

    return prog


def build_null_model(variant, n, k):
    """Same input shape, no arithmetic worth measuring. Its median predict time IS the framework
    overhead, and subtracting it is what turns a round-trip number into a compute number."""
    shape = (n, k) if variant == "matmul_2d" else (1, k, 1, n)

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        return mb.identity(x=x, name="null")

    return prog


def input_array(variant, n, k, seed=1234):
    rng = np.random.default_rng(seed)
    shape = (n, k) if variant == "matmul_2d" else (1, k, 1, n)
    return (rng.standard_normal(shape) * 0.05).astype(np.float16)


def compile_model(prog, compute_units, tag):
    t0 = time.time()
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    log(f"{tag}: compiled in {time.time() - t0:.1f}s, compute_units={compute_units}")
    return model


def device_assignment(model):
    """Witness 1: ask CoreML itself which processor each op was assigned to.

    Returns a dict of device name to op count, or a reason string when the API is unavailable.
    A missing plan is reported as UNKNOWN and never silently treated as ANE.
    """
    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=model.get_compiled_model_path(),
            compute_units=model.compute_unit,
        )
    except Exception as e:  # noqa: BLE001 - the API surface moves between OS versions
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    counts, supported = {}, set()
    try:
        prog = plan.model_structure.program
        main = prog.functions["main"]
        for op in main.block.operations:
            info = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if info is None:
                continue  # const ops carry no device usage
            dev = type(info.preferred_compute_device).__name__
            counts[dev] = counts.get(dev, 0) + 1
            for d in info.supported_compute_devices:
                supported.add(type(d).__name__)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"walk failed: {type(e).__name__}: {e}"}
    # SUPPORTED and PREFERRED are different questions and both are recorded. The first probe run
    # conflated them: at 512x512x512 both forms reported CPU as preferred, which reads as "the
    # ANE cannot do this" when the truth was "the ANE can, and CoreML chose not to at this size".
    # Only the preferred set decides `ran_on_ane`; supported is context.
    return {"available": True, "op_counts": counts, "supported_devices": sorted(supported)}


def time_predict(model, x, reps, warmup):
    key = list(model.get_spec().description.input)[0].name
    feed = {key: x}
    for _ in range(warmup):
        model.predict(feed)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append(time.perf_counter() - t0)
    return times


def run_ane(variant, n, k, m, reps, warmup):
    log(f"ANE arm [{variant}]: one linear layer, {n} tokens, {k} in, {m} out, fp16")
    gemm = compile_model(build_gemm_model(variant, n, k, m), ct.ComputeUnit.CPU_AND_NE,
                         f"{variant}/gemm")
    null = compile_model(build_null_model(variant, n, k), ct.ComputeUnit.CPU_AND_NE,
                         f"{variant}/null")

    assign = device_assignment(gemm)
    log(f"[{variant}] device assignment (witness 1): {assign}")

    x = input_array(variant, n, k)

    with ane_power_sampler() as sampler:
        null_times = time_predict(null, x, reps=max(8, reps // 4), warmup=warmup)
        gemm_times = time_predict(gemm, x, reps=reps, warmup=warmup)
    power = sampler.summary()
    log(f"[{variant}] ane power (witness 2): {power}")

    overhead_s = statistics.median(null_times)
    round_trip_s = statistics.median(gemm_times)
    compute_s = round_trip_s - overhead_s
    flops = 2.0 * n * k * m

    out = {
        "arm": "ane",
        "variant": variant,
        "shape": {"n": n, "k": k, "m": m},
        "reps": reps,
        "round_trip_median_s": round(round_trip_s, 6),
        "framework_overhead_median_s": round(overhead_s, 6),
        "compute_median_s": round(compute_s, 6),
        "tflops_round_trip": round(flops / round_trip_s / 1e12, 3),
        "tflops_compute": (round(flops / compute_s / 1e12, 3) if compute_s > 0 else None),
        "device_assignment": assign,
        "ran_on_ane": _ran_on_ane(assign),
        "power": power,
    }
    if compute_s <= 0:
        log("WARNING: overhead exceeded round trip. The shape is too small to measure; "
            "tflops_compute is reported as null rather than as a negative or an inflated number.")
    return out


def _ran_on_ane(assign):
    """True only when witness 1 positively says so. UNKNOWN stays None, never False, and never
    True: a missing compute plan is an absence of evidence and must not be reported as either
    verdict."""
    if not assign.get("available"):
        return None
    counts = assign.get("op_counts", {})
    if not counts:
        return None
    return any("Neural" in dev or "ANE" in dev for dev in counts)


def run_gpu(n, k, m, reps, warmup, python_bin):
    """GPU arm, same shape, same fp16, run out of process in the mlx venv.

    Out of process on purpose: importing mlx into this interpreter would put a Metal context
    alongside the CoreML one, and the two would then be sharing the machine during each other's
    timed sections.
    """
    src = f"""
import json, time, statistics
import mlx.core as mx
n, k, m, reps, warmup = {n}, {k}, {m}, {reps}, {warmup}
a = (mx.random.normal((n, k)) * 0.05).astype(mx.float16)
b = (mx.random.normal((k, m)) * 0.02).astype(mx.float16)
mx.eval(a, b)
for _ in range(warmup):
    mx.eval(a @ b)
ts = []
for _ in range(reps):
    t0 = time.perf_counter()
    mx.eval(a @ b)
    ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
print(json.dumps({{"median_s": med, "tflops": 2.0 * n * k * m / med / 1e12,
                   "dtype": "float16"}}))
"""
    log(f"GPU arm: same shape in fp16 via {python_bin}")
    with ane_power_sampler() as sampler:
        res = subprocess.run([python_bin, "-c", src], capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        log(f"GPU arm FAILED rc={res.returncode}: {res.stderr[-500:]}")
        return {"arm": "gpu", "error": res.stderr[-500:]}
    d = json.loads(res.stdout.strip().splitlines()[-1])
    return {
        "arm": "gpu",
        "shape": {"n": n, "k": k, "m": m},
        "reps": reps,
        "median_s": round(d["median_s"], 6),
        "tflops": round(d["tflops"], 3),
        "dtype": d["dtype"],
        "power": sampler.summary(),
    }


def run_concurrent(variant, n, k, m, reps, warmup, python_bin):
    """Run the ANE and GPU arms AT THE SAME TIME and report what each keeps.

    THIS IS THE ONLY QUESTION THAT DECIDES THE BACKEND'S SHAPE. The solo arms already say the
    ANE is the slower processor, so an ANE backend can never be a replacement. It can only be
    worth building if the ANE's throughput ADDS to the GPU's rather than being taken out of it.
    Two processors on one memory system may or may not add: at this shape the arithmetic
    intensity is high (~1400 FLOP per byte), so bandwidth should not be the binding constraint
    and the sum should hold. Should is not a measurement.

    Reported as a retention fraction per processor (concurrent divided by solo) plus the
    aggregate, so a result where one processor simply starves the other is visible rather than
    hidden inside a single summed number.
    """
    log(f"CONCURRENT arm [{variant}]: ANE and GPU running the same shape simultaneously")
    gemm = compile_model(build_gemm_model(variant, n, k, m), ct.ComputeUnit.CPU_AND_NE,
                         f"concurrent/{variant}")
    assign = device_assignment(gemm)
    if not _ran_on_ane(assign):
        log(f"concurrent arm ABORTED: witness 1 says {assign}, not the ANE")
        return {"arm": "concurrent", "variant": variant, "error": "not scheduled on ANE",
                "device_assignment": assign}
    x = input_array(variant, n, k)

    # The GPU side runs out of process and writes its own median to stdout. It is started
    # first and given a longer rep count so it is still busy for the whole ANE window; the
    # ANE median is therefore taken under genuine contention rather than after the GPU
    # finished early.
    gpu_src = f"""
import json, time, statistics
import mlx.core as mx
n, k, m = {n}, {k}, {m}
a = (mx.random.normal((n, k)) * 0.05).astype(mx.float16)
b = (mx.random.normal((k, m)) * 0.02).astype(mx.float16)
mx.eval(a, b)
for _ in range({warmup}):
    mx.eval(a @ b)
ts = []
for _ in range({reps * 6}):
    t0 = time.perf_counter()
    mx.eval(a @ b)
    ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
print(json.dumps({{"median_s": med, "tflops": 2.0 * n * k * m / med / 1e12}}))
"""
    flops = 2.0 * n * k * m
    with ane_power_sampler() as sampler:
        proc = subprocess.Popen([python_bin, "-c", gpu_src], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        time.sleep(2.0)  # let the GPU side finish loading mlx and get into its loop
        ane_times = time_predict(gemm, x, reps=reps, warmup=warmup)
        gpu_out, gpu_err = proc.communicate(timeout=900)
    power = sampler.summary()

    ane_round_trip = statistics.median(ane_times)
    out = {
        "arm": "concurrent",
        "variant": variant,
        "shape": {"n": n, "k": k, "m": m},
        "ane_round_trip_median_s": round(ane_round_trip, 6),
        "ane_tflops_round_trip": round(flops / ane_round_trip / 1e12, 3),
        "power": power,
        "device_assignment": assign,
    }
    if proc.returncode == 0 and gpu_out.strip():
        d = json.loads(gpu_out.strip().splitlines()[-1])
        out["gpu_median_s"] = round(d["median_s"], 6)
        out["gpu_tflops"] = round(d["tflops"], 3)
    else:
        out["gpu_error"] = (gpu_err or "")[-400:]
        log(f"concurrent GPU side failed rc={proc.returncode}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4096, help="rows of the activation")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--concurrent", action="store_true",
                    help="also run ANE and GPU simultaneously, the arm that decides whether an "
                         "ANE backend can ADD throughput rather than merely relocate it")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help=f"comma-separated subset of {VARIANTS}")
    ap.add_argument("--gpu-python", default=DEFAULT_GPU_PY,
                    help="interpreter with mlx for the GPU arm (env SURGE_GPU_PY)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log(f"coremltools {ct.__version__} | numpy {np.__version__} | pid {os.getpid()}")
    census_before = machine_census.guard("before")
    results = {"shape": {"n": a.n, "k": a.k, "m": a.m}, "arms": []}

    for variant in [v.strip() for v in a.variants.split(",") if v.strip()]:
        if variant not in VARIANTS:
            log(f"unknown variant {variant!r}, skipping")
            continue
        results["arms"].append(run_ane(variant, a.n, a.k, a.m, a.reps, a.warmup))
    if not a.skip_gpu:
        results["arms"].append(run_gpu(a.n, a.k, a.m, a.reps, a.warmup, a.gpu_python))

    gpu = next((r for r in results["arms"] if r["arm"] == "gpu"), None)
    # Compare the GPU only against an arm that PROVABLY ran on the ANE. Ratioing against a
    # CPU-assigned arm would publish a number labelled "ane/gpu" that measured neither.
    ane = next((r for r in results["arms"]
                if r["arm"] == "ane" and r.get("ran_on_ane") and r.get("tflops_compute")), None)
    if ane and gpu and gpu.get("tflops"):
        results["ane_over_gpu"] = round(ane["tflops_compute"] / gpu["tflops"], 4)
        results["ane_variant_compared"] = ane["variant"]
        log(f"VERDICT ratio ane/gpu = {results['ane_over_gpu']}x on {ane['variant']} "
            f"({ane['tflops_compute']} vs {gpu['tflops']} TFLOPS fp16)")
    else:
        results["ane_over_gpu"] = None
        log("NO RATIO PUBLISHED: no arm both ran on the ANE (witness 1) and produced a "
            "positive compute time. That absence is the result, not a failure of the probe.")

    if a.concurrent and ane:
        con = run_concurrent(ane["variant"], a.n, a.k, a.m, a.reps, a.warmup, a.gpu_python)
        results["arms"].append(con)
        # Retention is measured against each processor's OWN solo number, so a result where the
        # GPU simply absorbs the ANE's share is visible instead of being hidden in the sum.
        if gpu and gpu.get("tflops") and con.get("gpu_tflops") and con.get("ane_tflops_round_trip"):
            ane_solo_rt = flops_rt = ane["tflops_round_trip"]
            results["concurrency"] = {
                "ane_retention": round(con["ane_tflops_round_trip"] / ane_solo_rt, 3),
                "gpu_retention": round(con["gpu_tflops"] / gpu["tflops"], 3),
                "aggregate_tflops": round(con["ane_tflops_round_trip"] + con["gpu_tflops"], 3),
                "gpu_solo_tflops": gpu["tflops"],
                "note": ("ANE side is ROUND-TRIP TFLOPS (dispatch included), compared against "
                         "its own round-trip solo number, so the retention ratio is like for "
                         "like. It is not comparable to the compute-only figure."),
            }
            c = results["concurrency"]
            log(f"CONCURRENCY: ane keeps {c['ane_retention']}x, gpu keeps {c['gpu_retention']}x, "
                f"aggregate {c['aggregate_tflops']} vs gpu-alone {c['gpu_solo_tflops']} TFLOPS")
            _ = flops_rt

    census_after = machine_census.census("after")
    machine_census.report(census_after)
    census_deltas = machine_census.compare(census_before, census_after)
    results["census"] = {"before": census_before, "after": census_after,
                        "deltas": census_deltas}

    text = json.dumps(results, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        log(f"wrote {a.out}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
