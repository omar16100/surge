#!/usr/bin/env python3
"""Map what the Apple Neural Engine on this M3 Ultra is actually good for, by measurement.

WHAT THE FIRST PROBE SETTLED (tools/ane_gemm_probe.py, 2026-08-28):
  - ANE fp16 GEMM at 4096^3 is 8.5 to 9.0 TFLOPS against the GPU's 23.6, so 0.36x. The ANE
    cannot REPLACE the GPU at anything.
  - Run together, ANE keeps 1.004x and GPU keeps 0.999x of their solo rates, aggregate 1.198x.
    So the ANE ADDS. That is the only reason any of this is worth measuring further.
  - macmon's `ane_power` counter works here: 0.00 W idle, 2.5 to 4.1 W loaded.

WHAT THIS HARNESS ANSWERS, and why each one decides something:

  Q1 IS THE ANE BANDWIDTH-BOUND OR COMPUTE-BOUND, AND AT WHAT RATE?
     This is the crux. Decode at long context on this machine is memory-bandwidth-bound
     (llm-rnd Findings 16, 22, 25, 60), so the ANE can only ever help decode if it has real
     bandwidth. Sweeping the token count n from 1 to 4096 through a fixed weight stack walks
     the workload from purely bandwidth-bound (n=1, one FLOP per weight byte) to purely
     compute-bound (n=4096) without changing anything else. Effective GB/s at n=1 IS the ANE's
     usable memory bandwidth for inference.

  Q2 HOW MUCH ANE IS THERE? ioreg shows ane0 and ane1 behind an H1xANELoadBalancer, one per
     die. Whether ONE CoreML model reaches both, or whether two concurrent models are needed to
     get both, is unpublished for M3 Ultra and worth up to 2x.

  Q3 DOES INT8 BUY THE ADVERTISED ~1.9x? The ANE's int8 path is its fastest. LLM weights are
     already quantized, so if int8 doubles the ANE's rate the additive gain roughly doubles too.

  Q4 DOES IT CLAMP UNDER SUSTAINED LOAD? This harness was written against the premise that
     this machine's GPU is clamped by firmware to 338 MHz after ~4.5 minutes at high power
     (llm-rnd Finding 67). surge's own 256K telemetry did not support that premise on
     2026-08-15 (docs/15082026_prefill_duty_cycle_plan.md), so treat it as unverified. A burst
     measurement is still worth checking against a sustained one, and if the ANE holds its rate
     for ten minutes that is useful to know either way.

DESIGN NOTE, THE ONE THAT MAKES THE NUMBERS MEAN ANYTHING. A single CoreML predict carries
13 to 15 ms of round-trip overhead from Python, which at these shapes swamps the op. So the
model under test is a STACK of `--layers` sequential linear layers, not one. That amortizes
dispatch across real work, and it is also the honest shape: an ANE backend would submit a whole
layer stack per dispatch, never one matmul. Dispatch overhead is separately measured by a null
model of the same input shape and reported, never silently folded in.

Requires a venv with coremltools (the Makefile's ANE_PY). No torch: models are built with the MIL
builder.
"""
import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import machine_census
from ane_gemm_probe import (  # noqa: E402
    DEFAULT_GPU_PY, ane_power_sampler, device_assignment, _ran_on_ane)


# A point whose framework overhead is more than this share of its round trip is not reported as
# a rate. Set at 0.25 rather than 0.5 so the published numbers carry at most ~33% subtraction
# leverage; the fix for a dominated point is more layers, not a looser limit.
OVERHEAD_FRACTION_LIMIT = 0.25


def log(msg):
    print(f"[ane_envelope] {msg}", flush=True)


def build_stack(n, k, layers, seed=0):
    """`layers` sequential K->K 1x1 convs over a (1, K, 1, n) fp16 tensor.

    The conv form is not a stylistic choice. The first probe measured that at n=1, the single
    most important shape for decode, a rank-2 `matmul` is assigned to the CPU while the
    equivalent 1x1 conv is assigned to the ANE. Everything here therefore uses conv.

    Weights differ per layer so nothing can be cached or folded across the stack.
    """
    rng = np.random.default_rng(seed)
    ws = [(rng.standard_normal((k, k, 1, 1)) * 0.02).astype(np.float16) for _ in range(layers)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, k, 1, n), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        for i, w in enumerate(ws):
            x = mb.conv(x=x, weight=w, strides=[1, 1], pad_type="valid", name=f"l{i}")
        return x

    return prog


def build_null(n, k):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, k, 1, n), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        return mb.identity(x=x, name="null")

    return prog


def convert(prog, quantize_int8=False):
    model = ct.convert(prog, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_AND_NE,
                       minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16)
    if quantize_int8:
        from coremltools.optimize.coreml import (OpLinearQuantizerConfig,
                                                 OptimizationConfig,
                                                 linear_quantize_weights)
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
        model = linear_quantize_weights(model, config=cfg)
    return model


def time_model(model, x, reps, warmup):
    key = list(model.get_spec().description.input)[0].name
    feed = {key: x}
    for _ in range(warmup):
        model.predict(feed)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.predict(feed)
        ts.append(time.perf_counter() - t0)
    return ts


def measure_point(n, k, layers, reps, warmup, quantize_int8=False, sample_power=True):
    """One (n, k, layers) point: TFLOPS, effective GB/s over the weight stack, and both witnesses.

    Effective GB/s counts the WEIGHT bytes the stack must read once per predict. That is the
    number that matters for decode, where the whole cost is streaming parameters. Activation
    traffic is excluded because at n=1 it is three orders of magnitude smaller and including it
    would flatter the result.
    """
    prog = build_stack(n, k, layers)
    model = convert(prog, quantize_int8=quantize_int8)
    null = convert(build_null(n, k))
    assign = device_assignment(model)
    x = (np.random.default_rng(7).standard_normal((1, k, 1, n)) * 0.05).astype(np.float16)

    if sample_power:
        with ane_power_sampler() as sampler:
            null_ts = time_model(null, x, reps=max(6, reps // 3), warmup=warmup)
            ts = time_model(model, x, reps=reps, warmup=warmup)
        power = sampler.summary()
    else:
        null_ts = time_model(null, x, reps=max(6, reps // 3), warmup=warmup)
        ts = time_model(model, x, reps=reps, warmup=warmup)
        power = {}

    overhead = statistics.median(null_ts)
    round_trip = statistics.median(ts)
    compute = round_trip - overhead
    flops = 2.0 * n * k * k * layers
    weight_bytes = float(k) * k * layers * (1 if quantize_int8 else 2)
    overhead_fraction = overhead / round_trip if round_trip > 0 else 1.0

    # REFUSE TO REPORT AN OVERHEAD-DOMINATED POINT. `compute` is a difference of two measured
    # medians; when the framework overhead is most of the round trip, that difference is noise
    # amplified by a division. The smoke run of this harness produced "ANE 23.018 TFLOPS at
    # n=256", which beat the GPU by 2.6x and was pure subtraction error: 13.7 ms round trip
    # minus a 13.0 ms overhead. A number like that is worse than a missing one, because it
    # looks like a discovery. The point is reported with its timings and a flag instead.
    dominated = compute <= 0 or overhead_fraction > OVERHEAD_FRACTION_LIMIT

    return {
        "n": n, "k": k, "layers": layers, "int8": quantize_int8,
        "round_trip_median_s": round(round_trip, 6),
        "overhead_median_s": round(overhead, 6),
        "compute_median_s": round(compute, 6),
        "overhead_fraction": round(overhead_fraction, 3),
        "overhead_dominated": bool(dominated),
        "tflops": (None if dominated else round(flops / compute / 1e12, 3)),
        "weight_gbps": (None if dominated else round(weight_bytes / compute / 1e9, 1)),
        "ran_on_ane": _ran_on_ane(assign),
        "op_counts": assign.get("op_counts"),
        "power": power,
    }


def gpu_point(n, k, layers, reps, warmup, python_bin):
    """Same stack on the GPU: `layers` sequential (k,k) @ (k,n) fp16 matmuls."""
    src = f"""
import json, time, statistics
import mlx.core as mx
n, k, layers = {n}, {k}, {layers}
ws = [(mx.random.normal((k, k)) * 0.02).astype(mx.float16) for _ in range(layers)]
x0 = (mx.random.normal((k, n)) * 0.05).astype(mx.float16)
mx.eval(x0, *ws)
def run():
    x = x0
    for w in ws:
        x = w @ x
    mx.eval(x)
for _ in range({warmup}):
    run()
ts = []
for _ in range({reps}):
    t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
print(json.dumps({{"median_s": med,
                   "tflops": 2.0 * n * k * k * layers / med / 1e12,
                   "weight_gbps": float(k) * k * layers * 2 / med / 1e9}}))
"""
    import subprocess
    res = subprocess.run([python_bin, "-c", src], capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        return {"error": res.stderr[-300:]}
    d = json.loads(res.stdout.strip().splitlines()[-1])
    return {"n": n, "k": k, "layers": layers,
            "median_s": round(d["median_s"], 6),
            "tflops": round(d["tflops"], 3),
            "weight_gbps": round(d["weight_gbps"], 1)}


def _worker(args):
    """One ANE instance in its own process, for the dual-die test."""
    n, k, layers, reps, warmup, idx = args
    r = measure_point(n, k, layers, reps, warmup, sample_power=False)
    r["instance"] = idx
    return r


def dual_die(n, k, layers, reps, warmup, instances):
    """Q2: does a second concurrent ANE model add throughput?

    Separate PROCESSES, not threads: CoreML holds per-process state and the GIL would serialize
    the predict calls in one interpreter, which would produce a confident-looking 1.0x that
    measured Python rather than silicon.
    """
    log(f"dual-die: {instances} concurrent ANE instances at n={n}")
    with ane_power_sampler() as sampler:
        with mp.get_context("spawn").Pool(instances) as pool:
            rs = pool.map(_worker, [(n, k, layers, reps, warmup, i) for i in range(instances)])
    total = sum(r["tflops"] for r in rs if r.get("tflops"))
    return {"instances": instances, "per_instance": rs,
            "aggregate_tflops": round(total, 3) if total else None,
            "power": sampler.summary()}


def _gpu_loop_src(n, k, layers, seconds):
    """A GPU workload that runs for a wall-clock duration rather than a rep count, so it is
    still loaded for the whole window the ANE instances are measured in."""
    return f"""
import json, time, statistics
import mlx.core as mx
n, k, layers = {n}, {k}, {layers}
ws = [(mx.random.normal((k, k)) * 0.02).astype(mx.float16) for _ in range(layers)]
x0 = (mx.random.normal((k, n)) * 0.05).astype(mx.float16)
mx.eval(x0, *ws)
def run():
    x = x0
    for w in ws:
        x = w @ x
    mx.eval(x)
run()
ts, t_end = [], time.time() + {seconds}
while time.time() < t_end:
    t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
print(json.dumps({{"median_s": med, "n_iters": len(ts),
                   "tflops": 2.0 * n * k * k * layers / med / 1e12,
                   "weight_gbps": float(k) * k * layers * 2 / med / 1e9}}))
"""


def full_machine(n, k, layers, reps, warmup, ane_instances, python_bin, gpu_seconds=45):
    """THE DECISIVE CONCURRENCY TEST: every ANE die AND the GPU, all loaded at once.

    The earlier 4096^3 concurrency arm found ~0 interference, but that shape is compute-bound
    for both processors, so it never asked the question that matters. At a bandwidth-bound shape
    the ANE dies want ~254 GB/s and the GPU wants ~573 GB/s, and their sum is above what this
    machine has. Whether the additive result survives is therefore a property of the SHAPE, not
    of the hardware, and publishing the compute-bound number alone would overstate what an ANE
    backend can deliver on a real layer stack.

    Retention is reported per processor against its own solo rate, so starvation of one by the
    other is visible rather than hidden inside an aggregate.

    KNOWN FLAW (review, 2026-09-27): the GPU loop starts before the ANE workers compile their
    models, runs for a fixed 45 s, and its median is summed with independently timed ANE rates.
    Nothing guarantees the two timed intervals overlap, so a GPU median taken mostly while the
    ANE was idle would still read as ~1.0x retention. Results from this arm are provisional
    until both sides are timed over one synchronized window after compilation.
    """
    import subprocess
    log(f"full machine: {ane_instances} ANE instances + GPU, n={n}")
    proc = subprocess.Popen([python_bin, "-c", _gpu_loop_src(n, k, layers, gpu_seconds)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(3.0)  # let mlx load and enter its loop before the ANE side is timed
    with ane_power_sampler() as sampler:
        with mp.get_context("spawn").Pool(ane_instances) as pool:
            rs = pool.map(_worker, [(n, k, layers, reps, warmup, i)
                                    for i in range(ane_instances)])
    gpu_out, gpu_err = proc.communicate(timeout=600)

    ane_tflops = sum(r["tflops"] for r in rs if r.get("tflops"))
    ane_gbps = sum(r["weight_gbps"] for r in rs if r.get("weight_gbps"))
    out = {"ane_instances": ane_instances, "per_instance": rs,
           "ane_aggregate_tflops": round(ane_tflops, 3),
           "ane_aggregate_gbps": round(ane_gbps, 1),
           "power": sampler.summary()}
    if proc.returncode == 0 and gpu_out.strip():
        d = json.loads(gpu_out.strip().splitlines()[-1])
        out["gpu_tflops"] = round(d["tflops"], 3)
        out["gpu_gbps"] = round(d["weight_gbps"], 1)
        out["gpu_iters"] = d["n_iters"]
        out["machine_tflops"] = round(ane_tflops + d["tflops"], 3)
        out["machine_gbps"] = round(ane_gbps + d["weight_gbps"], 1)
    else:
        out["gpu_error"] = (gpu_err or "")[-400:]
    return out


def sustained(n, k, layers, minutes, python_bin=None):
    """Q4: does the ANE hold its rate under sustained load?

    Reports the rate in successive one-minute windows. A flat series means the ANE is not
    subject to the firmware GPU limiter this harness was written against (a clamp to 338 MHz
    after ~4.5 minutes; surge's own 256K telemetry did not support that premise, see
    docs/15082026_prefill_duty_cycle_plan.md), which would make it the more dependable of the
    two processors here despite being the slower one.
    """
    prog = build_stack(n, k, layers)
    model = convert(prog)
    x = (np.random.default_rng(7).standard_normal((1, k, 1, n)) * 0.05).astype(np.float16)
    key = list(model.get_spec().description.input)[0].name
    feed = {key: x}
    flops = 2.0 * n * k * k * layers

    log(f"sustained: {minutes} min at n={n}, k={k}, layers={layers}")
    windows = []
    with ane_power_sampler() as sampler:
        t_end = time.time() + minutes * 60
        while time.time() < t_end:
            w_end = min(time.time() + 60, t_end)
            ts = []
            while time.time() < w_end:
                t0 = time.perf_counter()
                model.predict(feed)
                ts.append(time.perf_counter() - t0)
            if ts:
                med = statistics.median(ts)
                windows.append({"n_preds": len(ts), "median_s": round(med, 6),
                                "tflops": round(flops / med / 1e12, 3)})
                log(f"  window {len(windows)}: {windows[-1]['tflops']} TFLOPS "
                    f"over {len(ts)} predicts")
    rates = [w["tflops"] for w in windows]
    return {"windows": windows,
            "first_window_tflops": rates[0] if rates else None,
            "last_window_tflops": rates[-1] if rates else None,
            "retention": (round(rates[-1] / rates[0], 3) if len(rates) > 1 and rates[0] else None),
            "power": sampler.summary()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-sweep", default="1,8,64,256,1024,4096")
    ap.add_argument("--gpu-python", default=DEFAULT_GPU_PY,
                    help="interpreter with mlx for the GPU arm (env SURGE_GPU_PY)")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--int8", action="store_true", help="Q3: also measure an int8 stack")
    ap.add_argument("--dual-die", type=int, default=0, help="Q2: N concurrent ANE instances")
    ap.add_argument("--full-machine", type=int, default=0,
                    help="N ANE instances AND the GPU at once, at --sustained-n")
    ap.add_argument("--sustained-min", type=float, default=0, help="Q4: minutes of sustained load")
    ap.add_argument("--sustained-n", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    log(f"coremltools {ct.__version__} | k={a.k} layers={a.layers}")
    census_before = machine_census.guard("before")
    out = {"k": a.k, "layers": a.layers, "sweep": [], "gpu_sweep": []}

    ns = [int(v) for v in a.n_sweep.split(",") if v.strip()]
    for n in ns:
        r = measure_point(n, a.k, a.layers, a.reps, a.warmup)
        out["sweep"].append(r)
        flag = " OVERHEAD-DOMINATED, not reported" if r["overhead_dominated"] else ""
        log(f"ANE  n={n:5d}: {r['tflops']} TFLOPS | {r['weight_gbps']} GB/s weights | "
            f"ane={r['ran_on_ane']} | oh {r['overhead_fraction']:.2f} | "
            f"power max {r['power'].get('ane_power_max_w')} W{flag}")
        if not a.skip_gpu:
            g = gpu_point(n, a.k, a.layers, a.reps, a.warmup, a.gpu_python)
            out["gpu_sweep"].append(g)
            log(f"GPU  n={n:5d}: {g.get('tflops')} TFLOPS | {g.get('weight_gbps')} GB/s weights")

    if a.int8:
        out["int8_sweep"] = []
        for n in ns:
            r = measure_point(n, a.k, a.layers, a.reps, a.warmup, quantize_int8=True)
            out["int8_sweep"].append(r)
            log(f"ANE8 n={n:5d}: {r['tflops']} TFLOPS | {r['weight_gbps']} GB/s | "
                f"ane={r['ran_on_ane']}")

    if a.dual_die > 0:
        out["dual_die"] = dual_die(a.sustained_n, a.k, a.layers, a.reps, a.warmup, a.dual_die)
        log(f"dual-die aggregate: {out['dual_die']['aggregate_tflops']} TFLOPS "
            f"across {a.dual_die} instances")

    if a.full_machine > 0:
        fm = full_machine(a.sustained_n, a.k, a.layers, a.reps, a.warmup,
                          a.full_machine, a.gpu_python)
        out["full_machine"] = fm
        log(f"FULL MACHINE: ane {fm['ane_aggregate_tflops']} TFLOPS / "
            f"{fm['ane_aggregate_gbps']} GB/s + gpu {fm.get('gpu_tflops')} TFLOPS / "
            f"{fm.get('gpu_gbps')} GB/s = {fm.get('machine_tflops')} TFLOPS / "
            f"{fm.get('machine_gbps')} GB/s")

    if a.sustained_min > 0:
        out["sustained"] = sustained(a.sustained_n, a.k, a.layers, a.sustained_min)
        log(f"sustained retention: {out['sustained']['retention']}x "
            f"({out['sustained']['first_window_tflops']} -> "
            f"{out['sustained']['last_window_tflops']} TFLOPS)")

    census_after = machine_census.census("after")
    machine_census.report(census_after)
    census_deltas = machine_census.compare(census_before, census_after)
    out["census"] = {"before": census_before, "after": census_after,
                        "deltas": census_deltas}

    text = json.dumps(out, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
        log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
