#!/usr/bin/env python3
"""ANE vs GPU vs CPU on the same GEMM: throughput, power, efficiency, and sustained behaviour.

THE QUESTION THIS ANSWERS is narrower than "which is fastest", which is already settled: the GPU
is, at 23.6 TFLOPS fp16 against the ANE's 8.2 per die. This asks what the ANE is BETTER at, which
needs axes where raw throughput is not the figure of merit:

  PERF PER WATT. The ANE draws single-digit watts. If it delivers a third of the GPU's throughput
  for a tenth of the power it is the right processor for anything running continuously, and the
  wrong one for anything latency-critical.

  SUSTAINED RATE. The ANE held a flat 11-minute series (llm-rnd Finding 78). Whether the GPU does
  the same under an equivalent load has never been tested head to head on this box, and the two
  existing accounts CONTRADICT each other: llm-rnd records a firmware limiter clamping to 338 MHz
  after ~4.5 minutes, while surge's own B8 analysis of a 30-hour run found clock FLAT across
  bursts, 338 MHz in 0.3 percent of loaded samples, and context length rather than time as the
  predictor. One of those is wrong for this workload and a matched sustained arm settles it.

  THE CPU IS IN THE COMPARISON because it has been assumed irrelevant and never measured. numpy
  here is built against Accelerate, so a CPU matmul reaches Apple's AMX units, which is not
  obviously slower than the ANE.

POWER IS ATTRIBUTED TWO WAYS AND BOTH ARE REPORTED. The per-unit counters (`ane_power`,
`gpu_power`, `cpu_power`) are what the SoC reports for each block. `sys_power` minus a measured
idle baseline is the whole-package delta, which catches memory and fabric power that a per-unit
counter attributes to nobody. They disagree, and the disagreement is informative rather than an
error: quoting only the per-unit number would flatter whichever processor moves the most bytes.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import machine_census
from ane_gemm_probe import DEFAULT_GPU_PY, ane_power_sampler  # noqa: E402

# Env-overridable (SURGE_GPU_PY); see ane_gemm_probe.DEFAULT_GPU_PY.
GPU_PY = DEFAULT_GPU_PY


def log(msg):
    print(f"[ane_vs_all] {msg}", flush=True)


class power_trace:
    """Full macmon trace, not just ane_power, so every arm is scored on the same fields."""

    def __init__(self, interval_ms=400):
        self.interval_ms = interval_ms
        self.rows = []
        self._proc = None
        self._stop = False

    def __enter__(self):
        import threading
        self._proc = subprocess.Popen(
            ["macmon", "pipe", "-s", "0", "-i", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def run():
            for line in self._proc.stdout:
                if self._stop:
                    break
                line = line.strip()
                if line.startswith("{"):
                    try:
                        self.rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        deadline = time.time() + 5
        while not self.rows and time.time() < deadline:
            time.sleep(0.1)
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._proc:
            self._proc.terminate()
        return False

    def stats(self):
        def med(key):
            vals = [r.get(key) for r in self.rows if isinstance(r.get(key), (int, float))]
            return round(statistics.median(vals), 3) if vals else None
        return {"n": len(self.rows), "ane_w": med("ane_power"), "gpu_w": med("gpu_power"),
                "cpu_w": med("cpu_power"), "ram_w": med("ram_power"),
                "sys_w": med("sys_power"), "gpu_mhz": med("gpu_freq_mhz"),
                "gpu_temp": med("temp.gpu_temp_avg")}


def run_subproc(src, timeout=3600):
    res = subprocess.run([GPU_PY, "-c", src], capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        return {"error": res.stderr[-400:]}
    return json.loads(res.stdout.strip().splitlines()[-1])


def gpu_src(n, k, m, seconds):
    return f"""
import json, time, statistics
import mlx.core as mx
n, k, m = {n}, {k}, {m}
a = (mx.random.normal((n, k))).astype(mx.float16)
b = (mx.random.normal((k, m))).astype(mx.float16)
mx.eval(a, b); mx.eval(a @ b)
ts, t_end = [], time.time() + {seconds}
while time.time() < t_end:
    t0 = time.perf_counter(); mx.eval(a @ b); ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
print(json.dumps({{"median_s": med, "iters": len(ts),
                   "tflops": 2.0 * n * k * m / med / 1e12}}))
"""


def gpu_sustained_src(n, k, m, minutes):
    """Windowed exactly like ane_envelope.sustained, so the two series are comparable."""
    return f"""
import json, time, statistics
import mlx.core as mx
n, k, m = {n}, {k}, {m}
a = (mx.random.normal((n, k))).astype(mx.float16)
b = (mx.random.normal((k, m))).astype(mx.float16)
mx.eval(a, b); mx.eval(a @ b)
windows, t_end = [], time.time() + {minutes} * 60
while time.time() < t_end:
    w_end = min(time.time() + 60, t_end)
    ts = []
    while time.time() < w_end:
        t0 = time.perf_counter(); mx.eval(a @ b); ts.append(time.perf_counter() - t0)
    if ts:
        med = statistics.median(ts)
        windows.append({{"iters": len(ts), "tflops": 2.0 * n * k * m / med / 1e12}})
print(json.dumps({{"windows": windows}}))
"""


def cpu_arm(n, k, m, seconds):
    """Accelerate BLAS via numpy. fp32, because numpy has no native fp16 GEMM: it would upcast
    and the number would then measure a conversion, not the CPU's matmul."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, k), dtype=np.float32)
    b = rng.standard_normal((k, m), dtype=np.float32)
    a @ b
    ts, t_end = [], time.time() + seconds
    while time.time() < t_end:
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    return {"median_s": med, "iters": len(ts), "tflops": 2.0 * n * k * m / med / 1e12,
            "dtype": "float32 (Accelerate BLAS, no native fp16 GEMM in numpy)"}


def ane_arm(n, k, m, seconds, instances):
    """`instances` concurrent ANE processes. One CoreML model reaches one die, so two instances
    are what it takes to use the whole ANE (measured 2.001x, llm-rnd Finding 78)."""
    src = f"""
import sys, json, time, statistics
sys.path.insert(0, "{__file__.rsplit('/', 1)[0]}")
import numpy as np, coremltools as ct
from ane_envelope import build_stack, convert
n, k, layers = {n}, {k}, 32
model = convert(build_stack(n, k, layers))
x = (np.random.default_rng(7).standard_normal((1, k, 1, n)) * 0.05).astype(np.float16)
key = list(model.get_spec().description.input)[0].name
feed = {{key: x}}
model.predict(feed)
ts, t_end = [], time.time() + {seconds}
while time.time() < t_end:
    t0 = time.perf_counter(); model.predict(feed); ts.append(time.perf_counter() - t0)
med = statistics.median(ts)
print(json.dumps({{"median_s": med, "iters": len(ts),
                   "tflops": 2.0 * n * k * k * layers / med / 1e12}}))
"""
    procs = [subprocess.Popen([sys.executable, "-c", src], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True) for _ in range(instances)]
    outs = [p.communicate(timeout=1800)[0] for p in procs]
    total = 0.0
    per = []
    for o in outs:
        if not o.strip():
            continue
        d = json.loads(o.strip().splitlines()[-1])
        per.append(round(d["tflops"], 3))
        total += d["tflops"]
    return {"instances": instances, "per_instance_tflops": per,
            "tflops": round(total, 3)}


def efficiency(tflops, stats, idle, unit_key):
    """TFLOPS per watt, both ways. `unit_delta` uses the block's own counter minus its idle
    reading; `sys_delta` uses whole-package power minus the idle baseline and is the honest
    upper bound on what the work actually costs the wall socket."""
    out = {}
    u, ui = stats.get(unit_key), idle.get(unit_key)
    if tflops and u is not None and ui is not None and (u - ui) > 0.05:
        out["unit_w"] = round(u - ui, 2)
        out["tflops_per_w_unit"] = round(tflops / (u - ui), 3)
    s, si = stats.get("sys_w"), idle.get("sys_w")
    if tflops and s is not None and si is not None and (s - si) > 0.5:
        out["sys_delta_w"] = round(s - si, 2)
        out["tflops_per_w_sys"] = round(tflops / (s - si), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--seconds", type=int, default=40, help="per throughput arm")
    ap.add_argument("--sustained-min", type=float, default=0,
                    help="GPU sustained arm, to compare against the ANE's flat 11-minute series")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n, k, m = a.n, a.k, a.k
    out = {"shape": {"n": n, "k": k, "m": m}, "arms": {}}

    log("idle baseline (12s, nothing running)")
    census_before = machine_census.guard("before")
    with power_trace() as t:
        time.sleep(12)
    idle = t.stats()
    out["idle"] = idle
    log(f"idle: sys {idle['sys_w']} W | gpu {idle['gpu_w']} W | ane {idle['ane_w']} W "
        f"| cpu {idle['cpu_w']} W")

    log("GPU arm, fp16")
    with power_trace() as t:
        g = run_subproc(gpu_src(n, k, m, a.seconds))
    gs = t.stats()
    g["power"] = gs
    g["efficiency"] = efficiency(g.get("tflops"), gs, idle, "gpu_w")
    out["arms"]["gpu"] = g
    log(f"GPU: {g.get('tflops')} TFLOPS | {gs['gpu_w']} W unit | sys {gs['sys_w']} W "
        f"| {gs['gpu_mhz']} MHz | {g['efficiency']}")

    log("CPU arm, fp32 via Accelerate")
    with power_trace() as t:
        c = cpu_arm(n, k, m, a.seconds)
    cs = t.stats()
    c["power"] = cs
    c["efficiency"] = efficiency(c.get("tflops"), cs, idle, "cpu_w")
    out["arms"]["cpu"] = c
    log(f"CPU: {c['tflops']:.3f} TFLOPS | {cs['cpu_w']} W unit | sys {cs['sys_w']} W "
        f"| {c['efficiency']}")

    for inst in (1, 2):
        log(f"ANE arm, {inst} die(s)")
        with power_trace() as t:
            r = ane_arm(64, k, m, a.seconds, inst)
        rs = t.stats()
        r["power"] = rs
        r["efficiency"] = efficiency(r.get("tflops"), rs, idle, "ane_w")
        out["arms"][f"ane_{inst}die"] = r
        log(f"ANE x{inst}: {r.get('tflops')} TFLOPS | {rs['ane_w']} W unit "
            f"| sys {rs['sys_w']} W | {r['efficiency']}")

    if a.sustained_min > 0:
        log(f"GPU sustained arm, {a.sustained_min} min, matched to the ANE protocol")
        with power_trace() as t:
            s = run_subproc(gpu_sustained_src(n, k, m, a.sustained_min),
                            timeout=int(a.sustained_min * 60) + 300)
        ss = t.stats()
        s["power"] = ss
        out["gpu_sustained"] = s
        ws = [w["tflops"] for w in s.get("windows", [])]
        for i, w in enumerate(ws, 1):
            log(f"  gpu window {i}: {w:.3f} TFLOPS")
        if len(ws) > 1:
            s["retention"] = round(ws[-1] / ws[0], 3)
            log(f"GPU sustained retention: {s['retention']}x ({ws[0]:.3f} -> {ws[-1]:.3f}), "
                f"median clock {ss['gpu_mhz']} MHz, temp {ss['gpu_temp']} C")

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
