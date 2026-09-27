#!/usr/bin/env python3
"""Record what else is using this machine, BEFORE and AFTER every measurement.

WHY THIS EXISTS, and it is not hypothetical. Two runs in one session were invalidated by
contention that a thirty-second check would have caught:

  1. An 11-minute ANE sustained series read half rate for nine windows. Cause: my own
     `make ane-fixture` and `test_ane` were using the ANE. Discovered only after the run.
  2. An 11-minute GPU sustained series read 2.006 TFLOPS. Cause: two booted iOS Simulators were
     holding 60 GB of GPU memory and keeping the GPU from ever going idle, so it never cleared
     a 338 MHz clamp. Re-running clean gave 1.650, which changed the published number.

Both were silent. Neither harness had any way to notice, and the numbers looked plausible. This
module is the analogue of the GEMM gate and the fan check that every GPU cell in llm-rnd already
runs: a precondition that fails loudly rather than a result that misleads quietly.

TAKE A CENSUS BEFORE AND AFTER, NOT JUST BEFORE. A before-only check cannot see a background
daemon that starts mid-run, which is exactly what happened in case 1. The after census is
compared against the before, and any new heavy consumer is reported as a reason to distrust the
interval.

NOTHING HERE KILLS ANYTHING. It observes and reports. Deciding to shut down a simulator is the
operator's call, not a benchmark harness's.
"""
import json
import os
import re
import subprocess
import time

# Processes that can take GPU, ANE or large memory from a measurement. Matched against the full
# command line. Deliberately broad: a false positive costs one line of output, a false negative
# costs an 11-minute run.
CONTENDER_PATTERNS = [
    ("ml_server", r"mlx_lm\.server|llama-server|ollama|vllm"),
    ("ml_job", r"mlx_lm\.(generate|chat)|llama-cli|llama-bench|surge-bench|bench_niah|lcb_local"),
    ("ane_job", r"ane_envelope|ane_gemm_probe|ane_vs_all|ane_accum_probe|test_ane"),
    ("simulator", r"CoreSimulator|simruntime|Simulator\.app"),
    ("media_ane", r"mediaanalysisd|photoanalysisd"),
    ("browser", r"Google Chrome Helper \(GPU\)|Brave Browser Helper \(GPU\)|firefox"),
    ("creative", r"blender|ffmpeg|Final Cut|Compressor|DaVinci"),
]


def _sh(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:  # noqa: BLE001
        return ""


def _macmon(samples=3, interval_ms=700):
    out = _sh(f"macmon pipe -s {samples} -i {interval_ms}", timeout=samples * 3 + 20)
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return {}
    def med(key, default=None):
        vals = []
        for r in rows:
            v = r
            for part in key.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)
        if not vals:
            return default
        vals.sort()
        return round(vals[len(vals) // 2], 3)
    fans = rows[-1].get("fans", [])
    return {
        "gpu_w": med("gpu_power"), "gpu_mhz": med("gpu_freq_mhz"),
        "gpu_busy_pct": round((med("gpu_active_ratio") or 0) * 100, 1),
        "ane_w": med("ane_power"), "cpu_w": med("cpu_power"), "sys_w": med("sys_power"),
        "gpu_temp_c": med("temp.gpu_temp_avg"),
        "ram_gb": round((med("memory.ram_usage") or 0) / 1e9, 1),
        "fan_rpm": [f.get("rpm") for f in fans] if fans else None,
    }


def _gpu_memory():
    """Accelerator-wide in-use bytes and the GPU restart counter.

    recoveryCount is worth carrying: a GPU that has restarted mid-session is one whose earlier
    numbers may not be comparable to its later ones, and surge's own B8 notes record WindowServer
    watchdog kills on this machine.
    """
    out = _sh("ioreg -r -c IOAccelerator -w0 2>/dev/null")
    in_use = re.search(r'"In use system memory"=(\d+)', out)
    rec = re.search(r'"recoveryCount"=(\d+)', out)
    return {"gpu_mem_in_use_gb": round(int(in_use.group(1)) / 1e9, 2) if in_use else None,
            "gpu_recovery_count": int(rec.group(1)) if rec else None}


def _self_pids():
    """This process and every descendant of it.

    WITHOUT THIS THE CENSUS CRIES WOLF AT ITSELF. `ane_gemm_probe.py` matches the `ane_job`
    pattern, so the first run of this module reported the very harness that called it as a
    competing ANE job, and its worker subprocesses too. A guard that fires on every clean run
    gets ignored, which is worse than no guard.
    """
    me = os.getpid()
    pids = {me}
    out = _sh("ps -Ao pid,ppid 2>/dev/null")
    kids, parent = {}, {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
            parent[int(parts[0])] = int(parts[1])

    # DESCENDANTS: workers this run spawned are this run's own load, not foreign contention.
    stack = [me]
    while stack:
        for child in kids.get(stack.pop(), []):
            if child not in pids:
                pids.add(child)
                stack.append(child)

    # ANCESTORS TOO, and the first version of this function missed them. The shell that launches
    # `ane_gemm_probe.py` carries that name on its own command line, so it matched the `ane_job`
    # pattern and was reported as a competing job on every single run. Walking up stops at pid 1
    # rather than at a fixed depth, because the launch chain here is zsh inside a tool wrapper
    # inside a session and its depth is not fixed.
    p = parent.get(me)
    while p and p > 1:
        pids.add(p)
        p = parent.get(p)
    return pids


def _contenders():
    mine = _self_pids()
    out = _sh("ps -Ao pid,pcpu,command 2>/dev/null")
    found = {}
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, pcpu, cmd = parts[0], parts[1], parts[2]
        if not pid.isdigit() or int(pid) in mine:
            continue
        for label, pat in CONTENDER_PATTERNS:
            if re.search(pat, cmd):
                found.setdefault(label, []).append(
                    {"pid": int(pid), "pcpu": float(pcpu), "cmd": cmd[:110]})
                break
    return {k: v for k, v in found.items()}


def _booted_simulators():
    out = _sh("xcrun simctl list devices booted 2>/dev/null")
    return [ln.strip() for ln in out.splitlines() if "Booted" in ln]


def census(label):
    """One full snapshot. Cheap: about 3 seconds."""
    c = {"label": label, "wall": time.strftime("%Y-%m-%dT%H:%M:%S")}
    c.update(_macmon())
    c.update(_gpu_memory())
    c["contenders"] = _contenders()
    c["booted_simulators"] = _booted_simulators()
    c["warnings"] = _warnings(c)
    return c


def _warnings(c):
    """Reasons to distrust a measurement taken in this state.

    Each threshold is tied to an incident rather than chosen for neatness. The ANE one is strict
    because `ane_power` is the ONLY signal available: there is no per-process ANE attribution on
    macOS at all, so a non-zero idle reading is the whole warning system.
    """
    w = []
    ane = c.get("ane_w")
    if ane is not None and ane > 0.05:
        w.append(f"ANE is already busy at {ane} W.  There is no per-process ANE "
                 f"attribution on macOS, so this is the only signal you get.")
    busy = c.get("gpu_busy_pct")
    if busy is not None and busy > 20:
        w.append(f"GPU is {busy}% busy.")
    mhz = c.get("gpu_mhz")
    if mhz is not None and mhz <= 400:
        w.append(f"GPU clock is {mhz} MHz, at or near the 338 MHz limiter floor. A run started "
                 f"here measures the clamp, not the engine.")
    if c.get("booted_simulators"):
        w.append(f"{len(c['booted_simulators'])} iOS Simulator(s) booted. These hold GPU memory "
                 f"and prevent the GPU from idling long enough to clear a clamp.")
    for label in ("ml_job", "ane_job", "media_ane", "creative"):
        if label in c.get("contenders", {}):
            pids = [p["pid"] for p in c["contenders"][label]]
            w.append(f"competing {label}: pids {pids}")
    mem = c.get("gpu_mem_in_use_gb")
    if mem is not None and mem > 10:
        w.append(f"{mem} GB of GPU memory already in use.")
    return w


def report(c, printer=print):
    printer(f"[census:{c['label']}] gpu {c.get('gpu_w')} W / {c.get('gpu_mhz')} MHz / "
            f"{c.get('gpu_busy_pct')}% busy / {c.get('gpu_mem_in_use_gb')} GB | "
            f"ane {c.get('ane_w')} W | ram {c.get('ram_gb')} GB | fans {c.get('fan_rpm')}")
    for w in c["warnings"]:
        printer(f"[census:{c['label']}] WARNING: {w}")
    if not c["warnings"]:
        printer(f"[census:{c['label']}] clean")


def compare(before, after, printer=print):
    """What CHANGED across the interval.

    A before-only check cannot see a daemon that starts mid-run, which is precisely how the ANE
    sustained series was lost. Anything that appears here invalidates the interval even if the
    before census was clean.
    """
    deltas = []
    b_pids = {l: {p["pid"] for p in v} for l, v in before.get("contenders", {}).items()}
    for label, procs in after.get("contenders", {}).items():
        new = [p["pid"] for p in procs if p["pid"] not in b_pids.get(label, set())]
        if new:
            deltas.append(f"{label} APPEARED during the run: pids {new}")
    if not before.get("booted_simulators") and after.get("booted_simulators"):
        deltas.append("an iOS Simulator booted during the run")
    br, ar = before.get("gpu_recovery_count"), after.get("gpu_recovery_count")
    if br is not None and ar is not None and ar > br:
        deltas.append(f"THE GPU RESTARTED during the run ({br} to {ar}). Every number from this "
                      f"interval is suspect.")
    for w in deltas:
        printer(f"[census:delta] WARNING: {w}")
    if not deltas:
        printer("[census:delta] nothing new appeared during the run")
    return deltas


def guard(label="before"):
    """Census plus a printed report, returned for embedding in a result file."""
    c = census(label)
    report(c)
    return c


if __name__ == "__main__":
    import sys
    c = census(sys.argv[1] if len(sys.argv) > 1 else "manual")
    report(c)
    print(json.dumps(c, indent=2))
