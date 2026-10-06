"""Native Surge thermal snapshot for Python benchmark harnesses (stdlib only)."""
import json
import math
import os
from pathlib import Path
import subprocess


def _number(value, *, temperature=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if temperature:
        return value if 0 < value <= 150 else None
    return value if 0 <= value <= 100000 else None


def _temperature(row, component):
    info = row.get("sensors", {}).get(component, {})
    count = info.get("sensors")
    if (row.get("discovery_errors") != 0 or info.get("status") != "ok"
            or isinstance(count, bool) or not isinstance(count, int) or count < 1
            or info.get("valid") != count or info.get("invalid") != 0 or info.get("read_errors") != 0):
        return None
    return _number(row.get("temp", {}).get(f"{component}_temp_avg"), temperature=True)


def native_thermal(monitor=None):
    """Return temperatures only for a complete fresh native sensor sample.

    Partial averages remain in `thermal` for diagnostics. There is deliberately
    no temperature fallback to macmon, a cached value, or a different component.
    """
    result = {"temperature_source": "surge-smc", "cpu_temp_c": None, "gpu_temp_c": None,
              "gpu_temp_status": "unavailable", "fan_rpm": None}
    binary = monitor or os.environ.get("SURGE_MONITOR") or Path(__file__).resolve().parents[1] / "surge-monitor"
    try:
        proc = subprocess.run([str(binary), "--samples", "1", "--retry-components", "gpu"],
                              capture_output=True, text=True, timeout=10)
        if proc.returncode:
            raise ValueError(proc.stderr.strip() or f"surge-monitor exited {proc.returncode}")
        row = json.loads(proc.stdout)
        if row.get("schema_version") != 1 or row.get("source") != "surge-smc":
            raise ValueError("unexpected surge-monitor schema/source")
        result.update(cpu_temp_c=_temperature(row, "cpu"), gpu_temp_c=_temperature(row, "gpu"),
                      gpu_temp_status=row["sensors"]["gpu"]["status"],
                      fan_rpm=[_number(fan["rpm"]) for fan in row["fans"]], thermal=row)
        if result["gpu_temp_c"] is None and result["gpu_temp_status"] == "ok":
            result["gpu_temp_status"] = "invalid_coverage"
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["temperature_error"] = str(exc)
    return result
