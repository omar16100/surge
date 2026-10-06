"""Native thermal integration, with hardware and external commands mocked."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import thermal


def row():
    info = {"status": "ok", "sensors": 2, "valid": 2, "invalid": 0, "read_errors": 0}
    return {"schema_version": 1, "source": "surge-smc", "discovery_errors": 0,
            "temp": {"cpu_temp_avg": 45.0, "gpu_temp_avg": 49.5},
            "sensors": {"cpu": copy.copy(info), "gpu": copy.copy(info)},
            "fans": [{"rpm": 3600}, {"rpm": 3610}]}


def snapshot(data):
    proc = subprocess.CompletedProcess([], 0, json.dumps(data), "")
    with patch.object(thermal.subprocess, "run", return_value=proc):
        return thermal.native_thermal()


class ThermalTests(unittest.TestCase):
    def test_production_json_encoder(self):
        binary = Path(__file__).resolve().parent / "telemetry_fixture.bin"
        proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=3, check=True)
        def reject_constant(value):
            raise ValueError(value)
        healthy, outage = [json.loads(line, parse_constant=reject_constant)
                           for line in proc.stdout.splitlines()]
        self.assertEqual(healthy["timestamp"], "2023-11-14T22:13:20.250000Z")
        self.assertEqual(healthy["sensors"]["cpu"]["status"], "partial")
        self.assertEqual(healthy["fans"], [{"key": "F0Ac", "rpm": 0},
                                          {"key": "F1Ac", "rpm": None}])
        self.assertEqual(snapshot(healthy)["gpu_temp_c"], 50)
        self.assertIsNone(snapshot(healthy)["cpu_temp_c"])
        self.assertIsNone(outage["temp"]["gpu_temp_avg"])
        self.assertIsNone(outage["sensors"]["gpu"]["min_c"])
        self.assertEqual(outage["sensors"]["gpu"]["status"], "read_error")
        self.assertEqual(outage["attempts"], 3)
        self.assertEqual(outage["discovery_errors"], 1)
        self.assertIsNone(snapshot(outage)["gpu_temp_c"])

    def test_adapter_requests_gpu_retry_policy(self):
        proc = subprocess.CompletedProcess([], 0, json.dumps(row()), "")
        with patch.object(thermal.subprocess, "run", return_value=proc) as run:
            thermal.native_thermal("monitor")
        self.assertEqual(run.call_args.args[0],
                         ["monitor", "--samples", "1", "--retry-components", "gpu"])

    def test_complete_native_read(self):
        result = snapshot(row())
        self.assertEqual(result["gpu_temp_c"], 49.5)
        self.assertEqual(result["fan_rpm"], [3600, 3610])

    def test_missing_invalid_and_nonfinite_never_become_cold(self):
        for value in (None, 0, -0.65, 151, float("nan"), float("inf"), True, "49.5"):
            with self.subTest(value=value):
                data = row()
                data["temp"]["gpu_temp_avg"] = value
                result = snapshot(data)
                self.assertIsNone(result["gpu_temp_c"])
                self.assertNotEqual(result["gpu_temp_status"], "ok")

    def test_partial_and_incomplete_discovery_are_not_valid_gate_values(self):
        for change in ("partial", "no_sensors", "read_error", "discovery", "wrong_count"):
            with self.subTest(change=change):
                data = row()
                if change == "discovery":
                    data["discovery_errors"] = 1
                elif change == "wrong_count":
                    data["sensors"]["gpu"]["valid"] = 1
                else:
                    data["sensors"]["gpu"]["status"] = change
                self.assertIsNone(snapshot(data)["gpu_temp_c"])

    def test_sensor_outage_preserves_other_telemetry(self):
        data = row()
        data["sensors"]["gpu"].update(status="no_valid_readings", valid=0, invalid=2)
        data["temp"]["gpu_temp_avg"] = None
        result = snapshot(data)
        self.assertEqual(result["cpu_temp_c"], 45)
        self.assertEqual(result["fan_rpm"], [3600, 3610])
        self.assertIsNone(result["gpu_temp_c"])

    def test_missing_binary_timeout_and_bad_output_are_unavailable(self):
        for failure in (FileNotFoundError("not built"), subprocess.TimeoutExpired("monitor", 10)):
            with patch.object(thermal.subprocess, "run", side_effect=failure):
                self.assertIn("temperature_error", thermal.native_thermal())
        for output in ("", "not JSON", "[]", "{}"):
            proc = subprocess.CompletedProcess([], 0, output, "")
            with patch.object(thermal.subprocess, "run", return_value=proc):
                self.assertIsNone(thermal.native_thermal()["gpu_temp_c"])

    def test_cli_rejects_bad_arguments_without_hardware(self):
        binary = Path(__file__).resolve().parents[1] / "surge-monitor"
        for args in (["--samples", "-1"], ["--interval-ms", "0"], ["--retries", "0"],
                     ["--retries", "11"], ["--samples", "bad"], ["--gpu-temp", "--samples", "0"],
                     ["--retry-components"], ["--retry-components", "invalid"]):
            with self.subTest(args=args):
                p = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=3)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(p.stdout, "")


if __name__ == "__main__":
    unittest.main()
