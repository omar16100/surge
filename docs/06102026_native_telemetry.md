# Native thermal telemetry

Category: Feature (dated).
Date: 6 October 2026.

## Interface

The `sg_thermal_*` API reads AppleSMC through IOKit and CoreFoundation.
The native reader needs no model, GPU workload, subprocess, or root privilege.
Each handle serves one thread.
The SMC transport uses commands 5, 8, and 9 only.
The API does not write fan controls.

`sg_thermal_read` requires complete CPU and GPU coverage before it stops its retries.
`sg_thermal_read_components` selects CPU, GPU, or both as the retry condition.
Each attempt still reads all discovered sensors.
The retry count is 1 through 10, with a 50 ms interval between attempts.
The API discards each previous attempt before the next attempt.
A successful API call can contain unavailable values.

The C API represents unavailable values as `NAN`.
Temperature values must be finite, above zero, and at most 150 degrees Celsius.
A stopped fan can have a valid zero RPM value.
Sensor metadata controls discovery.
The known temperature namespaces are `Tp`, `Te`, `Ts`, and `Tg`.
Tests on other chip families must verify these namespaces.

## JSON schema version 1

| Field | Meaning |
| --- | --- |
| `schema_version`, `source` | Version 1 and `surge-smc` |
| `timestamp` | UTC time at sample completion |
| `sample_ms`, `attempts` | Sample duration and actual attempt count |
| `discovery_errors` | Count of failed discovery operations |
| `temp` | CPU and GPU averages; unavailable values are null |
| `sensors` | Counts, status, minimum, and maximum for each component |
| `fans` | SMC key and RPM for each fan; unavailable RPM is null |

Statuses are `ok`, `partial`, `no_sensors`, `read_error`, and `no_valid_readings`.
A partial average remains visible for diagnosis.
A temperature check must require complete coverage and zero discovery errors.
The CLI scalar mode and Python adapter enforce this requirement.
Partial CPU coverage does not cause extra retries for a GPU-only request.
Partial CPU coverage remains visible and the Python CPU value remains unavailable.

The CLI defaults to one sample, a 1000 ms interval, and three attempts.
The minimum interval is 10 ms.
`--samples 0` selects continuous JSONL output.
`--gpu-temp` requires one sample and selects the GPU retry condition.
A closed output pipe returns exit code 1.

## Tests

`make monitor-check` executes wire-level fault tests and Python interface tests.
The fixture executable uses the same JSON encoder as the CLI.
The tests require no SMC connection or model weights.
`make monitor-debug` repeats these tests with ASan and UBSan.
The isolated contribution excludes the separate ANE changes and machine census tool.

The source test suite covers invalid values, transport faults, metadata faults, retry recovery, and sample freshness.
It also covers component selection, partial coverage, stopped fans, JSON null, and invalid CLI arguments.
The [validation record](06102026_native_telemetry_validation.json) contains the test counts and live sample summary.
All ten live samples had complete GPU coverage and one attempt.
CPU coverage stayed partial.
A natural sensor outage did not occur in this short trace.
