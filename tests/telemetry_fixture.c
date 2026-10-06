/* Exercise the production JSON encoder without an SMC connection. */
#include "telemetry_json.h"
#include <math.h>

int main(void) {
    sg_thermal_sample s = {
        .wall_time = 1700000000.25, .sample_ms = 12.5, .attempts = 1,
        .cpu = {.avg_c = 45, .min_c = 45, .max_c = 45, .sensors = 2,
                .valid = 1, .invalid = 1},
        .gpu = {.avg_c = 50, .min_c = 40, .max_c = 60, .sensors = 2, .valid = 2},
        .fans = {{.key = "F0Ac", .rpm = 0}, {.key = "F1Ac", .rpm = NAN}},
        .fan_count = 2
    };
    sg_thermal_json(stdout, &s);
    s.gpu = (sg_temperature){.avg_c = NAN, .min_c = NAN, .max_c = NAN,
                            .sensors = 2, .read_errors = 2};
    s.attempts = 3;
    s.discovery_errors = 1;
    sg_thermal_json(stdout, &s);
    return fflush(stdout) == EOF ? 1 : 0;
}
