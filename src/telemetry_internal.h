/* SMC wire layout follows fanpro's read transport (MIT, Omar Shabab 2026).
 * This private interface also permits tests to inject hardware failures. */
#ifndef SG_TELEMETRY_INTERNAL_H
#define SG_TELEMETRY_INTERNAL_H
#include "surge.h"
typedef struct {
    uint32_t size, type;
    uint8_t attributes;
} sg_smc_info;
typedef struct {
    uint32_t key, version;
    uint8_t limits[16], padding0[4];
    sg_smc_info info;
    uint8_t result, status, command, padding1;
    uint32_t index;
    uint8_t bytes[32];
} sg_smc_wire;
/* Zero means transport success. Firmware status is checked separately. */
typedef int (*sg_smc_transport)(void *, const sg_smc_wire *, sg_smc_wire *);
sg_err sg_thermal_open_transport(sg_smc_transport call, void *ctx, sg_thermal **out);
uint32_t sg_smc_fourcc(const char key[4]);
bool sg_smc_number(uint32_t type, uint32_t size, const uint8_t *bytes, double *out);
#endif
