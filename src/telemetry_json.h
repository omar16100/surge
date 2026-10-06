#ifndef SG_TELEMETRY_JSON_H
#define SG_TELEMETRY_JSON_H
#include "surge.h"
#include <stdio.h>

/* Write one JSONL sample. The caller must check the stream for I/O errors. */
void sg_thermal_json(FILE *stream, const sg_thermal_sample *sample);
#endif
