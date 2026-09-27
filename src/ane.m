/* ane.m - surge's Apple Neural Engine host layer, over CoreML.
 *
 * WHY THIS EXISTS, AND WHAT IT IS NOT FOR. Measured on this M3 Ultra
 * (tools/ane_gemm_probe.py and tools/ane_envelope.py, 2026-08-28):
 *
 *   ANE fp16 GEMM        ~8.2 TFLOPS per die, 2 dies, so ~16.3 TFLOPS
 *   ANE int8 GEMM        1.97x that, so ~32.7 TFLOPS across both dies
 *   ANE weight bandwidth ~126 GB/s per die, ~254 GB/s across both
 *   GPU fp16 GEMM        ~23.6 TFLOPS, ~573 GB/s at the n=1 shape
 *
 * So the ANE is the SLOWER processor on every axis and can never replace the
 * Metal path. The one thing it does that matters: it runs CONCURRENTLY with
 * the GPU at close to zero interference. Both dies plus the GPU at a
 * bandwidth-bound shape measured 31.7 TFLOPS against the GPU's own 15.8,
 * with the ANE keeping 0.963x and the GPU 1.002x of their solo rates.
 *
 * THIS BACKEND IS THEREFORE AN ADDITIVE PREFILL PATH, NOT A DECODE PATH.
 * Decode at long context is memory-bandwidth-bound (llm-rnd Findings 16, 22,
 * 25 and 60) and the ANE has roughly a quarter of the GPU's bandwidth, so
 * moving decode here would make it about four times slower. Nothing in this
 * file should ever be wired into sg_gpu_forward.
 *
 * SHAPES DECIDE WHETHER THE ANE IS USED AT ALL. CoreML assigns ops to a
 * processor by size: at 512x512x512 it puts this same conv on the CPU, and at
 * 1024x1024x1024 and above it puts it on the Neural Engine. A program built
 * for too small a shape is silently a CPU program. sg_ane_on_ane() reports
 * what CoreML actually decided, read once at open from the manifest the
 * generator wrote, so a caller can refuse to use a program that is not what it
 * asked for rather than quietly benchmarking the CPU.
 *
 * ACTIVATIONS ARE CHANNEL-MAJOR, (1, K, 1, N), element (c, s) at c*n + s.
 * That is the layout Apple's own ml-ane-transformers uses and it is not a
 * preference: the probe measured that at n=1 a rank-2 matmul goes to the CPU
 * while the equivalent 1x1 conv over this 4D layout goes to the ANE. surge's
 * GEMM is token-major, so a caller crossing this boundary transposes, and that
 * cost is the caller's. It is not hidden here because hiding it would hide it
 * from the measurement.
 *
 * The compiled program is a BUILD ARTIFACT like src/kernels.metallib, emitted
 * by tools/ane_build_model.py. It is loaded by path; nothing is committed.
 *
 * Memory management is manual (no -fobjc-arc anywhere in this project), so
 * every +1 here is balanced in sg_ane_close, exactly as src/metal.m does it.
 */
#ifdef SURGE_NO_ANE

/* Built out of the ASan/UBSan run the same way the Metal layer is: CoreML
 * spins up XPC services and its own threads, which the sanitizers report on
 * and which have nothing to do with surge's own code. */

#else

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "surge.h"

/* Local, because surge exposes no public clock: src/bench.c and
 * src/cli_bench.c each keep their own now_s(). Following that convention
 * rather than promoting one is deliberate. This clock only ever feeds
 * sg_ane_last_predict_s, which is diagnostic, so a third private copy costs
 * nothing, while a new public symbol would have to be kept in step with two
 * existing private ones forever. CLOCK_MONOTONIC so a wall-clock adjustment
 * mid-run cannot produce a negative duration. */
static double ane_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

struct sg_ane {
    MLModel *model;          /* +1 */
    MLModelConfiguration *cfg;   /* +1 */
    NSString *input_name;    /* +1 */
    NSString *output_name;   /* +1 */
    MLMultiArray *in_arr;    /* +1, reused across calls */
    uint32_t n, k, m;
    int on_ane;              /* 1 yes, 0 no, -1 unknown */
    double last_predict_s;
    uint64_t n_predicts;
    char err[256];
};

/* Formats into the handle's own buffer, so the returned pointer stays valid
 * until the next failing call on the same handle. Mirrors sg_gpu_errf's
 * contract minus the shared static, since an ANE handle already exists to own
 * the storage. */
static sg_err ane_errf(sg_ane *a, const char *fmt, ...) {
    static char fallback[256];
    char *buf = a ? a->err : fallback;
    size_t cap = a ? sizeof a->err : sizeof fallback;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, cap, fmt, ap);
    va_end(ap);
    return (sg_err){buf};
}

/* Reads `"on_ane": true|false|null` out of the generator's manifest.
 *
 * DELIBERATELY A TEXT SCAN, NOT A JSON PARSER. surge has no JSON dependency
 * and this is one boolean whose only job is to let a caller refuse a
 * CPU-assigned program. NSJSONSerialization would do it, but pulling the whole
 * manifest through Foundation to read one field, in a file this project
 * generated itself two lines earlier, is not worth the surface. Returns -1
 * (unknown) on any doubt, and -1 is never treated as yes.
 */
static int read_on_ane(const char *dir) {
    char path[1024];
    int nw = snprintf(path, sizeof path, "%s/manifest.json", dir);
    if (nw <= 0 || (size_t)nw >= sizeof path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[8192];
    size_t got = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[got] = '\0';
    const char *p = strstr(buf, "\"on_ane\"");
    if (!p) return -1;
    p += strlen("\"on_ane\"");
    while (*p == ':' || *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return -1;
}

int sg_ane_available(void) {
    /* CoreML itself is always present on a supported macOS, so the honest
     * check is whether a Neural Engine program can actually be scheduled.
     * That cannot be answered without a compiled program, so this reports the
     * weaker fact it can establish: the framework loaded and a configuration
     * requesting the ANE is accepted. sg_ane_on_ane() after sg_ane_open is
     * the real answer, and callers that care must use it. */
    @autoreleasepool {
        MLModelConfiguration *c = [[MLModelConfiguration alloc] init];
        if (!c) return 0;
        c.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        int ok = (c.computeUnits == MLComputeUnitsCPUAndNeuralEngine);
        [c release];
        return ok;
    }
}

sg_err sg_ane_open(const char *dir, sg_ane **out) {
    if (!out) return (sg_err){"ane: sg_ane_open needs an out pointer"};
    *out = NULL;
    if (!dir) return (sg_err){"ane: sg_ane_open needs a directory path"};

    sg_ane *a = calloc(1, sizeof *a);
    if (!a) return (sg_err){"ane: out of memory"};
    a->on_ane = -1;

    @autoreleasepool {
        char prog[1024];
        int nw = snprintf(prog, sizeof prog, "%s/program.mlmodelc", dir);
        if (nw <= 0 || (size_t)nw >= sizeof prog) {
            sg_err e = ane_errf(a, "ane: path too long: %s", dir);
            free(a);
            return e;
        }

        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:prog]];
        a->cfg = [[MLModelConfiguration alloc] init];
        a->cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        NSError *err = nil;
        a->model = [[MLModel modelWithContentsOfURL:url configuration:a->cfg
                                              error:&err] retain];
        if (!a->model) {
            sg_err e = ane_errf(a, "ane: cannot load %s: %s", prog,
                                err ? [[err localizedDescription] UTF8String] : "unknown");
            sg_ane_close(a);
            return e;
        }

        MLModelDescription *d = a->model.modelDescription;
        if (d.inputDescriptionsByName.count != 1 || d.outputDescriptionsByName.count != 1) {
            sg_err e = ane_errf(a, "ane: expected 1 input and 1 output, got %lu and %lu",
                                (unsigned long)d.inputDescriptionsByName.count,
                                (unsigned long)d.outputDescriptionsByName.count);
            sg_ane_close(a);
            return e;
        }
        a->input_name = [[[d.inputDescriptionsByName allKeys] objectAtIndex:0] retain];
        a->output_name = [[[d.outputDescriptionsByName allKeys] objectAtIndex:0] retain];

        /* Shape is read from the model, never from the caller: a program built
         * for a different shape than the caller believes is exactly the bug
         * that would otherwise surface as silent garbage. */
        MLFeatureDescription *ind = d.inputDescriptionsByName[a->input_name];
        NSArray<NSNumber *> *shape = ind.multiArrayConstraint.shape;
        if (shape.count != 4) {
            sg_err e = ane_errf(a, "ane: input rank %lu, expected 4 (1, K, 1, N)",
                                (unsigned long)shape.count);
            sg_ane_close(a);
            return e;
        }
        a->k = (uint32_t)[shape[1] unsignedIntValue];
        a->n = (uint32_t)[shape[3] unsignedIntValue];

        MLFeatureDescription *outd = d.outputDescriptionsByName[a->output_name];
        NSArray<NSNumber *> *oshape = outd.multiArrayConstraint.shape;
        a->m = (oshape.count == 4) ? (uint32_t)[oshape[1] unsignedIntValue] : 0;
        if (!a->k || !a->n || !a->m) {
            sg_err e = ane_errf(a, "ane: degenerate shape n=%u k=%u m=%u", a->n, a->k, a->m);
            sg_ane_close(a);
            return e;
        }

        /* One reusable input array. Allocating a fresh MLMultiArray per call
         * would put an allocation on the per-dispatch path, which is precisely
         * the overhead this backend exists to keep small. */
        NSError *aerr = nil;
        a->in_arr = [[MLMultiArray alloc]
            initWithShape:@[@1, @(a->k), @1, @(a->n)]
                 dataType:MLMultiArrayDataTypeFloat16
                    error:&aerr];
        if (!a->in_arr) {
            sg_err e = ane_errf(a, "ane: cannot allocate input array: %s",
                                aerr ? [[aerr localizedDescription] UTF8String] : "unknown");
            sg_ane_close(a);
            return e;
        }

        a->on_ane = read_on_ane(dir);
    }

    *out = a;
    return (sg_err){NULL};
}

void sg_ane_close(sg_ane *a) {
    if (!a) return;
    [a->in_arr release];
    [a->output_name release];
    [a->input_name release];
    [a->model release];
    [a->cfg release];
    free(a);
}

void sg_ane_shape(const sg_ane *a, uint32_t *n, uint32_t *k, uint32_t *m) {
    if (n) *n = a ? a->n : 0;
    if (k) *k = a ? a->k : 0;
    if (m) *m = a ? a->m : 0;
}

int sg_ane_on_ane(const sg_ane *a) { return a ? a->on_ane : -1; }
double sg_ane_last_predict_s(const sg_ane *a) { return a ? a->last_predict_s : 0.0; }
uint64_t sg_ane_predicts(const sg_ane *a) { return a ? a->n_predicts : 0; }

sg_err sg_ane_matmul(sg_ane *a, const void *in, void *out) {
    if (!a) return (sg_err){"ane: sg_ane_matmul needs a handle"};
    if (!in || !out) return ane_errf(a, "ane: sg_ane_matmul needs in and out");

    @autoreleasepool {
        /* getMutableBytesWithHandler: is the supported way to touch an
         * MLMultiArray's storage without the framework copying it behind us.
         * The alternative, initWithDataPointer:, hands CoreML a buffer whose
         * lifetime this layer would then have to guarantee across an async
         * prediction; reusing one array and filling it is both simpler and
         * keeps the allocation off the per-call path. */
        size_t in_bytes = (size_t)a->k * a->n * 2;
        __block int copied = 0;
        [a->in_arr getMutableBytesWithHandler:^(void *ptr, NSInteger len,
                                                NSArray<NSNumber *> *strides) {
            (void)strides;
            if ((size_t)len >= in_bytes) {
                memcpy(ptr, in, in_bytes);
                copied = 1;
            }
        }];
        if (!copied)
            return ane_errf(a, "ane: input array smaller than %zu bytes", in_bytes);

        MLDictionaryFeatureProvider *feed = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{a->input_name: a->in_arr} error:NULL];
        if (!feed) return ane_errf(a, "ane: cannot build the feature provider");

        NSError *err = nil;
        double t0 = ane_now_s();
        id<MLFeatureProvider> res = [a->model predictionFromFeatures:feed error:&err];
        a->last_predict_s = ane_now_s() - t0;
        [feed release];
        if (!res) {
            return ane_errf(a, "ane: prediction failed: %s",
                            err ? [[err localizedDescription] UTF8String] : "unknown");
        }
        a->n_predicts++;

        MLMultiArray *o = [[res featureValueForName:a->output_name] multiArrayValue];
        if (!o) return ane_errf(a, "ane: output '%s' is not a multiarray",
                                [a->output_name UTF8String]);
        if (o.dataType != MLMultiArrayDataTypeFloat16)
            return ane_errf(a, "ane: output dtype %ld, expected float16", (long)o.dataType);

        size_t out_bytes = (size_t)a->m * a->n * 2;
        __block int ok = 0;
        [o getBytesWithHandler:^(const void *ptr, NSInteger len) {
            if ((size_t)len >= out_bytes) {
                memcpy(out, ptr, out_bytes);
                ok = 1;
            }
        }];
        if (!ok) return ane_errf(a, "ane: output smaller than %zu bytes", out_bytes);
    }
    return (sg_err){NULL};
}

#endif /* SURGE_NO_ANE */
