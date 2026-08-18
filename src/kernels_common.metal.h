/* kernels_common.metal.h - the shader helpers src/kernels.metal and
 * src/kernels_splitk.metal both use.
 *
 * A METAL header, not a C one: the only things that include it are the two
 * .metal translation units the Makefile compiles into src/kernels.metallib.
 * The host-side contract lives in surge.h, as it always has.
 *
 * WHAT BELONGS HERE IS EXACTLY WHAT BOTH FILES CALL, AND NOTHING ELSE. The
 * split-K decode kernels moved out of kernels.metal into kernels_splitk.metal
 * when that file passed this project's ~2000-line guideline, and the three
 * declarations below are the entire surface the two share. A helper with
 * callers in only one of them stays in that file. This header exists so a
 * SHARED helper has exactly one definition, not as a place to collect
 * helpers: two copies that drift is precisely the failure the determinism
 * mandate exists to prevent.
 *
 * THAT MANDATE IS STATED ONCE, at the top of src/kernels.metal (:7-27), and
 * governs both files. The two fold trees below are its main instrument: a
 * fixed stride schedule over a compile-time SG_TG lanes, folded in an order
 * that cannot depend on the data or on the dispatched threads_per_threadgroup,
 * with no atomics and no simd_sum / simd_max anywhere.
 */
#ifndef SG_KERNELS_COMMON_METAL_H
#define SG_KERNELS_COMMON_METAL_H

#include <metal_stdlib>
using namespace metal;

/* The one threadgroup width every reduction kernel is dispatched with. A
 * power of two, so the fold tree is exactly log2(SG_TG) levels. */
constant uint SG_TG = 256u;

/* Fixed-shape tree sum over a threadgroup. Callable repeatedly: the leading
 * barrier makes sure every thread has finished READING the previous result
 * out of red[0] before this call overwrites red[lid]. */
static inline float tg_sum(threadgroup float *red, uint lid, float v) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[lid] = v;
    for (uint stride = SG_TG / 2u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) red[lid] = red[lid] + red[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

/* Same tree, maximum. A NaN on EITHER side wins, so a NaN anywhere in the
 * row survives the fold; fmax() would swallow it and hand back a plausible
 * maximum computed from poisoned data. */
static inline float tg_max(threadgroup float *red, uint lid, float v) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[lid] = v;
    for (uint stride = SG_TG / 2u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            float a = red[lid], b = red[lid + stride];
            red[lid] = (a != a) ? a : ((b != b) ? b : ((b > a) ? b : a));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

#endif  /* SG_KERNELS_COMMON_METAL_H */
