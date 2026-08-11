/* greedy.c - the ONE greedy-decode argmax, shared by every decode driver.
 *
 * PURE C. No Metal, no GPU, no Foundation -- it lives in LIB_SRC (the
 * Makefile's src wildcard) so it links into `surge` (cli_metal.c),
 * `surge-bench` (cli_bench.c) and every pure-C test binary from a single
 * translation unit.
 *
 * Why it is its own file: surge's M2 gate and surge-bench's B5 gate both
 * assert that the two binaries emit BYTE-IDENTICAL greedy token ids on the
 * same input. That can only hold if they pick the argmax the exact same way.
 * Duplicating the loop in both CLIs would let the two drift (a >= vs > tie
 * rule, a different start index) without any single compile catching it;
 * calling one shared symbol makes the drift impossible by construction.
 */
#include "surge.h"

uint32_t sg_argmax_f32(const float *v, uint32_t n) {
    /* Lowest index wins an exact tie (strict >, scanning upward from 0),
     * which is the rule the CPU reference (ref.c / cli_ref.c) uses too, so a
     * genuinely close position cannot flip on tie-break convention alone. */
    if (!v || n == 0) return 0;
    uint32_t arg = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > v[arg]) arg = i;
    }
    return arg;
}
