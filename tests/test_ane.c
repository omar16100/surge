/* test_ane.c - the ANE backend's correctness gate, against an independent
 * double-precision reference.
 *
 * THE ORACLE IS NOT THE MODEL. tools/ane_build_model.py writes the weight
 * matrix to weights.f16 alongside the compiled program, and this file recomputes
 * the product in double from those bytes. So a bug in the CoreML program, in the
 * layout translation, or in src/ane.m's marshalling lands on exactly one side.
 * A test that fed the model its own output back would pass through any of them.
 *
 * THE BAR. src/kernels.metal's tiled GEMM is gated at 1e-5 relative against a
 * host f64 reference (M5.3, worst measured 2.5e-6). This path cannot hold that,
 * and the reason was WRONG HERE UNTIL 2026-08-29. This comment used to say "the
 * ANE accumulates in fp16", predicting error growing like sqrt(k) * 2^-11.
 * tools/ane_accum_probe.py tested that on this machine and it is FALSE: swept
 * over k, relative error FALLS rather than rises (7.57e-4 at k=4096 to 5.41e-4
 * at k=16384, fitted exponent -0.24, where a sequential fp16 running sum
 * requires +0.5). So the reduction is not a sequential fp16 running sum,
 * consistent with Bryngelson arXiv:2606.22283 on M1 and M5 (the probe cannot
 * rule out an fp16 TREE reduction; see its header).
 *
 * WHAT ACTUALLY COSTS THE ACCURACY IS CANCELLATION over fp16-rounded operands.
 * The same probe measures 2.00e-3 with random-sign inputs against 7.57e-4 with
 * positive ones at identical k, a 2.6x penalty for cancellation alone. This
 * test's own input is a sin() oscillation against normal weights, which is a
 * heavy-cancellation case: max |want| is only 0.075 while individual products
 * are far larger. That is why it measures 1.1e-2 and why the bar is 2e-2.
 *
 * SO THE LOOSE BAR IS A PROPERTY OF THIS TEST'S INPUT, not a hardware ceiling.
 * A well-conditioned workload should hold roughly 1e-3 here. Anything that ever
 * moves this path past additive prefill should re-measure rather than inherit
 * this number.
 *
 * The normalization is max |err| / max |want|, the same tests/test_metal_ops.c
 * uses and for the same reason: a near-zero output element must not manufacture
 * a huge relative error out of an absolute one that is fine.
 *
 * WHAT THIS TEST DOES NOT CLAIM. It does not assert the work ran on the Neural
 * Engine, because sg_ane_on_ane() reports what CoreML decided and that decision
 * is per-shape: below roughly 1024x1024x1024 the same program is a CPU program.
 * The assignment is REPORTED here and asserted only when the fixture is large
 * enough to expect it.
 *
 * Portability: with no fixture directory this prints a skip and exits 0, so
 * `make check` stays green on a machine that has not built one. A fixture that
 * EXISTS but does not open is a failure, not a skip. A fixture built for the
 * ANE also asserts sg_ane_on_ane() == 1, which reads the generator's manifest,
 * so a fixture must be built on the machine that runs it. It is also
 * compiled to a bare skip under -DSURGE_NO_ANE, which is how `make debug`
 * keeps CoreML out of the ASan run.
 */
#ifdef SURGE_NO_ANE

#include <stdio.h>
int main(void) {
    fprintf(stderr, "SKIP test_ane: built with -DSURGE_NO_ANE "
                    "(CoreML and the ASan/UBSan run do not mix)\n");
    return 0;
}

#else

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "surge.h"
#include "tinytest.h"

/* The fixture directory, built by:
 *   make ane-fixture
 * which runs tools/ane_build_model.py. Overridable so a larger shape can be
 * pointed at the same test without editing it. */
static const char *fixture_dir(void) {
    const char *d = getenv("SURGE_ANE_FIXTURE");
    return d ? d : "tests/fixtures/ane";
}

static int read_file(const char *dir, const char *name, void **out, size_t *len) {
    char path[1024];
    int nw = snprintf(path, sizeof path, "%s/%s", dir, name);
    if (nw <= 0 || (size_t)nw >= sizeof path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return 0; }
    void *buf = malloc((size_t)n);
    if (!buf) { fclose(f); return 0; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return 0; }
    *out = buf;
    *len = got;
    return 1;
}

/* __fp16 is clang's native ARM half. Using it rather than hand-rolling a
 * bit-twiddling converter keeps the reference free of a second thing that
 * could be wrong. */
static double h2d(const __fp16 *p, size_t i) { return (double)p[i]; }

int main(void) {
    const char *dir = fixture_dir();

    /* ABSENT is a skip; PRESENT BUT UNLOADABLE is a failure. Folding the two
     * together would let a broken fixture or a loader regression pass as
     * "no fixture". */
    char prog[1024];
    int nw = snprintf(prog, sizeof prog, "%s/program.mlmodelc", dir);
    if (nw <= 0 || (size_t)nw >= sizeof prog) {
        fprintf(stderr, "FAIL test_ane: fixture path too long: %s\n", dir);
        return 1;
    }
    if (access(prog, F_OK) != 0) {
        if (errno != ENOENT) {
            fprintf(stderr, "FAIL test_ane: cannot check %s: %s\n", prog, strerror(errno));
            return 1;
        }
        fprintf(stderr, "SKIP test_ane: no fixture at %s.\n"
                        "  Build one with: make ane-fixture\n", dir);
        return 0;
    }

    sg_ane *a = NULL;
    sg_err e = sg_ane_open(dir, &a);
    if (e.msg) {
        fprintf(stderr, "FAIL test_ane: a fixture exists at %s but does not open: %s\n",
                dir, e.msg);
        return 1;
    }

    uint32_t n = 0, k = 0, m = 0;
    sg_ane_shape(a, &n, &k, &m);
    printf("test_ane: program n=%u k=%u m=%u, on_ane=%d\n", n, k, m, sg_ane_on_ane(a));

    void *wbuf = NULL;
    size_t wlen = 0;
    if (!read_file(dir, "weights.f16", &wbuf, &wlen)) {
        fprintf(stderr, "FAIL test_ane: cannot read %s/weights.f16\n", dir);
        sg_ane_close(a);
        return 1;
    }
    /* The weights file is the model's own weights, so a size mismatch means the
     * fixture is inconsistent with the program and every later comparison would
     * be meaningless. Checked before anything is computed from it. */
    tt_assert(wlen == (size_t)m * k * 2, "weights.f16 is m*k halves");
    if (wlen != (size_t)m * k * 2) {
        /* Fatal, not just counted: the reference loop below reads m*k halves. */
        free(wbuf);
        sg_ane_close(a);
        return tt_report();
    }
    const __fp16 *w = (const __fp16 *)wbuf;

    size_t in_elems = (size_t)k * n, out_elems = (size_t)m * n;
    __fp16 *in = malloc(in_elems * sizeof *in);
    __fp16 *got = malloc(out_elems * sizeof *got);
    double *want = malloc(out_elems * sizeof *want);
    if (!in || !got || !want) {
        fprintf(stderr, "FAIL test_ane: out of memory\n");
        return 1;
    }

    /* Deterministic input, no rand(): a test that cannot be re-run on the same
     * numbers cannot be debugged when it fails once in fifty. */
    for (size_t i = 0; i < in_elems; i++)
        in[i] = (__fp16)(0.05 * sin((double)i * 0.7 + 1.3));

    e = sg_ane_matmul(a, in, got);
    tt_assert(e.msg == NULL, "sg_ane_matmul succeeds");
    if (e.msg) {
        fprintf(stderr, "  error: %s\n", e.msg);
        return 1;
    }

    /* want[o][s] = sum_c W[o][c] * in[c][s], accumulated in double.
     * Both activation buffers are channel-major, so index (c, s) is c*n + s. */
    for (uint32_t o = 0; o < m; o++) {
        for (uint32_t s = 0; s < n; s++) {
            double acc = 0.0;
            for (uint32_t c = 0; c < k; c++)
                acc += h2d(w, (size_t)o * k + c) * h2d(in, (size_t)c * n + s);
            want[(size_t)o * n + s] = acc;
        }
    }

    /* Non-finite outputs are counted separately: `NaN > max_err` is false, so
     * an all-NaN result would otherwise sail through the error bar (and, being
     * repeatable, through the determinism check too). */
    double max_err = 0.0, max_want = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < out_elems; i++) {
        double g = (double)got[i];
        if (!isfinite(g)) { nonfinite++; continue; }
        double d = fabs(g - want[i]);
        if (d > max_err) max_err = d;
        if (fabs(want[i]) > max_want) max_want = fabs(want[i]);
    }
    tt_assert(nonfinite == 0, "every ANE output is finite (%zu are not)", nonfinite);
    double rel = (max_want > 0.0) ? max_err / max_want : max_err;
    printf("test_ane: worst relative error %.3e (bar 2e-2), max|want| %.4f\n", rel, max_want);
    tt_assert(max_want > 0.0, "reference output is not identically zero");
    tt_assert(rel < 2e-2, "ANE matmul matches the f64 reference within 2e-2 relative");

    /* DETERMINISM. Every Metal reduction kernel in this project is gated on
     * byte-identical reruns, and the ANE gets the same question asked of it.
     * A processor that is not deterministic cannot sit behind a byte-exact
     * greedy gate, so the answer decides how this path may ever be used. */
    __fp16 *again = malloc(out_elems * sizeof *again);
    int identical = 1;
    for (int r = 0; r < 8 && identical; r++) {
        e = sg_ane_matmul(a, in, again);
        tt_assert(e.msg == NULL, "rerun succeeds");
        if (e.msg) break;
        if (memcmp(got, again, out_elems * sizeof *again) != 0) identical = 0;
    }
    printf("test_ane: 8 reruns byte-identical: %s\n", identical ? "yes" : "NO");
    tt_assert(identical, "ANE output is byte-identical across 8 reruns");

    /* Per-dispatch floor from C. This is the number Python could not give:
     * tools/ane_envelope.py measured 13 to 15 ms of round-trip overhead through
     * numpy and the Python bridge, which would swamp any fine-grained use. What
     * the C path costs decides whether this backend can carry per-layer work or
     * only whole-stack submissions. Reported, not asserted: it is a property of
     * the machine and the shape, not a correctness bar. */
    double best = 1e9;
    for (int r = 0; r < 20; r++) {
        e = sg_ane_matmul(a, in, again);
        if (e.msg) break;
        double t = sg_ane_last_predict_s(a);
        if (t < best) best = t;
    }
    double gflop = 2.0 * (double)n * k * m / 1e9;
    printf("test_ane: best predict %.3f ms over %llu calls (%.1f GFLOP, %.2f TFLOPS)\n",
           best * 1e3, (unsigned long long)sg_ane_predicts(a), gflop,
           gflop / 1e3 / best);

    /* The assignment is only ASSERTED where CoreML is expected to choose the
     * ANE. Below that size it legitimately chooses the CPU, and a test that
     * demanded otherwise would fail on a correct system. */
    if ((double)n * k * m >= 1024.0 * 1024.0 * 1024.0)
        tt_assert(sg_ane_on_ane(a) == 1, "a program this large is assigned to the ANE");

    free(again);
    free(want);
    free(got);
    free(in);
    free(wbuf);
    sg_ane_close(a);
    return tt_report();
}

#endif /* SURGE_NO_ANE */
