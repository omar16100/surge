/* sched.c - decode pacing + GPU clamp detection (Task P3.0).
 *
 * PURE C11. No Metal, no Foundation, no GPU. It lives in LIB_SRC (the
 * Makefile's src wildcard), so it links into `surge` (cli_metal.c),
 * `surge-bench` (cli_bench.c) and every pure-C test binary from one
 * translation unit, and `make debug` (-DSURGE_NO_METAL + ASan/UBSan) covers
 * all of it with no GPU present.
 *
 * WHY IT IS ITS OWN FILE. B8's prefill duty cycle had to live in src/metal.m
 * because the loop it paces (the chunk loop) is inside sg_gpu_prefill. The
 * decode loop is not inside metal.m at all -- sg_gpu_forward is ONE step and
 * the per-token loop belongs to the caller -- so nothing here needs to be
 * near Metal. Keeping it out matters: src/metal.m is already 4616 lines
 * against this project's ~2000-line guideline, and adding pacing there would
 * have made the worst offender worse.
 *
 * WHAT IS DELIBERATELY NOT HERE. No fan control, no power management, no
 * shelling out to any external daemon. surge is a public engine; driving a
 * machine's cooling is not its business. Clamp detection and self-pacing
 * only.
 *
 * THE SPLIT THAT MAKES THIS TESTABLE. sg_decode_pace_decide is the whole
 * policy and is a pure function of (step timings, budget, rest) plus the
 * pacer's own counters: no clock_gettime, no nanosleep, no syscall of any
 * kind. sg_decode_pace_step is decide() plus the sleep. So the interesting
 * logic is exercised by feeding synthetic series in tests/test_sched.c,
 * deterministically, at full speed, under the sanitizers -- and the only
 * thing NOT covered that way is a nanosleep call.
 *
 * See surge.h for the interface contract, the clamp signal's exact rule and
 * its false-positive modes, and docs/18082026_decode_pacing.md for the
 * measurement behind the chosen ratio.
 */
#include "surge.h"

#include <errno.h>
#include <math.h>
#include <time.h>

/* --------------------------------------------------------------------
 * Internal helpers.
 * -------------------------------------------------------------------- */

/* Median of n samples, n >= 1. Copies into a small fixed buffer and
 * insertion-sorts it: n is bounded by SG_PACE_MAX_BASELINE (32), so this is
 * at most a few hundred comparisons ONCE per baseline, not per step, and an
 * insertion sort avoids pulling in qsort's comparator indirection for a
 * 32-element array. Even n takes the mean of the two middle elements, which
 * is the ordinary convention and keeps a 2-sample baseline sane.
 *
 * Why a median at all rather than a mean: see surge.h. The first decode
 * steps after a prefill carry first-touch and pipeline warm-up and can be
 * several times steady state, and a baseline biased LOW is the harmful
 * direction (it inserts rests that buy nothing). */
static double pace_median(const double *v, uint32_t n) {
    double s[SG_PACE_MAX_BASELINE];
    uint32_t i, j;

    if (!v || n == 0) return 0.0;
    if (n > SG_PACE_MAX_BASELINE) n = SG_PACE_MAX_BASELINE;
    for (i = 0; i < n; i++) s[i] = v[i];
    for (i = 1; i < n; i++) {
        double key = s[i];
        j = i;
        while (j > 0 && s[j - 1] > key) { s[j] = s[j - 1]; j--; }
        s[j] = key;
    }
    if ((n & 1u) != 0u) return s[n / 2u];
    return 0.5 * (s[n / 2u - 1u] + s[n / 2u]);
}

/* Clears ONLY the detector: the sample ring, the baseline, both run counters
 * and the latch. Leaves the duty cycle's accumulator and every lifetime
 * total (rests, rest_total_ms, steps, clamp_events, clamp_steps) alone, so a
 * tune() mid-run re-measures the baseline without erasing the history a
 * report is built from. */
static void pace_detector_clear(sg_decode_pacer *p) {
    uint32_t i;
    for (i = 0; i < SG_PACE_MAX_BASELINE; i++) p->samples[i] = 0.0;
    p->n_samples = 0;
    p->baseline_ms = 0.0;
    p->baseline_seeded = false;
    p->over_run = 0;
    p->under_run = 0;
    p->clamped = false;
}

/* Restarted on EINTR so a signal cannot cut the rest short and leave
 * rest_total_ms overstating the idle time actually taken. Same shape as
 * src/metal.m's pf_sleep_ms, deliberately not shared with it: that one is a
 * file-static in an Objective-C translation unit this file must not depend
 * on, and duplicating six lines is cheaper than coupling the pure-C
 * scheduler to the Metal backend. */
static void pace_sleep_ms(uint32_t ms) {
    struct timespec req = { .tv_sec = (time_t)(ms / 1000u),
                            .tv_nsec = (long)(ms % 1000u) * 1000000L };
    while (nanosleep(&req, &req) != 0 && errno == EINTR) {
        /* req now holds the remaining time; loop to finish it out. */
    }
}

/* --------------------------------------------------------------------
 * Configuration.
 * -------------------------------------------------------------------- */

void sg_decode_pace_init(sg_decode_pacer *p, uint32_t work_budget_ms, uint32_t rest_ms) {
    if (!p) return;
    /* Field-by-field rather than a struct-literal assignment so a field added
     * to sg_decode_pacer later that this function forgets is a compiler
     * warning about an uninitialised read somewhere, not a silently stale
     * value carried in from the caller's stack. */
    pace_detector_clear(p);
    p->work_budget_ms = work_budget_ms;
    p->rest_ms = rest_ms;
    p->baseline_n = SG_PACE_DEF_BASELINE_N;
    p->confirm_n = SG_PACE_DEF_CONFIRM_N;
    p->clamp_ratio = SG_PACE_DEF_CLAMP_RATIO;
    p->clamp_div = SG_PACE_DEF_CLAMP_DIV;
    p->work_acc_ms = 0.0;
    p->clamp_events = 0;
    p->clamp_steps = 0;
    p->rests = 0;
    p->rest_total_ms = 0;
    p->steps = 0;
}

void sg_decode_pace_tune(sg_decode_pacer *p, uint32_t baseline_n, uint32_t confirm_n,
                         double clamp_ratio, uint32_t clamp_div) {
    if (!p) return;
    /* Clamped, not rejected: every one of these has a defensible nearest
     * legal value, and a scheduler knob is not worth an error path in a
     * decode loop. The clamp_ratio floor of 1.0 is load-bearing -- a ratio
     * below 1 would make an at-baseline step "over" its own baseline, so the
     * detector would latch on every clean run. */
    if (baseline_n == 0) baseline_n = 1;
    if (baseline_n > SG_PACE_MAX_BASELINE) baseline_n = SG_PACE_MAX_BASELINE;
    if (confirm_n == 0) confirm_n = 1;
    if (clamp_div == 0) clamp_div = 1;
    if (!(clamp_ratio >= 1.0)) clamp_ratio = 1.0;   /* also catches NaN */
    p->baseline_n = baseline_n;
    p->confirm_n = confirm_n;
    p->clamp_ratio = clamp_ratio;
    p->clamp_div = clamp_div;
    pace_detector_clear(p);
}

void sg_decode_pace_set_baseline(sg_decode_pacer *p, double baseline_ms) {
    if (!p) return;
    pace_detector_clear(p);            /* also zeroes baseline_seeded */
    if (isfinite(baseline_ms) && baseline_ms > 0.0) {
        uint32_t n = p->baseline_n;
        /* Bounded even though init/tune already clamp baseline_n, because the
         * struct is transparent: a caller that wrote the field directly could
         * otherwise leave n_samples past the ring with the guard in
         * sg_decode_pace_decide never reached. */
        if (n > SG_PACE_MAX_BASELINE) n = SG_PACE_MAX_BASELINE;
        p->baseline_ms = baseline_ms;
        /* Mark the baseline window already satisfied, so the very next step
         * is CLASSIFIED rather than spent re-measuring. Skipping this is what
         * would silently reduce a seeded baseline to a no-op: the first
         * baseline_n steps would overwrite it with the run's own median, i.e.
         * exactly the blind spot seeding is meant to remove. */
        p->n_samples = n;
        p->baseline_seeded = true;
    }
}

void sg_decode_pace_reset(sg_decode_pacer *p) {
    if (!p) return;
    pace_detector_clear(p);
    p->work_acc_ms = 0.0;
    p->clamp_events = 0;
    p->clamp_steps = 0;
    p->rests = 0;
    p->rest_total_ms = 0;
    p->steps = 0;
}

/* --------------------------------------------------------------------
 * The policy. Pure: no clock, no sleep, no syscall.
 * -------------------------------------------------------------------- */

uint32_t sg_decode_pace_decide(sg_decode_pacer *p, double step_ms) {
    bool enabled;
    bool over;
    uint32_t threshold_ms;

    if (!p) return 0;

    /* A non-finite or non-positive step time is not a measurement. Dropping
     * it (rather than accumulating it or feeding it to the baseline) is what
     * keeps a single bad clock read from poisoning a baseline for the whole
     * run, and it means the caller does not have to filter. Not counted in
     * `steps` either, so `steps` is the number of real measurements. */
    if (!isfinite(step_ms) || step_ms <= 0.0) return 0;

    p->steps++;

    /* --- (1) the detector. Runs whether or not pacing is armed. ---
     *
     * It costs a compare and a couple of increments per step and issues no
     * syscall, so an UNARMED run behaves exactly as it did before this task
     * (no sleeps, identical output) while still reporting whether its own
     * step time rose. That is the whole value of running it by default: a
     * bench row can say "this decode number was taken while step time was
     * climbing" instead of leaving the reader to guess. */
    if (p->n_samples < p->baseline_n) {
        /* Still measuring the baseline. Cannot classify a step against a
         * baseline that does not exist yet, so no latch is possible here.
         * The bound is belt-and-braces: baseline_n is already clamped to
         * SG_PACE_MAX_BASELINE by init/tune, so this can only matter if a
         * caller wrote the field directly. */
        if (p->n_samples < SG_PACE_MAX_BASELINE) {
            p->samples[p->n_samples] = step_ms;
        }
        p->n_samples++;
        if (p->n_samples >= p->baseline_n) {
            p->baseline_ms = pace_median(p->samples, p->n_samples);
        }
    } else {
        over = (step_ms > p->baseline_ms * p->clamp_ratio);
        if (over) {
            p->under_run = 0;
            p->over_run++;
            /* Latch on the EDGE only, so clamp_events counts transitions
             * into a slow regime and not steps spent in one (clamp_steps is
             * that). Without the !clamped guard a long clamp would inflate
             * clamp_events by one per step and the two counters would carry
             * the same information. */
            if (!p->clamped && p->over_run >= p->confirm_n) {
                p->clamped = true;
                p->clamp_events++;
            }
        } else {
            p->over_run = 0;
            p->under_run++;
            /* Symmetric hysteresis: confirm_n consecutive steps back at
             * baseline to clear, so a single fast step in the middle of a
             * clamp does not drop the latch and then need re-confirming. */
            if (p->clamped && p->under_run >= p->confirm_n) {
                p->clamped = false;
            }
        }
        if (p->clamped) p->clamp_steps++;
    }

    /* --- (2) the duty cycle. Off unless BOTH knobs are > 0. ---
     *
     * Same disabled rule as sg_gpu_set_prefill_rest, and the same reason:
     * one field defaulting to 0 must not leave a half-armed mechanism that
     * rests with a zero budget (every step) or rests for zero ms (never, but
     * while claiming to). Returning before touching work_acc_ms also keeps
     * the accumulator at 0 for the whole of a disabled run, so
     * sg_decode_pace_rests() and sg_decode_pace_rest_ms() are exactly 0
     * there and a test can assert that as an equality, not a bound. */
    enabled = (p->work_budget_ms > 0u && p->rest_ms > 0u);
    if (!enabled) return 0;

    p->work_acc_ms += step_ms;

    /* A confirmed clamp shortens the budget by clamp_div, which DEFAULTS TO
     * 1: out of the box the detector changes no rest schedule at all, it
     * only reports. That is the conservative default on purpose -- the
     * detector cannot identify a cause, so coupling it to rest insertion by
     * default would multiply each of its false positives into idle time
     * bought for nothing (see SG_PACE_DEF_CLAMP_DIV in surge.h for why the
     * "rest harder and the clock returns" rationale is specifically NOT the
     * justification offered here).
     *
     * The divisor form is what bounds the cost when it IS opted into: the
     * densest possible schedule is one rest_ms per (work_budget_ms /
     * clamp_div) of work, a number an operator can compute before the run,
     * unlike "rest whenever the detector says so". A budget small enough
     * that the division reaches 0 is floored at 1 ms, so a clamp can never
     * degenerate into "rest after every step regardless of work". */
    threshold_ms = p->work_budget_ms;
    if (p->clamped && p->clamp_div > 1u) {
        threshold_ms /= p->clamp_div;
        if (threshold_ms == 0u) threshold_ms = 1u;
    }

    if (p->work_acc_ms >= (double)threshold_ms) {
        /* NOT predictive, unlike B8's prefill test. B8 has to predict because
         * one prefill chunk at long context can cost minutes, so overshooting
         * by a whole chunk overshoots the budget by more than the budget (367
         * of 367 bursts in the 2026-08-14 256K run overran a 150 s budget,
         * median 199.5 s). A decode step is one token: tens of ms at worst,
         * so the overshoot is bounded by one step and a predictive test would
         * buy nothing but a way to be wrong about the next step's cost. */
        p->work_acc_ms = 0.0;
        p->rests++;
        p->rest_total_ms += p->rest_ms;
        return p->rest_ms;
    }
    return 0;
}

uint32_t sg_decode_pace_step(sg_decode_pacer *p, double step_ms) {
    uint32_t ms = sg_decode_pace_decide(p, step_ms);
    if (ms > 0u) pace_sleep_ms(ms);
    return ms;
}

/* --------------------------------------------------------------------
 * Read-back. All NULL-safe, all returning the "nothing happened" value.
 * -------------------------------------------------------------------- */

uint64_t sg_decode_pace_rest_ms(const sg_decode_pacer *p) {
    return p ? p->rest_total_ms : 0;
}

uint32_t sg_decode_pace_clamp_events(const sg_decode_pacer *p) {
    return p ? p->clamp_events : 0;
}

uint32_t sg_decode_pace_clamp_steps(const sg_decode_pacer *p) {
    return p ? p->clamp_steps : 0;
}

bool sg_decode_pace_clamped(const sg_decode_pacer *p) {
    return p ? p->clamped : false;
}

double sg_decode_pace_baseline_ms(const sg_decode_pacer *p) {
    return p ? p->baseline_ms : 0.0;
}

uint32_t sg_decode_pace_rests(const sg_decode_pacer *p) {
    return p ? p->rests : 0;
}
