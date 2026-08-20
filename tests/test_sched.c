/* test_sched.c - src/sched.c: the decode pacing policy and the clamp
 * detector (Task P3.0). PURE C, no Metal, no GPU, so this runs under a plain
 * `make check` AND under `make debug` (SURGE_NO_METAL + ASan/UBSan), which is
 * where the real correctness gate for this task lives: the policy is a pure
 * function of (step timings, budget, rest), so a synthetic series exercises
 * 100% of the decision logic deterministically and at full speed.
 *
 * Almost everything below drives sg_decode_pace_DECIDE rather than _step, on
 * purpose: _step is _decide plus a nanosleep, and sleeping through hundreds
 * of synthetic steps would buy nothing but wall time. Two cases do call
 * _step, and case (k) is the load-bearing one: it holds a real clock against
 * it, because the accounting lives in _decide and so a caller that took the
 * accounting without taking the sleep would satisfy every other assertion
 * here.
 *
 * The cases exist against specific failure modes, not for coverage:
 *   (a) DISABLED IS INERT, and inert by equality (== 0), not by bound.
 *   (b) The armed duty cycle rests an EXACT, predicted number of times.
 *   (c) A single slow step does NOT latch the detector. This is the one that
 *       matters most: a detector that fires on ordinary variance would insert
 *       rests that cost throughput for nothing.
 *   (d) confirm_n consecutive slow steps DO latch it, once (an edge, not a
 *       count of steps).
 *   (e) Hysteresis: it clears only after confirm_n consecutive normal steps.
 *   (f) The clamp escalation is OFF by default and, when opted into,
 *       MEASURABLY raises rest density. This is the "cannot be enabled and do
 *       nothing" gate, in the pure build.
 *   (g) The median baseline survives an outlier that a mean would not.
 *   (h) A seeded baseline makes a clamped-from-start series detectable, which
 *       a self-measured baseline provably cannot see.
 *   (i) Junk inputs (NaN, inf, <= 0) are dropped, not accumulated.
 *   (j) NULL-safety and argument clamping.
 *   (k) _step ACTUALLY SLEEPS, asserted against the wall clock. The
 *       accounting lives in _decide, so without this a caller that took the
 *       accounting and skipped the sleep would pass every other gate here.
 */

#include "surge.h"
#include "tinytest.h"
#include <math.h>
#include <string.h>
#include <time.h>

/* Feeds n steps of the same cost; returns how many rests came back. */
static uint32_t feed(sg_decode_pacer *p, double step_ms, uint32_t n) {
    uint32_t i, rests = 0;
    for (i = 0; i < n; i++) {
        if (sg_decode_pace_decide(p, step_ms) > 0u) rests++;
    }
    return rests;
}

/* --------------------------------------------------------------------
 * (a) Disabled is inert. Both knobs 0 (the sg_decode_pace_init default a
 * caller gets by not asking for pacing), and each knob alone, since a
 * half-armed mechanism is the interesting bug: budget-only must not mean
 * "rest 0 ms very often" and rest-only must not mean "rest after every
 * step". Asserted as EQUALITY to zero, so a single stray rest fails.
 * -------------------------------------------------------------------- */
static void test_disabled_is_inert(void) {
    sg_decode_pacer p;
    struct { uint32_t w, r; const char *what; } cases[] = {
        { 0, 0,   "both 0" },
        { 10, 0,  "budget only" },
        { 0, 50,  "rest only" },
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        sg_decode_pace_init(&p, cases[c].w, cases[c].r);
        uint32_t rests = feed(&p, 20.0, 500);   /* 10 s of "work" */
        tt_assert(rests == 0, "%s: got %u rests, want 0", cases[c].what, rests);
        tt_assert(sg_decode_pace_rest_ms(&p) == 0,
                  "%s: rest_total_ms=%llu, want 0", cases[c].what,
                  (unsigned long long)sg_decode_pace_rest_ms(&p));
        tt_assert(sg_decode_pace_rests(&p) == 0, "%s: rests counter nonzero", cases[c].what);
        /* The DETECTOR still ran: it needs no arming, and its whole value on
         * an unpaced run is telling the reader the run's step time was
         * steady. A baseline of 0 here would mean it never observed
         * anything. */
        tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 20.0) < 1e-9,
                  "%s: baseline=%.6g, want 20 (detector must run when unarmed)",
                  cases[c].what, sg_decode_pace_baseline_ms(&p));
        tt_assert(sg_decode_pace_clamp_events(&p) == 0,
                  "%s: %u clamp events on a perfectly flat series",
                  cases[c].what, sg_decode_pace_clamp_events(&p));
    }
}

/* --------------------------------------------------------------------
 * (b) Armed: an EXACT rest count, not "> 0". 200 steps of 10 ms is 2000 ms
 * of work against a 100 ms budget, so a correct accumulator rests exactly
 * 20 times and no other number is close: an off-by-one reset, a missing
 * reset, or a > vs >= slip all land on a different integer.
 * -------------------------------------------------------------------- */
static void test_duty_cycle_exact_count(void) {
    sg_decode_pacer p;
    sg_decode_pace_init(&p, 100, 50);
    uint32_t rests = feed(&p, 10.0, 200);
    tt_assert(rests == 20, "got %u rests, want exactly 20 (200 x 10ms / 100ms budget)", rests);
    tt_assert(sg_decode_pace_rests(&p) == 20, "rests counter=%u, want 20", sg_decode_pace_rests(&p));
    tt_assert(sg_decode_pace_rest_ms(&p) == 20ull * 50ull,
              "rest_total_ms=%llu, want %llu",
              (unsigned long long)sg_decode_pace_rest_ms(&p), 20ull * 50ull);
    tt_assert(p.steps == 200, "steps=%llu, want 200", (unsigned long long)p.steps);

    /* A step bigger than the whole budget rests once per step, not many
     * times per step: the accumulator resets to 0 on a rest, it does not
     * subtract the budget repeatedly. */
    sg_decode_pace_init(&p, 10, 5);
    rests = feed(&p, 1000.0, 7);
    tt_assert(rests == 7, "oversized steps: got %u rests, want 7 (one per step)", rests);
}

/* --------------------------------------------------------------------
 * (c) THE false-positive gate. A baseline, then ONE 10x step, repeatedly,
 * never two in a row. confirm_n is 3, so this must never latch. If this
 * fails the detector fires on ordinary variance, which the brief calls
 * worse than having no detector at all.
 * -------------------------------------------------------------------- */
static void test_single_spike_does_not_latch(void) {
    sg_decode_pacer p;
    sg_decode_pace_init(&p, 0, 0);
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N);            /* establish baseline */
    for (uint32_t i = 0; i < 50; i++) {
        sg_decode_pace_decide(&p, 200.0);              /* one 10x outlier */
        sg_decode_pace_decide(&p, 20.0);               /* back to normal */
        sg_decode_pace_decide(&p, 20.0);
    }
    tt_assert(sg_decode_pace_clamp_events(&p) == 0,
              "%u clamp events from 50 ISOLATED 10x spikes (want 0)",
              sg_decode_pace_clamp_events(&p));
    tt_assert(!sg_decode_pace_clamped(&p), "latched on isolated spikes");

    /* Two in a row is still one short of confirm_n = 3. */
    sg_decode_pace_init(&p, 0, 0);
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N);
    for (uint32_t i = 0; i < 20; i++) {
        sg_decode_pace_decide(&p, 200.0);
        sg_decode_pace_decide(&p, 200.0);
        sg_decode_pace_decide(&p, 20.0);
    }
    tt_assert(sg_decode_pace_clamp_events(&p) == 0,
              "%u clamp events from PAIRS of spikes with confirm_n=3 (want 0)",
              sg_decode_pace_clamp_events(&p));

    /* And a rise that stays UNDER the ratio never latches however long it
     * lasts: 1.4x against a 1.5x threshold, for 1000 steps. This is the
     * ordinary-drift case (a slowly growing KV cache), and it is the reason
     * the threshold is a ratio and not "slower than baseline". */
    sg_decode_pace_init(&p, 0, 0);
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N);
    feed(&p, 20.0 * 1.4, 1000);
    tt_assert(sg_decode_pace_clamp_events(&p) == 0,
              "%u clamp events from a sustained 1.4x rise under a 1.5x threshold",
              sg_decode_pace_clamp_events(&p));
}

/* --------------------------------------------------------------------
 * (d) confirm_n consecutive over-threshold steps latch it, ONCE. A long
 * clamp must count as one EVENT and many clamp STEPS, otherwise the two
 * counters carry the same information and neither says how many transitions
 * happened.
 * -------------------------------------------------------------------- */
static void test_sustained_slowdown_latches_once(void) {
    sg_decode_pacer p;
    sg_decode_pace_init(&p, 0, 0);
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 20.0) < 1e-9,
              "baseline=%.6g, want 20", sg_decode_pace_baseline_ms(&p));

    /* Two over steps: still not latched (confirm_n = 3). */
    sg_decode_pace_decide(&p, 60.0);
    sg_decode_pace_decide(&p, 60.0);
    tt_assert(!sg_decode_pace_clamped(&p), "latched after 2 over steps, confirm_n=3");
    /* The third latches. */
    sg_decode_pace_decide(&p, 60.0);
    tt_assert(sg_decode_pace_clamped(&p), "did NOT latch after 3 consecutive over steps");
    tt_assert(sg_decode_pace_clamp_events(&p) == 1,
              "clamp_events=%u after one transition, want 1", sg_decode_pace_clamp_events(&p));

    feed(&p, 60.0, 97);
    tt_assert(sg_decode_pace_clamp_events(&p) == 1,
              "clamp_events=%u after 100 slow steps, want 1 (edges, not steps)",
              sg_decode_pace_clamp_events(&p));
    /* 100 over-threshold steps, but the first TWO happened before the latch,
     * so they are not steps spent clamped: 98. Asserting 100 here would be
     * asserting that the detector latches before it has confirmed. */
    tt_assert(sg_decode_pace_clamp_steps(&p) == 98,
              "clamp_steps=%u, want 98 (100 slow steps minus the 2 pre-latch ones)",
              sg_decode_pace_clamp_steps(&p));
}

/* --------------------------------------------------------------------
 * (e) Hysteresis. One fast step in the middle of a clamp must not clear it
 * (or the latch would flap and clamp_events would count noise); confirm_n
 * consecutive fast steps must.
 * -------------------------------------------------------------------- */
static void test_hysteresis(void) {
    sg_decode_pacer p;
    sg_decode_pace_init(&p, 0, 0);
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N);
    feed(&p, 60.0, 10);
    tt_assert(sg_decode_pace_clamped(&p), "setup: not latched");

    sg_decode_pace_decide(&p, 20.0);
    tt_assert(sg_decode_pace_clamped(&p), "one fast step cleared the latch");
    sg_decode_pace_decide(&p, 20.0);
    tt_assert(sg_decode_pace_clamped(&p), "two fast steps cleared the latch (confirm_n=3)");
    sg_decode_pace_decide(&p, 20.0);
    tt_assert(!sg_decode_pace_clamped(&p), "3 consecutive fast steps did NOT clear the latch");

    /* Re-entering counts a SECOND event. */
    feed(&p, 60.0, 3);
    tt_assert(sg_decode_pace_clamp_events(&p) == 2,
              "clamp_events=%u after clamp/clear/clamp, want 2",
              sg_decode_pace_clamp_events(&p));
}

/* --------------------------------------------------------------------
 * (f) The escalation: OFF by default, and provably NOT INERT when opted
 * into. Same series both ways, so the only difference is clamp_div.
 *
 * Series: 8 baseline steps at 10 ms, then 400 slow steps at 40 ms (4x
 * baseline, so over the 1.5x threshold from the first one).
 *
 *   clamp_div = 1: the schedule must be EXACTLY the unclamped one. Budget
 *     100 ms, and the accumulator resets to 0 on a rest (B8's rule, so the
 *     overshoot past the budget is discarded rather than carried): the 8
 *     baseline steps leave 80 ms accumulated, the first 40 ms step reaches
 *     120 >= 100 and rests, and from there it is one rest per 3 steps
 *     (40/80/120). 1 + floor(399/3) = 134. The exact number is the point --
 *     it moves if the accumulator ever starts carrying overshoot.
 *   clamp_div = 4: the threshold becomes 25 ms once latched, so past the 3
 *     steps the latch costs, every 40 ms step trips it. Asserted as a RATIO
 *     >= 2 rather than an exact count, since the exact count depends on
 *     latch lag that is not what this case is about.
 * -------------------------------------------------------------------- */
static void test_clamp_escalation_off_by_default_and_effective_when_set(void) {
    sg_decode_pacer p;
    uint32_t rests_plain, rests_escalated;

    sg_decode_pace_init(&p, 100, 5);
    tt_assert(p.clamp_div == SG_PACE_DEF_CLAMP_DIV && p.clamp_div == 1u,
              "clamp_div default is %u, want 1 (detector must not drive rests by default)",
              p.clamp_div);
    feed(&p, 10.0, SG_PACE_DEF_BASELINE_N);
    rests_plain = feed(&p, 40.0, 400);
    tt_assert(sg_decode_pace_clamped(&p), "setup: detector did not latch on a 4x slowdown");
    tt_assert(rests_plain == 134, "clamp_div=1: %u rests, want 134 (schedule must be unchanged)",
              rests_plain);

    sg_decode_pace_init(&p, 100, 5);
    sg_decode_pace_tune(&p, SG_PACE_DEF_BASELINE_N, SG_PACE_DEF_CONFIRM_N,
                        SG_PACE_DEF_CLAMP_RATIO, 4);
    feed(&p, 10.0, SG_PACE_DEF_BASELINE_N);
    rests_escalated = feed(&p, 40.0, 400);
    tt_assert(rests_escalated >= 2 * rests_plain,
              "clamp_div=4 gave %u rests vs %u unescalated: the escalation is inert",
              rests_escalated, rests_plain);
    tt_assert(sg_decode_pace_rest_ms(&p) == (uint64_t)rests_escalated * 5ull,
              "rest_total_ms=%llu does not match %u rests x 5ms",
              (unsigned long long)sg_decode_pace_rest_ms(&p), rests_escalated);

    /* The escalation must not fire without a clamp: same config, flat
     * series, so the rest count has to match the unescalated schedule. */
    sg_decode_pace_init(&p, 100, 5);
    sg_decode_pace_tune(&p, SG_PACE_DEF_BASELINE_N, SG_PACE_DEF_CONFIRM_N,
                        SG_PACE_DEF_CLAMP_RATIO, 4);
    uint32_t rests_flat = feed(&p, 10.0, 408);
    tt_assert(sg_decode_pace_clamp_events(&p) == 0, "flat series latched the detector");
    tt_assert(rests_flat == 40, "clamp_div=4 on a FLAT series: %u rests, want 40 "
              "(4080ms / 100ms budget) -- escalation must need a clamp", rests_flat);
}

/* --------------------------------------------------------------------
 * (g) Median baseline. 8 samples, one of them 10x: a mean would be pulled
 * to 2.1x the true value and the detector would then need a 3.2x slowdown
 * to notice anything. The median must be unmoved.
 * -------------------------------------------------------------------- */
static void test_median_baseline_survives_an_outlier(void) {
    sg_decode_pacer p;
    sg_decode_pace_init(&p, 0, 0);
    sg_decode_pace_decide(&p, 200.0);            /* the warm-up step */
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N - 1);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 20.0) < 1e-9,
              "baseline=%.6g with one 10x outlier in the window, want 20 "
              "(a mean would give 42.5)", sg_decode_pace_baseline_ms(&p));
    feed(&p, 40.0, 3);
    tt_assert(sg_decode_pace_clamped(&p),
              "a 2x slowdown went undetected, so the outlier moved the baseline");

    /* Even n: mean of the two middle samples. 4 samples 10,20,30,40 -> 25. */
    sg_decode_pace_init(&p, 0, 0);
    sg_decode_pace_tune(&p, 4, SG_PACE_DEF_CONFIRM_N, SG_PACE_DEF_CLAMP_RATIO, 1);
    sg_decode_pace_decide(&p, 30.0);
    sg_decode_pace_decide(&p, 10.0);
    sg_decode_pace_decide(&p, 40.0);
    sg_decode_pace_decide(&p, 20.0);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 25.0) < 1e-9,
              "even-n baseline=%.6g, want 25", sg_decode_pace_baseline_ms(&p));
}

/* --------------------------------------------------------------------
 * (h) The blind spot, and that seeding covers it. A run that is slow from
 * its FIRST token (the P2.9 case) cannot be seen by a self-measured
 * baseline, because that baseline is defined by those very steps. Both
 * halves are asserted: the miss, so the limitation is pinned by a test
 * rather than only described in a comment, and the fix.
 * -------------------------------------------------------------------- */
static void test_clamped_from_start_needs_a_seeded_baseline(void) {
    sg_decode_pacer p;

    /* Self-measured: a uniformly 3x-slow run looks perfectly normal. */
    sg_decode_pace_init(&p, 0, 0);
    feed(&p, 60.0, 300);
    tt_assert(sg_decode_pace_clamp_events(&p) == 0,
              "self-measured baseline reported %u events on a uniformly slow run; "
              "the documented blind spot has changed", sg_decode_pace_clamp_events(&p));
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 60.0) < 1e-9,
              "baseline=%.6g, want 60 (it IS the slow rate)", sg_decode_pace_baseline_ms(&p));

    /* Seeded from a known-idle run: the same series latches immediately. */
    sg_decode_pace_init(&p, 0, 0);
    sg_decode_pace_set_baseline(&p, 20.0);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 20.0) < 1e-9,
              "seed did not take: baseline=%.6g", sg_decode_pace_baseline_ms(&p));
    feed(&p, 60.0, 3);
    tt_assert(sg_decode_pace_clamped(&p),
              "seeded baseline did not latch on a 3x-slow run after confirm_n steps");
    tt_assert(sg_decode_pace_clamp_events(&p) == 1, "clamp_events=%u, want 1",
              sg_decode_pace_clamp_events(&p));
    /* The seed must SURVIVE the baseline window rather than being overwritten
     * by the run's own median, which is the way this feature would silently
     * become a no-op. That overwrite could only happen once baseline_n steps
     * had arrived, so this feeds MORE than baseline_n of them: checking after
     * only the 3 above would pass against the very bug it names. */
    feed(&p, 60.0, SG_PACE_DEF_BASELINE_N + 2u);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 20.0) < 1e-9,
              "baseline drifted to %.6g after %u steps; the seed was overwritten "
              "by the run's own median", sg_decode_pace_baseline_ms(&p),
              SG_PACE_DEF_BASELINE_N + 5u);

    /* A non-positive or non-finite seed reverts to self-measurement. */
    sg_decode_pace_init(&p, 0, 0);
    sg_decode_pace_set_baseline(&p, 20.0);
    sg_decode_pace_set_baseline(&p, 0.0);
    tt_assert(sg_decode_pace_baseline_ms(&p) == 0.0, "seed 0 did not clear the baseline");
    sg_decode_pace_set_baseline(&p, -1.0);
    tt_assert(sg_decode_pace_baseline_ms(&p) == 0.0, "negative seed installed a baseline");
    sg_decode_pace_set_baseline(&p, NAN);
    tt_assert(sg_decode_pace_baseline_ms(&p) == 0.0, "NaN seed installed a baseline");
    feed(&p, 33.0, SG_PACE_DEF_BASELINE_N);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 33.0) < 1e-9,
              "did not revert to self-measurement: baseline=%.6g",
              sg_decode_pace_baseline_ms(&p));
}

/* --------------------------------------------------------------------
 * (i) Junk step times are DROPPED. A NaN from a bad clock read must not
 * poison the baseline (NaN compares false against everything, so a NaN
 * baseline would silently disable the detector for the whole run) and a
 * zero/negative step must not count as work.
 * -------------------------------------------------------------------- */
static void test_junk_steps_are_dropped(void) {
    sg_decode_pacer p;
    sg_decode_pace_init(&p, 100, 50);
    tt_assert(sg_decode_pace_decide(&p, NAN) == 0, "NaN step returned a rest");
    tt_assert(sg_decode_pace_decide(&p, INFINITY) == 0, "inf step returned a rest");
    tt_assert(sg_decode_pace_decide(&p, 0.0) == 0, "zero step returned a rest");
    tt_assert(sg_decode_pace_decide(&p, -5.0) == 0, "negative step returned a rest");
    tt_assert(p.steps == 0, "steps=%llu after 4 junk inputs, want 0",
              (unsigned long long)p.steps);
    tt_assert(p.work_acc_ms == 0.0, "work accumulated from junk: %.6g", p.work_acc_ms);
    feed(&p, 20.0, SG_PACE_DEF_BASELINE_N);
    tt_assert(fabs(sg_decode_pace_baseline_ms(&p) - 20.0) < 1e-9,
              "baseline=%.6g after junk + 8 clean 20ms steps, want 20 (junk poisoned it)",
              sg_decode_pace_baseline_ms(&p));
}

/* --------------------------------------------------------------------
 * (j) NULL-safety, argument clamping, reset, and that _step agrees with
 * _decide. The _step case is small on purpose: it sleeps, so it is the one
 * thing in this file that costs real time.
 * -------------------------------------------------------------------- */
static void test_api_edges(void) {
    sg_decode_pacer p;

    sg_decode_pace_init(NULL, 1, 1);
    sg_decode_pace_tune(NULL, 1, 1, 1.0, 1);
    sg_decode_pace_set_baseline(NULL, 1.0);
    sg_decode_pace_reset(NULL);
    tt_assert(sg_decode_pace_decide(NULL, 10.0) == 0, "NULL decide returned nonzero");
    tt_assert(sg_decode_pace_step(NULL, 10.0) == 0, "NULL step returned nonzero");
    tt_assert(sg_decode_pace_rest_ms(NULL) == 0, "NULL rest_ms nonzero");
    tt_assert(sg_decode_pace_clamp_events(NULL) == 0, "NULL clamp_events nonzero");
    tt_assert(sg_decode_pace_clamp_steps(NULL) == 0, "NULL clamp_steps nonzero");
    tt_assert(sg_decode_pace_rests(NULL) == 0, "NULL rests nonzero");
    tt_assert(!sg_decode_pace_clamped(NULL), "NULL clamped true");
    tt_assert(sg_decode_pace_baseline_ms(NULL) == 0.0, "NULL baseline nonzero");

    /* Clamping, including the load-bearing ratio floor: a ratio below 1
     * would classify an exactly-at-baseline step as over its own baseline. */
    sg_decode_pace_init(&p, 0, 0);
    sg_decode_pace_tune(&p, 0, 0, 0.1, 0);
    tt_assert(p.baseline_n == 1, "baseline_n=%u after tune(0), want 1", p.baseline_n);
    tt_assert(p.confirm_n == 1, "confirm_n=%u after tune(0), want 1", p.confirm_n);
    tt_assert(p.clamp_div == 1, "clamp_div=%u after tune(0), want 1", p.clamp_div);
    tt_assert(p.clamp_ratio == 1.0, "clamp_ratio=%.6g after tune(0.1), want 1.0", p.clamp_ratio);
    sg_decode_pace_tune(&p, SG_PACE_MAX_BASELINE + 99u, 1, NAN, 1);
    tt_assert(p.baseline_n == SG_PACE_MAX_BASELINE, "baseline_n=%u, want %u",
              p.baseline_n, SG_PACE_MAX_BASELINE);
    tt_assert(p.clamp_ratio == 1.0, "NaN ratio became %.6g, want 1.0", p.clamp_ratio);
    /* With ratio floored to exactly 1.0, an at-baseline step is NOT over it
     * (the comparison is strict >), so a flat run still cannot latch. */
    feed(&p, 20.0, 100);
    tt_assert(sg_decode_pace_clamp_events(&p) == 0,
              "ratio 1.0 latched %u times on a flat series (want 0: > is strict)",
              sg_decode_pace_clamp_events(&p));

    /* reset clears counters but keeps configuration. */
    sg_decode_pace_init(&p, 100, 50);
    sg_decode_pace_tune(&p, 4, 2, 2.0, 3);
    feed(&p, 60.0, 20);
    tt_assert(sg_decode_pace_rest_ms(&p) > 0, "setup: no rests to clear");
    sg_decode_pace_reset(&p);
    tt_assert(sg_decode_pace_rest_ms(&p) == 0 && sg_decode_pace_rests(&p) == 0 &&
              sg_decode_pace_clamp_events(&p) == 0 && p.steps == 0 &&
              sg_decode_pace_baseline_ms(&p) == 0.0,
              "reset left state behind");
    tt_assert(p.work_budget_ms == 100 && p.rest_ms == 50 && p.baseline_n == 4 &&
              p.confirm_n == 2 && p.clamp_ratio == 2.0 && p.clamp_div == 3,
              "reset lost configuration");

    /* _step must return the same ms _decide would and must not double-count. */
    sg_decode_pace_init(&p, 1, 5);
    uint32_t got = sg_decode_pace_step(&p, 50.0);
    tt_assert(got == 5, "step returned %u, want 5", got);
    tt_assert(sg_decode_pace_rest_ms(&p) == 5, "step accounted %llu ms, want 5",
              (unsigned long long)sg_decode_pace_rest_ms(&p));
    tt_assert(sg_decode_pace_rests(&p) == 1, "step counted %u rests, want 1",
              sg_decode_pace_rests(&p));
}

/* --------------------------------------------------------------------
 * (k) _step ACTUALLY SLEEPS. This is the one case in this file that reads a
 * clock, and it exists to close a specific hole: the accounting lives in
 * _decide, so a caller (or a future edit to the decode loop) that called
 * _decide and never slept would still report rests, still satisfy the
 * accounting identity, and still make decode_compute_tps exceed the
 * wall-clock rate -- every other gate in this task would pass while the
 * mechanism did nothing at all. That is the exact shape of the P2.4
 * silently-inert switch, so it gets its own assertion against the wall clock
 * rather than against a counter.
 *
 * 120 ms of prescribed rest, asserted to have taken at least 90 ms. The
 * generous floor is deliberate: nanosleep may return early only on a signal
 * (which the EINTR loop restarts) but the clock and the scheduler both add
 * slack, and the failure this guards against is 0 ms, not 115 ms.
 * -------------------------------------------------------------------- */
static void test_step_actually_sleeps(void) {
    sg_decode_pacer p;
    struct timespec a, b;
    double elapsed_ms;
    uint32_t slept = 0;

    sg_decode_pace_init(&p, 1, 40);          /* 1 ms budget: every step rests */
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (uint32_t i = 0; i < 3; i++) slept += sg_decode_pace_step(&p, 50.0);
    clock_gettime(CLOCK_MONOTONIC, &b);
    elapsed_ms = ((double)b.tv_sec - (double)a.tv_sec) * 1000.0
               + ((double)b.tv_nsec - (double)a.tv_nsec) / 1e6;

    tt_assert(slept == 120, "step returned %u ms of rest over 3 steps, want 120", slept);
    tt_assert(sg_decode_pace_rest_ms(&p) == 120,
              "accounted %llu ms, want 120", (unsigned long long)sg_decode_pace_rest_ms(&p));
    tt_assert(elapsed_ms >= 90.0,
              "sg_decode_pace_step prescribed and accounted 120 ms of rest but only "
              "%.3f ms of wall time passed -- it is not sleeping", elapsed_ms);
}

int main(void) {
    tt_run("disabled_is_inert", test_disabled_is_inert);
    tt_run("duty_cycle_exact_count", test_duty_cycle_exact_count);
    tt_run("single_spike_does_not_latch", test_single_spike_does_not_latch);
    tt_run("sustained_slowdown_latches_once", test_sustained_slowdown_latches_once);
    tt_run("hysteresis", test_hysteresis);
    tt_run("clamp_escalation_off_by_default_and_effective_when_set",
           test_clamp_escalation_off_by_default_and_effective_when_set);
    tt_run("median_baseline_survives_an_outlier", test_median_baseline_survives_an_outlier);
    tt_run("clamped_from_start_needs_a_seeded_baseline",
           test_clamped_from_start_needs_a_seeded_baseline);
    tt_run("junk_steps_are_dropped", test_junk_steps_are_dropped);
    tt_run("api_edges", test_api_edges);
    tt_run("step_actually_sleeps", test_step_actually_sleeps);
    return tt_report();
}
