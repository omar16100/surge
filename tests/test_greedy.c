/* test_greedy.c - src/greedy.c: the ONE shared argmax (sg_argmax_f32).
 *
 * Pure C, no Metal, no GPU; runs under a plain `make check` and under
 * `make debug` (ASan/UBSan, SURGE_NO_METAL). This pins the tie-break rule
 * that B5's byte-exact gen_ids gate depends on: `surge` and `surge-bench`
 * call THIS function, so if the rule here is right, their greedy tokens
 * cannot disagree on a tie.
 */
#include "surge.h"
#include "tinytest.h"

static void test_plain_max(void) {
    float v[5] = { -1.0f, 0.5f, 3.25f, 3.0f, -8.0f };
    tt_assert(sg_argmax_f32(v, 5) == 2, "plain max at index 2");
}

/* Lowest index wins an EXACT tie -- strict >, scanning upward from 0. This is
 * the property the byte-exact greedy gate rests on. */
static void test_lowest_index_wins_tie(void) {
    float v[6] = { 1.0f, 7.0f, 7.0f, 7.0f, 2.0f, 7.0f };
    tt_assert(sg_argmax_f32(v, 6) == 1, "first of four equal maxima");

    float w[3] = { 5.0f, 5.0f, 5.0f };
    tt_assert(sg_argmax_f32(w, 3) == 0, "all equal -> index 0");
}

static void test_max_at_ends(void) {
    float lo[4] = { 9.0f, 1.0f, 2.0f, 3.0f };
    tt_assert(sg_argmax_f32(lo, 4) == 0, "max at index 0");
    float hi[4] = { 1.0f, 2.0f, 3.0f, 9.0f };
    tt_assert(sg_argmax_f32(hi, 4) == 3, "max at last index");
}

static void test_negatives_only(void) {
    float v[4] = { -9.0f, -2.0f, -2.5f, -100.0f };
    tt_assert(sg_argmax_f32(v, 4) == 1, "least-negative wins");
}

/* n == 1 returns 0; NULL / n == 0 are defensive no-ops returning 0. */
static void test_edge_cases(void) {
    float one[1] = { -3.0f };
    tt_assert(sg_argmax_f32(one, 1) == 0, "single element -> 0");
    tt_assert(sg_argmax_f32(NULL, 5) == 0, "NULL -> 0");
    tt_assert(sg_argmax_f32(one, 0) == 0, "n==0 -> 0");
}

int main(void) {
    tt_run("plain max", test_plain_max);
    tt_run("lowest index wins an exact tie", test_lowest_index_wins_tie);
    tt_run("max at first / last index", test_max_at_ends);
    tt_run("negatives only", test_negatives_only);
    tt_run("edge cases (NULL / n=0 / n=1)", test_edge_cases);
    return tt_report();
}
