#ifndef TINYTEST_H
#define TINYTEST_H
#include <stdio.h>
#include <stdlib.h>
static int tt_checks = 0, tt_fails = 0;
#define tt_assert(cond, ...) do { tt_checks++; if (!(cond)) { tt_fails++; \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
#define tt_run(name, fn) do { fprintf(stderr, "== %s\n", name); fn(); } while (0)
static inline int tt_report(void) {
    fprintf(stderr, "%d checks, %d failures\n", tt_checks, tt_fails);
    return tt_fails ? 1 : 0;
}
#endif
