#include "tinytest.h"
#include "../surge.h"

void smoke(void) {
    tt_assert(sg_failed(SG_OK) == false, "SG_OK should not be failed");
}

int main(void) {
    tt_run("smoke", smoke);
    return tt_report();
}
