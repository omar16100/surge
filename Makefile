CC = cc
CFLAGS = -std=c11 -Wall -Wextra -Werror -O2 -Isrc -I.
# -lm is a no-op on macOS (libSystem already has it) but ref.c's pow/exp/
# log1p/sin/cos/sqrt need it explicitly on GCC/glibc.
LDLIBS = -lm
FRAMEWORKS = -framework Metal -framework Foundation
TESTS = $(wildcard tests/test_*.c)
check: $(TESTS:.c=.bin)
	@set -e; for t in $^; do ./$$t; done
LIB_SRC = $(filter-out src/metal.m $(wildcard src/cli_*.c),$(wildcard src/*.c))
tests/%.bin: tests/%.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_SRC) $(LDLIBS)
surge-info: src/cli_info.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
surge-ref: src/cli_ref.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# --- Metal (Task 9) ---------------------------------------------------
#
# The shader compile needs Xcode's Metal toolchain, which is a separately
# downloaded component on macOS 26 (`xcodebuild -downloadComponent
# MetalToolchain`); without it `xcrun metal` exits with a message saying so.
#
# -fno-fast-math is not optional. The default enables reassociation and the
# fast transcendentals, and the kernels are checked against ref.c's double
# accumulators to 1e-4 relative with precise::exp / precise::sqrt.
METALLIB = src/kernels.metallib
METAL_DEFS = -DSG_METALLIB_PATH='"$(CURDIR)/src/kernels.metallib"'
$(METALLIB): src/kernels.metal
	xcrun -sdk macosx metal -fno-fast-math -Wall -c $< -o src/kernels.air
	xcrun -sdk macosx metallib src/kernels.air -o $@

# The tests that link src/metal.m need the frameworks and the metallib, so
# they get a static pattern rule instead of the generic one above (an
# explicit rule wins). Under -DSURGE_NO_METAL -- which is how `debug` runs --
# these sources compile down to a skip notice, so nothing Metal is built or
# linked and ASan's MallocStackLogging never meets the driver.
METAL_TESTS = tests/test_metal_ops.bin tests/test_gpu_fwd.bin
ifeq (,$(findstring SURGE_NO_METAL,$(CFLAGS)))
$(METAL_TESTS): tests/%.bin: tests/%.c $(LIB_SRC) src/metal.m $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ src/metal.m $< \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)
else
$(METAL_TESTS): tests/%.bin: tests/%.c
	$(CC) $(CFLAGS) -o $@ $<
endif

# `surge` (Task 10): the Metal decode path, with --ref selecting the scalar
# CPU forward for an A/B against the identical driver loop.
surge: src/cli_metal.c $(LIB_SRC) src/metal.m $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ src/cli_metal.c src/metal.m \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)

# `debug` must DELETE the test binaries first. Without that, a preceding
# `make check` leaves them newer than the sources, make considers them up to
# date (it does not track CFLAGS), and `make debug` cheerfully re-runs the
# uninstrumented binaries and reports a green sanitizer run that never ran a
# sanitizer. Recursing with CFLAGS on the command line also beats the
# `debug: CFLAGS += ...` target-specific form, which does not reach the
# pattern rule's recipe in every make.
debug:
	@rm -f $(TESTS:.c=.bin)
	@rm -rf $(TESTS:.c=.bin.dSYM)
	$(MAKE) CFLAGS="$(CFLAGS) -DSURGE_NO_METAL -fsanitize=address,undefined -fno-omit-frame-pointer -g -O0" check
	@rm -f $(TESTS:.c=.bin)
	@rm -rf $(TESTS:.c=.bin.dSYM)
clean:
	rm -f $(TESTS:.c=.bin) surge-info surge-ref surge $(METALLIB) src/kernels.air
	rm -rf $(TESTS:.c=.bin.dSYM) surge-info.dSYM surge-ref.dSYM surge.dSYM
.PHONY: check debug clean
