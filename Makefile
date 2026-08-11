CC = cc
CFLAGS = -std=c11 -Wall -Wextra -Werror -O2 -Isrc -I.
# -lm is a no-op on macOS (libSystem already has it) but ref.c's pow/exp/
# log1p/sin/cos/sqrt need it explicitly on GCC/glibc.
LDLIBS = -lm
FRAMEWORKS = -framework Metal -framework Foundation
TESTS = $(wildcard tests/test_*.c)
check: $(TESTS:.c=.bin)
	@set -e; for t in $^; do ./$$t; done
# Gate 4 (M5.6): the `surge` CLI's prefill (default) vs --no-prefill must emit
# byte-identical gen_ids. This drives the real binary, so it is skipped under
# -DSURGE_NO_METAL (how `debug` runs) exactly like the Metal C tests; the ifeq
# excludes these recipe lines there, so `make debug` never builds or runs surge.
ifeq (,$(findstring SURGE_NO_METAL,$(CFLAGS)))
	@$(MAKE) --no-print-directory surge
	@bash tests/test_cli_prefill.sh ./surge
endif
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
METAL_TESTS = tests/test_metal_ops.bin tests/test_gpu_fwd.bin tests/test_gpu_prefill.bin

# tests/test_gpu_mem.bin (Task B2) is METAL-AWARE but not METAL-ONLY like the
# three above: its tier-1 pure-C tracker checks must actually RUN (not skip)
# under `make debug`, unlike test_metal_ops.c/test_gpu_fwd.c/test_gpu_prefill.c,
# whose whole file collapses to a bare stub under -DSURGE_NO_METAL. So under
# `check` it needs metal.m + the frameworks exactly like METAL_TESTS; under
# `debug` it needs LIB_SRC (for src/bench.c's tracker + sg_proc_phys_footprint)
# but NOT metal.m/the frameworks, so Metal still stays out of the ASan run.
METAL_HYBRID_TESTS = tests/test_gpu_mem.bin
ifeq (,$(findstring SURGE_NO_METAL,$(CFLAGS)))
$(METAL_TESTS) $(METAL_HYBRID_TESTS): tests/%.bin: tests/%.c $(LIB_SRC) src/metal.m $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ src/metal.m $< \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)
else
$(METAL_TESTS): tests/%.bin: tests/%.c
	$(CC) $(CFLAGS) -o $@ $<
$(METAL_HYBRID_TESTS): tests/%.bin: tests/%.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_SRC) $(LDLIBS)
endif

# `surge` (Task 10): the Metal decode path, with --ref selecting the scalar
# CPU forward for an A/B against the identical driver loop.
surge: src/cli_metal.c $(LIB_SRC) src/metal.m $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ src/cli_metal.c src/metal.m \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)

# surge-bench CLI stub (Task B1). src/bench.c's math (this task) already
# links into LIB_SRC via the src/*.c wildcard above and is exercised by
# tests/test_bench.c; the CLI itself (src/cli_bench.c, wiring B1-B4 plus the
# forward pass) is Task B5. This placeholder keeps `make surge-bench` from
# failing with "No rule to make target" in the meantime.
.PHONY: surge-bench
surge-bench:
	@echo "surge-bench: CLI not built yet (Task B5); src/bench.c math is in LIB_SRC now" >&2
	@exit 1

# --- M3.4 Q8_0 forward correctness gate (manual, NOT wired into `make check`)
#
# Proves surge's Metal Q8_0 decode of the 27B GGUF is NUMERICALLY correct, not
# just coherent. Two gates (see tools/gate_q8.sh):
#   (A) surge Metal Q8_0 vs surge CPU-ref Q8_0, teacher-forced, 100% top-1.
#   (B) surge greedy vs llama.cpp (llama-simple) greedy, no early divergence.
# Slow (one scalar-C 27B CPU forward ~10 min) and needs the 28 GB GGUF plus
# llama.cpp, so it stays a manual/env-gated target -- `make check` never runs
# it. Override GGUF=/path or PY=/interp; FREEZE=1 re-freezes the digest.
GGUF ?= /Users/macmini/models/gguf/Qwen3.6-27B-Q8_0.gguf
PY   ?= /Users/macmini/models/dsv4-venv/bin/python
.PHONY: gate gate-a gate-b
gate: surge surge-ref
	@FREEZE=$(FREEZE) bash tools/gate_q8.sh "$(GGUF)" "$(PY)"
gate-a: surge surge-ref
	$(PY) tools/tf_compare_q8.py --gguf "$(GGUF)" $(if $(FREEZE),--freeze,)
gate-b: surge
	$(PY) tools/xcheck_llama_q8.py --gguf "$(GGUF)"

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
