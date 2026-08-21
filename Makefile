CC = cc
CFLAGS = -std=c11 -Wall -Wextra -Werror -O2 -Isrc -I.
# -lm is a no-op on macOS (libSystem already has it) but ref.c's pow/exp/
# log1p/sin/cos/sqrt need it explicitly on GCC/glibc.
LDLIBS = -lm
FRAMEWORKS = -framework Metal -framework Foundation
TESTS = $(wildcard tests/test_*.c)
check: $(TESTS:.c=.bin)
# Guard on the Metal host layer's UNPREFIXED global set, which task R4 emptied
# by sg_ prefixing the twelve the R2/R3 splits had promoted (sg_check_params,
# sg_check_sizes, sg_gpu_grid, sg_enc_op, ...). Pure grep over source: no
# compiler, no GPU, no model, no ordering dependence, so it is free and
# deterministic in both `check` and `debug`'s -DSURGE_NO_METAL recursion, and it
# asserts NOTHING about test counts. It now fails if ANY unprefixed global joins
# src/metal_internal.h, which is what keeps the next src/metal.m cut from
# promoting its own dozen bare names. See the script's header comment.
	@bash tools/check_metal_globals.sh
	@set -e; for t in $^; do ./$$t; done
# Gate 4 (M5.6): the `surge` CLI's prefill (default) vs --no-prefill must emit
# byte-identical gen_ids. This drives the real binary, so it is skipped under
# -DSURGE_NO_METAL (how `debug` runs) exactly like the Metal C tests; the ifeq
# excludes these recipe lines there, so `make debug` never builds or runs surge.
ifeq (,$(findstring SURGE_NO_METAL,$(CFLAGS)))
	@$(MAKE) --no-print-directory surge
	@bash tests/test_cli_prefill.sh ./surge
	@$(MAKE) --no-print-directory surge-bench
	@bash tests/test_cli_bench.sh ./surge ./surge-bench
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
# THREE OBJECTIVE-C SOURCES, ONE HOST LAYER (tasks R2 and R3). src/metal.m
# passed the ~2000-line guideline, so the chunked-prefill half moved to
# src/metal_prefill.m and the per-dispatch validation trio to
# src/metal_validate.m; all three share src/metal_internal.h (struct sg_gpu,
# the KI_ and SG_K_ enums, sg_enc, the helpers that cross the seams). Unlike
# the .metal sources these are REAL translation units that link, so all three
# must appear on every link line that used to name src/metal.m and none may be
# dropped -- the link would fail loudly (undefined sg_gpu_prefill, undefined
# sg_check_params), which is the point.
#
# metal_internal.h is listed as a prerequisite for the same reason
# kernels_common.metal.h is below: make does not scan #include lines, so
# without it an edit to the shared struct sg_gpu would leave every Metal
# binary stale while still passing every test. THAT LIST IS HAND-MAINTAINED
# and is exhaustive today (metal_internal.h and surge.h; the system and
# framework headers are not tracked, as everywhere else in this Makefile).
METAL_M = src/metal.m src/metal_prefill.m src/metal_validate.m
METAL_M_DEPS = $(METAL_M) src/metal_internal.h surge.h

METALLIB = src/kernels.metallib
METAL_DEFS = -DSG_METALLIB_PATH='"$(CURDIR)/src/kernels.metallib"'
#
# TWO .metal SOURCES, ONE METALLIB (task R1). Metal compiles each translation
# unit to its own .air and `metallib` links them, so moving the split-K decode
# kernels into src/kernels_splitk.metal (done when kernels.metal passed the
# ~2000-line guideline) changes nothing the host can see: metal.m still looks
# every kernel up by name in this one library. THE FLAGS MUST STAY IDENTICAL
# ON BOTH -- losing -fno-fast-math on one of the two would be a silent
# numerics change across half the library -- which is why they are one
# variable used by one pattern rule rather than typed out per source.
#
# kernels_common.metal.h is a prerequisite of every .air because make does not
# scan #include lines: without it, editing the shared tg_sum/tg_max/SG_TG
# header would leave a stale metallib that still passes every test. THAT LIST
# IS HAND-MAINTAINED. It is exhaustive today (one header, no other #include in
# either source but <metal_stdlib>), and any #include added to either .metal
# source has to be added here too or its edits will not trigger a rebuild.
METAL_SRC = src/kernels.metal src/kernels_splitk.metal
METAL_AIR = $(METAL_SRC:.metal=.air)
METAL_CFLAGS = -fno-fast-math -Wall
src/%.air: src/%.metal src/kernels_common.metal.h
	xcrun -sdk macosx metal $(METAL_CFLAGS) -c $< -o $@
$(METALLIB): $(METAL_AIR)
	xcrun -sdk macosx metallib $^ -o $@

# The tests that link $(METAL_M) need the frameworks and the metallib, so
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
$(METAL_TESTS) $(METAL_HYBRID_TESTS): tests/%.bin: tests/%.c $(LIB_SRC) $(METAL_M_DEPS) $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ $(METAL_M) $< \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)
else
$(METAL_TESTS): tests/%.bin: tests/%.c
	$(CC) $(CFLAGS) -o $@ $<
$(METAL_HYBRID_TESTS): tests/%.bin: tests/%.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_SRC) $(LDLIBS)
endif

# `surge` (Task 10): the Metal decode path, with --ref selecting the scalar
# CPU forward for an A/B against the identical driver loop.
surge: src/cli_metal.c $(LIB_SRC) $(METAL_M_DEPS) $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ src/cli_metal.c $(METAL_M) \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)

# `surge-bench` (Task B5): the benchmark harness. Wires src/bench.c's B1-B4
# math + the B2 peak-memory probe to the M5 tiled prefill + the shared greedy
# driver (src/greedy.c's sg_argmax_f32, the SAME argmax `surge` uses, so their
# gen_ids cannot drift). Same link shape as `surge`.
surge-bench: src/cli_bench.c $(LIB_SRC) $(METAL_M_DEPS) $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ src/cli_bench.c $(METAL_M) \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)

# `make bench-check` (Task B5 gate, mirrors test_cli_prefill's pattern): builds
# both binaries and runs the surge-vs-surge-bench gen_ids parity + VOID/exit-3 +
# BOS-toggle shell test. The test SKIPs cleanly when Metal is unavailable, so
# this is safe to run anywhere; it is not part of `make debug` (SURGE_NO_METAL).
.PHONY: bench-check
bench-check: surge surge-bench
	@bash tests/test_cli_bench.sh ./surge ./surge-bench

# --- P2.3a: split-K decode-attention timing harness (NOT part of `make check`)
#
# tests/bench_splitk.c sweeps n_splits for k_attn_decode_splitk_partial +
# _combine against the incumbent k_attn_decode_f16, across the real 27B/4B
# shapes at seq 8192..262144 (see that file's header for the full
# methodology). It is a MEASUREMENT tool, not a correctness gate -- it has no
# pass/fail assertion, so it must never be folded into `check`'s $(TESTS)
# list. That list is `$(wildcard tests/test_*.c)`, and this file is
# deliberately named tests/bench_splitk.c (not test_*.c) so it is excluded
# structurally, not by a rule someone has to remember to keep following.
#
# Same Metal-aware build shape as METAL_TESTS above (collapses to a bare skip
# stub under SURGE_NO_METAL, so `make debug` never links Metal for it either),
# kept as its own rule rather than folded into METAL_TESTS so that list's own
# comment ("Task 9 Metal kernels ... checks its threadgroup width") stays
# accurate for every file in it -- this one is not a per-op correctness gate.
BENCH_SPLITK = tests/bench_splitk.bin
ifeq (,$(findstring SURGE_NO_METAL,$(CFLAGS)))
$(BENCH_SPLITK): tests/%.bin: tests/%.c $(LIB_SRC) $(METAL_M_DEPS) $(METALLIB)
	$(CC) $(CFLAGS) $(METAL_DEFS) -o $@ $(METAL_M) $< \
	  $(LIB_SRC) $(FRAMEWORKS) $(LDLIBS)
else
$(BENCH_SPLITK): tests/%.bin: tests/%.c
	$(CC) $(CFLAGS) -o $@ $<
endif

# `make bench-splitk`: builds AND RUNS the sweep (`./tests/bench_splitk.bin
# --reps N` to override the default rep count). Never invoked by `check` or
# `debug`. Slow and GPU-bound (many shapes x seq lengths x n_splits); prints a
# results table to stdout, one row per (shape, seq, n_splits), with a SKIP
# line wherever a row's buffers could not be allocated rather than a silently
# missing row.
.PHONY: bench-splitk
bench-splitk: $(BENCH_SPLITK)
	./$(BENCH_SPLITK)

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
	rm -f $(TESTS:.c=.bin) surge-info surge-ref surge surge-bench $(BENCH_SPLITK) $(METALLIB) $(METAL_AIR)
	rm -rf $(TESTS:.c=.bin.dSYM) surge-info.dSYM surge-ref.dSYM surge.dSYM surge-bench.dSYM
.PHONY: check debug clean
