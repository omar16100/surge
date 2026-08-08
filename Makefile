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
	$(MAKE) CFLAGS="$(CFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer -g -O0" check
	@rm -f $(TESTS:.c=.bin)
	@rm -rf $(TESTS:.c=.bin.dSYM)
clean:
	rm -f $(TESTS:.c=.bin) surge-info
	rm -rf $(TESTS:.c=.bin.dSYM) surge-info.dSYM
.PHONY: check debug clean
