CC = cc
CFLAGS = -std=c11 -Wall -Wextra -Werror -O2 -Isrc -I.
FRAMEWORKS = -framework Metal -framework Foundation
TESTS = $(wildcard tests/test_*.c)
check: $(TESTS:.c=.bin)
	@set -e; for t in $^; do ./$$t; done
LIB_SRC = $(filter-out src/metal.m $(wildcard src/cli_*.c),$(wildcard src/*.c))
tests/%.bin: tests/%.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_SRC)
surge-info: src/cli_info.c $(LIB_SRC)
	$(CC) $(CFLAGS) -o $@ $^
debug: CFLAGS += -fsanitize=address -g -O0
debug: check
.PHONY: check debug
