#!/usr/bin/env bash
# check_metal_globals.sh - freeze the Metal host layer's UNPREFIXED global set.
#
# WHY THIS EXISTS. Tasks R2 and R3 split src/metal.m into three translation
# units. Helpers that used to be `static` had to gain external linkage so the
# other .m files could call them, and they kept their original unqualified
# names: twelve of them, including names as generic as `check_params` and
# `check_sizes`. Every link in this project is a direct `cc` of objects into an
# executable, so a duplicate DEFINITION would fail the link loudly. The QUIET
# failure mode is different: C has no name mangling, so a future translation
# unit that DECLARES `sg_err check_params(const char *, const uint32_t *)`
# itself, without including src/metal_internal.h, binds to the Metal host
# layer's definition silently. All twelve were also in the executable's dynamic
# export trie (`xcrun dyld_info -exports ./surge`), so the surface was wider
# than "another .c in this repo".
#
# TASK R4 (2026-08-21) LANDED THE REAL FIX: all twelve are now sg_-prefixed
# (sg_gpu_errf, sg_scratch_ensure, sg_gpu_elem_width, sg_gemm_kernel_for,
# sg_gpu_embed_row, sg_gpu_alloc_f32, sg_enc_op, sg_enc_kv_store,
# sg_enc_matmul, sg_check_sizes, sg_check_params, sg_gpu_grid), so FROZEN below
# is EMPTY and the export trie carries no unprefixed entry from that set. R4 was
# kept out of R3 because bundling a 12-symbol rename into a code move would have
# destroyed R3's byte-identity evidence; as its own commit it produced the
# stronger result instead (every __text section byte-identical).
#
# THE SCRIPT STAYS, because an empty FROZEN is a state to DEFEND, not a reason
# to delete the guard. The next src/metal.m cut promotes roughly another dozen
# helpers, and CHECK 1 is what stops them landing unprefixed. It does two
# things:
#
#   CHECK 1  The set of UNPREFIXED external-linkage prototypes declared in
#            src/metal_internal.h is exactly the frozen set below, which is now
#            EMPTY. Any unprefixed global joining fails here, which forces the
#            author to either sg_-prefix it (right answer) or edit this list
#            deliberately. Adding an `sg_`-prefixed prototype is always fine
#            and is invisible to this check. THIS IS THE CHECK WITH TEETH NOW.
#   CHECK 2  No file outside the four that own them declares or defines a
#            frozen name. Comment mentions are fine and are skipped; this looks
#            only at declaration/definition context. With FROZEN empty it has
#            nothing to look for and is SKIPPED ENTIRELY -- it must be, because
#            an empty name list would build the alternation `\b()\b`, which
#            matches the empty string and therefore fires on every declaration
#            in the tree. It comes back automatically if a name is ever added
#            to FROZEN again.
#
# Pure grep over tracked source. No compiler, no GPU, no network, no model, no
# ordering dependence: safe inside `make check` and inside `make debug`'s
# -DSURGE_NO_METAL recursion, where the .m files are not built at all.

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hdr="$root/src/metal_internal.h"

# EMPTY SINCE TASK R4 (2026-08-21), and it should stay that way. It held the
# twelve names R2 and R3 promoted; R4 renamed every one of them to sg_<name>,
# so there is nothing left to permit. A name only belongs here if some future
# change genuinely cannot carry an sg_ prefix, and that needs an argument in the
# task entry, not a quiet edit.
FROZEN=""

fail=0

if [ ! -f "$hdr" ]; then
    echo "metal-globals: FAIL: $hdr not found" >&2
    exit 1
fi

# CHECK 1. Extract every external-linkage function prototype from the header:
# a line starting in column 1 that is not `static`, `typedef` or a directive,
# and that names an identifier immediately before a `(`. Continuation lines of
# a multi-line prototype are indented and therefore never match. Then drop the
# sg_-prefixed ones, which since R4 is every one of them.
have="$(grep -E '^[A-Za-z_]' "$hdr" \
        | grep -vE '^(static|typedef|#)' \
        | sed -nE 's/^[A-Za-z_][A-Za-z_0-9 ]*[ *]([A-Za-z_][A-Za-z_0-9]*)[[:space:]]*\(.*/\1/p' \
        | grep -vE '^sg_' \
        | sort -u)"
want="$(printf '%s\n' "$FROZEN" | grep -v '^$' | sort -u || true)"

added="$(comm -13 <(printf '%s\n' "$want") <(printf '%s\n' "$have"))"
removed="$(comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$have"))"

if [ -n "$added" ]; then
    fail=1
    echo "metal-globals: FAIL: new UNPREFIXED global(s) in src/metal_internal.h:" >&2
    printf '  %s\n' $added >&2
    echo "  Give them an sg_ prefix, the way task R4 did with the previous twelve," >&2
    echo "  or, if the name really must stay bare, add it to FROZEN in $0" >&2
    echo "  and say why in that task's entry." >&2
fi
if [ -n "$removed" ]; then
    fail=1
    echo "metal-globals: FAIL: frozen global(s) no longer declared:" >&2
    printf '  %s\n' $removed >&2
    echo "  If they were renamed, delete them from FROZEN in $0." >&2
fi

# CHECK 2. No declaration or definition of any frozen name outside the four
# files that own them. First non-space character `*`, `/` or `#` means the hit
# is a comment or a directive and is not a declaration.
#
# SKIPPED WHOLESALE WHEN FROZEN IS EMPTY, which it is since R4. This is not an
# optimisation: `paste -sd'|'` over nothing yields the empty pattern, the
# alternation becomes `\b()[[:space:]]*\(`, that matches the empty string before
# any `(`, and the check would report every declaration in src, tests and tools
# as a violation. Tested: with the guard left unguarded, an empty FROZEN reports
# src/bench.c and everything after it.
owners="src/metal.m src/metal_prefill.m src/metal_validate.m src/metal_internal.h"
pattern="$(printf '%s\n' "$want" | paste -sd'|' -)"

if [ -n "$want" ]; then
    while IFS= read -r f; do
        rel="${f#"$root"/}"
        case " $owners " in *" $rel "*) continue ;; esac
        hits="$(grep -nE "^[[:space:]]*[A-Za-z_].*\b(${pattern})[[:space:]]*\(" "$f" 2>/dev/null \
                | grep -vE '^[0-9]+:[[:space:]]*[*/#]' || true)"
        if [ -n "$hits" ]; then
            fail=1
            echo "metal-globals: FAIL: $rel declares or defines a Metal host global:" >&2
            printf '%s\n' "$hits" | sed 's/^/  /' >&2
        fi
    done < <(find "$root/src" "$root/tests" "$root/tools" -type f \
                  \( -name '*.c' -o -name '*.h' -o -name '*.m' \) 2>/dev/null | sort
             printf '%s\n' "$root/surge.h")
fi

n="$(printf '%s\n' "$want" | grep -c . || true)"
if [ "$fail" -ne 0 ]; then
    exit 1
elif [ "$n" -eq 0 ]; then
    echo "metal-globals: ok, 0 unprefixed host globals (task R4 sg_ prefixed all twelve)"
else
    echo "metal-globals: ok, $n unprefixed host global(s), exactly the frozen set"
fi
