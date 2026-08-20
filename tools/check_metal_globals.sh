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
# layer's definition silently. All twelve are also in the executable's dynamic
# export trie (`xcrun dyld_info -exports ./surge`), so the surface is wider
# than "another .c in this repo".
#
# The real fix is task R4: sg_-prefix all twelve. That is a rename, it must
# land BEFORE the next src/metal.m cut (which promotes roughly another dozen),
# and it deliberately did NOT happen inside R3, because bundling a 12-symbol
# rename into a code move would have destroyed R3's byte-identity evidence
# (metal_prefill.o byte-identical, _gpu_grid instruction-identical 54/54,
# _sg_gpu_forward bit-identical at 1408 instructions).
#
# THIS SCRIPT IS THE INTERIM GUARD, not the fix. It does two things:
#
#   CHECK 1  The set of UNPREFIXED external-linkage prototypes declared in
#            src/metal_internal.h is exactly the frozen twelve below. A new
#            unprefixed global joining the set fails here, which forces the
#            author to either sg_-prefix it (right answer) or edit this list
#            deliberately. Adding an `sg_`-prefixed prototype is always fine
#            and is invisible to this check.
#   CHECK 2  None of the twelve is declared or defined anywhere outside the
#            four files that own them. Comment mentions are fine and are
#            skipped; this looks only at declaration/definition context.
#
# Pure grep over tracked source. No compiler, no GPU, no network, no model, no
# ordering dependence: safe inside `make check` and inside `make debug`'s
# -DSURGE_NO_METAL recursion, where the .m files are not built at all.
#
# After R4 lands, FROZEN becomes empty and both checks still pass; keep the
# script so the set cannot silently grow back.

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hdr="$root/src/metal_internal.h"

# The twelve, in the order they were promoted.
# R2 (2026-08-20, src/metal.m -> src/metal_prefill.m seam): nine.
# R3 (2026-08-20, src/metal.m -> src/metal_validate.m seam): three.
FROZEN="check_params
check_sizes
enc_kv_store
enc_matmul
enc_op
gemm_kernel_for
gpu_alloc_f32
gpu_elem_width
gpu_embed_row
gpu_errf
gpu_grid
scratch_ensure"

fail=0

if [ ! -f "$hdr" ]; then
    echo "metal-globals: FAIL: $hdr not found" >&2
    exit 1
fi

# CHECK 1. Extract every external-linkage function prototype from the header:
# a line starting in column 1 that is not `static`, `typedef` or a directive,
# and that names an identifier immediately before a `(`. Continuation lines of
# a multi-line prototype are indented and therefore never match. Then drop the
# sg_-prefixed ones, which are exactly what R4 is moving everything towards.
have="$(grep -E '^[A-Za-z_]' "$hdr" \
        | grep -vE '^(static|typedef|#)' \
        | sed -nE 's/^[A-Za-z_][A-Za-z_0-9 ]*[ *]([A-Za-z_][A-Za-z_0-9]*)[[:space:]]*\(.*/\1/p' \
        | grep -vE '^sg_' \
        | sort -u)"
want="$(printf '%s\n' "$FROZEN" | sort -u)"

added="$(comm -13 <(printf '%s\n' "$want") <(printf '%s\n' "$have"))"
removed="$(comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$have"))"

if [ -n "$added" ]; then
    fail=1
    echo "metal-globals: FAIL: new UNPREFIXED global(s) in src/metal_internal.h:" >&2
    printf '  %s\n' $added >&2
    echo "  Give them an sg_ prefix (see task R4), or, if the name really must" >&2
    echo "  stay bare, add it to FROZEN in $0 and say why in the R4 entry." >&2
fi
if [ -n "$removed" ]; then
    fail=1
    echo "metal-globals: FAIL: frozen global(s) no longer declared:" >&2
    printf '  %s\n' $removed >&2
    echo "  If R4 renamed them, delete them from FROZEN in $0." >&2
fi

# CHECK 2. No declaration or definition of any of the twelve outside the four
# files that own them. First non-space character `*`, `/` or `#` means the hit
# is a comment or a directive and is not a declaration.
owners="src/metal.m src/metal_prefill.m src/metal_validate.m src/metal_internal.h"
pattern="$(printf '%s\n' "$FROZEN" | paste -sd'|' -)"

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

n="$(printf '%s\n' "$want" | grep -c . || true)"
if [ "$fail" -eq 0 ]; then
    echo "metal-globals: ok, $n unprefixed host globals, exactly the frozen set (R4 will sg_ prefix them)"
else
    exit 1
fi
