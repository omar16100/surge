/* tok.c - byte-level BPE tokenizer built from GGUF tokenizer.ggml.* metadata.
 *
 * Implements the GPT-2 style byte-level BPE used by "gpt2"-model GGUFs
 * (tokenizer.ggml.model == "gpt2"): every input byte is first mapped through
 * a fixed byte-to-unicode table (bytes 0-255 -> printable codepoints), then
 * adjacent symbols within each pretokenizer-delimited "word" are repeatedly
 * merged by lowest learned-merge rank until no further merge applies.
 *
 * Pretokenization implements the qwen/GPT-4 style regex used by this
 * tokenizer's `tokenizer.ggml.pre = "qwen35"` (verified against the real HF
 * tokenizer's pre_tokenizer.pretokenizers[0].pattern):
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
 *   |\p{N}
 *   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 *
 * as a hand-written priority-ordered scanner over decoded codepoints (no
 * regex engine / external dependency). Unicode letter/number/mark/space
 * classification is a pragmatic range-table subset (ASCII, Latin
 * incl. accented, common CJK/Hangul/Kana blocks, Unicode whitespace),
 * sufficient for and verified against tests/fixtures/tok_cases.jsonl; see
 * the task-4 report for the exact coverage and its limits (no full Unicode
 * NFC normalization, no non-ASCII digit BPE numeral coverage beyond a few
 * common blocks).
 *
 * Two open-addressing FNV-1a hash maps (no external libs) back the tokenizer:
 * vocab_map (string -> id) and merge_map ((left,right) pair -> rank). The
 * merge OPERATION itself is O(1): since the byte-to-unicode mapping is
 * applied in order, symbol i's mapped bytes are always immediately followed
 * by symbol i+1's in a single contiguous buffer, so merging two adjacent
 * symbols is just extending a length, never a copy. Finding which pair to
 * merge next is done with a lazy generation-stamped min-heap of merge
 * candidates (see push_candidate/heap_push/heap_pop below), giving
 * O(word_len log word_len) per pretoken word overall rather than the
 * O(word_len^2) a naive full-rescan-per-merge approach would pay on long
 * unbroken runs (long dividers, hashes, ids, ascii art, ...).
 */
#include "surge.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Codepoints produced by the GPT-2 byte-to-unicode table never exceed ~323
 * (256 base bytes, ~188 already printable, the rest remapped to 256+n); 512
 * leaves comfortable margin. */
#define TOK_CP_MAX 512u

/* ---------------------------------------------------------------------
 * FNV-1a hashing and open-addressing hash maps (no external libs).
 * --------------------------------------------------------------------- */

static uint64_t fnv1a(const char *s, uint32_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv1a_pair(const char *l, uint32_t llen, const char *r, uint32_t rlen) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < llen; i++) {
        h ^= (unsigned char)l[i];
        h *= 1099511628211ULL;
    }
    h ^= 0xFFu;
    h *= 1099511628211ULL;
    for (uint32_t i = 0; i < rlen; i++) {
        h ^= (unsigned char)r[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* vocab_map: byte-encoded token string -> vocab id. */

typedef struct {
    uint64_t hash;
    char *key;      /* owned copy */
    uint32_t key_len;
    int32_t val;
    bool used;
} vocab_entry;

typedef struct {
    vocab_entry *slots;
    uint64_t cap;   /* power of two */
} vocab_map;

static bool vocab_map_init(vocab_map *m, uint64_t expected) {
    uint64_t cap = next_pow2(expected * 2 + 2);
    if (cap < 8) cap = 8;
    m->slots = calloc(cap, sizeof(*m->slots));
    m->cap = cap;
    return m->slots != NULL;
}

static bool vocab_map_put(vocab_map *m, const char *key, uint32_t key_len, int32_t val) {
    uint64_t h = fnv1a(key, key_len);
    uint64_t mask = m->cap - 1;
    uint64_t idx = h & mask;
    while (m->slots[idx].used) {
        vocab_entry *e = &m->slots[idx];
        if (e->hash == h && e->key_len == key_len && memcmp(e->key, key, key_len) == 0) {
            return true; /* duplicate: keep the first (lowest id) */
        }
        idx = (idx + 1) & mask;
    }
    char *copy = malloc(key_len ? key_len : 1);
    if (!copy) return false;
    memcpy(copy, key, key_len);
    m->slots[idx].hash = h;
    m->slots[idx].key = copy;
    m->slots[idx].key_len = key_len;
    m->slots[idx].val = val;
    m->slots[idx].used = true;
    return true;
}

static bool vocab_map_get(const vocab_map *m, const char *key, uint32_t key_len, int32_t *out) {
    uint64_t h = fnv1a(key, key_len);
    uint64_t mask = m->cap - 1;
    uint64_t idx = h & mask;
    while (m->slots[idx].used) {
        const vocab_entry *e = &m->slots[idx];
        if (e->hash == h && e->key_len == key_len && memcmp(e->key, key, key_len) == 0) {
            *out = e->val;
            return true;
        }
        idx = (idx + 1) & mask;
    }
    return false;
}

static void vocab_map_free(vocab_map *m) {
    if (!m->slots) return;
    for (uint64_t i = 0; i < m->cap; i++) {
        if (m->slots[i].used) free(m->slots[i].key);
    }
    free(m->slots);
    m->slots = NULL;
}

/* merge_map: (left, right) byte-encoded symbol pair -> merge rank. */

typedef struct {
    uint64_t hash;
    char *buf;      /* owned: left_len bytes then right_len bytes, no separator */
    uint32_t left_len, right_len;
    uint32_t rank;
    bool used;
} merge_entry;

typedef struct {
    merge_entry *slots;
    uint64_t cap;
} merge_map;

static bool merge_map_init(merge_map *m, uint64_t expected) {
    uint64_t cap = next_pow2(expected * 2 + 2);
    if (cap < 8) cap = 8;
    m->slots = calloc(cap, sizeof(*m->slots));
    m->cap = cap;
    return m->slots != NULL;
}

static bool merge_map_put(merge_map *m, const char *l, uint32_t llen,
                          const char *r, uint32_t rlen, uint32_t rank) {
    uint64_t h = fnv1a_pair(l, llen, r, rlen);
    uint64_t mask = m->cap - 1;
    uint64_t idx = h & mask;
    while (m->slots[idx].used) {
        merge_entry *e = &m->slots[idx];
        if (e->hash == h && e->left_len == llen && e->right_len == rlen &&
            memcmp(e->buf, l, llen) == 0 && memcmp(e->buf + llen, r, rlen) == 0) {
            return true; /* duplicate pair: keep the earlier (lower) rank */
        }
        idx = (idx + 1) & mask;
    }
    uint32_t total = llen + rlen;
    char *buf = malloc(total ? total : 1);
    if (!buf) return false;
    memcpy(buf, l, llen);
    memcpy(buf + llen, r, rlen);
    m->slots[idx].hash = h;
    m->slots[idx].buf = buf;
    m->slots[idx].left_len = llen;
    m->slots[idx].right_len = rlen;
    m->slots[idx].rank = rank;
    m->slots[idx].used = true;
    return true;
}

static bool merge_map_get(const merge_map *m, const char *l, uint32_t llen,
                          const char *r, uint32_t rlen, uint32_t *rank_out) {
    uint64_t h = fnv1a_pair(l, llen, r, rlen);
    uint64_t mask = m->cap - 1;
    uint64_t idx = h & mask;
    while (m->slots[idx].used) {
        const merge_entry *e = &m->slots[idx];
        if (e->hash == h && e->left_len == llen && e->right_len == rlen &&
            memcmp(e->buf, l, llen) == 0 && memcmp(e->buf + llen, r, rlen) == 0) {
            *rank_out = e->rank;
            return true;
        }
        idx = (idx + 1) & mask;
    }
    return false;
}

static void merge_map_free(merge_map *m) {
    if (!m->slots) return;
    for (uint64_t i = 0; i < m->cap; i++) {
        if (m->slots[i].used) free(m->slots[i].buf);
    }
    free(m->slots);
    m->slots = NULL;
}

/* ---------------------------------------------------------------------
 * UTF-8 encode/decode helpers.
 * --------------------------------------------------------------------- */

static int utf8_encode(uint32_t cp, char *out) {
    unsigned char *o = (unsigned char *)out;
    if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) {
        o[0] = (unsigned char)(0xC0 | (cp >> 6));
        o[1] = (unsigned char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        o[0] = (unsigned char)(0xE0 | (cp >> 12));
        o[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (unsigned char)(0xF0 | (cp >> 18));
    o[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    o[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Lenient: on a malformed/truncated sequence, consumes 1 byte and returns
 * that byte's value so callers always make forward progress. Not expected to
 * be exercised on well-formed UTF-8 input (all fixtures are valid UTF-8). */
static uint32_t utf8_decode_one(const char *s, size_t avail, size_t *consumed) {
    const unsigned char *u = (const unsigned char *)s;
    if (avail == 0) { *consumed = 0; return 0; }
    unsigned char b0 = u[0];
    if (b0 < 0x80) { *consumed = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0 && avail >= 2 && (u[1] & 0xC0) == 0x80) {
        *consumed = 2;
        return ((uint32_t)(b0 & 0x1F) << 6) | (u[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0 && avail >= 3 && (u[1] & 0xC0) == 0x80 && (u[2] & 0xC0) == 0x80) {
        *consumed = 3;
        return ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)(u[1] & 0x3F) << 6) | (u[2] & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0 && avail >= 4 && (u[1] & 0xC0) == 0x80 &&
        (u[2] & 0xC0) == 0x80 && (u[3] & 0xC0) == 0x80) {
        *consumed = 4;
        return ((uint32_t)(b0 & 0x07) << 18) | ((uint32_t)(u[1] & 0x3F) << 12) |
               ((uint32_t)(u[2] & 0x3F) << 6) | (u[3] & 0x3F);
    }
    *consumed = 1;
    return b0;
}

/* Decodes the whole string into a codepoint array plus a parallel array of
 * each codepoint's starting byte offset (with a trailing sentinel entry
 * equal to len), so pretoken spans can be sliced back into raw UTF-8 bytes.
 * Returns false on allocation failure (out params are left unset/freed by
 * the caller isn't required: this function frees its own partial state). */
static bool decode_utf8_string(const char *s, size_t len, uint32_t **cps_out,
                               uint32_t **byte_off_out, uint32_t *n_out) {
    uint32_t cap = 16, n = 0;
    uint32_t *cps = malloc(cap * sizeof(*cps));
    uint32_t *off = malloc((cap + 1) * sizeof(*off));
    if (!cps || !off) { free(cps); free(off); return false; }

    size_t pos = 0;
    while (pos < len) {
        if (n + 1 > cap) {
            /* Realloc one array at a time and update its variable
             * immediately on success, so a failure on the second never
             * frees/uses a pointer already invalidated by the first. */
            uint32_t new_cap = cap * 2;
            uint32_t *ncps = realloc(cps, new_cap * sizeof(*cps));
            if (!ncps) { free(cps); free(off); return false; }
            cps = ncps;
            uint32_t *noff = realloc(off, (new_cap + 1) * sizeof(*off));
            if (!noff) { free(cps); free(off); return false; }
            off = noff;
            cap = new_cap;
        }
        size_t consumed;
        uint32_t cp = utf8_decode_one(s + pos, len - pos, &consumed);
        off[n] = (uint32_t)pos;
        cps[n] = cp;
        n++;
        pos += consumed;
    }
    if (n + 1 > cap) {
        uint32_t new_cap = cap + 1;
        uint32_t *ncps = realloc(cps, new_cap * sizeof(*cps));
        if (!ncps) { free(cps); free(off); return false; }
        cps = ncps;
        uint32_t *noff = realloc(off, (new_cap + 1) * sizeof(*off));
        if (!noff) { free(cps); free(off); return false; }
        off = noff;
    }
    off[n] = (uint32_t)pos;
    *cps_out = cps;
    *byte_off_out = off;
    *n_out = n;
    return true;
}

/* ---------------------------------------------------------------------
 * Pragmatic Unicode classification (ASCII, accented Latin, common CJK /
 * Hangul / Kana blocks, Unicode whitespace). See the file header comment.
 * --------------------------------------------------------------------- */

static bool is_space(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680:
        case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

static bool is_number(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 0x0660 && cp <= 0x0669) return true; /* Arabic-Indic */
    if (cp >= 0x06F0 && cp <= 0x06F9) return true; /* Extended Arabic-Indic */
    if (cp >= 0x0966 && cp <= 0x096F) return true; /* Devanagari */
    if (cp >= 0xFF10 && cp <= 0xFF19) return true; /* fullwidth digits */
    return false;
}

static bool is_mark(uint32_t cp) {
    if (cp >= 0x0300 && cp <= 0x036F) return true; /* combining diacritics */
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return true;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return true;
    if (cp >= 0x20D0 && cp <= 0x20FF) return true;
    if (cp >= 0x3099 && cp <= 0x309A) return true; /* Japanese combining marks */
    return false;
}

static bool is_letter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp == 0xAA || cp == 0xB5 || cp == 0xBA) return true;
    if (cp >= 0xC0 && cp <= 0xD6) return true;
    if (cp >= 0xD8 && cp <= 0xF6) return true;
    if (cp >= 0xF8 && cp <= 0x2C1) return true;    /* Latin-1/Ext-A/Ext-B, IPA */
    if (cp >= 0x370 && cp <= 0x3FF && cp != 0x374 && cp != 0x375 && cp != 0x37E) return true;
    if (cp >= 0x400 && cp <= 0x52F) return true;   /* Cyrillic + supplement */
    if (cp >= 0x531 && cp <= 0x58A) return true;   /* Armenian */
    if (cp >= 0x5D0 && cp <= 0x5EA) return true;   /* Hebrew */
    if (cp >= 0x620 && cp <= 0x64A) return true;   /* Arabic */
    if (cp >= 0x900 && cp <= 0x97F) return true;   /* Devanagari */
    if (cp >= 0x1100 && cp <= 0x11FF) return true; /* Hangul Jamo */
    if (cp >= 0x3040 && cp <= 0x309F) return true; /* Hiragana */
    if (cp >= 0x30A0 && cp <= 0x30FF) return true; /* Katakana */
    if (cp >= 0x3400 && cp <= 0x4DBF) return true; /* CJK ext A */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true; /* CJK unified */
    if (cp >= 0xAC00 && cp <= 0xD7A3) return true; /* Hangul syllables */
    if (cp >= 0xF900 && cp <= 0xFAFF) return true; /* CJK compat ideographs */
    return false;
}

static bool is_other(uint32_t cp) {
    return !is_space(cp) && !is_letter(cp) && !is_mark(cp) && !is_number(cp);
}

/* ---------------------------------------------------------------------
 * Pretokenizer scanner: hand-written priority-ordered matcher for the qwen
 * regex documented in the file header. Returns the end codepoint index of
 * the next pretoken span starting at i (i < n guaranteed by the caller).
 * --------------------------------------------------------------------- */

static bool contraction_len_at(const uint32_t *cp, uint32_t n, uint32_t i, uint32_t *out_len) {
    if (cp[i] != '\'') return false;
    if (i + 1 >= n) return false;
    uint32_t c1 = cp[i + 1];
    uint32_t lc1 = (c1 >= 'A' && c1 <= 'Z') ? c1 + 32 : c1;
    if (lc1 == 's' || lc1 == 't' || lc1 == 'm' || lc1 == 'd') { *out_len = 2; return true; }
    if (i + 2 >= n) return false;
    uint32_t c2 = cp[i + 2];
    uint32_t lc2 = (c2 >= 'A' && c2 <= 'Z') ? c2 + 32 : c2;
    if ((lc1 == 'r' && lc2 == 'e') || (lc1 == 'v' && lc2 == 'e') || (lc1 == 'l' && lc2 == 'l')) {
        *out_len = 3;
        return true;
    }
    return false;
}

static uint32_t scan_pretoken(const uint32_t *cp, uint32_t n, uint32_t i) {
    uint32_t clen;
    if (contraction_len_at(cp, n, i, &clen)) return i + clen;

    /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
    {
        uint32_t p = i;
        if (!(is_letter(cp[i]) || is_mark(cp[i]))) {
            if (cp[i] != '\r' && cp[i] != '\n' && !is_number(cp[i]) &&
                i + 1 < n && (is_letter(cp[i + 1]) || is_mark(cp[i + 1]))) {
                p = i + 1;
            }
        }
        if (p < n && (is_letter(cp[p]) || is_mark(cp[p]))) {
            uint32_t q = p;
            while (q < n && (is_letter(cp[q]) || is_mark(cp[q]))) q++;
            return q;
        }
    }

    /* \p{N} */
    if (is_number(cp[i])) return i + 1;

    /* ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* -- by elimination cp[i] is space or
     * "other" here (letter/mark/number already matched above). */
    {
        uint32_t p = i;
        if (cp[i] == ' ' && i + 1 < n && is_other(cp[i + 1])) p = i + 1;
        if (p < n && is_other(cp[p])) {
            uint32_t q = p;
            while (q < n && is_other(cp[q])) q++;
            while (q < n && (cp[q] == '\r' || cp[q] == '\n')) q++;
            return q;
        }
    }

    /* \s*[\r\n]+ | \s+(?!\S) | \s+ -- cp[i] is guaranteed whitespace here. */
    {
        uint32_t j = i;
        while (j < n && is_space(cp[j])) j++;
        uint32_t run_len = j - i;

        int64_t last_nl = -1;
        for (uint32_t k = i; k < j; k++) {
            if (cp[k] == '\r' || cp[k] == '\n') last_nl = (int64_t)k;
        }
        if (last_nl >= 0) return (uint32_t)last_nl + 1;

        bool followed_by_nonspace = (j < n);
        uint32_t consumed = (run_len > 1 && followed_by_nonspace) ? run_len - 1 : run_len;
        if (consumed == 0) consumed = 1; /* always make forward progress */
        return i + consumed;
    }
}

/* ---------------------------------------------------------------------
 * sg_tok.
 * --------------------------------------------------------------------- */

struct sg_tok {
    uint32_t byte_to_cp[256];
    int32_t cp_to_byte[TOK_CP_MAX]; /* -1 = unmapped */

    vocab_map vmap; /* byte-encoded token string -> id */
    merge_map mmap; /* (left, right) byte-encoded pair -> rank */

    char **decoded;        /* id -> owned raw-byte buffer, for decode */
    uint32_t *decoded_len; /* id -> length of that buffer */
    uint64_t vocab_count;

    int32_t eos_id;
};

static void build_byte_tables(sg_tok *t) {
    bool printable[256] = {0};
    for (int b = '!'; b <= '~'; b++) printable[b] = true;
    for (int b = 0xA1; b <= 0xAC; b++) printable[b] = true;
    for (int b = 0xAE; b <= 0xFF; b++) printable[b] = true;

    for (uint32_t c = 0; c < TOK_CP_MAX; c++) t->cp_to_byte[c] = -1;

    uint32_t n = 0;
    for (int b = 0; b < 256; b++) {
        uint32_t cp = printable[b] ? (uint32_t)b : (256u + n);
        if (!printable[b]) n++;
        t->byte_to_cp[b] = cp;
        t->cp_to_byte[cp] = b;
    }
}

/* Symbols within one pretoken word, doubly linked with a generation counter
 * bumped whenever a symbol's content grows (absorbs a neighbor). `start` is
 * a byte offset into a per-word scratch buffer where every symbol's
 * byte-encoded content is laid out contiguously and in original order, so
 * absorbing a neighbor is just extending `len` (no copy; see file header).
 * `dead` marks a symbol removed by having been absorbed. */
typedef struct {
    uint32_t start, len;
    int32_t next, prev;
    uint32_t gen;
    bool dead;
} bpe_symbol;

/* Lazy min-heap of merge candidates, ordered by (rank asc, left-index asc)
 * -- the second key reproduces the reference "leftmost lowest-rank pair
 * wins ties" BPE rule. gen_l/gen_r pin each endpoint's content version at
 * push time, so a stale entry (an endpoint that has since died, moved to a
 * different neighbor, or grown by absorbing something else) is detected in
 * O(1) at pop time and skipped, rather than trusting a rank computed
 * against content that no longer exists. This turns merge selection from
 * O(word_len) work per merge (rescan every surviving pair every time) into
 * O(log word_len) amortized, avoiding O(word_len^2) blowup on long
 * unbroken pretokens (long dividers, hashes, ids, ascii art, ...). */
typedef struct { uint32_t rank; int32_t l, r; uint32_t gen_l, gen_r; } heap_entry;

static bool heap_less(const heap_entry *a, const heap_entry *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->l < b->l;
}

static void heap_push(heap_entry *heap, uint32_t *len, heap_entry e) {
    uint32_t i = (*len)++;
    heap[i] = e;
    while (i > 0) {
        uint32_t parent = (i - 1) / 2;
        if (!heap_less(&heap[i], &heap[parent])) break;
        heap_entry tmp = heap[i]; heap[i] = heap[parent]; heap[parent] = tmp;
        i = parent;
    }
}

static bool heap_pop(heap_entry *heap, uint32_t *len, heap_entry *out) {
    if (*len == 0) return false;
    *out = heap[0];
    (*len)--;
    heap[0] = heap[*len];
    uint32_t i = 0;
    while (1) {
        uint32_t l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < *len && heap_less(&heap[l], &heap[smallest])) smallest = l;
        if (r < *len && heap_less(&heap[r], &heap[smallest])) smallest = r;
        if (smallest == i) break;
        heap_entry tmp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = tmp;
        i = smallest;
    }
    return true;
}

/* Looks up the merge rank for the pair currently at symbols a,b and, if a
 * merge rule exists for it, pushes a fresh candidate stamped with a and b's
 * current generations. */
static void push_candidate(const merge_map *mmap, const char *mapped, const bpe_symbol *sym,
                           heap_entry *heap, uint32_t *heap_len, int32_t a, int32_t b) {
    uint32_t rank;
    if (merge_map_get(mmap, mapped + sym[a].start, sym[a].len,
                      mapped + sym[b].start, sym[b].len, &rank)) {
        heap_push(heap, heap_len, (heap_entry){rank, a, b, sym[a].gen, sym[b].gen});
    }
}

static sg_err bpe_encode_word(const sg_tok *t, const char *word, uint32_t word_len,
                              int32_t **ids, uint64_t *ids_len, uint64_t *ids_cap) {
    if (word_len == 0) return SG_OK;

    char *mapped = malloc((size_t)word_len * 2);
    bpe_symbol *sym = malloc((size_t)word_len * sizeof(*sym));
    /* Worst case total pushes: word_len-1 initial candidates, plus at most 2
     * fresh pushes per completed merge (at most word_len-1 merges) -- always
     * comfortably under 3*word_len even accounting for pop/push interleaving. */
    uint64_t heap_cap = (uint64_t)word_len * 3 + 4;
    heap_entry *heap = malloc(heap_cap * sizeof(*heap));
    if (!mapped || !sym || !heap) {
        free(mapped); free(sym); free(heap);
        return (sg_err){"tok: out of memory"};
    }

    uint32_t off = 0;
    for (uint32_t k = 0; k < word_len; k++) {
        uint32_t cp = t->byte_to_cp[(unsigned char)word[k]];
        int elen = utf8_encode(cp, mapped + off);
        sym[k].start = off;
        sym[k].len = (uint32_t)elen;
        sym[k].next = (k + 1 < word_len) ? (int32_t)(k + 1) : -1;
        sym[k].prev = (k > 0) ? (int32_t)(k - 1) : -1;
        sym[k].gen = 0;
        sym[k].dead = false;
        off += (uint32_t)elen;
    }

    uint32_t heap_len = 0;
    for (uint32_t k = 0; k + 1 < word_len; k++) {
        push_candidate(&t->mmap, mapped, sym, heap, &heap_len, (int32_t)k, (int32_t)(k + 1));
    }

    heap_entry e;
    while (heap_pop(heap, &heap_len, &e)) {
        int32_t l = e.l, r = e.r;
        /* Stale entry: either endpoint died, l moved on to a different
         * right neighbor, or an endpoint's content changed since push
         * (grew by absorbing something) -- any of these means this rank
         * was computed against content that no longer exists. Skip it;
         * the merge it would have represented either already happened via
         * a fresher entry, or a fresher entry for the current pairing is
         * (or will be) in the heap instead. */
        if (sym[l].dead || sym[r].dead) continue;
        if (sym[l].next != r) continue;
        if (sym[l].gen != e.gen_l || sym[r].gen != e.gen_r) continue;

        /* Valid, current-content merge: absorb r into l. */
        int32_t nr = sym[r].next;
        sym[l].len += sym[r].len;
        sym[l].next = nr;
        sym[l].gen++;
        sym[r].dead = true;
        if (nr != -1) sym[nr].prev = l;

        /* l's content and right neighbor changed: refresh both boundaries. */
        if (nr != -1) push_candidate(&t->mmap, mapped, sym, heap, &heap_len, l, nr);
        int32_t pl = sym[l].prev;
        if (pl != -1) push_candidate(&t->mmap, mapped, sym, heap, &heap_len, pl, l);
    }

    sg_err result = SG_OK;
    for (int32_t cur = 0; cur != -1; cur = sym[cur].next) {
        int32_t id;
        if (!vocab_map_get(&t->vmap, mapped + sym[cur].start, sym[cur].len, &id)) {
            result = (sg_err){"tok: BPE produced a symbol absent from vocab (corrupt merges/vocab?)"};
            break;
        }
        if (*ids_len == *ids_cap) {
            uint64_t new_cap = (*ids_cap == 0) ? 64 : (*ids_cap * 2);
            int32_t *nids = realloc(*ids, new_cap * sizeof(**ids));
            if (!nids) { result = (sg_err){"tok: out of memory"}; break; }
            *ids = nids;
            *ids_cap = new_cap;
        }
        (*ids)[(*ids_len)++] = id;
    }

    free(heap);
    free(sym);
    free(mapped);
    return result;
}

sg_err sg_tok_from_gguf(const sg_gguf *g, sg_tok **out) {
    if (out) *out = NULL;
    if (!g || !out) return (sg_err){"tok: invalid arguments"};

    const char *model_kind = NULL;
    if (!sg_gguf_get_str(g, "tokenizer.ggml.model", &model_kind) ||
        strcmp(model_kind, "gpt2") != 0) {
        return (sg_err){"tok: tokenizer.ggml.model is not \"gpt2\" (byte-level BPE required)"};
    }

    sg_gguf_kv_type tok_elem_type;
    uint64_t vocab_count = 0;
    if (!sg_gguf_get_arr(g, "tokenizer.ggml.tokens", &tok_elem_type, NULL, &vocab_count) ||
        tok_elem_type != SG_GGUF_STR || vocab_count == 0) {
        return (sg_err){"tok: tokenizer.ggml.tokens missing or not a string array"};
    }

    sg_gguf_kv_type merge_elem_type;
    uint64_t merge_count = 0;
    if (!sg_gguf_get_arr(g, "tokenizer.ggml.merges", &merge_elem_type, NULL, &merge_count) ||
        merge_elem_type != SG_GGUF_STR) {
        return (sg_err){"tok: tokenizer.ggml.merges missing or not a string array"};
    }

    sg_tok *t = calloc(1, sizeof(*t));
    if (!t) return (sg_err){"tok: out of memory"};

    build_byte_tables(t);
    t->vocab_count = vocab_count;
    t->decoded = calloc(vocab_count, sizeof(*t->decoded));
    t->decoded_len = calloc(vocab_count, sizeof(*t->decoded_len));
    if (!t->decoded || !t->decoded_len) { sg_tok_free(t); return (sg_err){"tok: out of memory"}; }

    if (!vocab_map_init(&t->vmap, vocab_count) || !merge_map_init(&t->mmap, merge_count)) {
        sg_tok_free(t);
        return (sg_err){"tok: out of memory"};
    }

    sg_err e = SG_OK;
    for (uint64_t i = 0; i < vocab_count; i++) {
        const char *s = NULL;
        if (!sg_gguf_get_arr_str(g, "tokenizer.ggml.tokens", i, &s)) {
            e = (sg_err){"tok: failed to read tokenizer.ggml.tokens element"};
            break;
        }
        uint32_t slen = (uint32_t)strlen(s);
        if (!vocab_map_put(&t->vmap, s, slen, (int32_t)i)) {
            e = (sg_err){"tok: out of memory"};
            break;
        }

        /* Precompute this token's raw decoded bytes for O(1)-per-token
         * sg_tok_decode: walk its byte-encoded UTF-8 codepoints back through
         * the inverse byte table, one raw byte per codepoint. */
        char *dec = malloc(slen ? slen : 1);
        if (!dec) { e = (sg_err){"tok: out of memory"}; break; }
        uint32_t dn = 0;
        size_t pos = 0;
        bool bad = false;
        while (pos < slen) {
            size_t consumed;
            uint32_t cp = utf8_decode_one(s + pos, slen - pos, &consumed);
            if (cp >= TOK_CP_MAX || t->cp_to_byte[cp] < 0) { bad = true; break; }
            dec[dn++] = (char)t->cp_to_byte[cp];
            pos += consumed;
        }
        if (bad) {
            free(dec);
            e = (sg_err){"tok: vocab token contains a codepoint outside the byte-level table"};
            break;
        }
        t->decoded[i] = dec;
        t->decoded_len[i] = dn;
    }

    if (!sg_failed(e)) {
        for (uint64_t i = 0; i < merge_count; i++) {
            const char *s = NULL;
            if (!sg_gguf_get_arr_str(g, "tokenizer.ggml.merges", i, &s)) {
                e = (sg_err){"tok: failed to read tokenizer.ggml.merges element"};
                break;
            }
            const char *sp = strchr(s, ' ');
            if (!sp) {
                e = (sg_err){"tok: merges element missing the left/right separator space"};
                break;
            }
            uint32_t llen = (uint32_t)(sp - s);
            uint32_t rlen = (uint32_t)strlen(sp + 1);
            if (!merge_map_put(&t->mmap, s, llen, sp + 1, rlen, (uint32_t)i)) {
                e = (sg_err){"tok: out of memory"};
                break;
            }
        }
    }

    if (sg_failed(e)) { sg_tok_free(t); return e; }

    /* eos_token_id's exact int width varies by GGUF writer; scan the kv
     * table and pull it through the generic scalar accessor rather than
     * assuming u32 (surge.h's typed getters don't cover every int width). */
    int64_t eos_val = -1;
    for (uint64_t i = 0; i < sg_gguf_kv_count(g); i++) {
        const char *key; sg_gguf_kv_type type;
        if (!sg_gguf_kv_at(g, i, &key, &type)) continue;
        if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            int64_t v;
            if (sg_gguf_kv_scalar_at(g, i, NULL, &v, NULL, NULL)) eos_val = v;
            break;
        }
    }
    t->eos_id = (int32_t)eos_val;

    *out = t;
    return SG_OK;
}

void sg_tok_free(sg_tok *t) {
    if (!t) return;
    if (t->decoded) {
        for (uint64_t i = 0; i < t->vocab_count; i++) free(t->decoded[i]);
    }
    free(t->decoded);
    free(t->decoded_len);
    vocab_map_free(&t->vmap);
    merge_map_free(&t->mmap);
    free(t);
}

/* decode_utf8_string's internal codepoint-count/byte-offset arithmetic uses
 * uint32_t for speed and memory (codepoint counts and pretoken byte spans
 * never approach 2^32 for realistic prompts); this ceiling keeps that
 * arithmetic (capacity doubling in particular) from ever wrapping, rather
 * than relying on inputs staying small by convention. 256 MiB is already
 * far beyond any realistic single tokenize call (a 128k-token context is a
 * few MB of text at most). */
#define SG_TOK_MAX_INPUT_BYTES ((size_t)1 << 28)

sg_err sg_tok_encode(const sg_tok *t, const char *utf8, int32_t **ids, uint64_t *n) {
    if (!t || !utf8 || !ids || !n) return (sg_err){"tok: invalid arguments"};
    *ids = NULL;
    *n = 0;

    size_t len = strlen(utf8);
    if (len > SG_TOK_MAX_INPUT_BYTES) {
        return (sg_err){"tok: input exceeds sg_tok_encode's single-call size limit (256 MiB)"};
    }

    uint32_t *cps = NULL, *byte_off = NULL, cp_n = 0;
    if (!decode_utf8_string(utf8, len, &cps, &byte_off, &cp_n)) {
        return (sg_err){"tok: out of memory"};
    }

    int32_t *out_ids = NULL;
    uint64_t out_len = 0, out_cap = 0;
    sg_err e = SG_OK;

    uint32_t i = 0;
    while (i < cp_n) {
        uint32_t q = scan_pretoken(cps, cp_n, i);
        if (q <= i) q = i + 1; /* defensive: always make forward progress */
        uint32_t bstart = byte_off[i], bend = byte_off[q];
        e = bpe_encode_word(t, utf8 + bstart, bend - bstart, &out_ids, &out_len, &out_cap);
        if (sg_failed(e)) break;
        i = q;
    }

    free(cps);
    free(byte_off);

    if (sg_failed(e)) {
        free(out_ids);
        return e;
    }
    *ids = out_ids;
    *n = out_len;
    return SG_OK;
}

int64_t sg_tok_decode(const sg_tok *t, const int32_t *ids, uint64_t n,
                      char *buf, uint64_t buf_cap) {
    if (!t || (n > 0 && (!ids || !buf))) return -1;
    uint64_t written = 0;
    for (uint64_t i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || (uint64_t)id >= t->vocab_count) return -1;
        uint32_t dlen = t->decoded_len[id];
        if (dlen > buf_cap - written) return -1;
        if (dlen > 0) memcpy(buf + written, t->decoded[id], dlen);
        written += dlen;
    }
    return (int64_t)written;
}

int32_t sg_tok_eos(const sg_tok *t) {
    return t ? t->eos_id : -1;
}
