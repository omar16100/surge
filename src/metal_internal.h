/* metal_internal.h - the definitions src/metal.m shares with the other
 * Objective-C translation units of the Metal host layer (tasks R2 and R3).
 *
 * THIS IS NOT A PUBLIC HEADER. The public contract (buffer layouts, params[]
 * per kernel, aliasing rules) lives in surge.h; this file exists only because
 * src/metal.m passed the ~2000-line guideline and pieces of it moved out:
 * its chunked-prefill half to src/metal_prefill.m (task R2) and its per-
 * dispatch validation trio to src/metal_validate.m (task R3). Both need to
 * see the same sg_gpu, the same kernel index enum and the same handful of
 * helpers. Nothing outside src/metal*.m may include it.
 *
 * WHAT BELONGS HERE, AND WHAT DOES NOT. Only declarations that genuinely
 * CROSS a seam between the three .m files. A helper with callers on one side
 * only stays static in that file. There is exactly one definition of
 * everything below; nothing here is a copy of anything in src/metal.m.
 *
 * TWO KINDS OF FUNCTION LIVE HERE, and the difference is deliberate:
 *
 *   - `static inline` DEFINITIONS, for the one-line accessors, bit casts and
 *     predicates (bufof, offof, fbits, mul_ck, add_ck, buf_big_enough). These
 *     were `static` in metal.m and are called from the innermost encode loops
 *     or from dozens of size checks, so giving them external linkage would
 *     have turned every one into a real call. `static inline` in a shared
 *     header keeps them file-local and inlinable in EVERY translation unit,
 *     which is the closest thing to "nothing changed".
 *   - PROTOTYPES ONLY, for the larger helpers (the encode primitives, the
 *     allocators, the error formatter). Those really do gain external linkage
 *     and lose cross-translation-unit inlining; they are per-dispatch or
 *     per-allocation, never per-element, so the cost is a call each.
 *
 * Objective-C: struct sg_gpu holds `id<MTL...>` members, so this header is
 * only includable from a .m compiled with the Metal and Foundation frameworks.
 */
#ifndef SURGE_METAL_INTERNAL_H
#define SURGE_METAL_INTERNAL_H

#include "surge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string.h>

/* The threadgroup width every reduction kernel folds its tree over. See the
 * kernel-table comment in src/metal.m for why `kind` is what guarantees the
 * reduction kernels are dispatched at exactly this width. */
#define SG_TG 256u

/* The GRID KIND of a kernel: how its grid is derived from params[]. Moved here
 * from src/metal.m by task R3, because sg_gpu_grid (the switch that reads it) is
 * now in src/metal_validate.m while the SG_KERNELS table that assigns it and
 * sg_gpu_init's threadgroup-width check that consults it both stayed in
 * src/metal.m. An anonymous enum is a declaration with no storage and no
 * linkage, so this move creates no symbol in any object, exactly like the KI_
 * enum below.
 *
 * `kind` is how the grid is derived from params[], and it is also what
 * guarantees the reduction kernels get exactly SG_TG threads per
 * threadgroup: kernels.metal folds its trees over the compile-time constant
 * SG_TG, so dispatching a different width would silently drop lanes. */
enum {
    SG_K_ELEM,    /* params[0] threads, non-uniform threadgroups */
    SG_K_ROWS,    /* params[0] threadgroups of SG_TG (one per output row) */
    SG_K_TG1,     /* exactly one threadgroup of SG_TG */
    SG_K_ATTN,    /* params[0] threadgroups of SG_TG, plus a scores scratch */
    SG_K_GATED,   /* params[1] threadgroups of SG_TG (one per value head) */
    SG_K_ELEM01,  /* params[0] * params[1] threads */
    SG_K_ELEM02,  /* params[0] * params[2] threads */
    SG_K_GROUPS2, /* params[2] threadgroups of SG_TG */
    /* M5.3: a 2D grid of ceil(params[1]/SG_GEMM_TN) x ceil(params[0]/SG_GEMM_TM)
     * threadgroups of SG_TG, one per GEMM output tile. Computed by hand in
     * sg_gpu_run_op (not by sg_gpu_grid, which only returns a 1D count) because
     * this is the only kind whose group count needs two dimensions. */
    SG_K_TILES2D,
    /* M5.4: k_rope_chunk, one thread per (token, head, element) of the chunk,
     * params[0]*params[2]*params[4] = head_dim*heads*n_tok threads, non-uniform
     * threadgroups. An elementwise kernel with no reduction, dispatched through
     * sg_gpu_grid's *elems path like SG_K_ELEM but with a 3-factor count. */
    SG_K_ROPE_CHUNK,
    /* P2.2: the split-K decode-attention partial kernel's grid, a 2D
     * params[6] x params[0] (split, query head) block of threadgroups of
     * SG_TG. The SECOND kind after SG_K_TILES2D that needs two group
     * dimensions and for the same reason SG_K_ATTN cannot be reused: that
     * class carries exactly one *groups count (see sg_gpu_grid in
     * src/metal_validate.m), so it can express "one threadgroup per head" and
     * nothing wider. Computed by hand in sg_gpu_run_attn_splitk_partial, which
     * is the only path that reaches the kernel at all (it takes six device
     * buffers, so sg_gpu_run_op's (a, b, out) shape cannot).
     *
     * P2.4's k_attn_decode_splitk_partial_gqa shares this kind with a params[6]
     * x params[1] (split, KV head) grid: one threadgroup per GQA GROUP rather
     * than per query head. Same class because the dispatchers compute both
     * extents by hand anyway; the `kind` column is only read by sg_gpu_init's
     * threadgroup-width check. */
    SG_K_HEADS2D
};

/* The byte counts below are products of up to three caller-supplied uint32
 * params, and (uint64)p[0] * p[1] * 4 genuinely wraps for large ones. A
 * wrapped `need` is worse than no check at all: it would be SMALL, so an
 * undersized buffer would sail through and the kernel would index with the
 * original, unwrapped dimensions. Everything therefore goes through these,
 * and an overflow is an error rather than a number. */
static inline bool mul_ck(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static inline bool add_ck(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

/* sg_gpu.pipes is indexed by position here, so the enum and the SG_KERNELS
 * table (src/metal.m, which is where the table and its _Static_assert stayed)
 * must stay in lockstep; that assert is what enforces it. The first
 * thirteen are Task 9 / M3.1's per-op kernels, reachable through
 * sg_gpu_run_op; the rest are Task 10's fused/strided variants, which take
 * buffer layouts sg_gpu_run_op has no size rule for and are therefore encoded
 * only by sg_gpu_forward. */
enum {
    KI_RMSNORM = 0, KI_RMSNORM_GATED, KI_ROPE, KI_MATVEC_BF16, KI_MATVEC_F32,
    KI_MATVEC_Q8, KI_SOFTMAX, KI_SWIGLU, KI_SILU, KI_GATE_SIGMOID, KI_ATTN,
    KI_CONV1D, KI_DELTA, KI_RMSNORM_HEADS, KI_ROPE_HEADS, KI_GATE_STRIDED,
    KI_SCALE, KI_ADD, KI_DELTA_GATES, KI_DELTA_MULTI,
    KI_KV_STORE_F16, KI_ATTN_F16,
    KI_MATMUL_BF16, KI_MATMUL_F32, KI_MATMUL_Q8,
    KI_ROPE_CHUNK, KI_ATTN_PREFILL,
    KI_CONV1D_CHUNK, KI_DELTA_GATES_CHUNK, KI_DELTA_CHUNK, KI_RMSNORM_GATED_CHUNK,
    KI_ATTN_SPLITK_PARTIAL, KI_ATTN_SPLITK_COMBINE,
    KI_ATTN_SPLITK_PARTIAL_GQA,
    KI_ATTN_SPLITK_PARTIAL_GQA_ONLINE,
    KI_COUNT
};

/* A wrapped or allocated buffer. The offset is what makes sg_gpu_wrap
 * possible at all: Metal demands a page-aligned base and a tensor inside a
 * checkpoint mmap never is one, so the handle remembers how far into the
 * page its data starts and every bind adds it. */
typedef struct {
    id<MTLBuffer> buf;
    uint64_t offset;
    uint64_t nbytes;
} sg_gpu_buf;

/* Every kernel reads its inputs from indices computed out of params[], so a
 * params/buffer mismatch is an out-of-bounds DEVICE read: not a crash the
 * process can catch, but a GPU fault that takes down the whole context (and
 * on a bad day the display driver). Hence a size precondition per kernel,
 * checked by sg_check_sizes in src/metal_validate.m, before anything is encoded.
 *
 * `static inline` and in this header since task R3, for the same reason the
 * other one-liners here are. sg_check_sizes moved to src/metal_validate.m and
 * needs it, but twenty-six of its twenty-nine call sites are the one-shot
 * entry points that stayed in src/metal.m. A two-instruction predicate behind
 * a real call at all twenty-nine would be pure loss, and it would add one more
 * unprefixed global with a very generic name; `static inline` gives each
 * translation unit its own inlinable copy and creates no symbol at all (nm
 * shows none in any object, at this revision or at the parent). */
static inline bool buf_big_enough(const sg_gpu_buf *b, uint64_t need) {
    return b != NULL && b->nbytes >= need;
}

/* Per-layer weights and state for the full decode path (Task 10). Exactly
 * one of the two attention groups is populated, by the same tensor-presence
 * rule sg_ref_state_new uses. The `w_*` handles wrap checkpoint memory with
 * no copy; everything else is surge-owned and zero-filled at allocation. */
typedef struct {
    bool is_attn;
    /* wrapped matmul weights */
    void *w_q, *w_k, *w_v, *w_o;                       /* full attention */
    void *w_qkv, *w_z, *w_a, *w_b, *w_out;             /* gated DeltaNet */
    void *w_gate, *w_up, *w_down;                      /* shared MLP */
    /* owned f32 copies of the small tensors, norm shift already applied */
    void *ln1, *ln2;        /* [hidden] */
    void *qk_norm;          /* [2*head_dim]: q_norm then k_norm */
    void *conv_w;           /* [conv_dim, conv_kernel] */
    void *a_dt;             /* [2*n_v_heads]: ssm_a then dt_bias */
    /* [value_dim + head_v_dim]: the in_proj_z output, with this layer's
     * ssm_norm weight parked immediately after it, because k_rmsnorm_gated
     * reads z and w out of ONE buffer. Per layer rather than shared for
     * exactly that reason. */
    void *zw;
    /* state */
    /* [2, max_ctx, n_kv_heads, head_dim]: K then V, f32. Populated ONLY when
     * SURGE_KV_DTYPE selects f32 (sg_gpu.kv_dtype == SG_T_F32): that path is
     * kept byte-for-bit as it was before M5.2, combined buffer and all, so
     * the pre-existing M2 gate never sees a numerical or allocation change.
     * The default fp16 path does not use this field at all; its K/V live in
     * sg_gpu.kv (a separate-buffer sg_kv object, indexed by layer). */
    void *kv;
    void *conv_buf;   /* [conv_kernel, conv_dim]: output row then carried tail */
    void *ssm;        /* [n_v_heads, head_v_dim, head_k_dim] */
} sg_gpu_layer;

struct sg_gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    /* Sized by KI_COUNT rather than by src/metal.m's SG_N_KERNELS (which is
     * sizeof SG_KERNELS / sizeof SG_KERNELS[0] and so cannot be evaluated
     * without the table): the same number, because metal.m's
     * _Static_assert(SG_N_KERNELS == KI_COUNT) is what makes the table and the
     * enum agree, and it is still there. Layout is unchanged. */
    id<MTLComputePipelineState> pipes[KI_COUNT];
    /* k_attn_decode's per-head score row. Owned and grown on demand rather
     * than living in threadgroup memory, which caps out at 32 KB and would
     * put a ceiling of ~8k tokens on the context length.
     *
     * ONE ALLOCATION, SHARED BY EVERY USER, ALWAYS BOUND AT OFFSET 0:
     * k_attn_decode, k_attn_decode_f16, k_attn_prefill and sg_gpu_forward's
     * encoder all point at these same bytes. sg_scratch_ensure only ever GROWS
     * it; it never partitions it and hands nobody a private range. That is
     * safe today because each of those is either its own commit-and-wait or
     * the only scratch user in the command buffer it is encoded into, and it
     * is a real constraint on anything that batches two DIFFERENT scratch
     * users into ONE open command buffer: their rows would be the same bytes.
     * The split-K partial (P2.2) therefore does NOT use this buffer, see
     * splitk_scratch below. */
    id<MTLBuffer> scratch;
    uint64_t scratch_bytes;

    /* k_attn_decode_splitk_partial's per-(head, split) score row (P2.2 review
     * finding 1). A SEPARATE allocation from `scratch` above, on purpose: the
     * split-K pair is meant to be dispatched from the batched decode encoder
     * next to kernels that use `scratch`, and sharing one grown-not-
     * partitioned buffer between two users in one open command buffer is the
     * kind of hazard that has to be structural rather than a comment somebody
     * remembers. Costs one extra device allocation, sized
     * n_heads * n_splits * ceil(seq/n_splits) floats, which is the same order
     * as `scratch`'s own n_heads * seq. */
    id<MTLBuffer> splitk_scratch;
    uint64_t splitk_scratch_bytes;

    /* --- the loaded model (sg_gpu_load_model) --- */
    const sg_model *model;
    sg_cfg cfg;
    uint32_t key_dim, value_dim, conv_dim, q_width, kv_width, attn_width;
    int mat_kernel;            /* KI_MATVEC_BF16, KI_MATVEC_F32 or KI_MATVEC_Q8 */
    sg_gpu_layer *ls;          /* cfg.n_layers */
    void *lm_head;             /* wrapped [vocab, hidden] */
    void *out_norm;            /* owned [hidden] f32, shift applied */

    /* --- the decode state (sg_gpu_state_new) --- */
    uint32_t max_ctx, used;
    bool have_state;
    void *b_x, *b_h, *b_r;             /* [hidden] */
    void *b_qg;                        /* [n_heads, 2*head_dim] */
    void *b_ctx;                       /* [n_heads, head_dim] */
    void *b_ffg, *b_ffu;               /* [ffn_hidden] */
    void *b_qkv;                       /* [conv_dim] */
    void *b_ab, *b_gates;              /* [2*n_v_heads] */
    void *b_y;                         /* [value_dim] */
    void *b_cs;                        /* [rope_dim]: cos half then sin half */
    void *b_logits;                    /* [vocab] */
    float *h_x, *h_cs, *h_logits;      /* host views of the three above */

    /* --- M5.2: fp16 KV cache --- */
    /* SURGE_KV_DTYPE at the last sg_gpu_state_new call: SG_T_F16 (default) or
     * SG_T_F32. Decides which of {g->kv, per-layer L->kv} is live. */
    sg_tensor_type kv_dtype;
    /* Full-attention K/V, SEPARATE per-layer buffers, allocated through
     * sg_kv (Task M5.1) ONLY on the fp16 path; NULL on the f32 path, where
     * L->kv (the pre-M5.2 combined buffer) is used instead. */
    sg_kv *kv;
    /* [kv_width] f32 landing spots for this token's freshly computed K and V,
     * fp16 path only: q/k-norm and RoPE run here in f32 exactly as they did
     * on the old combined buffer, and k_kv_store_f16 then casts the result
     * into g->kv's per-layer half buffers. NULL on the f32 path (matvec
     * writes straight into L->kv there, as before). */
    void *b_k32, *b_v32;

    /* --- Task P2.3: split-K decode attention ---
     * Read from SURGE_ATTN_SPLITK at the last sg_gpu_state_new call. true (the
     * default) makes enc_attn's fp16 branch dispatch the
     * k_attn_decode_splitk_partial + _combine PAIR instead of the incumbent
     * single-threadgroup-per-head k_attn_decode_f16, once the sequence is long
     * enough (see splitk_n_splits). false pins the incumbent, which is what the
     * A/B measurement needs and why the incumbent stays reachable rather than
     * being deleted. Always false on the f32 KV path: the split-K kernels read
     * half-typed separate K and V buffers, which only the fp16 cache has. */
    bool attn_splitk;
    /* --- Task P2.4: GQA-shared split-K threadgroups ---
     * Read from SURGE_ATTN_SPLITK_GQA at the last sg_gpu_state_new call, and
     * only consulted when attn_splitk above is already true. true makes
     * enc_attn_splitk dispatch k_attn_decode_splitk_partial_gqa (one
     * threadgroup per GQA GROUP, each K/V element read once for all the query
     * heads that share it) instead of the per-head
     * k_attn_decode_splitk_partial; the two write the same bytes, so this
     * changes memory traffic and grid shape, never the answer. DEFAULT TRUE
     * SINCE TASK P4.0 (2026-08-18): this is the shipped decode partial, and
     * SURGE_ATTN_SPLITK_GQA=0 is what pins the per-head one. See
     * splitk_gqa_use for the shape and occupancy conditions that still apply. */
    bool attn_splitk_gqa;
    /* --- Task P2.8: online (streaming) softmax in the GQA partial ---
     * Read from SURGE_ATTN_SPLITK_ONLINE at the last sg_gpu_state_new call, and
     * only consulted when attn_splitk above is already true. true makes
     * enc_attn_splitk dispatch k_attn_decode_splitk_partial_gqa_online, which
     * keeps a running (m, s, acc) per head instead of writing a score row into
     * splitk_scratch and walking it three more times.
     *
     * IT IS A SEPARATE SWITCH FROM attn_splitk_gqa, not a modifier of it: the
     * online kernel IS a GQA-shared kernel (same grid, same group-size band,
     * same threadgroup floor, same split policy), so requiring both switches
     * would only mean two ways to spell one choice. When both are set the online
     * kernel wins, because it is the more specific request; see
     * splitk_online_use.
     *
     * UNLIKE THE OTHER TWO KERNEL SWITCHES, THIS ONE CHANGES THE ANSWER'S LAST
     * BITS. Streaming reorders the exponential sums, so the online arm is NOT
     * byte-identical to the four-pass arm at a fixed n_splits (only `m` is
     * exact, and the whole triple happens to be exact when a split fits one
     * SG_TG-wide tile). The bar is accuracy against sg_ref_attn_decode_splitk
     * plus determinism plus byte-exact greedy tokens, the P2.2 standard, not the
     * memcmp P2.4 could claim.
     *
     * DEFAULT FALSE, and for the same evidence reason P2.4's switch is: written
     * and compiled with the GPU held by a 256K benchmark, so no timing and no
     * accuracy number for it had been observed when it landed. */
    bool attn_splitk_online;
    /* --- Task P2.6: the GQA split policy's saturation cap, overridable ---
     * Read from SURGE_SPLITK_GQA_CAP at the last sg_gpu_state_new call. 0 means
     * "never set on this state", which splitk_gqa_cap_of resolves to the
     * measured default SG_SPLITK_GQA_N_SPLITS_CAP == 256; it is deliberately
     * NOT a cap of 0, because a literal 0 reaching splitk_gqa_n_splits would
     * silently clamp every step down to SG_SPLITK_MIN.
     *
     * IT EXISTS FOR THE P2.6 GATE, not for users. The cap is what makes
     * splitk_gqa_n_splits diverge from splitk_n_splits, and at the shipped 256
     * that divergence only starts at seq 65792 (SG_TG * 257), a depth no test
     * can reach in a `make check`. Lowering the cap moves the SAME divergence
     * mechanism down to SG_TG * (cap + 1), so the greedy-token gate can run in
     * seconds instead of hours. Its second use is retuning the cap without a
     * recompile if a future GPU moves the saturation point.
     *
     * Validated at parse time and REJECTED (sg_gpu_state_new returns an error)
     * outside [SG_SPLITK_MIN, SG_SPLITK_MAX], because a silently ignored value
     * here would make the gate that depends on it pass vacuously. */
    uint32_t splitk_gqa_cap;
    /* WHICH PARTIAL enc_attn_splitk ACTUALLY ENCODED, counted per dispatch
     * (P2.4 fix round 1, review finding I1). Pure diagnostics: nothing reads
     * them inside a kernel, no buffer size or dispatch shape depends on them,
     * so they cannot change computed output.
     *
     * THEY EXIST BECAUSE THE END-TO-END A/B IS VACUOUS WITHOUT THEM. The two
     * partials are contracted to produce the SAME bytes, so
     * "SURGE_ATTN_SPLITK_GQA=0 and =1 give byte-identical logits" is exactly
     * what you also see when the GQA kernel is never selected at all (a
     * narrowed band in splitk_gqa_use, an unset attn_splitk_gqa, a lost
     * dispatch). A gate that cannot tell those two apart would stay green while
     * the whole point of the task silently disappeared. sg_gpu_splitk_dispatch_
     * counts exposes them so the gate can assert WHICH kernel ran, the way
     * P2.3's wiring subtest asserted the threshold. Reset by sg_gpu_state_new
     * and by gpu_free_state; they count ENCODER dispatches only, never the
     * one-shot entry points.
     *
     * P2.7: with the switch ON, BOTH counters are expected to be nonzero over a
     * run that starts short, because the threadgroup floor in splitk_gqa_use
     * declines the GQA kernel until the grid is big enough. The split between
     * them is exactly where that floor sits, which is what makes them a
     * two-sided gate on the floor itself and not only on the selection. */
    uint64_t splitk_partial_dispatches;
    uint64_t splitk_gqa_dispatches;
    /* P2.8's third arm, counted separately rather than folded into
     * splitk_gqa_dispatches. Two reasons: the P2.4/P2.6/P2.7 gates assert EXACT
     * values of the two counters above and must keep their meaning ("the
     * four-pass GQA partial"), and the online arm's own A/B needs to distinguish
     * "the online kernel ran" from "some GQA kernel ran", which is the same
     * vacuity argument that put the first two counters here. Read through
     * sg_gpu_splitk_online_dispatches. */
    uint64_t splitk_online_dispatches;
    /* The largest n_splits any decode step at this max_ctx can ask for, i.e.
     * splitk_n_splits(max_ctx). n_splits is nondecreasing in seq and seq is
     * capped at max_ctx, so the three buffers below and g->splitk_scratch can
     * be sized ONCE here and never grown from inside an open command buffer. */
    uint32_t splitk_max_splits;
    /* The per-(head, split) partial triples: m, s are [n_heads, n_splits] and
     * acc is [n_heads, n_splits, head_dim], all f32, allocated for
     * splitk_max_splits. One set is shared by every full-attention layer in a
     * forward, the same way g->b_ctx and g->scratch already are: dispatches in
     * one encoder run in encode order with an implicit barrier between them
     * (MTLDispatchTypeSerial), so a layer's combine has consumed these before
     * the next layer's partial overwrites them. NULL unless the fp16 path
     * allocated them. */
    void *b_sk_m, *b_sk_s, *b_sk_acc;

    /* --- Task B8: prefill duty-cycle (yields the GPU between chunks) ---
     * Set by sg_gpu_set_prefill_rest; both 0 (the calloc default in
     * sg_gpu_init) means DISABLED, i.e. sg_gpu_prefill never sleeps and its
     * OUTPUT (gen_ids, logits, KV/decode state, g->used) is byte-identical to
     * before this task -- the chunk loop still takes two extra clock_gettime
     * reads per chunk either way, but they feed only rest accounting, never
     * anything output-affecting. prefill_rest_total_ms is reset to 0 at the
     * start of every sg_gpu_prefill call (whether or not the feature is
     * enabled, and even one that fails validation) and accumulates the wall
     * time actually slept during that call; read it back with
     * sg_gpu_prefill_rest_ms after the call returns. */
    uint32_t prefill_work_budget_ms;
    uint32_t prefill_rest_ms;
    uint64_t prefill_rest_total_ms;

    /* --- prefill command-buffer segmentation -------------------------
     * Set by sg_gpu_set_prefill_max_burst; 0 (the calloc default) means
     * DISABLED, i.e. one command buffer per chunk exactly as before.
     *
     * The duty-cycle rest above can only yield the GPU BETWEEN chunks, so it
     * cannot help once a single chunk's command buffer runs longer than the
     * macOS userspace watchdog allows WindowServer to go without rendering
     * (80 s; measured ~130 s for one 256-token chunk at 220k context on
     * 2026-08-14, which killed WindowServer twice). Splitting a chunk's layer
     * sweep across several command buffers bounds how long the GPU is held by
     * any one submission.
     *
     * This is safe in a way that changing `chunk` would not be: command buffer
     * boundaries carry no state. The same kernels are dispatched with the same
     * arguments in the same order, and buffers committed in sequence on one
     * queue execute in order, so segmentation cannot change a single output
     * value. That is why the segment count may also adapt mid-run.
     *
     * prefill_seg_layers is the live, adapting value; it is reset from
     * n_layers at the start of every sg_gpu_prefill call. */
    uint32_t prefill_max_burst_ms;
    uint32_t prefill_seg_layers;
    /* Command buffers submitted by the most recent sg_gpu_prefill call. Equals
     * the chunk count when segmentation is off. Exposed so a parity test can
     * prove segmentation actually engaged rather than passing vacuously. */
    uint64_t prefill_segments;
};

static inline uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* Zero-padded params array as a call argument. Every kernel reads at most
 * seven of the eight slots; the rest must still be defined, because
 * setBytes: uploads all 32 bytes. */
#define PARAMS(...) ((const uint32_t[8]){ __VA_ARGS__ })

static inline id<MTLBuffer> bufof(void *h) { return h ? ((sg_gpu_buf *)h)->buf : nil; }
static inline uint64_t offof(void *h) { return h ? ((sg_gpu_buf *)h)->offset : 0; }

typedef struct {
    sg_gpu *g;
    id<MTLComputeCommandEncoder> enc;
} sg_enc;

/* --------------------------------------------------------------------
 * The helpers that cross the seam, DEFINED IN src/metal.m
 * --------------------------------------------------------------------
 *
 * Each of these was `static` in src/metal.m and is still DEFINED there,
 * exactly once; the `static` was dropped so src/metal_prefill.m can call it.
 * Listed in the order they appear in src/metal.m. */

/* Format an error whose text quotes a runtime detail into metal.m's single
 * static message buffer. Single-threaded contract, as ever. */
__attribute__((format(printf, 1, 2)))
sg_err sg_gpu_errf(const char *fmt, ...);

/* Grow g->scratch to at least nbytes (the shared per-head score row). */
sg_err sg_scratch_ensure(sg_gpu *g, uint64_t nbytes);

/* Threadgroup width for an elementwise dispatch of `elems` threads. */
NSUInteger sg_gpu_elem_width(sg_gpu *g, int ki, uint64_t elems);

/* The tiled-GEMM kernel index for a weight tensor of the given dtype. */
int sg_gemm_kernel_for(sg_tensor_type t);

/* One embedding row widened into f32, bit-identical to ref.c's wrow. */
void sg_gpu_embed_row(const void *w, sg_tensor_type t, uint64_t row,
                      uint32_t cols, float *out);

/* Allocate n floats of shared-storage device memory and hand back both the
 * handle and, optionally, the host pointer. */
sg_err sg_gpu_alloc_f32(sg_gpu *g, uint64_t n, void **buf, float **host);

/* One dispatch into an already-open encoder; see the definition's comment for
 * what `ao`/`bo`/`oo` and `aux` mean. */
void sg_enc_op(sg_enc *E, int ki, void *a, uint64_t ao, void *b, uint64_t bo,
               void *o, uint64_t oo, id<MTLBuffer> aux, uint64_t auxoff,
               const uint32_t *p);

/* Cast n f32 elements into the fp16 KV cache at a destination offset. */
void sg_enc_kv_store(sg_enc *E, void *src, void *dst, uint64_t dst_off, uint32_t n);

/* One tiled-GEMM dispatch, Y[n, m] = X[n, k] @ W[m, k]^T. */
void sg_enc_matmul(sg_enc *E, int ki, void *x, uint64_t xoff, void *w,
                   void *y, uint64_t yoff, uint32_t nn, uint32_t mm, uint32_t kk);

/* --------------------------------------------------------------------
 * The helpers that cross the seam, DEFINED IN src/metal_validate.m
 * --------------------------------------------------------------------
 *
 * Task R3's direction of travel is the OPPOSITE of R2's. Nothing in
 * src/metal_validate.m is called from there; all six call sites of the three
 * below stayed in src/metal.m, so it is the MOVED code that had to lose
 * `static`, not the code it calls. All three are per-DISPATCH (once per
 * sg_gpu_run_op call, once per encoded dispatch for sg_gpu_grid), never per
 * element, so the cost is one call each. */

/* Buffer-size preconditions per kernel name; see the definition. One caller,
 * sg_gpu_run_op. */
sg_err sg_check_sizes(const char *kernel, const sg_gpu_buf *a, const sg_gpu_buf *b,
                      const sg_gpu_buf *o, const uint32_t *p);

/* Per-kernel preconditions that are not about buffer sizes. Three callers, all
 * one-shot entry points in src/metal.m. */
sg_err sg_check_params(const char *kernel, const uint32_t *p);

/* Grid geometry from the kernel's SG_K_* kind and its params. Two callers,
 * sg_gpu_run_op and sg_enc_op. */
void sg_gpu_grid(int kind, const uint32_t *p, uint64_t *groups, uint64_t *elems);

#endif /* SURGE_METAL_INTERNAL_H */
