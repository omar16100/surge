/* metal.m - the host side of the Task 9 kernels: device/queue/metallib setup,
 * no-copy buffer wrapping, and a one-shot per-op dispatch.
 *
 * The public contract (buffer layouts, params[] per kernel, aliasing rules)
 * lives in surge.h next to sg_gpu_run_op and is not repeated here.
 *
 * Objective-C, MANUAL retain/release (no ARC): the file is compiled in the
 * same clang invocation as the C sources, and -fobjc-arc there would either
 * be rejected for the C translation units or need a second, differently
 * flagged compile. There are exactly five owned objects (device, queue,
 * library, one pipeline per kernel, and the scratch buffer) plus one per
 * sg_gpu_buf, each created with a +1 method and released in the matching
 * free; everything else is autoreleased inside an @autoreleasepool.
 *
 * THREADING: an sg_gpu is single-threaded by design. sg_gpu_run_op commits
 * one command buffer and waits for it, the scratch buffer is grown in place,
 * and the error string below is a single static buffer. Task 10 batches a
 * whole layer into one command buffer; it does not make this concurrent.
 */
#include "surge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Errors that quote a runtime detail (a Metal error string, a size) need
 * storage; sg_err holds a bare const char *. One static buffer, consistent
 * with the single-threaded contract above. Static messages are returned
 * directly and never touch it. */
static char g_errbuf[512];

__attribute__((format(printf, 1, 2)))
static sg_err gpu_errf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_errbuf, sizeof g_errbuf, fmt, ap);
    va_end(ap);
    return (sg_err){g_errbuf};
}

/* --------------------------------------------------------------------
 * Kernel table
 * --------------------------------------------------------------------
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
     * sg_gpu_run_op (not by gpu_grid, which only returns a 1D count) because
     * this is the only kind whose group count needs two dimensions. */
    SG_K_TILES2D,
    /* M5.4: k_rope_chunk, one thread per (token, head, element) of the chunk,
     * params[0]*params[2]*params[4] = head_dim*heads*n_tok threads, non-uniform
     * threadgroups. An elementwise kernel with no reduction, dispatched through
     * gpu_grid's *elems path like SG_K_ELEM but with a 3-factor count. */
    SG_K_ROPE_CHUNK,
    /* P2.2: the split-K decode-attention partial kernel's grid, a 2D
     * params[6] x params[0] (split, query head) block of threadgroups of
     * SG_TG. The SECOND kind after SG_K_TILES2D that needs two group
     * dimensions and for the same reason SG_K_ATTN cannot be reused: that
     * class carries exactly one *groups count (see gpu_grid below), so it can
     * express "one threadgroup per head" and nothing wider. Computed by hand
     * in sg_gpu_run_attn_splitk_partial, which is the only path that reaches
     * the kernel at all (it takes six device buffers, so sg_gpu_run_op's
     * (a, b, out) shape cannot).
     *
     * P2.4's k_attn_decode_splitk_partial_gqa shares this kind with a params[6]
     * x params[1] (split, KV head) grid: one threadgroup per GQA GROUP rather
     * than per query head. Same class because the dispatchers compute both
     * extents by hand anyway; the `kind` column is only read by sg_gpu_init's
     * threadgroup-width check. */
    SG_K_HEADS2D
};

#define SG_TG 256u

/* The largest GQA group (n_heads / n_kv_heads) k_attn_decode_splitk_partial_gqa
 * keeps in registers, mirroring kernels.metal's SG_SPLITK_GQA_MAX constant of
 * the same value (which static_asserts on it); keep both in sync. Past this
 * the kernel still answers correctly but one head at a time, with no reuse, so
 * splitk_gqa_use below routes those shapes to the per-head partial instead. */
#define SG_SPLITK_GQA_MAX 8u

/* The tiled-GEMM output tile shape (Task M5.3), mirroring kernels.metal's
 * SG_GEMM_TM / SG_GEMM_TN constants of the same value; keep both in sync.
 * TM * TN must equal SG_TG, since each threadgroup dispatches SG_TG threads
 * and each thread owns exactly one tile element (see k_matmul_bf16 et al.). */
#define SG_GEMM_TM 16u
#define SG_GEMM_TN 16u

typedef struct {
    const char *name;
    int kind;
} sg_kernel_desc;

/* sg_gpu.pipes is indexed by position here, so the enum and the table must
 * stay in lockstep; the static assert below is what enforces that. The first
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

static const sg_kernel_desc SG_KERNELS[] = {
    { "k_rmsnorm",             SG_K_TG1     },
    { "k_rmsnorm_gated",       SG_K_GATED   },
    { "k_rope",                SG_K_ELEM    },
    { "k_matvec_bf16",         SG_K_ROWS    },
    { "k_matvec_f32",          SG_K_ROWS    },
    { "k_matvec_q8",           SG_K_ROWS    },
    { "k_softmax",             SG_K_TG1     },
    { "k_swiglu",              SG_K_ELEM    },
    { "k_silu",                SG_K_ELEM    },
    { "k_gate_sigmoid",        SG_K_ELEM    },
    { "k_attn_decode",         SG_K_ATTN    },
    { "k_conv1d_step",         SG_K_ELEM    },
    { "k_delta_step",          SG_K_TG1     },
    { "k_rmsnorm_heads",       SG_K_GATED   },
    { "k_rope_heads",          SG_K_ELEM02  },
    { "k_gate_sigmoid_strided",SG_K_ELEM01  },
    { "k_scale",               SG_K_ELEM    },
    { "k_add",                 SG_K_ELEM    },
    { "k_delta_gates",         SG_K_ELEM    },
    { "k_delta_multi",         SG_K_GROUPS2 },
    /* M5.2: fp16 KV. k_kv_store_f16 is a plain elementwise cast (dispatched
     * through the generic sg_gpu_run_op path below); k_attn_decode_f16 takes
     * THREE device buffer inputs (Q, separate K, separate V) where every
     * other kernel here takes at most two, so it is dispatched by hand (see
     * enc_attn_f16 and sg_gpu_run_attn_decode_f16) and is listed here only so
     * sg_gpu_init builds its pipeline and checks its threadgroup width. */
    { "k_kv_store_f16",        SG_K_ELEM    },
    { "k_attn_decode_f16",     SG_K_ATTN    },
    /* M5.3: tiled GEMM (a whole chunk of N tokens through one weight matrix
     * in a single dispatch), used by the later M5 prefill tasks; the decode
     * path above is untouched and never reaches these. */
    { "k_matmul_bf16",         SG_K_TILES2D },
    { "k_matmul_f32",          SG_K_TILES2D },
    { "k_matmul_q8",           SG_K_TILES2D },
    /* M5.4: full-attention tiled prefill. k_rope_chunk is a plain elementwise
     * kernel dispatched through the generic sg_gpu_run_op path (a=x, b=cs,
     * out); k_attn_prefill takes THREE device buffer inputs (Q, separate K,
     * separate V) like k_attn_decode_f16, so it is dispatched by hand (see
     * sg_gpu_run_attn_prefill and enc_attn_prefill) and is listed here only so
     * sg_gpu_init builds its pipeline and checks its threadgroup width. */
    { "k_rope_chunk",          SG_K_ROPE_CHUNK },
    { "k_attn_prefill",        SG_K_ATTN       },
    /* M5.5: gated-DeltaNet tiled prefill. All four thread the recurrent state
     * (conv tail + S) across chunks and take an extra device buffer (the state
     * carrier or the shared weight) beyond sg_gpu_run_op's (a, b, out) shape,
     * so each is dispatched by hand -- through a dedicated public one-shot for
     * the per-op tests, and inside enc_gdn_prefill for the batched path -- and
     * is listed here only so sg_gpu_init builds its pipeline and (for the two
     * reduction/threadgroup kernels) checks its threadgroup width. The `kind`
     * column is the closest existing grid class and is used ONLY by that init
     * width check; the actual grids are computed in the dispatchers.
     * k_conv1d_chunk / k_delta_gates_chunk are one-thread-per-work-item with no
     * reduction (SG_K_ELEM, exempt from the SG_TG width assertion); k_delta_chunk
     * / k_rmsnorm_gated_chunk run SG_TG-wide threadgroups. */
    { "k_conv1d_chunk",        SG_K_ELEM    },
    { "k_delta_gates_chunk",   SG_K_ELEM    },
    { "k_delta_chunk",         SG_K_GROUPS2 },
    { "k_rmsnorm_gated_chunk", SG_K_GATED   },
    /* P2.2: split-K decode attention. The partial kernel takes SIX device
     * buffers (q, k, v, m, s, acc) plus a score scratch and the combine FOUR
     * (m, s, acc, out), so neither fits sg_gpu_run_op's (a, b, out) shape;
     * both are dispatched by hand from their own one-shot entry points
     * (sg_gpu_run_attn_splitk_partial / _combine) and are listed here only so
     * sg_gpu_init builds their pipelines and checks their threadgroup width.
     * The partial's SG_K_HEADS2D is the real shape of its grid; the combine's
     * SG_K_ROWS is literally right too (params[0] = n_heads threadgroups),
     * though neither `kind` is read outside that init width check. */
    { "k_attn_decode_splitk_partial", SG_K_HEADS2D },
    { "k_attn_decode_splitk_combine", SG_K_ROWS    },
    /* P2.4: the GQA-shared partial. Same eight bindings, same params array and
     * same output layout as the per-head partial above, so it is a drop-in
     * alternative for it; only the grid's y extent (KV heads, not query heads)
     * and the traffic differ. Dispatched by hand from
     * sg_gpu_run_attn_splitk_partial_gqa and enc_attn_splitk. */
    { "k_attn_decode_splitk_partial_gqa", SG_K_HEADS2D },
    /* P2.8: the ONLINE-SOFTMAX GQA partial. Same grid class, same params array
     * and same output layout as the two above, but SEVEN bindings instead of
     * eight: it keeps a running (m, s, acc) per head in registers and
     * threadgroup memory instead of writing a score row, so it has no `scores`
     * argument and never touches g->splitk_scratch. Dispatched by hand from
     * sg_gpu_run_attn_splitk_partial_gqa_online and enc_attn_splitk. */
    { "k_attn_decode_splitk_partial_gqa_online", SG_K_HEADS2D },
};
#define SG_N_KERNELS ((int)(sizeof SG_KERNELS / sizeof SG_KERNELS[0]))
_Static_assert(SG_N_KERNELS == KI_COUNT, "SG_KERNELS and the KI_ enum disagree");

/* A wrapped or allocated buffer. The offset is what makes sg_gpu_wrap
 * possible at all: Metal demands a page-aligned base and a tensor inside a
 * checkpoint mmap never is one, so the handle remembers how far into the
 * page its data starts and every bind adds it. */
typedef struct {
    id<MTLBuffer> buf;
    uint64_t offset;
    uint64_t nbytes;
} sg_gpu_buf;

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
    id<MTLComputePipelineState> pipes[SG_N_KERNELS];
    /* k_attn_decode's per-head score row. Owned and grown on demand rather
     * than living in threadgroup memory, which caps out at 32 KB and would
     * put a ceiling of ~8k tokens on the context length.
     *
     * ONE ALLOCATION, SHARED BY EVERY USER, ALWAYS BOUND AT OFFSET 0:
     * k_attn_decode, k_attn_decode_f16, k_attn_prefill and sg_gpu_forward's
     * encoder all point at these same bytes. scratch_ensure only ever GROWS
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

/* --------------------------------------------------------------------
 * Init
 * -------------------------------------------------------------------- */

/* Candidate metallib locations, in priority order. The env var is first so a
 * test or a packaged binary can point at a specific build; SG_METALLIB_PATH
 * is baked in by the Makefile; the last two cover "run from the repo root"
 * and "run the binary from anywhere". */
static NSString *gpu_metallib_path(void) {
    NSFileManager *fm = [NSFileManager defaultManager];

    const char *env = getenv("SURGE_METALLIB");
    if (env && *env) {
        NSString *p = [NSString stringWithUTF8String:env];
        if (p && [fm fileExistsAtPath:p]) return p;
    }
#ifdef SG_METALLIB_PATH
    {
        NSString *p = @SG_METALLIB_PATH;
        if ([fm fileExistsAtPath:p]) return p;
    }
#endif
    if ([fm fileExistsAtPath:@"src/kernels.metallib"]) return @"src/kernels.metallib";

    char exe[4096];
    uint32_t sz = sizeof exe;
    if (_NSGetExecutablePath(exe, &sz) == 0) {
        NSString *dir = [[NSString stringWithUTF8String:exe] stringByDeletingLastPathComponent];
        NSString *p = [dir stringByAppendingPathComponent:@"kernels.metallib"];
        if ([fm fileExistsAtPath:p]) return p;
        p = [[dir stringByAppendingPathComponent:@"../src"]
                stringByAppendingPathComponent:@"kernels.metallib"];
        if ([fm fileExistsAtPath:p]) return p;
    }
    return nil;
}

/* Defined with the rest of the decode path below; declared here because
 * sg_gpu_free has to release the model and state handles too. */
static void gpu_unload(sg_gpu *g);

void sg_gpu_free(sg_gpu *g) {
    if (!g) return;
    gpu_unload(g);
    for (int i = 0; i < SG_N_KERNELS; i++) {
        [g->pipes[i] release];
        g->pipes[i] = nil;
    }
    [g->scratch release];
    [g->splitk_scratch release];
    [g->lib release];
    [g->queue release];
    [g->dev release];
    g->scratch = nil;
    g->splitk_scratch = nil;
    g->lib = nil;
    g->queue = nil;
    g->dev = nil;
    free(g);
}

sg_err sg_gpu_init(sg_gpu **out) {
    if (!out) return (sg_err){"gpu: sg_gpu_init needs an out pointer"};
    *out = NULL;

    sg_gpu *g = calloc(1, sizeof *g);
    if (!g) return (sg_err){"gpu: out of memory"};

    @autoreleasepool {
        g->dev = MTLCreateSystemDefaultDevice();   /* +1 */
        if (!g->dev) {
            free(g);
            return (sg_err){"gpu: no Metal device on this machine"};
        }
        g->queue = [g->dev newCommandQueue];
        if (!g->queue) {
            sg_gpu_free(g);
            return (sg_err){"gpu: could not create a command queue"};
        }

        NSString *path = gpu_metallib_path();
        if (!path) {
            sg_gpu_free(g);
            return (sg_err){"gpu: kernels.metallib not found (set SURGE_METALLIB)"};
        }
        NSError *err = nil;
        /* newLibraryWithURL: is already +1, so no retain here. */
        g->lib = [g->dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
        if (!g->lib) {
            sg_err e = gpu_errf("gpu: cannot load %s: %s", [path UTF8String],
                                err ? [[err localizedDescription] UTF8String] : "unknown error");
            sg_gpu_free(g);
            return e;
        }

        /* Build every pipeline up front: a metallib that is stale (missing a
         * kernel this build expects) must fail at init, not on the one
         * dispatch that happens to reach the missing kernel. */
        for (int i = 0; i < SG_N_KERNELS; i++) {
            NSString *nm = [NSString stringWithUTF8String:SG_KERNELS[i].name];
            id<MTLFunction> fn = [g->lib newFunctionWithName:nm];
            if (!fn) {
                sg_err e = gpu_errf("gpu: kernel '%s' missing from the metallib",
                                    SG_KERNELS[i].name);
                sg_gpu_free(g);
                return e;
            }
            g->pipes[i] = [g->dev newComputePipelineStateWithFunction:fn error:&err];
            [fn release];
            if (!g->pipes[i]) {
                sg_err e = gpu_errf("gpu: pipeline for '%s' failed: %s", SG_KERNELS[i].name,
                                    err ? [[err localizedDescription] UTF8String] : "unknown error");
                sg_gpu_free(g);
                return e;
            }
            /* The reduction kernels fold over the compile-time SG_TG; a
             * device that cannot run that many threads per threadgroup would
             * silently lose the lanes above the limit. No Apple GPU is in
             * that position (the limit is 1024), but assert rather than
             * assume. */
            if (SG_KERNELS[i].kind != SG_K_ELEM &&
                [g->pipes[i] maxTotalThreadsPerThreadgroup] < SG_TG) {
                sg_err e = gpu_errf("gpu: '%s' allows only %lu threads per threadgroup, need %u",
                                    SG_KERNELS[i].name,
                                    (unsigned long)[g->pipes[i] maxTotalThreadsPerThreadgroup],
                                    SG_TG);
                sg_gpu_free(g);
                return e;
            }
        }
    }

    /* Register this gpu's alloc/free/host as sg_kv's backend, so sg_kv_new
     * (Task M5.1/M5.2) can allocate real GPU buffers rather than the NULL
     * stub every sg_kv call would otherwise hit. Must happen before the
     * first sg_kv_new call, i.e. before sg_gpu_state_new. This is a global
     * registration (sg_kv.c carries no gpu handle of its own), so the last
     * sg_gpu_init to run wins; the codebase runs one sg_gpu at a time. */
    sg_kv_set_backend(sg_gpu_alloc, sg_gpu_buf_free, sg_gpu_buf_host);

    *out = g;
    return SG_OK;
}

/* --------------------------------------------------------------------
 * Buffers
 * -------------------------------------------------------------------- */

static uint64_t gpu_page_size(void) {
    long ps = sysconf(_SC_PAGESIZE);
    return (ps > 0) ? (uint64_t)ps : 4096u;
}

sg_err sg_gpu_wrap(sg_gpu *g, const void *ptr, uint64_t nbytes, void **buf_out) {
    if (!g || !ptr || !buf_out) return (sg_err){"gpu: sg_gpu_wrap got a NULL argument"};
    *buf_out = NULL;
    if (nbytes == 0) return (sg_err){"gpu: cannot wrap a zero-length region"};

    uint64_t page = gpu_page_size();
    uintptr_t base = (uintptr_t)ptr;

    /* The saved offset is handed to setBuffer:offset:, whose granularity is
     * 4 bytes, and the kernels then cast the bound region to float* / ushort*.
     * A pointer that cannot satisfy that has to be caught HERE: accepting it
     * would produce either a validation failure four call frames away or,
     * worse, misaligned typed loads. (A safetensors tensor at an odd byte
     * offset is a real case; see sg_st_read_f32's note. Those get copied, not
     * wrapped.) */
    if (base % 4 != 0) {
        return gpu_errf("gpu: sg_gpu_wrap needs a 4-byte aligned pointer (offset %llu)",
                        (unsigned long long)(base % 4));
    }

    uintptr_t aligned = base & ~(uintptr_t)(page - 1);
    uint64_t offset = (uint64_t)(base - aligned);

    /* Round the LENGTH up too: Metal maps whole pages, so a buffer that ends
     * mid-page still covers the rest of that page. That is safe for a file
     * mapping (the tail of the last page reads as zero) and for any
     * page-granular allocation, which is what surge.h asks of the caller.
     * Both steps are checked for wraparound: nbytes is caller-supplied and a
     * length that wrapped to something small would sail past the device-limit
     * test below and then hand the GPU a buffer shorter than the region the
     * kernels index. */
    if (nbytes > UINT64_MAX - offset) {
        return (sg_err){"gpu: wrapped length overflows when the page offset is added"};
    }
    uint64_t len = offset + nbytes;
    if (len > UINT64_MAX - (page - 1)) {
        return (sg_err){"gpu: wrapped length overflows when rounded up to a page"};
    }
    len = (len + page - 1) & ~(page - 1);

    if (len > [g->dev maxBufferLength]) {
        return gpu_errf("gpu: %llu bytes exceeds the device limit of %llu",
                        (unsigned long long)len,
                        (unsigned long long)[g->dev maxBufferLength]);
    }

    sg_gpu_buf *b = calloc(1, sizeof *b);
    if (!b) return (sg_err){"gpu: out of memory"};

    @autoreleasepool {
        /* The const cast is deliberate: Metal has no read-only buffer type.
         * Wrapped memory is only ever bound to `a`/`b` inputs, which no
         * kernel writes (the two in-place kernels take allocated buffers). */
        b->buf = [g->dev newBufferWithBytesNoCopy:(void *)aligned
                                           length:(NSUInteger)len
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
        if (!b->buf) {
            free(b);
            return (sg_err){"gpu: newBufferWithBytesNoCopy failed (is the base mapped?)"};
        }
    }
    b->offset = offset;
    b->nbytes = nbytes;
    *buf_out = b;
    return SG_OK;
}

sg_err sg_gpu_alloc(sg_gpu *g, uint64_t nbytes, void **buf_out, void **host_out) {
    if (!g || !buf_out) return (sg_err){"gpu: sg_gpu_alloc got a NULL argument"};
    *buf_out = NULL;
    if (host_out) *host_out = NULL;
    if (nbytes == 0) return (sg_err){"gpu: cannot allocate a zero-length buffer"};
    if (nbytes > [g->dev maxBufferLength]) {
        return gpu_errf("gpu: %llu bytes exceeds the device limit of %llu",
                        (unsigned long long)nbytes,
                        (unsigned long long)[g->dev maxBufferLength]);
    }

    sg_gpu_buf *b = calloc(1, sizeof *b);
    if (!b) return (sg_err){"gpu: out of memory"};

    @autoreleasepool {
        b->buf = [g->dev newBufferWithLength:(NSUInteger)nbytes
                                     options:MTLResourceStorageModeShared];
        if (!b->buf) {
            free(b);
            return (sg_err){"gpu: buffer allocation failed"};
        }
    }
    b->offset = 0;
    b->nbytes = nbytes;
    /* newBufferWithLength does not promise zeroed memory; a test that forgets
     * to fill an input should fail loudly and repeatably, not read whatever
     * the last frame left behind. */
    memset([b->buf contents], 0, (size_t)nbytes);
    if (host_out) *host_out = [b->buf contents];
    *buf_out = b;
    return SG_OK;
}

void sg_gpu_buf_free(void *buf) {
    sg_gpu_buf *b = (sg_gpu_buf *)buf;
    if (!b) return;
    [b->buf release];
    b->buf = nil;
    free(b);
}

void *sg_gpu_buf_host(void *buf) {
    sg_gpu_buf *b = (sg_gpu_buf *)buf;
    if (!b || !b->buf) return NULL;
    return (uint8_t *)[b->buf contents] + b->offset;
}

/* Task B2: current bytes allocated by this Metal device across every live
 * buffer (MTLDevice.currentAllocatedSize) -- a snapshot, not a peak; callers
 * sample it into an sg_mem_tracker (src/bench.c) to build one. 0 for a NULL
 * g or g->dev, the same defensive-NULL convention as sg_gpu_buf_host above. */
uint64_t sg_gpu_current_alloc_bytes(const sg_gpu *g) {
    if (!g || !g->dev) return 0;
    return (uint64_t)[g->dev currentAllocatedSize];
}

/* --------------------------------------------------------------------
 * Dispatch
 * -------------------------------------------------------------------- */

/* Every kernel reads its inputs from indices computed out of params[], so a
 * params/buffer mismatch is an out-of-bounds DEVICE read: not a crash the
 * process can catch, but a GPU fault that takes down the whole context (and
 * on a bad day the display driver). Hence a size precondition per kernel,
 * checked here, before anything is encoded. */
static bool buf_big_enough(const sg_gpu_buf *b, uint64_t need) {
    return b != NULL && b->nbytes >= need;
}

/* Two handles collide when the HOST BYTE RANGES they describe intersect.
 *
 * Comparing MTLBuffer identity plus offsets is not enough, and the case it
 * misses is the one sg_gpu_wrap makes easy: newBufferWithBytesNoCopy called
 * twice on the same mapping returns two DIFFERENT MTLBuffer objects over the
 * same physical bytes, so an identity test reports "no overlap" for two
 * handles that alias completely. Every sg_gpu_buf is shared-storage (both
 * sg_gpu_wrap and sg_gpu_alloc ask for MTLResourceStorageModeShared), so
 * `contents` is always a real host pointer and the comparison is exact; the
 * identity fallback is there only so a future private-storage handle degrades
 * to the old, weaker test rather than to no test. */
static bool bufs_overlap(const sg_gpu_buf *x, const sg_gpu_buf *y) {
    if (!x || !y) return false;
    const uint8_t *xc = (const uint8_t *)[x->buf contents];
    const uint8_t *yc = (const uint8_t *)[y->buf contents];
    if (!xc || !yc) {
        if (x->buf != y->buf) return false;
        return x->offset < y->offset + y->nbytes && y->offset < x->offset + x->nbytes;
    }
    const uint8_t *xb = xc + x->offset, *yb = yc + y->offset;
    return xb < yb + y->nbytes && yb < xb + x->nbytes;
}

/* The byte counts below are products of up to three caller-supplied uint32
 * params, and (uint64)p[0] * p[1] * 4 genuinely wraps for large ones. A
 * wrapped `need` is worse than no check at all: it would be SMALL, so an
 * undersized buffer would sail through and the kernel would index with the
 * original, unwrapped dimensions. Everything therefore goes through these,
 * and an overflow is an error rather than a number. */
static bool mul_ck(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool add_ck(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static sg_err check_sizes(const char *kernel, const sg_gpu_buf *a, const sg_gpu_buf *b,
                          const sg_gpu_buf *o, const uint32_t *p) {
    uint64_t f = 4;  /* sizeof(float) */
    uint64_t need_a = 0, need_b = 0, need_o = 0;
    uint64_t t0 = 0, t1 = 0, t2 = 0;
    bool want_b = true;
    bool ok = true;

    if (strcmp(kernel, "k_rmsnorm") == 0) {
        ok = mul_ck(p[0], f, &need_a);
        need_o = need_a;
        want_b = p[2] != 0;
        need_b = want_b ? need_a : 0;
    } else if (strcmp(kernel, "k_rmsnorm_gated") == 0) {
        ok = mul_ck(p[0], p[1], &t0) && mul_ck(t0, f, &need_a) &&
             add_ck(t0, p[0], &t1) && mul_ck(t1, f, &need_b);
        need_o = need_a;
    } else if (strcmp(kernel, "k_rope") == 0) {
        ok = mul_ck(p[0], f, &need_a) && mul_ck(p[1], f, &need_b);
        need_o = need_a;                  /* b is cos[rope_dim/2] then sin[..] */
    } else if (strcmp(kernel, "k_rope_heads") == 0) {
        /* a = out = [heads, stride] f32 (a head's rotated tail stays inside its
         * own stride, guaranteed by check_params' stride>=head_dim); b is the
         * single-token cos/sin table [rope_dim] f32. p[0]=head_dim p[1]=rope_dim
         * p[2]=heads p[3]=stride. */
        ok = mul_ck(p[2], p[3], &t0) && mul_ck(t0, f, &need_a) && mul_ck(p[1], f, &need_b);
        need_o = need_a;
    } else if (strcmp(kernel, "k_matvec_bf16") == 0) {
        ok = mul_ck(p[0], p[1], &t0) && mul_ck(t0, 2, &need_a) &&
             mul_ck(p[1], f, &need_b) && mul_ck(p[0], f, &need_o);
    } else if (strcmp(kernel, "k_matvec_f32") == 0) {
        ok = mul_ck(p[0], p[1], &t0) && mul_ck(t0, f, &need_a) &&
             mul_ck(p[1], f, &need_b) && mul_ck(p[0], f, &need_o);
    } else if (strcmp(kernel, "k_matvec_q8") == 0) {
        /* Weight is rows * (cols/32) Q8_0 blocks of 34 bytes each; x is cols
         * floats, y is rows floats. cols is a nonzero multiple of 32, which
         * check_params enforces before this runs, so p[1]/32 is the exact
         * block count and does not truncate. */
        ok = mul_ck(p[0], p[1] / 32u, &t0) && mul_ck(t0, 34, &need_a) &&
             mul_ck(p[1], f, &need_b) && mul_ck(p[0], f, &need_o);
    } else if (strcmp(kernel, "k_softmax") == 0 || strcmp(kernel, "k_silu") == 0) {
        ok = mul_ck(p[0], f, &need_a);
        need_o = need_a;
        want_b = false;
    } else if (strcmp(kernel, "k_swiglu") == 0 || strcmp(kernel, "k_gate_sigmoid") == 0) {
        ok = mul_ck(p[0], f, &need_a);
        need_b = need_o = need_a;
    } else if (strcmp(kernel, "k_attn_decode") == 0) {
        ok = mul_ck(p[0], p[4], &t0) && mul_ck(t0, f, &need_a) &&
             mul_ck((uint64_t)p[3] * p[1], p[2], &t1) && add_ck(t1, p[5], &t1) &&
             mul_ck(t1, f, &need_b) &&
             mul_ck((uint64_t)p[0] * p[2], f, &need_o);
        /* p[3]*p[1] and p[0]*p[2] are each two uint32 params, so at most
         * 2^64-2^33; only the third factor can wrap, and that is checked. */
    } else if (strcmp(kernel, "k_conv1d_step") == 0) {
        ok = mul_ck(p[0], f, &need_a) && mul_ck((uint64_t)p[0] * p[1], f, &need_b);
        need_o = need_b;                  /* out[C] then state[(K-1)*C] */
    } else if (strcmp(kernel, "k_delta_step") == 0) {
        ok = mul_ck((uint64_t)p[0] * p[1], f, &need_a) &&
             add_ck(2ull * p[0], p[1], &t0) && mul_ck(t0, f, &need_b) &&
             mul_ck(p[1], f, &need_o);
    } else if (strcmp(kernel, "k_delta_gates") == 0) {
        /* ab = [a(n), b(n)], adt = [ssm_a(n), dt_bias(n)], gates = [beta(n),
         * decay(n)]; each is 2*n floats. p[0]=n. Made reachable through
         * sg_gpu_run_op so the M5.5 per-op test can use k_delta_gates as the
         * bit-identical oracle for k_delta_gates_chunk. */
        ok = mul_ck(2ull * p[0], f, &need_a);
        need_b = need_o = need_a;
    } else if (strcmp(kernel, "k_kv_store_f16") == 0) {
        /* a is p[0] f32 floats in; out is p[0] half (2-byte) elements out. */
        ok = mul_ck(p[0], f, &need_a) && mul_ck(p[0], 2, &need_o);
        want_b = false;
    } else if (strcmp(kernel, "k_matmul_bf16") == 0) {
        /* a = X [N, K] f32, b = W [M, K] bf16, o = Y [N, M] f32. p[0]=N
         * p[1]=M p[2]=K. */
        ok = mul_ck(p[0], p[2], &t0) && mul_ck(t0, f, &need_a) &&
             mul_ck(p[1], p[2], &t1) && mul_ck(t1, 2, &need_b) &&
             mul_ck(p[0], p[1], &t2) && mul_ck(t2, f, &need_o);
    } else if (strcmp(kernel, "k_matmul_f32") == 0) {
        ok = mul_ck(p[0], p[2], &t0) && mul_ck(t0, f, &need_a) &&
             mul_ck(p[1], p[2], &t1) && mul_ck(t1, f, &need_b) &&
             mul_ck(p[0], p[1], &t2) && mul_ck(t2, f, &need_o);
    } else if (strcmp(kernel, "k_matmul_q8") == 0) {
        /* b is M rows of (K/32) Q8_0 blocks of 34 bytes each. K is a nonzero
         * multiple of 32, which check_params enforces before this runs (the
         * same ordering k_matvec_q8 relies on), so p[2]/32 is the exact
         * block count and does not truncate. */
        ok = mul_ck(p[0], p[2], &t0) && mul_ck(t0, f, &need_a) &&
             mul_ck(p[1], p[2] / 32u, &t1) && mul_ck(t1, 34, &need_b) &&
             mul_ck(p[0], p[1], &t2) && mul_ck(t2, f, &need_o);
    } else if (strcmp(kernel, "k_rope_chunk") == 0) {
        /* a = out = [n_tok*heads, stride] f32 (a slice's rotated tail stays
         * inside its own stride, guaranteed by check_params' stride>=head_dim,
         * so n_tok*heads*stride is a safe cover); b = cos/sin table
         * [n_tok, rope_dim] f32. p[0]=head_dim p[1]=rope_dim p[2]=heads
         * p[3]=stride p[4]=n_tok. */
        ok = mul_ck((uint64_t)p[4], p[2], &t0) && mul_ck(t0, p[3], &t1) &&
             mul_ck(t1, f, &need_a) &&
             mul_ck((uint64_t)p[4], p[1], &t2) && mul_ck(t2, f, &need_b);
        need_o = need_a;
    } else if (strcmp(kernel, "k_attn_decode_splitk_partial") == 0 ||
               strcmp(kernel, "k_attn_decode_splitk_combine") == 0 ||
               strcmp(kernel, "k_attn_decode_splitk_partial_gqa") == 0 ||
               strcmp(kernel, "k_attn_decode_splitk_partial_gqa_online") == 0) {
        /* P2.2 (and P2.4's GQA partial, which binds exactly the same eight,
         * and P2.8's online partial, which binds seven of them).
         * A ROUTING rule rather than a size rule: the partial kernels bind six
         * device buffers (q, k, v, m, s, acc) plus a score scratch and the
         * combine four (m, s, acc, out), so the (a, b, o) triple this
         * function is handed cannot describe either and sg_gpu_run_op has no
         * way to dispatch them. Named here rather than left to the generic
         * fall-through below so the error points at the entry points that do
         * work. Their real byte counts go through splitk_sizes(), which the
         * one-shots call and which guards every product with mul_ck exactly
         * as the rules above do. */
        return gpu_errf("gpu: %s binds more device buffers than sg_gpu_run_op's "
                        "(a, b, out); use sg_gpu_run_attn_splitk_partial/_gqa/"
                        "_gqa_online/_combine", kernel);
    } else {
        return gpu_errf("gpu: no size rule for kernel '%s'", kernel);
    }

    if (!ok) {
        return gpu_errf("gpu: %s params describe a region that overflows 64 bits", kernel);
    }

    if (!buf_big_enough(a, need_a)) {
        return gpu_errf("gpu: %s input a is %llu bytes, needs %llu", kernel,
                        (unsigned long long)(a ? a->nbytes : 0),
                        (unsigned long long)need_a);
    }
    if (want_b && !buf_big_enough(b, need_b)) {
        return gpu_errf("gpu: %s input b is %llu bytes, needs %llu", kernel,
                        (unsigned long long)(b ? b->nbytes : 0),
                        (unsigned long long)need_b);
    }
    if (!buf_big_enough(o, need_o)) {
        return gpu_errf("gpu: %s output is %llu bytes, needs %llu", kernel,
                        (unsigned long long)(o ? o->nbytes : 0),
                        (unsigned long long)need_o);
    }
    return SG_OK;
}

/* Per-kernel preconditions that are not about buffer sizes. */
static sg_err check_params(const char *kernel, const uint32_t *p) {
    if (strcmp(kernel, "k_matvec_q8") == 0) {
        /* Q8_0 rows are whole 32-element blocks, so a cols that is not a
         * multiple of 32 has no valid byte layout; ref.c returns without
         * touching y in that case, and the kernel would index a truncated
         * block count. Reject it loudly instead. */
        if (p[1] == 0 || p[1] % 32 != 0) {
            return gpu_errf("gpu: k_matvec_q8 cols %u must be a nonzero multiple of 32", p[1]);
        }
    } else if (strcmp(kernel, "k_rope") == 0) {
        if (p[1] < 2 || p[1] > p[0] || p[1] % 2 != 0) {
            return gpu_errf("gpu: k_rope rope_dim %u must be even and in [2, head_dim %u]",
                            p[1], p[0]);
        }
    } else if (strcmp(kernel, "k_rope_heads") == 0) {
        /* Same rope_dim rule as k_rope; stride (p[3]) must be at least head_dim
         * or a head's rotated tail would spill into the next head's slice.
         * (This kernel is normally dispatched by the decode encoder, which
         * always passes valid values; the rule lets sg_gpu_run_op reach it too,
         * which the M5.4 per-op test uses as the k_rope_chunk oracle.) */
        if (p[0] == 0) return (sg_err){"gpu: k_rope_heads head_dim must be nonzero"};
        if (p[1] < 2 || p[1] > p[0] || p[1] % 2 != 0) {
            return gpu_errf("gpu: k_rope_heads rope_dim %u must be even and in [2, head_dim %u]",
                            p[1], p[0]);
        }
        if (p[3] < p[0]) {
            return gpu_errf("gpu: k_rope_heads stride %u is smaller than head_dim %u",
                            p[3], p[0]);
        }
    } else if (strcmp(kernel, "k_attn_decode") == 0) {
        if (p[1] == 0 || p[0] % p[1] != 0) {
            return gpu_errf("gpu: k_attn_decode n_heads %u is not a multiple of n_kv_heads %u",
                            p[0], p[1]);
        }
        if (p[4] < p[2]) {
            return gpu_errf("gpu: k_attn_decode q_stride %u is smaller than head_dim %u",
                            p[4], p[2]);
        }
    } else if (strcmp(kernel, "k_conv1d_step") == 0) {
        if (p[1] == 0) return (sg_err){"gpu: k_conv1d_step ksize must be nonzero"};
    } else if (strcmp(kernel, "k_matmul_bf16") == 0 || strcmp(kernel, "k_matmul_f32") == 0) {
        if (p[0] == 0) return gpu_errf("gpu: %s N must be nonzero", kernel);
        if (p[1] == 0) return gpu_errf("gpu: %s M must be nonzero", kernel);
        if (p[2] == 0) return gpu_errf("gpu: %s K must be nonzero", kernel);
    } else if (strcmp(kernel, "k_matmul_q8") == 0) {
        if (p[0] == 0) return (sg_err){"gpu: k_matmul_q8 N must be nonzero"};
        if (p[1] == 0) return (sg_err){"gpu: k_matmul_q8 M must be nonzero"};
        /* Q8_0 rows are whole 32-element blocks, exactly k_matvec_q8's rule,
         * checked here (before check_sizes divides by 32) for the same
         * reason: a truncated block count would silently undersize need_b. */
        if (p[2] == 0 || p[2] % 32 != 0) {
            return gpu_errf("gpu: k_matmul_q8 K %u must be a nonzero multiple of 32", p[2]);
        }
    } else if (strcmp(kernel, "k_rope_chunk") == 0) {
        /* Same rope_dim rule as k_rope. stride must be at least head_dim, or a
         * slice's rotated tail would spill into the next slice; heads and n_tok
         * must be nonzero or the dispatch is empty. */
        if (p[0] == 0) return (sg_err){"gpu: k_rope_chunk head_dim must be nonzero"};
        if (p[1] < 2 || p[1] > p[0] || p[1] % 2 != 0) {
            return gpu_errf("gpu: k_rope_chunk rope_dim %u must be even and in [2, head_dim %u]",
                            p[1], p[0]);
        }
        if (p[2] == 0) return (sg_err){"gpu: k_rope_chunk heads must be nonzero"};
        if (p[3] < p[0]) {
            return gpu_errf("gpu: k_rope_chunk stride %u is smaller than head_dim %u",
                            p[3], p[0]);
        }
        if (p[4] == 0) return (sg_err){"gpu: k_rope_chunk n_tok must be nonzero"};
        /* The kernel carries slices = n_tok*heads and slices*head_dim (the grid
         * element count, and thread_position_in_grid) in 32-bit uint, so the
         * total must fit u32 or those wrap in-kernel even though the byte sizes
         * in check_sizes are u64-guarded. Reject rather than truncate. */
        uint64_t rc_slices = (uint64_t)p[4] * p[2];
        if (rc_slices > UINT32_MAX || rc_slices * p[0] > UINT32_MAX) {
            return (sg_err){"gpu: k_rope_chunk head_dim*heads*n_tok exceeds the 32-bit grid range"};
        }
    } else if (strcmp(kernel, "k_attn_decode_splitk_partial") == 0 ||
               strcmp(kernel, "k_attn_decode_splitk_combine") == 0 ||
               strcmp(kernel, "k_attn_decode_splitk_partial_gqa") == 0 ||
               strcmp(kernel, "k_attn_decode_splitk_partial_gqa_online") == 0) {
        /* P2.2. ONE params array serves both dispatches (surge.h documents it
         * that way, and the test fills it once), so both kernels get ONE rule:
         * a caller must not be able to get an array past the partial only to
         * have the combine reject it, or the pair would be dispatchable
         * half-way. The combine ignores n_kv_heads and q_stride, but they are
         * still validated here for that reason.
         * [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=seq [4]=q_stride
         * [5]=scale bits [6]=n_splits.
         *
         * seq (p[3]) is deliberately NOT required to be nonzero, unlike
         * sg_gpu_run_attn_decode_f16's own guard: seq == 0 makes every split
         * empty, and these two kernels DEFINE that case (every triple is the
         * m=-INFINITY/s=0/acc=0 encoding and the combine writes out[d] = 0.0,
         * matching sg_ref_attn_decode_splitk) instead of leaving `out`
         * unwritten the way k_attn_decode_f16 does. */
        if (p[0] == 0) return gpu_errf("gpu: %s n_heads must be nonzero", kernel);
        if (p[1] == 0 || p[0] % p[1] != 0) {
            return gpu_errf("gpu: %s n_heads %u is not a multiple of n_kv_heads %u",
                            kernel, p[0], p[1]);
        }
        if (p[2] == 0) return gpu_errf("gpu: %s head_dim must be nonzero", kernel);
        if (p[4] < p[2]) {
            return gpu_errf("gpu: %s q_stride %u is smaller than head_dim %u",
                            kernel, p[4], p[2]);
        }
        /* Zero splits would be a zero-length grid dimension (a Metal API
         * violation that aborts the process) and, on the combine side, the
         * oracle's n_parts == 0 case, which has no partial triples to fold at
         * all. Rejected rather than improvised. */
        if (p[6] == 0) return gpu_errf("gpu: %s n_splits must be nonzero", kernel);
        /* P2.4's GQA partial shares this rule and needs no extra one. The
         * divisibility check above is exactly what makes its (n_splits, n_kv)
         * grid tile the query heads: group hk covers hk*repeat .. +repeat-1
         * with n_kv*repeat == n_heads, so no group runs off the end and none
         * is skipped. A group WIDER than SG_SPLITK_GQA_MAX is still answered
         * correctly (the kernel's default arm walks it one head at a time), so
         * it is a policy question for splitk_gqa_use, not a validity one.
         *
         * P2.8's online partial shares it too, and adds NO head_dim ceiling
         * here on purpose: head_dim > SG_TG makes that kernel re-stream the
         * split once per SG_TG-wide band of output dims, which is slower than
         * the four-pass kernel but still exactly correct, so it is the same
         * kind of policy question (splitk_online_use declines it) rather than a
         * validity one. A rejection here would instead break the one-shot for a
         * shape the kernel answers correctly. */
    }
    return SG_OK;
}

/* Grid geometry from the kernel's kind and its params. Split out of
 * sg_gpu_run_op because sg_gpu_forward's encoder needs exactly the same
 * mapping and a second copy of it would be a silent way for the batched path
 * to dispatch a different shape than the one-shot path. Products are u64:
 * both factors are u32 params, so they cannot wrap. */
static void gpu_grid(int kind, const uint32_t *p, uint64_t *groups, uint64_t *elems) {
    *groups = 1;
    *elems = 0;
    switch (kind) {
    case SG_K_ELEM:    *elems = p[0]; break;
    case SG_K_ELEM01:  *elems = (uint64_t)p[0] * p[1]; break;
    case SG_K_ELEM02:  *elems = (uint64_t)p[0] * p[2]; break;
    case SG_K_ROWS:    *groups = p[0]; break;
    case SG_K_ATTN:    *groups = p[0]; break;
    case SG_K_GATED:   *groups = p[1]; break;
    case SG_K_GROUPS2: *groups = p[2]; break;
    /* M5.4 k_rope_chunk: one thread per (token, head, element). Three u32
     * factors; the (uint64)p[0]*p[2] cannot wrap (two u32) and *p[4] cannot
     * either at any real chunk size (head_dim*heads*n_tok is far under 2^64).
     * check_sizes re-guards the byte counts with mul_ck regardless. */
    case SG_K_ROPE_CHUNK: *elems = (uint64_t)p[0] * p[2] * p[4]; break;
    /* SG_K_TILES2D and SG_K_HEADS2D each need two group dimensions, which this
     * function's (groups, elems) pair cannot carry; their dispatchers compute
     * both by hand instead (sg_gpu_run_op's SG_K_TILES2D case, and
     * sg_gpu_run_attn_splitk_partial for SG_K_HEADS2D). Left at the default
     * *groups = 1 so a caller that ignored this comment gets an
     * obviously-wrong single threadgroup rather than a plausible-looking wrong
     * number. */
    default:           *groups = 1; break;
    }
}

/* k_attn_decode's scores live in device memory, one private row of seq_len
 * floats per head, so nothing about the context length is capped by the
 * 32 KB threadgroup allocation. */
static sg_err scratch_ensure(sg_gpu *g, uint64_t nbytes) {
    if (g->scratch && g->scratch_bytes >= nbytes) return SG_OK;
    id<MTLBuffer> nb = nil;
    @autoreleasepool {
        nb = [g->dev newBufferWithLength:(NSUInteger)nbytes
                                 options:MTLResourceStorageModePrivate];
    }
    if (!nb) {
        return gpu_errf("gpu: cannot allocate %llu bytes of scratch",
                        (unsigned long long)nbytes);
    }
    [g->scratch release];
    g->scratch = nb;
    g->scratch_bytes = nbytes;
    return SG_OK;
}

/* The same grow-on-demand allocator for the split-K partial's OWN score
 * buffer (P2.2 review finding 1). Deliberately a second function rather than
 * a refactor of scratch_ensure into a shared (buffer, bytes) helper: that
 * would have rewritten the allocation path every existing Metal gate runs
 * through, and the point of the finding was to add isolation, not to churn
 * the code the isolated-from kernels depend on. The duplication is fifteen
 * lines and the distinct error text ("split-K score scratch") is what a
 * failure at 262,144 x n_splits would need to say anyway. */
static sg_err splitk_scratch_ensure(sg_gpu *g, uint64_t nbytes) {
    if (g->splitk_scratch && g->splitk_scratch_bytes >= nbytes) return SG_OK;
    id<MTLBuffer> nb = nil;
    @autoreleasepool {
        nb = [g->dev newBufferWithLength:(NSUInteger)nbytes
                                 options:MTLResourceStorageModePrivate];
    }
    if (!nb) {
        return gpu_errf("gpu: cannot allocate %llu bytes of split-K score scratch",
                        (unsigned long long)nbytes);
    }
    [g->splitk_scratch release];
    g->splitk_scratch = nb;
    g->splitk_scratch_bytes = nbytes;
    return SG_OK;
}

/* =====================================================================
 * P2.3: the decode path's split-K policy
 * =====================================================================
 *
 * n_splits = clamp(seq / SG_TG, 4, 1024). This is MEASURED (task P2.3a, run
 * with `make bench-splitk` on this machine), not a guess: sweeping n_splits
 * over {1..1024} at seq 8192 / 32768 / 131072 / 262144 on both the real 27B
 * decode shape (24 heads, 4 kv, head_dim 256) and the real 4B dense shape (32
 * heads, 8 kv, head_dim 128), the fastest n_splits was EXACTLY seq / SG_TG in
 * every one of those eight cells (8192 -> 32, 32768 -> 128, 131072 -> 512,
 * 262144 -> 1024), i.e. hand every split exactly SG_TG keys so every lane of
 * the threadgroup gets one. That is the top of the occupancy band documented
 * on k_attn_decode_splitk_partial (n_heads*n_splits >= GPU cores from below,
 * n_splits <= seq/SG_TG from above), so the closed form and the band agree.
 * At 262144 the pair beat k_attn_decode_f16 by 15.9x (27B shape) and 21.9x
 * (4B shape).
 *
 * The 1024 ceiling is the band's own upper bound at SG_KV_CAP_MAX == 262144
 * (262144/256), so it only ever binds for a hypothetical longer context; the
 * 4 floor is the band's lower bound.
 *
 * SHORT SEQUENCES KEEP THE INCUMBENT (splitk_use returns 0 below
 * SG_SPLITK_MIN_SEQ). Under seq == SG_TG * SG_SPLITK_MIN == 1024 the floor is
 * what binds, not seq / SG_TG, so the splits come out SHORTER than SG_TG keys
 * and the partial's lanes start idling: the closed form stops being the
 * measured optimum and becomes an extrapolation below the shortest sequence
 * P2.3a measured (8192). The threshold is therefore exactly the point where
 * the clamp's floor stops binding, not a tuned constant, and below it decode
 * runs the incumbent k_attn_decode_f16 it has always run. */
#define SG_SPLITK_MIN 4u
#define SG_SPLITK_MAX 1024u
#define SG_SPLITK_MIN_SEQ (SG_TG * SG_SPLITK_MIN)

/* The closed form, clamped into the occupancy band. Nondecreasing in seq
 * (floor division is, and clamping preserves that), which is what lets
 * sg_gpu_state_new size every split-K buffer once from max_ctx. */
static uint32_t splitk_n_splits(uint32_t seq) {
    uint32_t n = seq / SG_TG;
    if (n < SG_SPLITK_MIN) n = SG_SPLITK_MIN;
    if (n > SG_SPLITK_MAX) n = SG_SPLITK_MAX;
    return n;
}

/* The n_splits this decode step should use, or 0 meaning "dispatch the
 * incumbent k_attn_decode_f16 instead". Zero is returned when the policy is
 * off (SURGE_ATTN_SPLITK=0, or the f32 KV path, which has no half-typed K/V
 * for these kernels to read), when the sequence is below the threshold above,
 * or when any of the three partial buffers is missing -- the last is
 * defensive: sg_gpu_state_new allocates them together with the fp16 cache, so
 * a NULL here means an allocation path changed, and falling back to a kernel
 * that is known to work beats dispatching against a nil buffer. */
static uint32_t splitk_use(const sg_gpu *g, uint32_t seq) {
    if (!g->attn_splitk) return 0;
    if (seq < SG_SPLITK_MIN_SEQ) return 0;
    if (!g->b_sk_m || !g->b_sk_s || !g->b_sk_acc || !g->splitk_scratch) return 0;
    uint32_t n = splitk_n_splits(seq);
    if (n > g->splitk_max_splits) return 0;   /* sized from max_ctx; cannot happen */
    return n;
}

/* =====================================================================
 * P2.4: which of the two split-K partials a decode step dispatches
 * =====================================================================
 *
 * true means k_attn_decode_splitk_partial_gqa (one threadgroup per GQA GROUP,
 * so each K/V element is read once and used for all `repeat` query heads that
 * share it); false means the per-head k_attn_decode_splitk_partial P2.2/P2.3
 * ship. The two write the SAME bytes into the same m/s/acc layout, so this
 * chooses memory traffic and grid shape, never the answer.
 *
 * ON BY DEFAULT SINCE TASK P4.0 (2026-08-18), and that is a statement about
 * evidence, exactly as the earlier default of OFF was. P2.4 wrote this kernel
 * with the GPU held by a 256K benchmark, so nothing about it had been run and it
 * shipped switched off, the way P2.2 shipped its kernels. What moved the default
 * is that the three things it was waiting for are gated: byte-identity against
 * the per-head partial at a fixed n_splits (P2.4), byte-exact greedy tokens on a
 * real model where the two split policies DIVERGE (P2.6), and the measured
 * occupancy floor below, which keeps this kernel out of every point where the
 * sweep saw it lose (P2.7). SURGE_ATTN_SPLITK_GQA=0 pins the per-head partial,
 * which is what keeps the A/B runnable on ONE binary; see docs/17082026_splitk_
 * gqa_threadgroups.md for the gate results.
 *
 * THE GROUP MUST BE WORTH SHARING AND MUST FIT IN REGISTERS:
 *   - repeat < 2 means no sharing exists (the GQA grid would be the per-head
 *     grid, one query head per threadgroup, with extra index arithmetic);
 *   - repeat > SG_SPLITK_GQA_MAX falls into the kernel's correct-but-no-reuse
 *     arm, which is strictly worse than the per-head kernel because it also
 *     shrinks the grid by `repeat`.
 * Both real shapes sit inside the band (27B 24/4 = 6, 4B dense 32/8 = 4).
 *
 * AND THE GRID MUST STILL FILL THE MACHINE (task P2.7). Collapsing `repeat`
 * per-head threadgroups into one DIVIDES THE GRID BY `repeat`, so at short
 * context the GQA kernel trades redundant traffic for an idle GPU and loses.
 * The floor below is on the THREADGROUP COUNT the dispatch will actually have,
 * splitk_gqa_n_splits(seq, cap) * n_kv (computed here through the same two
 * functions enc_attn_splitk dispatches with, never restated), because that is
 * the quantity that starves; it is NOT on seq, and the measurement says so.
 *
 * MEASURED 2026-08-17, `./tests/bench_splitk.bin --reps 20` both arms, 3
 * alternating rounds, GPU idle-checked. Speedup is per-head time / GQA time at
 * the split count the decode policy would dispatch, so below 1.000 the GQA
 * kernel is the wrong choice. Median of the 3 rounds, by threadgroup count:
 *
 *   threadgroups     32    40    48    56    64    80    96   112   128   256   512
 *   27B 24h/4kv   0.938 0.867 0.980 0.971 0.990 0.988 1.025 1.110 1.148 1.248     -
 *   4B  32h/8kv       -     -     -     - 0.975 0.925 0.975 1.008 1.004 1.056 1.182
 *
 * THE 4B SHAPE IS THE BINDING ONE: it still loses at 96 threadgroups (0.975,
 * and 0.974 in a second study of 6 rounds at --reps 50) and only reaches parity
 * at 128 (1.004 in both studies). The 27B crosses earlier (0.956 at 80, 1.040
 * at 96 in that second study) because repeat 6 and head_dim 256 save more
 * traffic per threadgroup given up. 128 is the smallest floor that leaves NO
 * measured loss inside the GQA region on EITHER shape: the crossover is
 * bracketed in (112, 128] for the 4B and (80, 96] for the 27B, and this takes
 * the higher of the two rather than a floor that scales with `repeat`, which
 * two `repeat` values do not justify and which would extrapolate to 64
 * threadgroups -- below this machine's 80 GPU cores -- at repeat 8. It costs
 * the 27B its measured 1.03x and 1.11x at 96 and 112 threadgroups (seq 6144 and
 * 7168), deliberately: a wrong-way error there is a slowdown, and this switch
 * exists to be flippable without one.
 *
 * A SEQ THRESHOLD WOULD NOT FIT. The same crossovers in seq are 5120..6144 (27B)
 * and 3072..4096 (4B): 1.7x apart, and in the OPPOSITE order from the
 * threadgroup view, because the 4B has twice the kv heads. So one seq threshold
 * must either admit the 27B's 0.956 at seq 5120 or reject the 4B's wins from
 * 4096 up, while one threadgroup threshold separates all 20 measured points.
 * See docs/17082026_splitk_gqa_threadgroups.md for the full table and the
 * per-round numbers. */
#define SG_SPLITK_GQA_MIN_TG 128u  /* measured occupancy floor, see above */

/* P2.7 needs the DISPATCHED split count inside splitk_gqa_use, and the two
 * functions that produce it belong to the P2.5/P2.6 sections below (moving them
 * up here would separate splitk_gqa_n_splits from the measured table that
 * justifies it). Declared, not restated: the guard must see the same grid the
 * dispatch will use, or it guards a shape that never runs. */
static uint32_t splitk_gqa_n_splits(uint32_t seq, uint32_t cap);
static uint32_t splitk_gqa_cap_of(const sg_gpu *g);

/* Everything above that is about the SHAPE AND THE GRID rather than about which
 * switch is on: the group must be worth sharing, must fit in registers, and the
 * dispatch must still fill the machine. Factored out for P2.8, whose online
 * kernel is a GQA-shared kernel with the same grid, the same group-size band and
 * the same occupancy problem, so it must ask the same three questions. Splitting
 * this out rather than copying it is the point: a change to the measured floor
 * or to the band must reach both arms or the second one silently keeps the old
 * policy. */
static bool splitk_gqa_shape_ok(const sg_gpu *g, uint32_t n_heads, uint32_t n_kv,
                                uint32_t seq) {
    if (n_kv == 0 || n_heads % n_kv != 0) return false;
    uint32_t repeat = n_heads / n_kv;
    if (repeat < 2 || repeat > SG_SPLITK_GQA_MAX) return false;
    /* uint64_t so a caller's absurd n_kv cannot wrap the product into a pass. */
    uint64_t tgs = (uint64_t)splitk_gqa_n_splits(seq, splitk_gqa_cap_of(g)) * n_kv;
    return tgs >= SG_SPLITK_GQA_MIN_TG;
}

/* P2.8: forward declaration, because the four-pass GQA arm now has to yield to
 * the online arm when both switches are set (see below), and the online
 * predicate needs splitk_gqa_shape_ok above. */
static bool splitk_online_use(const sg_gpu *g, uint32_t n_heads, uint32_t n_kv,
                              uint32_t hd, uint32_t seq);

static bool splitk_gqa_use(const sg_gpu *g, uint32_t n_heads, uint32_t n_kv,
                           uint32_t seq) {
    if (!g->attn_splitk_gqa) return false;
    /* P2.8: with BOTH kernel switches set the online arm is the one that runs
     * (it is the more specific request), so this predicate must answer false
     * there or the encoder's two counters would double-count one dispatch. It
     * takes head_dim from the state's own config because this entry point is
     * not given one; g->cfg.head_dim is exactly the head_dim every decode step
     * of this state dispatches with. With attn_splitk_online false, which is the
     * default, splitk_online_use is false for every shape and this line changes
     * nothing about P2.4's through P2.7's behaviour. */
    if (splitk_online_use(g, n_heads, n_kv, g->cfg.head_dim, seq)) return false;
    return splitk_gqa_shape_ok(g, n_heads, n_kv, seq);
}

/* =====================================================================
 * P2.8: whether a decode step uses the ONLINE-SOFTMAX GQA partial
 * =====================================================================
 *
 * The same three shape/grid questions splitk_gqa_use asks (via the shared
 * splitk_gqa_shape_ok), plus ONE MORE that is specific to this kernel:
 *
 *   head_dim <= SG_TG.
 *
 * That is the accumulator-storage boundary, not a tuning choice. The online
 * kernel keeps its running acc in registers by giving each thread exactly one
 * output dim, which works while head_dim fits one SG_TG-wide band. Past that
 * the kernel re-streams the split (and re-reads K) once per band: still exactly
 * correct, which is why the one-shot entry point accepts it and check_params
 * does not reject it, but strictly more traffic than the four-pass kernel it is
 * supposed to beat. Both real shapes are inside the bound (27B head_dim 256 ==
 * SG_TG, 4B dense 128), so the decline costs surge nothing today.
 *
 * NO SEPARATE SPLIT POLICY. The online arm dispatches with
 * splitk_gqa_n_splits(seq, cap), P2.5's measured GQA policy, because it has the
 * same grid shape and the same per-threadgroup reuse; whether streaming moves
 * the saturation point is a MEASUREMENT nobody has taken, and inventing a second
 * policy without one would be a guess. Noted in
 * docs/17082026_splitk_gqa_threadgroups.md as an open question.
 *
 * OFF BY DEFAULT (attn_splitk_online), so this returns false for every shape
 * unless SURGE_ATTN_SPLITK_ONLINE=1 was accepted by sg_gpu_state_new. */
static bool splitk_online_use(const sg_gpu *g, uint32_t n_heads, uint32_t n_kv,
                              uint32_t hd, uint32_t seq) {
    if (!g->attn_splitk_online) return false;
    if (hd == 0 || hd > SG_TG) return false;
    return splitk_gqa_shape_ok(g, n_heads, n_kv, seq);
}

/* The same predicate, for the gate (P2.4 fix round 1, review finding I1/M4).
 * It CALLS splitk_gqa_use rather than restating it, so a test of the [2, 8]
 * band, of the repeat == 1 decline, of the not-a-multiple decline or of P2.7's
 * threadgroup floor is a test of the function the encoder actually consults,
 * not of a copy that could drift from it. Read-only; a NULL g answers false
 * rather than crashing, since this is reachable from a test harness.
 *
 * P2.7 gave it `seq`, because the floor cannot be answered without one: the same
 * shape is a GQA shape at depth and a per-head shape at short context. */
bool sg_gpu_splitk_gqa_selected(const sg_gpu *g, uint32_t n_heads,
                                uint32_t n_kv_heads, uint32_t seq) {
    if (!g) return false;
    return splitk_gqa_use(g, n_heads, n_kv_heads, seq);
}

/* What enc_attn_splitk actually encoded since the last sg_gpu_state_new, split
 * by kernel (P2.4 fix round 1, review finding I1). Either pointer may be NULL.
 *
 * P2.8: `gqa` still counts ONLY the four-pass GQA partial. The online arm has
 * its own counter (sg_gpu_splitk_online_dispatches) so the exact-count
 * assertions the P2.6/P2.7 gates make keep meaning what they meant. */
void sg_gpu_splitk_dispatch_counts(const sg_gpu *g, uint64_t *per_head, uint64_t *gqa) {
    if (per_head) *per_head = g ? g->splitk_partial_dispatches : 0;
    if (gqa) *gqa = g ? g->splitk_gqa_dispatches : 0;
}

/* P2.8, the same predicate the encoder consults, for the gate. It CALLS
 * splitk_online_use rather than restating it, so a test of the head_dim bound,
 * of the group-size band or of P2.7's threadgroup floor is a test of the
 * function that actually picks the kernel. Read-only; NULL g answers false. */
bool sg_gpu_splitk_online_selected(const sg_gpu *g, uint32_t n_heads,
                                   uint32_t n_kv_heads, uint32_t head_dim,
                                   uint32_t seq) {
    if (!g) return false;
    return splitk_online_use(g, n_heads, n_kv_heads, head_dim, seq);
}

/* How many online partials enc_attn_splitk encoded since the last
 * sg_gpu_state_new. Same vacuity argument as the two counters above: the online
 * arm's end-to-end A/B compares greedy TOKENS, which agree whether or not the
 * kernel was ever selected, so a gate has to be able to say that it was. */
uint64_t sg_gpu_splitk_online_dispatches(const sg_gpu *g) {
    return g ? g->splitk_online_dispatches : 0;
}

/* =====================================================================
 * P2.5: the GQA partial's OWN split count
 * =====================================================================
 *
 * n_splits_gqa = clamp(min(seq / SG_TG, SG_SPLITK_GQA_N_SPLITS_CAP),
 * SG_SPLITK_MIN, SG_SPLITK_MAX). MEASURED (task P2.5,
 * `./tests/bench_splitk.bin --seqs 8192,32768,131072,262144 --gqa` on this
 * machine, 2026-08-17, fresh GEMM gate 21.63 TFLOPS), not a guess and not a
 * rescaling of splitk_n_splits.
 *
 * Per-point optimum n_splits for the GQA kernel, against seq / SG_TG:
 *   seq                    8192   32768   131072   262144
 *   seq / SG_TG              32     128      512     1024
 *   27B 24h/4kv/256d         32     128      256      256
 *   4B  32h/8kv/128d         32      64      256      512
 * The optimum tracks seq / SG_TG at the two short lengths, then SATURATES
 * near 256 rather than climbing with it the way the per-head optimum does
 * (splitk_n_splits above always equals seq / SG_TG in this same band).
 *
 * Four candidate policies were scored by regret against that per-point
 * optimum (mean / worst over all eight points above, lower is better):
 *   clamp(seq/SG_TG, 4, 1024)           current per-head policy   3.1% / 7.3%
 *   clamp(min(seq/SG_TG,256), 4, 1024)  THE WINNER, used below    0.5% / 2.6%
 *   clamp(min(seq/SG_TG,512), 4, 1024)                            1.5% / 4.2%
 *   clamp(seq/(2*SG_TG), 4, 1024)       "half the per-head optimum" 3.8% / 13.4%
 *
 * THE INTUITIVE RULE IS THE WORST ONE. Halving the per-head closed form
 * fits the two longest sequences by construction (both optima there happen
 * to be roughly half of seq / SG_TG) and is 13.4% worst-case wrong overall:
 * the true optimum does not RESCALE with seq, it SATURATES, so a value that
 * keeps climbing and is merely divided by two eventually overshoots almost
 * as badly as never capping at all. The winner is the per-head closed form
 * with a measured cap, not a different closed form.
 *
 * SG_SPLITK_MIN / SG_SPLITK_MAX (defined above) are the SAME occupancy-band
 * floor and ceiling splitk_n_splits uses; only the cap in the middle differs,
 * and the cap can only ever LOWER the result relative to splitk_n_splits at
 * the same seq (min() cannot raise it), which is what keeps this policy
 * inside the m/s/acc buffers g->splitk_max_splits already sizes from
 * splitk_n_splits(max_ctx) -- see sg_gpu_state_new. No buffer changed size
 * for this task.
 *
 * See docs/17082026_splitk_gqa_threadgroups.md for the full sweep and the
 * regret re-measured against this implementation. */
#define SG_SPLITK_GQA_N_SPLITS_CAP 256u  /* measured saturation point, see above */

/* Task P2.6 made the cap a PARAMETER rather than reading the macro directly,
 * so the whole policy stays a pure function of its inputs (which is what keeps
 * the P2.5 policy assertions meaningful) while the cap the DECODE PATH uses can
 * be overridden at state-creation time. Every caller resolves the cap through
 * splitk_gqa_cap_of below or passes SG_SPLITK_GQA_N_SPLITS_CAP explicitly; no
 * caller passes 0.
 *
 * The <= splitk_n_splits(seq) invariant holds for EVERY cap, not just 256:
 * min(x, cap) <= x for any cap, and both sides then run the identical
 * SG_SPLITK_MIN/SG_SPLITK_MAX clamp, which is monotone. So no override can
 * make the GQA arm ask for more splits than the m/s/acc buffers
 * g->splitk_max_splits sized from splitk_n_splits(max_ctx). */
static uint32_t splitk_gqa_n_splits(uint32_t seq, uint32_t cap) {
    uint32_t n = seq / SG_TG;
    if (n > cap) n = cap;
    if (n < SG_SPLITK_MIN) n = SG_SPLITK_MIN;
    if (n > SG_SPLITK_MAX) n = SG_SPLITK_MAX;
    return n;
}

/* The cap THIS state will dispatch with: the SURGE_SPLITK_GQA_CAP override if
 * sg_gpu_state_new accepted one, otherwise the measured default. The 0 ->
 * default mapping is the reason splitk_gqa_cap is allowed to be 0 in a freshly
 * calloc'd or freshly freed sg_gpu: an unset field must mean "the shipped
 * policy", never "cap of 0" (which the clamp above would turn into
 * SG_SPLITK_MIN for every seq, silently). */
static uint32_t splitk_gqa_cap_of(const sg_gpu *g) {
    uint32_t cap = g ? g->splitk_gqa_cap : 0;
    return cap ? cap : SG_SPLITK_GQA_N_SPLITS_CAP;
}

/* The diagnostic counterpart of sg_gpu_splitk_gqa_selected above: calls
 * splitk_gqa_n_splits rather than restating it, so a test of the measured
 * table above tests the value enc_attn_splitk actually dispatches with once
 * it has picked the GQA kernel, not a copy that could drift from it. Pure
 * function of seq; needs no sg_gpu state, unlike sg_gpu_splitk_gqa_selected
 * (there is no on/off switch to gate here -- the caller only reaches this
 * once the GQA kernel is already selected).
 *
 * P2.6: this entry point reports the SHIPPED table, i.e. the compiled cap,
 * which is what the P2.5 assertions are about. For what a particular state
 * will dispatch, including a SURGE_SPLITK_GQA_CAP override, use
 * sg_gpu_splitk_gqa_n_splits_at below. */
uint32_t sg_gpu_splitk_gqa_n_splits(uint32_t seq) {
    return splitk_gqa_n_splits(seq, SG_SPLITK_GQA_N_SPLITS_CAP);
}

/* Task P2.6. sg_gpu_splitk_gqa_n_splits_at is the ONLY honest answer to "how
 * many splits will the GQA arm of enc_attn_splitk dispatch at this seq", since
 * that depends on the state's cap; it calls the same splitk_gqa_n_splits with
 * the same resolved cap the encoder uses, so a gate cannot drift from the
 * dispatch. sg_gpu_splitk_gqa_cap exposes the resolved cap itself, which is how
 * a gate proves an override was actually parsed rather than ignored.
 * sg_gpu_splitk_n_splits is the PER-HEAD arm's split count, the value
 * splitk_use hands enc_attn_splitk in p[6]; a gate needs both sides to assert
 * that the two arms really do partition the keys differently. */
uint32_t sg_gpu_splitk_gqa_cap(const sg_gpu *g) {
    return splitk_gqa_cap_of(g);
}

uint32_t sg_gpu_splitk_gqa_n_splits_at(const sg_gpu *g, uint32_t seq) {
    return splitk_gqa_n_splits(seq, splitk_gqa_cap_of(g));
}

uint32_t sg_gpu_splitk_n_splits(uint32_t seq) {
    return splitk_n_splits(seq);
}

sg_err sg_gpu_run_op(sg_gpu *g, const char *kernel, void *a, void *b, void *out,
                     const uint32_t params[8]) {
    if (!g || !kernel || !params) return (sg_err){"gpu: sg_gpu_run_op got a NULL argument"};

    int idx = -1;
    for (int i = 0; i < SG_N_KERNELS; i++) {
        if (strcmp(SG_KERNELS[i].name, kernel) == 0) { idx = i; break; }
    }
    if (idx < 0) return gpu_errf("gpu: unknown kernel '%s'", kernel);

    sg_gpu_buf *ab = (sg_gpu_buf *)a, *bb = (sg_gpu_buf *)b, *ob = (sg_gpu_buf *)out;
    sg_err e = check_params(kernel, params);
    if (sg_failed(e)) return e;
    e = check_sizes(kernel, ab, bb, ob, params);
    if (sg_failed(e)) return e;
    /* surge.h forbids `out` overlapping an input, and the reason is not
     * style: a threadgroup that has already written its output row would be
     * changing an input row another threadgroup has not read yet, which is
     * both wrong and NONDETERMINISTIC, the one property this whole layer
     * exists to provide. Enforce it rather than document it. Note this
     * catches the two in-place kernels correctly too: k_delta_step updates S,
     * which is `a`, and k_conv1d_step's carried state lives inside `out`, so
     * neither of them wants out to overlap an input either. */
    if (bufs_overlap(ab, ob) || bufs_overlap(bb, ob)) {
        return gpu_errf("gpu: %s output overlaps an input buffer", kernel);
    }

    /* Grid geometry. A zero threadgroup count is a Metal API violation
     * (it aborts the process), so an empty op is rejected here instead. */
    int kind = SG_KERNELS[idx].kind;
    uint64_t groups = 1, elems = 0;
    uint64_t tiles_m = 0, tiles_n = 0;
    gpu_grid(kind, params, &groups, &elems);
    if (kind == SG_K_ELEM || kind == SG_K_ELEM01 || kind == SG_K_ELEM02
        || kind == SG_K_ROPE_CHUNK) {
        if (elems == 0) return gpu_errf("gpu: %s dispatched with zero elements", kernel);
    } else if (kind == SG_K_TILES2D) {
        /* Two group dimensions: params[0]=N tiles vertically, params[1]=M
         * tiles horizontally. check_params has already rejected N == 0 and
         * M == 0 for every k_matmul_* kernel, so both ceil-divisions here are
         * already known nonzero; the explicit check below is a second,
         * cheap guard rather than trust across a function boundary. */
        tiles_n = ((uint64_t)params[0] + SG_GEMM_TM - 1) / SG_GEMM_TM;
        tiles_m = ((uint64_t)params[1] + SG_GEMM_TN - 1) / SG_GEMM_TN;
        if (tiles_n == 0 || tiles_m == 0) {
            return gpu_errf("gpu: %s dispatched with zero tiles", kernel);
        }
    } else if (groups == 0) {
        return gpu_errf("gpu: %s dispatched with zero threadgroups", kernel);
    }

    if (kind == SG_K_ATTN) {
        uint64_t need = (uint64_t)params[0] * params[3] * 4;
        if (need == 0) return gpu_errf("gpu: %s dispatched with an empty score row", kernel);
        e = scratch_ensure(g, need);
        if (sg_failed(e)) return e;
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            rc = (sg_err){"gpu: could not open a compute encoder"};
        } else {
            [enc setComputePipelineState:g->pipes[idx]];
            [enc setBuffer:ab->buf offset:(NSUInteger)ab->offset atIndex:0];
            /* Index 1 must be bound even for the single-input kernels: the
             * function signature declares the argument, and Metal's
             * validation layer rejects an unbound one. Reuse `a` rather than
             * keep a dummy buffer alive; those kernels never read it. */
            [enc setBuffer:(bb ? bb->buf : ab->buf)
                    offset:(NSUInteger)(bb ? bb->offset : ab->offset)
                   atIndex:1];
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:2];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:3];
            if (kind == SG_K_ATTN) [enc setBuffer:g->scratch offset:0 atIndex:4];

            if (elems != 0) {
                NSUInteger w = [g->pipes[idx] maxTotalThreadsPerThreadgroup];
                if (w > SG_TG) w = SG_TG;
                if (w > elems) w = (NSUInteger)elems;
                [enc dispatchThreads:MTLSizeMake((NSUInteger)elems, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
            } else if (kind == SG_K_TILES2D) {
                /* x = M tiles, y = N tiles, matching k_matmul_*'s
                 * tile.x/tile.y reading in kernels.metal. The per-threadgroup
                 * shape is 2D too (SG_GEMM_TN, SG_GEMM_TM), matching the
                 * kernel's uint2 thread_position_in_threadgroup -- Metal
                 * requires that attribute's vector width to match
                 * threadgroup_position_in_grid's, so it cannot be a flat
                 * SG_TG the way the reduction kernels dispatch. Total
                 * threads per threadgroup is still SG_TG (16*16). */
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tiles_m, (NSUInteger)tiles_n, 1)
                    threadsPerThreadgroup:MTLSizeMake(SG_GEMM_TN, SG_GEMM_TM, 1)];
            } else {
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) {
                rc = gpu_errf("gpu: %s failed: %s", kernel,
                              [[[cb error] localizedDescription] UTF8String]);
            }
        }
    }
    return rc;
}

/* One-shot dispatch for k_attn_decode_f16 (Task M5.2), sg_gpu_run_op's
 * synchronous commit-and-wait contract extended to a THREE-input kernel: q
 * is f32 [n_heads, q_stride], k and v are f16 [seq, n_kv_heads, head_dim]
 * SEPARATE buffers (the sg_kv layout), out is f32 [n_heads, head_dim].
 * sg_gpu_run_op cannot reach this kernel at all -- its (a, b, out) contract
 * has no slot for a third input -- so this is its dedicated entry point,
 * used by the per-op test and nowhere else (the batched decode path in
 * enc_attn calls enc_attn_f16 directly, inside the one open command buffer a
 * whole token's layers share).
 *
 * params: [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=seq_len [4]=q_stride
 * [5]=softmax scale bits (params[6] and params[7] are unused). */
sg_err sg_gpu_run_attn_decode_f16(sg_gpu *g, void *q, void *k, void *v, void *out,
                                  const uint32_t params[8]) {
    if (!g || !q || !k || !v || !out || !params) {
        return (sg_err){"gpu: sg_gpu_run_attn_decode_f16 got a NULL argument"};
    }
    uint32_t n_heads = params[0], n_kv = params[1], hd = params[2], seq = params[3];
    uint32_t q_stride = params[4];
    if (n_kv == 0 || n_heads % n_kv != 0) {
        return gpu_errf("gpu: k_attn_decode_f16 n_heads %u is not a multiple of n_kv_heads %u",
                        n_heads, n_kv);
    }
    if (q_stride < hd) {
        return gpu_errf("gpu: k_attn_decode_f16 q_stride %u is smaller than head_dim %u",
                        q_stride, hd);
    }
    if (n_heads == 0 || hd == 0 || seq == 0) {
        return (sg_err){"gpu: k_attn_decode_f16 dispatched with a zero dimension"};
    }

    sg_gpu_buf *qb = (sg_gpu_buf *)q, *kb = (sg_gpu_buf *)k, *vb = (sg_gpu_buf *)v,
               *ob = (sg_gpu_buf *)out;

    uint64_t need_q = 0, need_kv = 0, need_o = 0, t0 = 0;
    bool ok = mul_ck((uint64_t)n_heads, q_stride, &t0) && mul_ck(t0, 4, &need_q)
           && mul_ck((uint64_t)seq * n_kv, hd, &t0) && mul_ck(t0, 2, &need_kv)
           && mul_ck((uint64_t)n_heads, hd, &t0) && mul_ck(t0, 4, &need_o);
    if (!ok) {
        return (sg_err){"gpu: k_attn_decode_f16 params describe a region that overflows 64 bits"};
    }
    if (!buf_big_enough(qb, need_q)) {
        return gpu_errf("gpu: k_attn_decode_f16 q is %llu bytes, needs %llu",
                        (unsigned long long)(qb ? qb->nbytes : 0), (unsigned long long)need_q);
    }
    if (!buf_big_enough(kb, need_kv)) {
        return gpu_errf("gpu: k_attn_decode_f16 k is %llu bytes, needs %llu",
                        (unsigned long long)(kb ? kb->nbytes : 0), (unsigned long long)need_kv);
    }
    if (!buf_big_enough(vb, need_kv)) {
        return gpu_errf("gpu: k_attn_decode_f16 v is %llu bytes, needs %llu",
                        (unsigned long long)(vb ? vb->nbytes : 0), (unsigned long long)need_kv);
    }
    if (!buf_big_enough(ob, need_o)) {
        return gpu_errf("gpu: k_attn_decode_f16 out is %llu bytes, needs %llu",
                        (unsigned long long)(ob ? ob->nbytes : 0), (unsigned long long)need_o);
    }
    if (bufs_overlap(qb, ob) || bufs_overlap(kb, ob) || bufs_overlap(vb, ob)) {
        return (sg_err){"gpu: k_attn_decode_f16 output overlaps an input buffer"};
    }

    /* n_heads*seq alone cannot wrap (two uint32 factors, as above), but *4
     * can when both sit near UINT32_MAX; guard it like every other byte
     * count here rather than let scratch_ensure see a wrapped-small need. */
    uint64_t need_scratch = 0;
    if (!mul_ck((uint64_t)n_heads * seq, 4, &need_scratch)) {
        return (sg_err){"gpu: k_attn_decode_f16 score-scratch size overflows 64 bits"};
    }
    sg_err e = scratch_ensure(g, need_scratch);
    if (sg_failed(e)) return e;

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            rc = (sg_err){"gpu: could not open a compute encoder"};
        } else {
            [enc setComputePipelineState:g->pipes[KI_ATTN_F16]];
            [enc setBuffer:qb->buf offset:(NSUInteger)qb->offset atIndex:0];
            [enc setBuffer:kb->buf offset:(NSUInteger)kb->offset atIndex:1];
            [enc setBuffer:vb->buf offset:(NSUInteger)vb->offset atIndex:2];
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:3];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:4];
            [enc setBuffer:g->scratch offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) {
                rc = gpu_errf("gpu: k_attn_decode_f16 failed: %s",
                              [[[cb error] localizedDescription] UTF8String]);
            }
        }
    }
    return rc;
}

/* One-shot dispatch for k_attn_prefill (Task M5.4), the same synchronous
 * commit-and-wait, three-device-input contract as sg_gpu_run_attn_decode_f16,
 * but over a CHUNK of n query tokens instead of one. q is f32
 * [n, n_heads, q_stride]; k and v are f16 [base+n, n_kv_heads, head_dim]
 * SEPARATE buffers (the sg_kv layout) that already hold base+n positions
 * (prior context in 0..base-1, the chunk's own K/V in base..base+n-1, stored
 * by the caller before this runs); out is f32 [n, n_heads, head_dim]. Each of
 * the n*n_heads threadgroups attends causally over the first base+t+1 cache
 * positions for its token t. Used by the per-op test; enc_attn_prefill
 * dispatches the same kernel by hand inside an open command buffer.
 *
 * params: [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=base [4]=n
 * [5]=q_stride [6]=softmax scale bits (params[7] unused). */
sg_err sg_gpu_run_attn_prefill(sg_gpu *g, void *q, void *k, void *v, void *out,
                               const uint32_t params[8]) {
    if (!g || !q || !k || !v || !out || !params) {
        return (sg_err){"gpu: sg_gpu_run_attn_prefill got a NULL argument"};
    }
    uint32_t n_heads = params[0], n_kv = params[1], hd = params[2], base = params[3];
    uint32_t n = params[4], q_stride = params[5];
    if (n_kv == 0 || n_heads % n_kv != 0) {
        return gpu_errf("gpu: k_attn_prefill n_heads %u is not a multiple of n_kv_heads %u",
                        n_heads, n_kv);
    }
    if (q_stride < hd) {
        return gpu_errf("gpu: k_attn_prefill q_stride %u is smaller than head_dim %u",
                        q_stride, hd);
    }
    if (n_heads == 0 || hd == 0 || n == 0) {
        return (sg_err){"gpu: k_attn_prefill dispatched with a zero dimension"};
    }

    sg_gpu_buf *qb = (sg_gpu_buf *)q, *kb = (sg_gpu_buf *)k, *vb = (sg_gpu_buf *)v,
               *ob = (sg_gpu_buf *)out;

    /* base+n is the total cache position count; the kernel carries seq and the
     * per-threadgroup score-row stride in 32-bit uint, and tg = token*n_heads+h
     * is a 32-bit threadgroup index, so both base+n and n*n_heads must fit u32
     * or host sizing and kernel indexing would disagree (breaking the causal
     * window and the per-threadgroup scratch isolation). Reject rather than
     * truncate; real callers are far under this (sg_kv caps positions at
     * SG_KV_CAP_MAX == 262144). */
    uint64_t seq_max = 0, t0 = 0, t1 = 0;
    if (!add_ck((uint64_t)base, n, &seq_max)) {
        return (sg_err){"gpu: k_attn_prefill base+n overflows 64 bits"};
    }
    if (seq_max > UINT32_MAX) {
        return (sg_err){"gpu: k_attn_prefill base+n exceeds the 32-bit position range"};
    }
    if ((uint64_t)n * n_heads > UINT32_MAX) {
        return (sg_err){"gpu: k_attn_prefill n*n_heads exceeds the 32-bit threadgroup range"};
    }

    /* q [n, n_heads, q_stride] f32; k/v [base+n, n_kv, hd] f16;
     * out [n, n_heads, hd] f32; scores [n*n_heads, base+n] f32. Every product
     * is u64-guarded: the byte counts are products of up to three caller u32s
     * and a wrapped-small `need` would let an undersized buffer through. */
    uint64_t need_q = 0, need_kv = 0, need_o = 0, need_scratch = 0;
    bool ok = mul_ck((uint64_t)n * n_heads, q_stride, &t0) && mul_ck(t0, 4, &need_q)
           && mul_ck(seq_max, n_kv, &t0) && mul_ck(t0, hd, &t1) && mul_ck(t1, 2, &need_kv)
           && mul_ck((uint64_t)n * n_heads, hd, &t0) && mul_ck(t0, 4, &need_o)
           && mul_ck((uint64_t)n * n_heads, seq_max, &t0) && mul_ck(t0, 4, &need_scratch);
    if (!ok) {
        return (sg_err){"gpu: k_attn_prefill params describe a region that overflows 64 bits"};
    }
    if (!buf_big_enough(qb, need_q)) {
        return gpu_errf("gpu: k_attn_prefill q is %llu bytes, needs %llu",
                        (unsigned long long)(qb ? qb->nbytes : 0), (unsigned long long)need_q);
    }
    if (!buf_big_enough(kb, need_kv)) {
        return gpu_errf("gpu: k_attn_prefill k is %llu bytes, needs %llu",
                        (unsigned long long)(kb ? kb->nbytes : 0), (unsigned long long)need_kv);
    }
    if (!buf_big_enough(vb, need_kv)) {
        return gpu_errf("gpu: k_attn_prefill v is %llu bytes, needs %llu",
                        (unsigned long long)(vb ? vb->nbytes : 0), (unsigned long long)need_kv);
    }
    if (!buf_big_enough(ob, need_o)) {
        return gpu_errf("gpu: k_attn_prefill out is %llu bytes, needs %llu",
                        (unsigned long long)(ob ? ob->nbytes : 0), (unsigned long long)need_o);
    }
    if (bufs_overlap(qb, ob) || bufs_overlap(kb, ob) || bufs_overlap(vb, ob)) {
        return (sg_err){"gpu: k_attn_prefill output overlaps an input buffer"};
    }

    sg_err e = scratch_ensure(g, need_scratch);
    if (sg_failed(e)) return e;

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            rc = (sg_err){"gpu: could not open a compute encoder"};
        } else {
            [enc setComputePipelineState:g->pipes[KI_ATTN_PREFILL]];
            [enc setBuffer:qb->buf offset:(NSUInteger)qb->offset atIndex:0];
            [enc setBuffer:kb->buf offset:(NSUInteger)kb->offset atIndex:1];
            [enc setBuffer:vb->buf offset:(NSUInteger)vb->offset atIndex:2];
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:3];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:4];
            [enc setBuffer:g->scratch offset:0 atIndex:5];
            /* One threadgroup per (query token, query head), same SG_TG width
             * as k_attn_decode_f16. */
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n * n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) {
                rc = gpu_errf("gpu: k_attn_prefill failed: %s",
                              [[[cb error] localizedDescription] UTF8String]);
            }
        }
    }
    return rc;
}

/* =====================================================================
 * P2.2: split-K decode attention one-shots
 * =====================================================================
 *
 * The host side of k_attn_decode_splitk_partial / _combine, and (P2.4) of
 * k_attn_decode_splitk_partial_gqa, which shares every rule below. Full
 * contract (layout, the shared params array, the partition rule, the seq == 0
 * decision) is in surge.h next to the declarations. None of the three is
 * reachable through sg_gpu_run_op -- six and four device buffers against its
 * (a, b, out) triple -- so these are their only one-shot entry points. They
 * commit and wait, so the decode path does not call them; it encodes the same
 * kernels by hand in enc_attn_splitk. */

/* One buffer-size check with the standard message. Factored out because the
 * partial dispatch below has six of them and the combine four, and six
 * copy-pasted gpu_errf calls are six chances to name the wrong buffer. */
static sg_err splitk_need(const char *kernel, const char *what,
                          const sg_gpu_buf *b, uint64_t need) {
    if (buf_big_enough(b, need)) return SG_OK;
    return gpu_errf("gpu: %s %s is %llu bytes, needs %llu", kernel, what,
                    (unsigned long long)(b ? b->nbytes : 0), (unsigned long long)need);
}

/* Every byte count the split-K pair needs, from the shared params array.
 * Each `need_*` may be NULL when that size is not wanted (the combine has no
 * q, k, v or scratch; the partial has no out). An overflow in ANY of them
 * fails the whole call even for a caller that did not ask for that size: the
 * params array is contracted (surge.h) to be valid for BOTH dispatches, so a
 * q/k/v extent that cannot exist is a bad array rather than a bad question.
 *
 * Guarded end to end with mul_ck for the reason stated above it: these are
 * products of up to three caller uint32 params (n_heads * n_splits * head_dim
 * is the largest), and a wrapped `need` would be SMALL, so an undersized
 * buffer would pass the checks below and the kernel would then index with the
 * original, unwrapped dimensions.
 *
 * `span` is ceil(seq / n_splits), the exact upper bound on one split's length
 * under the t0 = i*seq/n_splits partition rule (t1 - t0 <= floor(seq/n_splits)
 * + 1 for every i, with equality reachable), and it is computed here with the
 * IDENTICAL 64-bit expression k_attn_decode_splitk_partial uses to find its
 * private score row, so host sizing and kernel indexing cannot drift apart. */
static bool splitk_sizes(const uint32_t *p, uint64_t *need_q, uint64_t *need_kv,
                         uint64_t *need_ms, uint64_t *need_acc, uint64_t *need_out,
                         uint64_t *need_scratch) {
    const uint64_t f = 4;   /* sizeof(float) */
    uint64_t n_heads = p[0], n_kv = p[1], hd = p[2], seq = p[3];
    uint64_t q_stride = p[4], n_splits = p[6];
    uint64_t q = 0, kv = 0, ms = 0, acc = 0, out = 0, scratch = 0, parts = 0, t = 0;

    /* check_params rejects this first; guarded again here because the span
     * division below would be a divide by zero. */
    if (n_splits == 0) return false;
    uint64_t span = (seq + n_splits - 1) / n_splits;   /* both u32-derived, cannot wrap u64 */

    bool ok = mul_ck(n_heads, q_stride, &t) && mul_ck(t, f, &q)
           && mul_ck(seq, n_kv, &t) && mul_ck(t, hd, &t) && mul_ck(t, 2, &kv)
           && mul_ck(n_heads, n_splits, &parts) && mul_ck(parts, f, &ms)
           && mul_ck(parts, hd, &t) && mul_ck(t, f, &acc)
           && mul_ck(n_heads, hd, &t) && mul_ck(t, f, &out)
           && mul_ck(parts, span, &t) && mul_ck(t, f, &scratch);
    if (!ok) return false;

    /* Metal rejects a zero-length buffer, and the partial kernel's `scores`
     * argument has to be BOUND even in the one case where no thread touches
     * it (seq == 0 makes every split empty, so span is 0). Ask for one float
     * rather than nothing, so the binding stays valid. */
    if (scratch == 0) scratch = f;

    if (need_q) *need_q = q;
    if (need_kv) *need_kv = kv;
    if (need_ms) *need_ms = ms;
    if (need_acc) *need_acc = acc;
    if (need_out) *need_out = out;
    if (need_scratch) *need_scratch = scratch;
    return true;
}

/* The shared body of ALL THREE split-K partial one-shots (P2.4, extended by
 * P2.8). The per-head k_attn_decode_splitk_partial, the GQA-shared
 * k_attn_decode_splitk_partial_gqa and the online
 * k_attn_decode_splitk_partial_gqa_online take the same params array, the same
 * six device buffers and the same buffer sizes; they differ only in the
 * pipeline, in the grid's y extent (query heads against KV heads) and in
 * whether they need the score scratch at all. Factored rather than copied so
 * they cannot drift on a validation rule: every rejection surge.h documents is
 * checked HERE, once.
 *
 * `kn` is the kernel name for the error messages, `ki` its pipeline index,
 * `gqa` picks the grid, and `scratch` says whether buffer 7 exists on this
 * kernel. The arguments are already NULL-checked by the three public entry
 * points below, which do it under their own names. */
static sg_err splitk_partial_run(sg_gpu *g, const char *kn, int ki, bool gqa,
                                 bool scratch, void *q, void *k, void *v,
                                 void *m, void *s, void *acc,
                                 const uint32_t params[8]) {
    sg_err e = check_params(kn, params);
    if (sg_failed(e)) return e;

    /* The grid's y extent: one threadgroup per QUERY head for the per-head
     * partial, per KV head (i.e. per GQA group) for the GQA one. check_params
     * has already rejected n_kv_heads == 0 and n_heads % n_kv_heads != 0, so
     * the groups tile the query heads exactly. */
    uint32_t n_splits = params[6];
    uint32_t n_rows = gqa ? params[1] : params[0];
    uint64_t need_q = 0, need_kv = 0, need_ms = 0, need_acc = 0, need_scratch = 0;
    if (!splitk_sizes(params, &need_q, &need_kv, &need_ms, &need_acc, NULL, &need_scratch)) {
        return gpu_errf("gpu: %s params describe a region that overflows 64 bits", kn);
    }

    sg_gpu_buf *ins[3] = {(sg_gpu_buf *)q, (sg_gpu_buf *)k, (sg_gpu_buf *)v};
    sg_gpu_buf *outs[3] = {(sg_gpu_buf *)m, (sg_gpu_buf *)s, (sg_gpu_buf *)acc};
    static const char *const in_name[3] = {"q", "k", "v"};
    static const char *const out_name[3] = {"m", "s", "acc"};
    const uint64_t in_need[3] = {need_q, need_kv, need_kv};
    const uint64_t out_need[3] = {need_ms, need_ms, need_acc};

    for (int i = 0; i < 3; i++) {
        e = splitk_need(kn, in_name[i], ins[i], in_need[i]);
        if (sg_failed(e)) return e;
        e = splitk_need(kn, out_name[i], outs[i], out_need[i]);
        if (sg_failed(e)) return e;
    }

    /* surge.h forbids an output overlapping an input for the reason
     * sg_gpu_run_op states: a threadgroup that has already written its
     * partial would be changing an input another threadgroup has not read
     * yet, which is both wrong and NONDETERMINISTIC. The three OUTPUTS must
     * also be disjoint from each other here, which the (a, b, out) kernels
     * never have to say: m, s and acc are written by different threadgroups
     * at different strides, so an overlap between two of them is the same
     * race with a different name. */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (bufs_overlap(ins[i], outs[j])) {
                return gpu_errf("gpu: %s output %s overlaps input %s", kn,
                                out_name[j], in_name[i]);
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (bufs_overlap(outs[i], outs[j])) {
                return gpu_errf("gpu: %s outputs %s and %s overlap", kn,
                                out_name[i], out_name[j]);
            }
        }
    }

    /* g->splitk_scratch, NOT the process-wide g->scratch every other attention
     * kernel binds at offset 0 (P2.2 review finding 1). See the two comments
     * on struct sg_gpu: the shared buffer is grown and never partitioned, so
     * two different users encoded into ONE command buffer would be addressing
     * the same bytes, and this kernel is specifically meant to be dispatched
     * from the batched decode encoder alongside kernels that use it. Owning a
     * separate allocation makes that a non-question instead of a rule the next
     * task has to remember.
     *
     * P2.8's online partial has no score row at all, so it does not ask for the
     * allocation and does not bind it below. That is the whole memory saving of
     * the streaming form, and it is expressed as "this arm never mentions the
     * buffer" rather than as a smaller size request. */
    if (scratch) {
        e = splitk_scratch_ensure(g, need_scratch);
        if (sg_failed(e)) return e;
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            rc = (sg_err){"gpu: could not open a compute encoder"};
        } else {
            [enc setComputePipelineState:g->pipes[ki]];
            for (int i = 0; i < 3; i++) {
                [enc setBuffer:ins[i]->buf offset:(NSUInteger)ins[i]->offset atIndex:(NSUInteger)i];
                [enc setBuffer:outs[i]->buf offset:(NSUInteger)outs[i]->offset
                       atIndex:(NSUInteger)(3 + i)];
            }
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:6];
            /* The DEDICATED split-K scratch, not g->scratch (see above), and
             * only for the two kernels that declare it: the online partial has
             * seven bindings and binding an eighth would be a claim about a
             * buffer it does not have. */
            if (scratch) [enc setBuffer:g->splitk_scratch offset:0 atIndex:7];
            /* The SG_K_HEADS2D grid: x = split, y = query head (per-head
             * partial) or KV head (GQA partial), matching the kernel's
             * tg.x / tg.y. Both grid attributes are declared uint2 there, the
             * same pairing the SG_K_TILES2D dispatch above uses and for the
             * same stated reason (the two position attributes must agree on
             * their vector width), so the kernel reads its lane index out of
             * tid.x of an SG_TG x 1 threadgroup. Total threads per threadgroup
             * is still exactly SG_TG, which is what the fixed tg_max/tg_sum
             * fold trees are shaped for. */
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_splits, (NSUInteger)n_rows, 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) {
                rc = gpu_errf("gpu: %s failed: %s", kn,
                              [[[cb error] localizedDescription] UTF8String]);
            }
        }
    }
    return rc;
}

sg_err sg_gpu_run_attn_splitk_partial(sg_gpu *g, void *q, void *k, void *v,
                                      void *m, void *s, void *acc,
                                      const uint32_t params[8]) {
    if (!g || !q || !k || !v || !m || !s || !acc || !params) {
        return (sg_err){"gpu: sg_gpu_run_attn_splitk_partial got a NULL argument"};
    }
    return splitk_partial_run(g, "k_attn_decode_splitk_partial",
                              KI_ATTN_SPLITK_PARTIAL, false, true,
                              q, k, v, m, s, acc, params);
}

/* P2.4. The GQA-shared twin: identical arguments, identical validation,
 * identical output bytes, one threadgroup per GQA group instead of per query
 * head. Separate entry point rather than a flag in the params array because
 * the params array is contracted (surge.h) to serve the combine dispatch too,
 * and because the per-op gate's whole job is to run the TWO of them on the
 * same inputs and memcmp the results. */
sg_err sg_gpu_run_attn_splitk_partial_gqa(sg_gpu *g, void *q, void *k, void *v,
                                          void *m, void *s, void *acc,
                                          const uint32_t params[8]) {
    if (!g || !q || !k || !v || !m || !s || !acc || !params) {
        return (sg_err){"gpu: sg_gpu_run_attn_splitk_partial_gqa got a NULL argument"};
    }
    return splitk_partial_run(g, "k_attn_decode_splitk_partial_gqa",
                              KI_ATTN_SPLITK_PARTIAL_GQA, true, true,
                              q, k, v, m, s, acc, params);
}

/* P2.8. The ONLINE-SOFTMAX twin of the GQA partial: identical arguments,
 * identical validation, identical output LAYOUT, and (unlike the pair above)
 * NOT identical output bytes, because streaming reorders the exponential sums.
 * See surge.h for what is and is not promised, and note what this entry point
 * does NOT touch: g->splitk_scratch is neither grown nor bound, so this is also
 * the dispatch that proves the streaming form needs no score row.
 *
 * Separate entry point rather than a flag in the params array for P2.4's
 * reason: the array is contracted to serve the combine dispatch too, and the
 * per-op gate's job is to run the arms on the SAME inputs and compare. */
sg_err sg_gpu_run_attn_splitk_partial_gqa_online(sg_gpu *g, void *q, void *k, void *v,
                                                 void *m, void *s, void *acc,
                                                 const uint32_t params[8]) {
    if (!g || !q || !k || !v || !m || !s || !acc || !params) {
        return (sg_err){"gpu: sg_gpu_run_attn_splitk_partial_gqa_online got a NULL argument"};
    }
    return splitk_partial_run(g, "k_attn_decode_splitk_partial_gqa_online",
                              KI_ATTN_SPLITK_PARTIAL_GQA_ONLINE, true, false,
                              q, k, v, m, s, acc, params);
}

sg_err sg_gpu_run_attn_splitk_combine(sg_gpu *g, void *m, void *s, void *acc,
                                      void *out, const uint32_t params[8]) {
    const char *kn = "k_attn_decode_splitk_combine";
    if (!g || !m || !s || !acc || !out || !params) {
        return (sg_err){"gpu: sg_gpu_run_attn_splitk_combine got a NULL argument"};
    }
    sg_err e = check_params(kn, params);
    if (sg_failed(e)) return e;

    uint32_t n_heads = params[0];
    uint64_t need_ms = 0, need_acc = 0, need_out = 0;
    if (!splitk_sizes(params, NULL, NULL, &need_ms, &need_acc, &need_out, NULL)) {
        return gpu_errf("gpu: %s params describe a region that overflows 64 bits", kn);
    }

    sg_gpu_buf *ins[3] = {(sg_gpu_buf *)m, (sg_gpu_buf *)s, (sg_gpu_buf *)acc};
    sg_gpu_buf *ob = (sg_gpu_buf *)out;
    static const char *const in_name[3] = {"m", "s", "acc"};
    const uint64_t in_need[3] = {need_ms, need_ms, need_acc};

    for (int i = 0; i < 3; i++) {
        e = splitk_need(kn, in_name[i], ins[i], in_need[i]);
        if (sg_failed(e)) return e;
    }
    e = splitk_need(kn, "out", ob, need_out);
    if (sg_failed(e)) return e;

    for (int i = 0; i < 3; i++) {
        if (bufs_overlap(ins[i], ob)) {
            return gpu_errf("gpu: %s output overlaps input %s", kn, in_name[i]);
        }
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) {
            rc = (sg_err){"gpu: could not open a compute encoder"};
        } else {
            [enc setComputePipelineState:g->pipes[KI_ATTN_SPLITK_COMBINE]];
            for (int i = 0; i < 3; i++) {
                [enc setBuffer:ins[i]->buf offset:(NSUInteger)ins[i]->offset atIndex:(NSUInteger)i];
            }
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:3];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:4];
            /* One SG_TG-wide threadgroup per query head, a plain 1D grid: the
             * combine folds n_splits triples per head and has no second
             * dimension to spread. */
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) {
                rc = gpu_errf("gpu: %s failed: %s", kn,
                              [[[cb error] localizedDescription] UTF8String]);
            }
        }
    }
    return rc;
}

/* =====================================================================
 * M5.5: gated-DeltaNet chunked-scan prefill one-shots
 * =====================================================================
 *
 * One synchronous commit-and-wait per call, the same contract as
 * sg_gpu_run_attn_prefill, extended to the extra device buffer each M5.5 kernel
 * needs (the state carrier, or the shared weight). These are the per-op test's
 * entry points; enc_gdn_prefill dispatches the same kernels by hand inside one
 * open command buffer. Every byte count is u64-guarded (mul_ck/add_ck): the
 * sizes are products of up to three caller u32s and a wrapped-small `need` would
 * let an undersized buffer through to a device-side out-of-bounds read. */

/* One dispatch of an elementwise (no-reduction) kernel over `elems` threads,
 * threadgroup width clamped to the pipeline max, SG_TG and `elems`. */
static NSUInteger gpu_elem_width(sg_gpu *g, int ki, uint64_t elems) {
    NSUInteger w = [g->pipes[ki] maxTotalThreadsPerThreadgroup];
    if (w > SG_TG) w = SG_TG;
    if (w > elems) w = (NSUInteger)elems;
    return w;
}

/* k_conv1d_chunk: x [n_tok, channels] f32, w [channels, ksize] f32,
 * out [n_tok, channels] f32, state [ksize-1, channels] f32 (in AND out, the
 * conv tail). params: [0]=channels [1]=ksize [2]=n_tok. */
sg_err sg_gpu_run_conv1d_chunk(sg_gpu *g, void *x, void *w, void *out, void *state,
                               const uint32_t params[8]) {
    if (!g || !x || !w || !out || !state || !params) {
        return (sg_err){"gpu: sg_gpu_run_conv1d_chunk got a NULL argument"};
    }
    uint32_t channels = params[0], ksize = params[1], n_tok = params[2];
    if (channels == 0 || ksize == 0 || n_tok == 0) {
        return (sg_err){"gpu: k_conv1d_chunk dispatched with a zero dimension"};
    }

    sg_gpu_buf *xb = (sg_gpu_buf *)x, *wb = (sg_gpu_buf *)w, *ob = (sg_gpu_buf *)out,
               *sb = (sg_gpu_buf *)state;
    uint64_t f = 4, need_x = 0, need_w = 0, need_s = 0;
    bool ok = mul_ck((uint64_t)n_tok * channels, f, &need_x)
           && mul_ck((uint64_t)channels * ksize, f, &need_w)
           && mul_ck((uint64_t)(ksize - 1u) * channels, f, &need_s);
    if (!ok) return (sg_err){"gpu: k_conv1d_chunk params describe a region that overflows 64 bits"};
    uint64_t need_o = need_x;
    if (!buf_big_enough(xb, need_x)) return gpu_errf("gpu: k_conv1d_chunk x is %llu bytes, needs %llu",
        (unsigned long long)(xb ? xb->nbytes : 0), (unsigned long long)need_x);
    if (!buf_big_enough(wb, need_w)) return gpu_errf("gpu: k_conv1d_chunk w is %llu bytes, needs %llu",
        (unsigned long long)(wb ? wb->nbytes : 0), (unsigned long long)need_w);
    if (!buf_big_enough(ob, need_o)) return gpu_errf("gpu: k_conv1d_chunk out is %llu bytes, needs %llu",
        (unsigned long long)(ob ? ob->nbytes : 0), (unsigned long long)need_o);
    if (need_s > 0 && !buf_big_enough(sb, need_s)) return gpu_errf("gpu: k_conv1d_chunk state is %llu bytes, needs %llu",
        (unsigned long long)(sb ? sb->nbytes : 0), (unsigned long long)need_s);
    /* out must not alias any input; state is written in place (thread c owns
     * column c of it), so it must not overlap out NOR either read-only input --
     * a thread rewriting the tail could otherwise clobber an x or w value
     * another thread or a later token still has to read. */
    if (bufs_overlap(ob, xb) || bufs_overlap(ob, wb) || bufs_overlap(ob, sb)
        || bufs_overlap(sb, xb) || bufs_overlap(sb, wb)) {
        return (sg_err){"gpu: k_conv1d_chunk output or state overlaps another buffer"};
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) { rc = (sg_err){"gpu: could not open a compute encoder"}; }
        else {
            [enc setComputePipelineState:g->pipes[KI_CONV1D_CHUNK]];
            [enc setBuffer:xb->buf offset:(NSUInteger)xb->offset atIndex:0];
            [enc setBuffer:wb->buf offset:(NSUInteger)wb->offset atIndex:1];
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:2];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:3];
            [enc setBuffer:sb->buf offset:(NSUInteger)sb->offset atIndex:4];
            NSUInteger wdt = gpu_elem_width(g, KI_CONV1D_CHUNK, channels);
            [enc dispatchThreads:MTLSizeMake(channels, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(wdt, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) rc = gpu_errf("gpu: k_conv1d_chunk failed: %s",
                                          [[[cb error] localizedDescription] UTF8String]);
        }
    }
    return rc;
}

/* k_delta_gates_chunk: a [n_tok, n] f32, b [n_tok, n] f32, gates [n_tok, 2n] f32
 * out, adt [ssm_a(n), dt_bias(n)] f32. params: [0]=n [1]=neg_exp [2]=n_tok. */
sg_err sg_gpu_run_delta_gates_chunk(sg_gpu *g, void *a, void *b, void *gates, void *adt,
                                    const uint32_t params[8]) {
    if (!g || !a || !b || !gates || !adt || !params) {
        return (sg_err){"gpu: sg_gpu_run_delta_gates_chunk got a NULL argument"};
    }
    uint32_t n = params[0], n_tok = params[2];
    if (n == 0 || n_tok == 0) {
        return (sg_err){"gpu: k_delta_gates_chunk dispatched with a zero dimension"};
    }
    /* The grid is one thread per (token, head), and the kernel carries n_tok*n
     * in a 32-bit thread index; reject rather than truncate. */
    if ((uint64_t)n_tok * n > UINT32_MAX) {
        return (sg_err){"gpu: k_delta_gates_chunk n_tok*n exceeds the 32-bit grid range"};
    }

    sg_gpu_buf *ab = (sg_gpu_buf *)a, *bb = (sg_gpu_buf *)b, *gb = (sg_gpu_buf *)gates,
               *db = (sg_gpu_buf *)adt;
    uint64_t f = 4, need_ab = 0, need_g = 0, need_adt = 0;
    bool ok = mul_ck((uint64_t)n_tok * n, f, &need_ab)
           && mul_ck(2ull * (uint64_t)n_tok * n, f, &need_g)
           && mul_ck(2ull * n, f, &need_adt);
    if (!ok) return (sg_err){"gpu: k_delta_gates_chunk params describe a region that overflows 64 bits"};
    if (!buf_big_enough(ab, need_ab)) return gpu_errf("gpu: k_delta_gates_chunk a is %llu bytes, needs %llu",
        (unsigned long long)(ab ? ab->nbytes : 0), (unsigned long long)need_ab);
    if (!buf_big_enough(bb, need_ab)) return gpu_errf("gpu: k_delta_gates_chunk b is %llu bytes, needs %llu",
        (unsigned long long)(bb ? bb->nbytes : 0), (unsigned long long)need_ab);
    if (!buf_big_enough(gb, need_g)) return gpu_errf("gpu: k_delta_gates_chunk gates is %llu bytes, needs %llu",
        (unsigned long long)(gb ? gb->nbytes : 0), (unsigned long long)need_g);
    if (!buf_big_enough(db, need_adt)) return gpu_errf("gpu: k_delta_gates_chunk adt is %llu bytes, needs %llu",
        (unsigned long long)(db ? db->nbytes : 0), (unsigned long long)need_adt);
    if (bufs_overlap(gb, ab) || bufs_overlap(gb, bb) || bufs_overlap(gb, db)) {
        return (sg_err){"gpu: k_delta_gates_chunk output overlaps an input buffer"};
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) { rc = (sg_err){"gpu: could not open a compute encoder"}; }
        else {
            uint64_t elems = (uint64_t)n_tok * n;
            [enc setComputePipelineState:g->pipes[KI_DELTA_GATES_CHUNK]];
            [enc setBuffer:ab->buf offset:(NSUInteger)ab->offset atIndex:0];
            [enc setBuffer:bb->buf offset:(NSUInteger)bb->offset atIndex:1];
            [enc setBuffer:gb->buf offset:(NSUInteger)gb->offset atIndex:2];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:3];
            [enc setBuffer:db->buf offset:(NSUInteger)db->offset atIndex:4];
            NSUInteger wdt = gpu_elem_width(g, KI_DELTA_GATES_CHUNK, elems);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)elems, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(wdt, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) rc = gpu_errf("gpu: k_delta_gates_chunk failed: %s",
                                          [[[cb error] localizedDescription] UTF8String]);
        }
    }
    return rc;
}

/* Shared validation + dispatch for the two SG_TG-wide DeltaNet chunk kernels
 * that read a qkv chunk and a gates chunk and update S in place: k_delta_chunk
 * (n_tok in params[6], conv_dim in params[7]) and k_delta_multi (single token,
 * params[6]/params[7] unused). `n_tok_eff` folds the two: k_delta_multi is the
 * n_tok == 1 case with per-token stride conv_dim = 2*key_dim + value_dim. */
static sg_err gpu_run_delta_common(sg_gpu *g, int ki, void *S, void *qkv, void *out,
                                   void *gates, uint32_t dk, uint32_t dv, uint32_t n_v,
                                   uint32_t n_k, uint32_t key_dim, uint32_t n_tok,
                                   uint32_t conv_dim, const uint32_t params[8]) {
    if (dk == 0 || dv == 0 || n_v == 0 || n_tok == 0) {
        return (sg_err){"gpu: k_delta_* dispatched with a zero dimension"};
    }
    if (n_k == 0 || n_v % n_k != 0) {
        return gpu_errf("gpu: k_delta_* n_v %u is not a multiple of n_k %u", n_v, n_k);
    }
    /* The kernel reads q at qkv+hk*dk and k at qkv+key_dim+hk*dk with hk < n_k,
     * so the q/k regions run up to n_k*dk and key_dim+n_k*dk; key_dim must be at
     * least n_k*dk or an inconsistent public param would index a k slice past
     * the region the conv_dim size rule guarantees. */
    if (key_dim < (uint64_t)n_k * dk) {
        return gpu_errf("gpu: k_delta_* key_dim %u is smaller than n_k*dk %llu",
                        key_dim, (unsigned long long)((uint64_t)n_k * dk));
    }
    uint64_t value_dim = (uint64_t)n_v * dv;
    if (conv_dim < 2ull * key_dim + value_dim) {
        return gpu_errf("gpu: k_delta_* conv_dim %u is smaller than 2*key_dim + value_dim %llu",
                        conv_dim, (unsigned long long)(2ull * key_dim + value_dim));
    }

    sg_gpu_buf *Sb = (sg_gpu_buf *)S, *qb = (sg_gpu_buf *)qkv, *ob = (sg_gpu_buf *)out,
               *gb = (sg_gpu_buf *)gates;
    /* Every product goes through mul_ck: n_tok, n_v, dv are caller u32s and a
     * bare (2*n_tok*n_v) or (n_tok*value_dim) can wrap u64 before the check. */
    uint64_t f = 4, need_s = 0, need_q = 0, need_o = 0, need_g = 0, t0 = 0;
    bool ok = mul_ck((uint64_t)n_v * dv, dk, &t0) && mul_ck(t0, f, &need_s)
           && mul_ck((uint64_t)n_tok, conv_dim, &t0) && mul_ck(t0, f, &need_q)
           && mul_ck((uint64_t)n_tok, value_dim, &t0) && mul_ck(t0, f, &need_o)
           && mul_ck((uint64_t)n_tok, n_v, &t0) && mul_ck(t0, 2, &t0) && mul_ck(t0, f, &need_g);
    if (!ok) return (sg_err){"gpu: k_delta_* params describe a region that overflows 64 bits"};
    if (!buf_big_enough(Sb, need_s)) return gpu_errf("gpu: k_delta_* S is %llu bytes, needs %llu",
        (unsigned long long)(Sb ? Sb->nbytes : 0), (unsigned long long)need_s);
    if (!buf_big_enough(qb, need_q)) return gpu_errf("gpu: k_delta_* qkv is %llu bytes, needs %llu",
        (unsigned long long)(qb ? qb->nbytes : 0), (unsigned long long)need_q);
    if (!buf_big_enough(ob, need_o)) return gpu_errf("gpu: k_delta_* out is %llu bytes, needs %llu",
        (unsigned long long)(ob ? ob->nbytes : 0), (unsigned long long)need_o);
    if (!buf_big_enough(gb, need_g)) return gpu_errf("gpu: k_delta_* gates is %llu bytes, needs %llu",
        (unsigned long long)(gb ? gb->nbytes : 0), (unsigned long long)need_g);
    /* out must not alias any input; S is read+written in place (row-private per
     * thread), so it must not overlap out NOR the read-only qkv/gates a thread
     * still reads while updating its rows. */
    if (bufs_overlap(ob, Sb) || bufs_overlap(ob, qb) || bufs_overlap(ob, gb)
        || bufs_overlap(Sb, qb) || bufs_overlap(Sb, gb)) {
        return (sg_err){"gpu: k_delta_* output or S overlaps another buffer"};
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) { rc = (sg_err){"gpu: could not open a compute encoder"}; }
        else {
            [enc setComputePipelineState:g->pipes[ki]];
            [enc setBuffer:Sb->buf offset:(NSUInteger)Sb->offset atIndex:0];
            [enc setBuffer:qb->buf offset:(NSUInteger)qb->offset atIndex:1];
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:2];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:3];
            [enc setBuffer:gb->buf offset:(NSUInteger)gb->offset atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_v, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) rc = gpu_errf("gpu: %s failed: %s", SG_KERNELS[ki].name,
                                          [[[cb error] localizedDescription] UTF8String]);
        }
    }
    return rc;
}

/* k_delta_chunk: S [n_v, dv, dk] f32 (in+out), qkv [n_tok, conv_dim] f32,
 * out [n_tok, value_dim] f32, gates [n_tok, 2*n_v] f32. params: [0]=dk [1]=dv
 * [2]=n_v [3]=n_k [4]=key_dim [5]=tiled [6]=n_tok [7]=conv_dim. */
sg_err sg_gpu_run_delta_chunk(sg_gpu *g, void *S, void *qkv, void *out, void *gates,
                              const uint32_t params[8]) {
    if (!g || !S || !qkv || !out || !gates || !params) {
        return (sg_err){"gpu: sg_gpu_run_delta_chunk got a NULL argument"};
    }
    return gpu_run_delta_common(g, KI_DELTA_CHUNK, S, qkv, out, gates,
                                params[0], params[1], params[2], params[3],
                                params[4], params[6], params[7], params);
}

/* k_delta_multi (single token), the per-op oracle for k_delta_chunk. S
 * [n_v, dv, dk] f32 (in+out), qkv [conv_dim] f32 (conv_dim = 2*key_dim +
 * value_dim), out [value_dim] f32, gates [2*n_v] f32. params: [0]=dk [1]=dv
 * [2]=n_v [3]=n_k [4]=key_dim [5]=tiled. */
sg_err sg_gpu_run_delta_multi(sg_gpu *g, void *S, void *qkv, void *out, void *gates,
                              const uint32_t params[8]) {
    if (!g || !S || !qkv || !out || !gates || !params) {
        return (sg_err){"gpu: sg_gpu_run_delta_multi got a NULL argument"};
    }
    uint32_t key_dim = params[4];
    uint64_t conv_dim = 2ull * key_dim + (uint64_t)params[2] * params[1];
    if (conv_dim > UINT32_MAX) {
        return (sg_err){"gpu: k_delta_multi conv_dim exceeds the 32-bit range"};
    }
    return gpu_run_delta_common(g, KI_DELTA_MULTI, S, qkv, out, gates,
                                params[0], params[1], params[2], params[3],
                                key_dim, 1u, (uint32_t)conv_dim, params);
}

/* k_rmsnorm_gated_chunk: y [n_tok, heads*dv] f32, z [n_tok, heads*dv] f32,
 * out [n_tok, heads*dv] f32, w [dv] f32 (shared norm weight). params: [0]=dv
 * [1]=heads [2]=eps bits [3]=n_tok. */
sg_err sg_gpu_run_rmsnorm_gated_chunk(sg_gpu *g, void *y, void *z, void *out, void *w,
                                      const uint32_t params[8]) {
    if (!g || !y || !z || !out || !w || !params) {
        return (sg_err){"gpu: sg_gpu_run_rmsnorm_gated_chunk got a NULL argument"};
    }
    uint32_t dv = params[0], heads = params[1], n_tok = params[3];
    if (dv == 0 || heads == 0 || n_tok == 0) {
        return (sg_err){"gpu: k_rmsnorm_gated_chunk dispatched with a zero dimension"};
    }
    /* One threadgroup per (token, head); tg is a 32-bit threadgroup index. */
    if ((uint64_t)n_tok * heads > UINT32_MAX) {
        return (sg_err){"gpu: k_rmsnorm_gated_chunk n_tok*heads exceeds the 32-bit range"};
    }

    sg_gpu_buf *yb = (sg_gpu_buf *)y, *zb = (sg_gpu_buf *)z, *ob = (sg_gpu_buf *)out,
               *wb = (sg_gpu_buf *)w;
    uint64_t f = 4, need_yzo = 0, need_w = 0, t0 = 0;
    bool ok = mul_ck((uint64_t)n_tok * heads, dv, &t0) && mul_ck(t0, f, &need_yzo)
           && mul_ck((uint64_t)dv, f, &need_w);
    if (!ok) return (sg_err){"gpu: k_rmsnorm_gated_chunk params describe a region that overflows 64 bits"};
    if (!buf_big_enough(yb, need_yzo)) return gpu_errf("gpu: k_rmsnorm_gated_chunk y is %llu bytes, needs %llu",
        (unsigned long long)(yb ? yb->nbytes : 0), (unsigned long long)need_yzo);
    if (!buf_big_enough(zb, need_yzo)) return gpu_errf("gpu: k_rmsnorm_gated_chunk z is %llu bytes, needs %llu",
        (unsigned long long)(zb ? zb->nbytes : 0), (unsigned long long)need_yzo);
    if (!buf_big_enough(ob, need_yzo)) return gpu_errf("gpu: k_rmsnorm_gated_chunk out is %llu bytes, needs %llu",
        (unsigned long long)(ob ? ob->nbytes : 0), (unsigned long long)need_yzo);
    if (!buf_big_enough(wb, need_w)) return gpu_errf("gpu: k_rmsnorm_gated_chunk w is %llu bytes, needs %llu",
        (unsigned long long)(wb ? wb->nbytes : 0), (unsigned long long)need_w);
    /* out may alias y (each thread rewrites only the elements it read, after
     * tg_sum's trailing barrier), exactly as k_rmsnorm_gated is used in place in
     * enc_gdn; it must not alias z or w. */
    if (bufs_overlap(ob, zb) || bufs_overlap(ob, wb)) {
        return (sg_err){"gpu: k_rmsnorm_gated_chunk output overlaps z or w"};
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) { rc = (sg_err){"gpu: could not open a compute encoder"}; }
        else {
            uint64_t groups = (uint64_t)n_tok * heads;
            [enc setComputePipelineState:g->pipes[KI_RMSNORM_GATED_CHUNK]];
            [enc setBuffer:yb->buf offset:(NSUInteger)yb->offset atIndex:0];
            [enc setBuffer:zb->buf offset:(NSUInteger)zb->offset atIndex:1];
            [enc setBuffer:ob->buf offset:(NSUInteger)ob->offset atIndex:2];
            [enc setBytes:params length:8 * sizeof(uint32_t) atIndex:3];
            [enc setBuffer:wb->buf offset:(NSUInteger)wb->offset atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) rc = gpu_errf("gpu: k_rmsnorm_gated_chunk failed: %s",
                                          [[[cb error] localizedDescription] UTF8String]);
        }
    }
    return rc;
}

/* =====================================================================
 * The full hybrid decode path (Task 10)
 * =====================================================================
 *
 * sg_gpu_forward encodes EVERY layer of the model into one command buffer
 * per token, commits it once and waits once. That is the whole reason this
 * exists: the same work through sg_gpu_run_op is ~540 commit/wait round
 * trips per token on the 2B, which is roughly 200x the GPU time.
 *
 * WHAT RUNS ON THE HOST, AND WHY. Exactly two things, and neither is a
 * numerical compromise:
 *
 *   - the embedding lookup, because the token id is known before the command
 *     buffer is opened and a gather kernel would only reproduce ref.c's
 *     wrow() less exactly;
 *   - the RoPE cos/sin table, computed in DOUBLE at the current position and
 *     uploaded as f32, exactly as Task 9's per-op test does. Metal has no
 *     f64 and the f32-rounded angle is 8e-3 wrong at position 262143, so
 *     this is the one place a kernel cannot be trusted with the arithmetic.
 *
 * The DeltaNet gates (beta and the decay) are the interesting case: ref.c
 * computes them on the CPU from two matvec outputs, which here are produced
 * on the GPU mid-layer. Reading them back would mean a commit-and-wait
 * inside every one of the 18 DeltaNet layers, so k_delta_gates computes them
 * on device instead. That is a deliberate f64 -> f32 step; see the task
 * report for the measured consequence.
 *
 * IN-PLACE OPS. sg_gpu_run_op forbids `out` aliasing an input because a
 * threadgroup could then overwrite a row another threadgroup has not read.
 * The batched path uses in-place forms where that cannot happen, and only
 * there: k_rmsnorm / k_rmsnorm_heads / k_rmsnorm_gated (thread lid writes
 * only elements it alone read, after tg_sum's trailing barrier),
 * k_rope_heads, k_scale, k_silu, k_swiglu, k_gate_sigmoid_strided and k_add
 * (all one-thread-per-output-element with no cross-thread reads). No
 * reduction kernel is ever asked to alias across threadgroups.
 *
 * ORDERING between dispatches is Metal's: computeCommandEncoder defaults to
 * MTLDispatchTypeSerial, which runs dispatches in encode order with an
 * implicit barrier between them. Nothing here would be correct under
 * MTLDispatchTypeConcurrent, so the encoder must not be created with it.
 */

#define SG_KV_GROUPS 2u   /* the kv cache holds K then V in one buffer */

static uint32_t fbits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* Zero-padded params array as a call argument. Every kernel reads at most
 * seven of the eight slots; the rest must still be defined, because
 * setBytes: uploads all 32 bytes. */
#define PARAMS(...) ((const uint32_t[8]){ __VA_ARGS__ })

static float gpu_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* IEEE binary16 -> f32, bit-identical to ref.c's f16_to_f32. Used only to
 * decode a Q8_0 block scale in the host-side embedding lookup, so the
 * embedding row the Metal path feeds in matches the CPU reference exactly. */
static float gpu_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp_bits = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp_bits == 0) {
        if (mant == 0) {
            bits = sign; /* +-0 */
        } else {
            int shift = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; shift++; }
            mant &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - shift + 1) << 23) | (mant << 13);
        }
    } else if (exp_bits == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / NaN */
    } else {
        bits = sign | ((exp_bits + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* The matvec kernel for a weight tensor of the given dtype. All of a model's
 * matmul weights share one dtype: sg_model_from_gguf requires every matmul
 * tensor -- and output.weight -- to equal token_embd's type, and the
 * safetensors loader reports a single wtype, so the decode encoder selects the
 * kernel once from m->wtype rather than per call site (a per-weight lookup
 * would return this same kernel at every matmul dispatch). This is the
 * per-tensor dispatch the M3 plan asks for, collapsed to one selection because
 * the loader guarantees the matmul weights are dtype-uniform. */
static int matmul_kernel_for(sg_tensor_type t) {
    switch (t) {
    case SG_T_Q8_0: return KI_MATVEC_Q8;
    case SG_T_BF16: return KI_MATVEC_BF16;
    default:        return KI_MATVEC_F32;   /* SG_T_F32 */
    }
}

/* The tiled-GEMM (M5.3) analog of matmul_kernel_for, for the M5.4 prefill
 * path: it projects a whole CHUNK of tokens through one weight matrix at once
 * (Y[N,M] = X[N,K] @ W[M,K]^T) rather than one token per matvec. Same
 * per-tensor dtype dispatch, same one-selection-per-model reasoning as
 * matmul_kernel_for (the loader guarantees the matmul weights are
 * dtype-uniform), just the batched kernel family. */
static int gemm_kernel_for(sg_tensor_type t) {
    switch (t) {
    case SG_T_Q8_0: return KI_MATMUL_Q8;
    case SG_T_BF16: return KI_MATMUL_BF16;
    default:        return KI_MATMUL_F32;   /* SG_T_F32 */
    }
}

/* ref.c's wwiden: widen a small (non-matmul) tensor into owned f32 storage,
 * adding the residual-norm shift in f32 AFTER widening. Same order, same
 * roundings, so a norm weight is bit-identical on the two paths. */
static void gpu_widen(const void *w, sg_tensor_type t, float *out, uint64_t n, float shift) {
    if (!w || !out) return;
    if (t == SG_T_BF16) {
        const uint16_t *b = (const uint16_t *)w;
        for (uint64_t i = 0; i < n; i++) out[i] = gpu_bf16_to_f32(b[i]) + shift;
    } else {
        const float *f = (const float *)w;
        for (uint64_t i = 0; i < n; i++) out[i] = f[i] + shift;
    }
}

/* ref.c's wrow for the three dtypes this path accepts. The Q8_0 branch
 * mirrors ref.c's wrow exactly (same f16 scale decode, same scale*int8 in
 * f32), so the embedding row is bit-identical to the CPU reference. */
static void gpu_embed_row(const void *w, sg_tensor_type t, uint64_t row,
                          uint32_t cols, float *out) {
    if (t == SG_T_BF16) {
        const uint16_t *b = (const uint16_t *)w + row * cols;
        for (uint32_t i = 0; i < cols; i++) out[i] = gpu_bf16_to_f32(b[i]);
    } else if (t == SG_T_Q8_0) {
        /* Q8_0 rows are whole 32-element blocks (gpu_check_model rejects a
         * hidden size that is not a multiple of 32), row `row` starts at byte
         * row*(cols/32)*34, each block is a little-endian f16 scale then 32
         * signed int8. */
        uint64_t blocks = cols / 32u;
        const uint8_t *p = (const uint8_t *)w + row * blocks * 34u;
        for (uint64_t b = 0; b < blocks; b++) {
            const uint8_t *blk = p + b * 34u;
            uint16_t sbits = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float d = gpu_f16_to_f32(sbits);
            const int8_t *q = (const int8_t *)(blk + 2);
            for (uint32_t i = 0; i < 32; i++) out[b * 32 + i] = d * (float)q[i];
        }
    } else {
        memcpy(out, (const float *)w + row * cols, (size_t)cols * sizeof *out);
    }
}

static id<MTLBuffer> bufof(void *h) { return h ? ((sg_gpu_buf *)h)->buf : nil; }
static uint64_t offof(void *h) { return h ? ((sg_gpu_buf *)h)->offset : 0; }

/* --------------------------------------------------------------------
 * Encoding
 * -------------------------------------------------------------------- */

typedef struct {
    sg_gpu *g;
    id<MTLComputeCommandEncoder> enc;
} sg_enc;

/* One dispatch into an already-open encoder. `ao`/`bo`/`oo` are offsets in
 * FLOATS from the start of the handle's data, which is what every buffer in
 * the decode path is made of; the wrapped bf16 weights are always bound at
 * 0. `aux` is buffer(4): k_attn_decode's score scratch or k_delta_multi's
 * gate vector, nil for everything else. */
static void enc_op(sg_enc *E, int ki, void *a, uint64_t ao, void *b, uint64_t bo,
                   void *o, uint64_t oo, id<MTLBuffer> aux, uint64_t auxoff,
                   const uint32_t *p) {
    sg_gpu *g = E->g;
    id<MTLComputeCommandEncoder> e = E->enc;

    [e setComputePipelineState:g->pipes[ki]];
    [e setBuffer:bufof(a) offset:(NSUInteger)(offof(a) + ao * 4) atIndex:0];
    /* Buffer 1 is declared by every kernel signature, so it must be bound
     * even where the kernel ignores it; rebind `a` in that case. */
    if (b) {
        [e setBuffer:bufof(b) offset:(NSUInteger)(offof(b) + bo * 4) atIndex:1];
    } else {
        [e setBuffer:bufof(a) offset:(NSUInteger)(offof(a) + ao * 4) atIndex:1];
    }
    [e setBuffer:bufof(o) offset:(NSUInteger)(offof(o) + oo * 4) atIndex:2];
    [e setBytes:p length:8 * sizeof(uint32_t) atIndex:3];
    if (aux) [e setBuffer:aux offset:(NSUInteger)auxoff atIndex:4];

    uint64_t groups = 1, elems = 0;
    gpu_grid(SG_KERNELS[ki].kind, p, &groups, &elems);
    if (elems != 0) {
        NSUInteger w = [g->pipes[ki] maxTotalThreadsPerThreadgroup];
        if (w > SG_TG) w = SG_TG;
        if (w > elems) w = (NSUInteger)elems;
        [e dispatchThreads:MTLSizeMake((NSUInteger)elems, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    } else if (groups != 0) {
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
    }
}

/* Casts n f32 values out of `src` (offset 0, always a whole small scratch
 * buffer here) into `dst` (an sg_kv K or V buffer) at element offset
 * `dst_off`. dst is HALF-typed storage, so its byte offset is dst_off * 2,
 * not the *4 enc_op assumes for its all-f32 buffers -- that mismatch is
 * exactly why this is a standalone dispatch rather than a call to enc_op. */
static void enc_kv_store(sg_enc *E, void *src, void *dst, uint64_t dst_off, uint32_t n) {
    sg_gpu *g = E->g;
    id<MTLComputeCommandEncoder> e = E->enc;
    const uint32_t p[8] = { n, 0, 0, 0, 0, 0, 0, 0 };

    [e setComputePipelineState:g->pipes[KI_KV_STORE_F16]];
    [e setBuffer:bufof(src) offset:(NSUInteger)offof(src) atIndex:0];
    /* Buffer 1 is unused by k_kv_store_f16 but declared in its signature. */
    [e setBuffer:bufof(src) offset:(NSUInteger)offof(src) atIndex:1];
    [e setBuffer:bufof(dst) offset:(NSUInteger)(offof(dst) + dst_off * 2) atIndex:2];
    [e setBytes:p length:8 * sizeof(uint32_t) atIndex:3];

    NSUInteger w = [g->pipes[KI_KV_STORE_F16] maxTotalThreadsPerThreadgroup];
    if (w > SG_TG) w = SG_TG;
    if (w > n) w = n;
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
}

/* k_attn_decode_f16's dispatch: THREE device buffer inputs (q, separate k,
 * separate v) where every other kernel in this file needs at most two, so it
 * cannot go through enc_op's (a, b, out) shape. Buffer indices follow the
 * kernel's own signature in kernels.metal: q=0, k=1, v=2, out=3, params=4,
 * scores=5. p must supply exactly k_attn_decode_f16's six params. */
static void enc_attn_f16(sg_enc *E, void *q, void *k, void *v, void *out,
                         const uint32_t *p) {
    sg_gpu *g = E->g;
    id<MTLComputeCommandEncoder> e = E->enc;

    [e setComputePipelineState:g->pipes[KI_ATTN_F16]];
    [e setBuffer:bufof(q) offset:(NSUInteger)offof(q) atIndex:0];
    [e setBuffer:bufof(k) offset:(NSUInteger)offof(k) atIndex:1];
    [e setBuffer:bufof(v) offset:(NSUInteger)offof(v) atIndex:2];
    [e setBuffer:bufof(out) offset:(NSUInteger)offof(out) atIndex:3];
    [e setBytes:p length:8 * sizeof(uint32_t) atIndex:4];
    [e setBuffer:g->scratch offset:0 atIndex:5];

    /* One threadgroup per query head, same geometry as k_attn_decode
     * (SG_K_ATTN: params[0] threadgroups of SG_TG). */
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)p[0], 1, 1)
      threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
}

/* Task P2.3: k_attn_decode_splitk_partial THEN k_attn_decode_splitk_combine
 * into an already-open encoder, the batched twin of the
 * sg_gpu_run_attn_splitk_partial / _combine one-shot pair. It replaces the
 * enc_attn_f16 call above when splitk_use says so; both write the same
 * [n_heads, head_dim] attention output into the same handle, so nothing
 * downstream of it changes.
 *
 * TWO DISPATCHES, ONE ENCODER, NO WAIT BETWEEN THEM. The default dispatch type
 * is MTLDispatchTypeSerial (stated at sg_gpu_forward's encoder), so the
 * combine cannot start before the partial has finished writing m/s/acc. This
 * is the whole point of the task: the pair used to be two commit-and-waits.
 *
 * THE SCRATCH HAZARD, HANDLED STRUCTURALLY. The partial binds
 * g->splitk_scratch (buffer 7), the DEDICATED allocation P2.2's fix round
 * added, never the process-wide g->scratch that k_attn_decode,
 * k_attn_decode_f16, k_attn_prefill and this same encoder all bind at offset
 * 0. That matters exactly here and only here: this is the first dispatch of
 * the partial from INSIDE an open command buffer that also carries other
 * scratch users, and g->scratch is grown, never partitioned, so the two would
 * be addressing the same bytes. There is no g->scratch reference anywhere in
 * this function, by construction rather than by rule.
 *
 * BUFFER SIZING IS DONE BEFORE THE COMMAND BUFFER OPENS. m, s, acc and
 * splitk_scratch are all allocated in sg_gpu_state_new for the worst case at
 * max_ctx (see splitk_max_splits), and splitk_use refuses to run if any of
 * them is missing, so nothing here can allocate, grow or release a buffer
 * mid-encode. The three are shared by every full-attention layer in the
 * forward, which is safe for the same serial-dispatch reason as above.
 *
 * P2.5 NARROWS THE BYTE-IDENTITY CLAIM ABOVE seq 65791 (the cap first binds,
 * i.e. first returns something below the per-head value, at seq 65792 ==
 * SG_TG * (SG_SPLITK_GQA_N_SPLITS_CAP + 1): that is where seq / SG_TG first
 * reaches 257, one past the 256 cap; every seq up to and including 65791
 * still floors to 256 or less, where the cap is a no-op). Below 65792,
 * splitk_gqa_n_splits(seq) equals splitk_n_splits(seq) exactly, so
 * SURGE_ATTN_SPLITK_GQA=0 and =1 dispatch with the SAME n_splits and still
 * agree bit for bit end to end -- which is what the existing gate
 * (test_gpu_fwd.c's mini_f16_splitk_gqa_dispatches_and_matches, seq up to
 * SPLITK_GATE_N == 1600) measures. From 65792 on, the two modes partition
 * the same keys differently, exactly the way SURGE_ATTN_SPLITK=0/1 already
 * does (see sg_gpu_run_attn_splitk_partial's doc in surge.h), so they only
 * agree to float rounding there. This is a property of picking FEWER
 * SPLITS, not a bug: the two KERNELS are still byte-identical to each other
 * at any FIXED n_splits, which is what metal_attn_splitk_gqa_bit_identical
 * (test_metal_ops.c) checks directly, bypassing this policy entirely.
 *
 * P2.6 GATES THAT NARROWED CLAIM instead of arguing it. The divergence point
 * is SG_TG * (cap + 1) for whatever cap the state resolved, so
 * SURGE_SPLITK_GQA_CAP=4 moves the identical mechanism from seq 65792 down to
 * seq 1280, where a `make check` subtest can run both arms end to end
 * (test_gpu_fwd.c's mini_f16_splitk_gqa_cap_override_greedy_matches): below
 * 1280 it still requires BYTE-identity, at and above it requires the logits to
 * DIFFER (or the override did nothing and the test is vacuous) while every
 * greedy argmax still agrees. That is surge's actual correctness standard,
 * byte-exact greedy tokens, asserted in the regime where the cap binds.
 *
 * `p` must carry the same 8 params the one-shots take (surge.h has the full
 * layout; p[3] = seq and p[6] = n_splits are the two this function itself
 * reads -- p[6] only for the per-head kernel as of P2.5, see below); the
 * buffer indices below are k_attn_decode_splitk_partial's and _combine's
 * own signatures in kernels.metal. */
static void enc_attn_splitk(sg_enc *E, void *q, void *k, void *v, void *out,
                            const uint32_t *p) {
    sg_gpu *g = E->g;
    id<MTLComputeCommandEncoder> e = E->enc;
    NSUInteger n_heads = (NSUInteger)p[0];

    /* Task P2.4: which partial, and therefore how tall the grid is. The GQA
     * kernel gives one threadgroup the whole GQA group, so it reads each K/V
     * element ONCE instead of once per query head sharing it; it writes the
     * same m/s/acc bytes into the same layout, so nothing else here (the
     * combine, the buffers, the scratch) changes with the choice. ON by default
     * since P4.0, wherever the policy admits it; SURGE_ATTN_SPLITK_GQA=0 pins
     * the per-head partial. See splitk_gqa_use.
     *
     * P2.7: p[3] is seq, and the predicate needs it for the measured
     * threadgroup floor, so the SAME step can pick the per-head partial early
     * in a sequence and the GQA one later. That is why the two dispatch
     * counters below are BOTH nonzero over a long run with the switch on. */
    /* P2.8: THREE arms now, and the online one is a GQA arm. splitk_gqa_use
     * already answers false when the online kernel is selected (see its body),
     * so the two predicates are mutually exclusive and `gqa` below is purely
     * "is the grid one threadgroup per KV head", which both GQA kernels are. */
    bool online = splitk_online_use(g, p[0], p[1], p[2], p[3]);
    bool gqa = online || splitk_gqa_use(g, p[0], p[1], p[3]);
    NSUInteger rows = gqa ? (NSUInteger)p[1] : n_heads;
    /* Record WHICH kernel this dispatch is, for the gate (finding I1): the
     * four-pass pair is contracted to produce the same bytes, so nothing
     * downstream can tell them apart, and a gate that cannot either would pass
     * while the GQA path quietly stopped being selected. The online arm's own
     * A/B has the same problem for a different reason: it agrees on greedy
     * TOKENS by design, so only a counter can show it ran. Diagnostics only,
     * read through sg_gpu_splitk_dispatch_counts and
     * sg_gpu_splitk_online_dispatches; no computed value depends on them. */
    if (online) g->splitk_online_dispatches++;
    else if (gqa) g->splitk_gqa_dispatches++;
    else g->splitk_partial_dispatches++;

    /* Task P2.5: p[6] as the caller built it is splitk_use's PER-HEAD
     * n_splits (splitk_n_splits(seq)); the GQA kernel saturates at a lower
     * split count (splitk_gqa_n_splits above), so once gqa is chosen this
     * dispatch needs its OWN n_splits rather than the caller's. A local copy
     * of the whole params array is required, not just a local n_splits
     * variable, because the combine dispatch below reads p[6] out of the
     * SAME array: the partial and the combine it is paired with must agree
     * with EACH OTHER on how many splits were written, not with whatever the
     * per-head caller computed. splitk_gqa_n_splits(seq, cap) <=
     * splitk_n_splits(seq) always (its cap only ever lowers the result, at ANY
     * cap), so this can only shrink the grid relative to p[6], never exceed the
     * m/s/acc buffers g->splitk_max_splits sized from the per-head policy. The
     * per-head arm is untouched: pd[6] stays exactly the caller's p[6] there,
     * so splitk_n_splits' own behaviour does not change at all.
     *
     * P2.6: the cap comes from splitk_gqa_cap_of(g), i.e. the state's
     * SURGE_SPLITK_GQA_CAP if one was accepted and the measured 256 otherwise.
     * This line is the single place the dispatched GQA n_splits is decided, and
     * sg_gpu_splitk_gqa_n_splits_at reports exactly it. */
    uint32_t pd[8];
    memcpy(pd, p, sizeof pd);
    /* P2.8: the online arm is a GQA arm and takes the SAME measured GQA policy;
     * `gqa` is true for it, so this line covers both without a second case. */
    if (gqa) pd[6] = splitk_gqa_n_splits(p[3], splitk_gqa_cap_of(g));
    NSUInteger n_splits = (NSUInteger)pd[6];

    /* The partial: q=0, kc=1, vc=2, m=3, s=4, acc=5, params=6, scores=7 (the
     * online partial has no buffer 7, see below). */
    int partial_ki = online ? KI_ATTN_SPLITK_PARTIAL_GQA_ONLINE
                            : (gqa ? KI_ATTN_SPLITK_PARTIAL_GQA : KI_ATTN_SPLITK_PARTIAL);
    [e setComputePipelineState:g->pipes[partial_ki]];
    [e setBuffer:bufof(q) offset:(NSUInteger)offof(q) atIndex:0];
    [e setBuffer:bufof(k) offset:(NSUInteger)offof(k) atIndex:1];
    [e setBuffer:bufof(v) offset:(NSUInteger)offof(v) atIndex:2];
    [e setBuffer:bufof(g->b_sk_m) offset:(NSUInteger)offof(g->b_sk_m) atIndex:3];
    [e setBuffer:bufof(g->b_sk_s) offset:(NSUInteger)offof(g->b_sk_s) atIndex:4];
    [e setBuffer:bufof(g->b_sk_acc) offset:(NSUInteger)offof(g->b_sk_acc) atIndex:5];
    [e setBytes:pd length:8 * sizeof(uint32_t) atIndex:6];
    /* Buffer 7 exists on the two four-pass partials only. The online kernel
     * holds its running (m, s, acc) in registers and threadgroup memory, so the
     * split-K score scratch is not part of its signature and is not bound here
     * (it stays ALLOCATED, because the other two arms of the same run still use
     * it; see the note in surge.h on what dropping it would save). */
    if (!online) [e setBuffer:g->splitk_scratch offset:0 atIndex:7];
    /* SG_K_HEADS2D: x = split, y = query head (per-head partial) or KV head
     * (GQA partial), matching the kernel's tg.x / tg.y. gpu_grid's (groups,
     * elems) pair cannot carry two group dimensions (it says so at its default
     * case), so this is computed here by hand, the same way enc_matmul does it
     * for SG_K_TILES2D and the one-shots do it for these kernels. Threads per
     * threadgroup stay exactly SG_TG, which is the width the fixed
     * tg_max/tg_sum fold trees are shaped for. */
    [e dispatchThreadgroups:MTLSizeMake(n_splits, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];

    /* The combine: m=0, s=1, acc=2, out=3, params=4. One threadgroup per query
     * head, a plain 1D grid, and the SAME params array (it reads only [0], [2]
     * and [6]) -- pd, not p, so it folds over exactly as many splits as the
     * partial above actually wrote. */
    [e setComputePipelineState:g->pipes[KI_ATTN_SPLITK_COMBINE]];
    [e setBuffer:bufof(g->b_sk_m) offset:(NSUInteger)offof(g->b_sk_m) atIndex:0];
    [e setBuffer:bufof(g->b_sk_s) offset:(NSUInteger)offof(g->b_sk_s) atIndex:1];
    [e setBuffer:bufof(g->b_sk_acc) offset:(NSUInteger)offof(g->b_sk_acc) atIndex:2];
    [e setBuffer:bufof(out) offset:(NSUInteger)offof(out) atIndex:3];
    [e setBytes:pd length:8 * sizeof(uint32_t) atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
}

/* One tiled-GEMM (M5.3) dispatch into an already-open encoder, the batched
 * analog of enc_op for a KI_MATMUL_* kernel. Buffer order is (X, W, Y) -- the
 * REVERSE of the matvec kernels enc_op drives -- so `x` is the [N, K]
 * activation chunk, `w` a wrapped [M, K] weight (bound at its own base, like
 * every wrapped weight), `y` the [N, M] output. `xoff`/`yoff` are element
 * offsets in FLOATS. The 2D tile grid (ceil(M/TN) x ceil(N/TM) threadgroups of
 * SG_GEMM_TN x SG_GEMM_TM threads) matches sg_gpu_run_op's SG_K_TILES2D branch
 * exactly; gpu_grid cannot express two group dimensions, so this is by hand. */
static void enc_matmul(sg_enc *E, int ki, void *x, uint64_t xoff, void *w,
                       void *y, uint64_t yoff, uint32_t nn, uint32_t mm, uint32_t kk) {
    sg_gpu *g = E->g;
    id<MTLComputeCommandEncoder> e = E->enc;
    const uint32_t p[8] = { nn, mm, kk, 0, 0, 0, 0, 0 };

    [e setComputePipelineState:g->pipes[ki]];
    [e setBuffer:bufof(x) offset:(NSUInteger)(offof(x) + xoff * 4) atIndex:0];
    [e setBuffer:bufof(w) offset:(NSUInteger)offof(w) atIndex:1];
    [e setBuffer:bufof(y) offset:(NSUInteger)(offof(y) + yoff * 4) atIndex:2];
    [e setBytes:p length:8 * sizeof(uint32_t) atIndex:3];

    uint64_t tiles_n = ((uint64_t)nn + SG_GEMM_TM - 1) / SG_GEMM_TM;
    uint64_t tiles_m = ((uint64_t)mm + SG_GEMM_TN - 1) / SG_GEMM_TN;
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)tiles_m, (NSUInteger)tiles_n, 1)
        threadsPerThreadgroup:MTLSizeMake(SG_GEMM_TN, SG_GEMM_TM, 1)];
}

/* Qwen3NextAttention for one token, mirroring ref.c's attn_layer statement
 * for statement. b_h holds rms_norm(x, ln1) on entry; b_r receives the
 * layer's residual contribution. `layer_idx` is only used on the fp16 path,
 * to look up this layer's K/V buffers in g->kv.
 *
 * The queries keep whatever per-head layout q_proj produced, the whole way
 * through, on both KV dtypes: q_norm and RoPE touch only the first head_dim
 * of each head (k_rmsnorm_heads / k_rope_heads take the stride).
 *
 * The per-head stride is ARCH-DEPENDENT (P1), not universally 2*head_dim:
 *   - hybrid qwen3_5/qwen35 (attn_output_gate == true): stride 2*head_dim,
 *     queries in the first half and the folded attention output gate in the
 *     second, so the gate reaches k_gate_sigmoid_strided exactly as q_proj
 *     produced it;
 *   - dense qwen3 (attn_output_gate == false): stride head_dim, queries only,
 *     there is no second half and the gate dispatch is skipped entirely.
 * Both read g->q_width, which sg_gpu_load_model sets from the flag.
 *
 * f32 path (SURGE_KV_DTYPE=f32): byte-for-bit the pre-M5.2 code. k_proj and
 * v_proj write STRAIGHT INTO the cache slot for this position and the
 * qk-norm and RoPE run in place there, which is the same values ref.c
 * computes into kbuf/vbuf and memcpy's afterwards, with one fewer copy.
 *
 * f16 path (default): k_proj and v_proj cannot write directly into the half
 * cache (a matvec kernel only ever produces f32), so they land in the f32
 * scratch b_k32/b_v32 instead; k-norm and RoPE run there in place exactly as
 * they did on the f32 cache slot; then enc_kv_store casts the finished K and
 * V into g->kv's per-layer half buffers at this position, and enc_attn_f16
 * reads them back widened to f32 for the dot products and softmax. */
static void enc_attn(sg_enc *E, sg_gpu_layer *L, uint32_t layer_idx, uint32_t pos) {
    sg_gpu *g = E->g;
    const sg_cfg *c = &g->cfg;
    uint32_t hd = c->head_dim;
    uint32_t used = pos + 1;
    float scale = (float)(1.0 / sqrt((double)hd));
    /* Task P1: per-head stride into g->b_qg. 2*hd (queries + folded gate) on
     * the hybrid, hd (queries only) on dense qwen3; must stay consistent
     * with g->q_width == c->n_heads * q_stride, set at load time. */
    uint32_t q_stride = c->attn_output_gate ? 2 * hd : hd;

    enc_op(E, g->mat_kernel, L->w_q, 0, g->b_h, 0, g->b_qg, 0, nil, 0,
           PARAMS(g->q_width, c->hidden));
    /* q_norm applies to the queries only, never to the gate. */
    enc_op(E, KI_RMSNORM_HEADS, g->b_qg, 0, L->qk_norm, 0, g->b_qg, 0, nil, 0,
           PARAMS(hd, c->n_heads, fbits(c->rms_eps), 1, q_stride));
    enc_op(E, KI_ROPE_HEADS, g->b_qg, 0, g->b_cs, 0, g->b_qg, 0, nil, 0,
           PARAMS(hd, c->rope_dim, c->n_heads, q_stride));

    if (g->kv_dtype == SG_T_F32) {
        uint64_t koff = (uint64_t)pos * g->kv_width;
        uint64_t vbase = (uint64_t)g->max_ctx * g->kv_width;

        enc_op(E, g->mat_kernel, L->w_k, 0, g->b_h, 0, L->kv, koff, nil, 0,
               PARAMS(g->kv_width, c->hidden));
        enc_op(E, g->mat_kernel, L->w_v, 0, g->b_h, 0, L->kv, vbase + koff, nil, 0,
               PARAMS(g->kv_width, c->hidden));

        enc_op(E, KI_RMSNORM_HEADS, L->kv, koff, L->qk_norm, hd, L->kv, koff, nil, 0,
               PARAMS(hd, c->n_kv_heads, fbits(c->rms_eps), 1, hd));
        enc_op(E, KI_ROPE_HEADS, L->kv, koff, g->b_cs, 0, L->kv, koff, nil, 0,
               PARAMS(hd, c->rope_dim, c->n_kv_heads, hd));

        enc_op(E, KI_ATTN, g->b_qg, 0, L->kv, 0, g->b_ctx, 0, g->scratch, 0,
               PARAMS(c->n_heads, c->n_kv_heads, hd, used, q_stride,
                      (uint32_t)vbase, fbits(scale)));
    } else {
        void *kbuf = sg_kv_k(g->kv, layer_idx);
        void *vbuf = sg_kv_v(g->kv, layer_idx);

        enc_op(E, g->mat_kernel, L->w_k, 0, g->b_h, 0, g->b_k32, 0, nil, 0,
               PARAMS(g->kv_width, c->hidden));
        enc_op(E, g->mat_kernel, L->w_v, 0, g->b_h, 0, g->b_v32, 0, nil, 0,
               PARAMS(g->kv_width, c->hidden));

        enc_op(E, KI_RMSNORM_HEADS, g->b_k32, 0, L->qk_norm, hd, g->b_k32, 0, nil, 0,
               PARAMS(hd, c->n_kv_heads, fbits(c->rms_eps), 1, hd));
        enc_op(E, KI_ROPE_HEADS, g->b_k32, 0, g->b_cs, 0, g->b_k32, 0, nil, 0,
               PARAMS(hd, c->rope_dim, c->n_kv_heads, hd));

        enc_kv_store(E, g->b_k32, kbuf, (uint64_t)pos * g->kv_width, g->kv_width);
        enc_kv_store(E, g->b_v32, vbuf, (uint64_t)pos * g->kv_width, g->kv_width);

        /* Task P2.3: the split-K pair when the sequence is long enough for it
         * (splitk_use returns the measured n_splits, or 0), the incumbent
         * single-threadgroup-per-head kernel otherwise. Both produce the same
         * [n_heads, head_dim] output in g->b_ctx; they differ in how the sum
         * over keys is PARTITIONED, so the two paths agree to float rounding
         * rather than bit for bit, exactly as sg_ref_attn_decode_splitk agrees
         * with sg_ref_attn_decode. */
        uint32_t n_splits = splitk_use(g, used);
        if (n_splits != 0) {
            enc_attn_splitk(E, g->b_qg, kbuf, vbuf, g->b_ctx,
                            PARAMS(c->n_heads, c->n_kv_heads, hd, used, q_stride,
                                   fbits(scale), n_splits));
        } else {
            enc_attn_f16(E, g->b_qg, kbuf, vbuf, g->b_ctx,
                         PARAMS(c->n_heads, c->n_kv_heads, hd, used, q_stride, fbits(scale)));
        }
    }

    /* The output gate is a sigmoid applied before o_proj -- same for both
     * dtypes, and only when the model actually has one (Task P1). Dense
     * qwen3 has no gate: g->b_ctx already holds the raw attention output, so
     * o_proj below reads it unmodified. */
    if (c->attn_output_gate) {
        enc_op(E, KI_GATE_STRIDED, g->b_ctx, 0, g->b_qg, 0, g->b_ctx, 0, nil, 0,
               PARAMS(hd, c->n_heads, 2 * hd, hd));
    }
    enc_op(E, g->mat_kernel, L->w_o, 0, g->b_ctx, 0, g->b_r, 0, nil, 0,
           PARAMS(c->hidden, g->attn_width));
}

/* Qwen3NextAttention for a CHUNK of `n` query tokens at absolute positions
 * base .. base+n-1, the chunked twin of enc_attn's fp16 path. It is the
 * single-full-attn-layer prefill encoder (Task M5.4); the whole-prompt,
 * all-layers prefill orchestration that drives it is Task M5.6, which is also
 * what sizes the buffers it reads. External linkage (rather than static) on
 * purpose: nothing in this build calls it yet, and a static uncalled function
 * would fail -Werror's -Wunused-function; M5.6's sg_gpu_prefill will call it.
 *
 * STORE-VS-ATTEND ORDER: store first, then attend. The chunk's own K and V
 * are cast into g->kv's per-layer fp16 buffers at positions base..base+n-1
 * BEFORE the attention dispatch, so at attend time the cache holds all base+n
 * positions and k_attn_prefill's threadgroup for token t attends over the
 * first base+t+1 of them (keys 0..base+t). That bound IS the causal mask: a
 * query at absolute position base+t never reads a key at a strictly-future
 * position (base+t' with t' > t is >= base+t+1, past the bound), while it does
 * read every earlier chunk token (t' < t) and all prior context.
 *
 * BUFFER SIZING (the caller's contract, since this reuses the decode field
 * names sized for one token): before calling, g->b_h must hold the chunk's
 * post-ln1 hidden [n, hidden]; g->b_qg [n, q_width] (q_width == 2*attn_width
 * when the model has an attention output gate, Task P1, else attn_width;
 * see cfg.attn_output_gate), g->b_k32/b_v32 [n, kv_width], g->b_ctx
 * [n, attn_width], g->b_r [n, hidden] must be
 * chunk-sized scratch; g->b_cs must hold the per-token RoPE table
 * [n, rope_dim] (cos half then sin half per absolute position, built in double
 * on the host exactly as sg_gpu_forward builds the one-token table); g->scratch
 * must be at least n*n_heads*(base+n) floats (ensured before the command
 * buffer opens, never mid-encode). g->kv must be the fp16 sg_kv cache.
 * Everything is left in g->b_r on exit, the same handle enc_attn writes. */
void enc_attn_prefill(sg_enc *E, sg_gpu_layer *L, uint32_t layer_idx,
                      uint32_t base, uint32_t n) {
    sg_gpu *g = E->g;
    const sg_cfg *c = &g->cfg;
    uint32_t hd = c->head_dim;
    uint32_t eps = fbits(c->rms_eps);
    float scale = (float)(1.0 / sqrt((double)hd));
    int gemm = gemm_kernel_for(g->model->wtype);
    /* Task P1: per-head stride into g->b_qg, same rule as enc_attn's
     * decode-step twin. */
    uint32_t q_stride = c->attn_output_gate ? 2 * hd : hd;

    /* Q/K/V projections for the whole chunk in one tiled GEMM each:
     * Y[n, width] = b_h[n, hidden] @ W[width, hidden]^T. */
    enc_matmul(E, gemm, g->b_h, 0, L->w_q, g->b_qg, 0, n, g->q_width, c->hidden);
    /* q_norm on every (token, head) query slice: n*n_heads slices of head_dim
     * at stride q_stride (2*head_dim when gated, head_dim when not), weight =
     * qk_norm[0..head_dim). Same in-place k_rmsnorm_heads the decode path
     * uses, applied across the chunk. */
    enc_op(E, KI_RMSNORM_HEADS, g->b_qg, 0, L->qk_norm, 0, g->b_qg, 0, nil, 0,
           PARAMS(hd, n * c->n_heads, eps, 1, q_stride));
    /* Partial RoPE per token at its absolute position, from the chunk's cos/sin
     * table in g->b_cs. */
    enc_op(E, KI_ROPE_CHUNK, g->b_qg, 0, g->b_cs, 0, g->b_qg, 0, nil, 0,
           PARAMS(hd, c->rope_dim, c->n_heads, q_stride, n));

    enc_matmul(E, gemm, g->b_h, 0, L->w_k, g->b_k32, 0, n, g->kv_width, c->hidden);
    enc_matmul(E, gemm, g->b_h, 0, L->w_v, g->b_v32, 0, n, g->kv_width, c->hidden);
    enc_op(E, KI_RMSNORM_HEADS, g->b_k32, 0, L->qk_norm, hd, g->b_k32, 0, nil, 0,
           PARAMS(hd, n * c->n_kv_heads, eps, 1, hd));
    enc_op(E, KI_ROPE_CHUNK, g->b_k32, 0, g->b_cs, 0, g->b_k32, 0, nil, 0,
           PARAMS(hd, c->rope_dim, c->n_kv_heads, hd, n));

    /* STORE: cast the chunk's finished K and V (f32) into the fp16 cache at
     * positions base..base+n-1. The chunk's [n, kv_width] rows map one-to-one
     * onto cache positions base..base+n-1 (both kv_width-contiguous), so one
     * store of n*kv_width elements at cache offset base*kv_width lands them. */
    void *kbuf = sg_kv_k(g->kv, layer_idx);
    void *vbuf = sg_kv_v(g->kv, layer_idx);
    enc_kv_store(E, g->b_k32, kbuf, (uint64_t)base * g->kv_width, n * g->kv_width);
    enc_kv_store(E, g->b_v32, vbuf, (uint64_t)base * g->kv_width, n * g->kv_width);

    /* ATTEND: k_attn_prefill over the now base+n-position cache, one
     * threadgroup per (token, head). Dispatched by hand (three device inputs),
     * exactly as sg_gpu_run_attn_prefill does. */
    {
        id<MTLComputeCommandEncoder> e = E->enc;
        const uint32_t pa[8] = { c->n_heads, c->n_kv_heads, hd, base, n,
                                 q_stride, fbits(scale), 0 };
        [e setComputePipelineState:g->pipes[KI_ATTN_PREFILL]];
        [e setBuffer:bufof(g->b_qg) offset:(NSUInteger)offof(g->b_qg) atIndex:0];
        [e setBuffer:bufof(kbuf) offset:(NSUInteger)offof(kbuf) atIndex:1];
        [e setBuffer:bufof(vbuf) offset:(NSUInteger)offof(vbuf) atIndex:2];
        [e setBuffer:bufof(g->b_ctx) offset:(NSUInteger)offof(g->b_ctx) atIndex:3];
        [e setBytes:pa length:8 * sizeof(uint32_t) atIndex:4];
        [e setBuffer:g->scratch offset:0 atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)n * c->n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
    }

    /* Output gate (sigmoid of the second half of each head's q_proj slice),
     * then o_proj, both chunked -- gate only when the model has one (Task
     * P1); dense qwen3 feeds g->b_ctx straight to o_proj unmodified. */
    if (c->attn_output_gate) {
        enc_op(E, KI_GATE_STRIDED, g->b_ctx, 0, g->b_qg, 0, g->b_ctx, 0, nil, 0,
               PARAMS(hd, n * c->n_heads, 2 * hd, hd));
    }
    enc_matmul(E, gemm, g->b_ctx, 0, L->w_o, g->b_r, 0, n, c->hidden, g->attn_width);
}

/* GatedDeltaNet for a CHUNK of `n` tokens, the chunked twin of enc_gdn (Task
 * M5.5). It is the single-DeltaNet-layer prefill encoder; the whole-prompt,
 * all-layers prefill orchestration that drives it (and sizes the chunk buffers
 * it reuses) is Task M5.6. External linkage on purpose: nothing in this build
 * calls it yet, and a static uncalled function would trip -Werror's
 * -Wunused-function; M5.6's sg_gpu_prefill will call it. Its four DeltaNet
 * chunk kernels are each gated in isolation by the per-op test through their
 * public one-shots above.
 *
 * STATE THREADING is the point of the task: the recurrent state (the conv tail
 * and the S matrix) is carried across chunks in sg_kv's per-layer conv/S
 * buffers, NOT in the ad hoc L->conv_buf / L->ssm the decode path uses. Both are
 * read at chunk entry and rewritten in place by the sequential within-chunk
 * scan, so feeding chunk k+1 the state chunk k left behind reproduces the
 * one-token-at-a-time decode exactly. `base` is unused: unlike full attention,
 * DeltaNet has no position-indexed cache and no RoPE, so a chunk depends on the
 * prior context ONLY through the carried state.
 *
 * BUFFER SIZING (the caller's contract, since this reuses the decode field
 * names sized for one token; M5.6 sizes them for the chunk): g->b_h holds the
 * chunk's post-ln1 hidden [n, hidden] on entry; g->b_qkv [n, conv_dim] is the
 * qkv projection, then the conv output IN PLACE (safe: k_conv1d_chunk reads and
 * writes each channel-position exactly once, own thread, read before write),
 * then the q|k|v working area, then (after the delta scan consumes it) reused as
 * the z scratch [n, value_dim]; g->b_ab [2*n*n_v] holds the a-chunk then the
 * b-chunk; g->b_gates [n, 2*n_v] the per-token [beta;decay]; g->b_y [n,
 * value_dim] the delta readout then the gated-norm output in place; g->b_r [n,
 * hidden] the layer's residual contribution on exit (the handle enc_gdn writes).
 * L->zw still parks this layer's ssm_norm weight at element offset value_dim,
 * bound to k_rmsnorm_gated_chunk as its separate w buffer. */
void enc_gdn_prefill(sg_enc *E, sg_gpu_layer *L, uint32_t layer_idx,
                     uint32_t base, uint32_t n) {
    sg_gpu *g = E->g;
    const sg_cfg *c = &g->cfg;
    uint32_t dk = c->head_k_dim, dv = c->head_v_dim;
    uint32_t key_dim = g->key_dim, value_dim = g->value_dim, conv_dim = g->conv_dim;
    double inv = 1.0 / sqrt((double)dk);
    uint32_t eps6 = fbits(1e-6f);
    uint32_t neg_exp = (g->model->ssm_a_form == SG_SSM_A_NEG_EXP) ? 1u : 0u;
    uint32_t tiled = g->model->v_heads_tiled ? 1u : 0u;
    int gemm = gemm_kernel_for(g->model->wtype);
    id<MTLComputeCommandEncoder> e = E->enc;
    (void)base;

    /* in_proj: qkv, a and b for the whole chunk (z is computed later, into the
     * b_qkv scratch the delta scan frees). a-chunk lands in b_ab[0..n*n_v), the
     * b-chunk in b_ab[n*n_v..2*n*n_v). */
    enc_matmul(E, gemm, g->b_h, 0, L->w_qkv, g->b_qkv, 0, n, conv_dim, c->hidden);
    enc_matmul(E, gemm, g->b_h, 0, L->w_a, g->b_ab, 0, n, c->n_v_heads, c->hidden);
    enc_matmul(E, gemm, g->b_h, 0, L->w_b, g->b_ab, (uint64_t)n * c->n_v_heads,
               n, c->n_v_heads, c->hidden);

    /* conv1d over the chunk (in place on b_qkv), threading the conv tail via
     * sg_kv, then SiLU in place. */
    {
        void *conv_state = sg_kv_conv(g->kv, layer_idx);
        const uint32_t pc[8] = { conv_dim, c->conv_kernel, n, 0, 0, 0, 0, 0 };
        [e setComputePipelineState:g->pipes[KI_CONV1D_CHUNK]];
        [e setBuffer:bufof(g->b_qkv) offset:(NSUInteger)offof(g->b_qkv) atIndex:0];
        [e setBuffer:bufof(L->conv_w) offset:(NSUInteger)offof(L->conv_w) atIndex:1];
        [e setBuffer:bufof(g->b_qkv) offset:(NSUInteger)offof(g->b_qkv) atIndex:2];
        [e setBytes:pc length:8 * sizeof(uint32_t) atIndex:3];
        [e setBuffer:bufof(conv_state) offset:(NSUInteger)offof(conv_state) atIndex:4];
        NSUInteger w = gpu_elem_width(g, KI_CONV1D_CHUNK, conv_dim);
        [e dispatchThreads:MTLSizeMake(conv_dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    enc_op(E, KI_SILU, g->b_qkv, 0, NULL, 0, g->b_qkv, 0, nil, 0,
           PARAMS(n * conv_dim));

    /* q and k RMS-normed per key head (no weight, hardcoded eps 1e-6) then
     * scaled (q by 1/head_k_dim, k by 1/sqrt(head_k_dim)), exactly enc_gdn's
     * arithmetic, applied per token: the q|k slices of a token are contiguous
     * within its conv_dim row but separated by the v half from the next token's,
     * so a single strided k_rmsnorm_heads cannot span the chunk. */
    for (uint32_t t = 0; t < n; t++) {
        uint64_t roff = (uint64_t)t * conv_dim;
        enc_op(E, KI_RMSNORM_HEADS, g->b_qkv, roff, NULL, 0, g->b_qkv, roff, nil, 0,
               PARAMS(dk, c->n_k_heads, eps6, 0, dk));
        enc_op(E, KI_RMSNORM_HEADS, g->b_qkv, roff + key_dim, NULL, 0,
               g->b_qkv, roff + key_dim, nil, 0,
               PARAMS(dk, c->n_k_heads, eps6, 0, dk));
        enc_op(E, KI_SCALE, g->b_qkv, roff, NULL, 0, g->b_qkv, roff, nil, 0,
               PARAMS(key_dim, fbits((float)(inv * inv))));
        enc_op(E, KI_SCALE, g->b_qkv, roff + key_dim, NULL, 0,
               g->b_qkv, roff + key_dim, nil, 0,
               PARAMS(key_dim, fbits((float)inv)));
    }

    /* gates for the whole chunk (a = b_ab[0..], b = b_ab[n*n_v..]). */
    {
        const uint32_t pg[8] = { c->n_v_heads, neg_exp, n, 0, 0, 0, 0, 0 };
        [e setComputePipelineState:g->pipes[KI_DELTA_GATES_CHUNK]];
        [e setBuffer:bufof(g->b_ab) offset:(NSUInteger)offof(g->b_ab) atIndex:0];
        [e setBuffer:bufof(g->b_ab)
                offset:(NSUInteger)(offof(g->b_ab) + (uint64_t)n * c->n_v_heads * 4)
               atIndex:1];
        [e setBuffer:bufof(g->b_gates) offset:(NSUInteger)offof(g->b_gates) atIndex:2];
        [e setBytes:pg length:8 * sizeof(uint32_t) atIndex:3];
        [e setBuffer:bufof(L->a_dt) offset:(NSUInteger)offof(L->a_dt) atIndex:4];
        uint64_t elems = (uint64_t)n * c->n_v_heads;
        NSUInteger w = gpu_elem_width(g, KI_DELTA_GATES_CHUNK, elems);
        [e dispatchThreads:MTLSizeMake((NSUInteger)elems, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }

    /* delta recurrence over the chunk, threading S via sg_kv, readout to b_y. */
    {
        void *s_state = sg_kv_s(g->kv, layer_idx);
        const uint32_t pd[8] = { dk, dv, c->n_v_heads, c->n_k_heads, key_dim,
                                 tiled, n, conv_dim };
        [e setComputePipelineState:g->pipes[KI_DELTA_CHUNK]];
        [e setBuffer:bufof(s_state) offset:(NSUInteger)offof(s_state) atIndex:0];
        [e setBuffer:bufof(g->b_qkv) offset:(NSUInteger)offof(g->b_qkv) atIndex:1];
        [e setBuffer:bufof(g->b_y) offset:(NSUInteger)offof(g->b_y) atIndex:2];
        [e setBytes:pd length:8 * sizeof(uint32_t) atIndex:3];
        [e setBuffer:bufof(g->b_gates) offset:(NSUInteger)offof(g->b_gates) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)c->n_v_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
    }

    /* z = w_z @ h into the now-free b_qkv scratch, then RMSNormGated
     * (silu(z) * rms_norm(y, ssm_norm)) over the chunk, in place on b_y. */
    enc_matmul(E, gemm, g->b_h, 0, L->w_z, g->b_qkv, 0, n, value_dim, c->hidden);
    {
        const uint32_t pn[8] = { dv, c->n_v_heads, fbits(c->rms_eps), n, 0, 0, 0, 0 };
        [e setComputePipelineState:g->pipes[KI_RMSNORM_GATED_CHUNK]];
        [e setBuffer:bufof(g->b_y) offset:(NSUInteger)offof(g->b_y) atIndex:0];
        [e setBuffer:bufof(g->b_qkv) offset:(NSUInteger)offof(g->b_qkv) atIndex:1];
        [e setBuffer:bufof(g->b_y) offset:(NSUInteger)offof(g->b_y) atIndex:2];
        [e setBytes:pn length:8 * sizeof(uint32_t) atIndex:3];
        [e setBuffer:bufof(L->zw)
                offset:(NSUInteger)(offof(L->zw) + (uint64_t)value_dim * 4)
               atIndex:4];
        uint64_t groups = (uint64_t)n * c->n_v_heads;
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SG_TG, 1, 1)];
    }

    enc_matmul(E, gemm, g->b_y, 0, L->w_out, g->b_r, 0, n, c->hidden, value_dim);
}

/* GatedDeltaNet for one token, mirroring ref.c's gdn_layer. The conv output
 * buffer doubles as the carried conv tail (k_conv1d_step's contract) and as
 * the q|k|v working area, so the silu, the two per-key-head normalizations
 * and the two scalings all run in place on it. */
static void enc_gdn(sg_enc *E, sg_gpu_layer *L) {
    sg_gpu *g = E->g;
    const sg_cfg *c = &g->cfg;
    uint32_t dk = c->head_k_dim, dv = c->head_v_dim;
    double inv = 1.0 / sqrt((double)dk);
    uint32_t neg_exp = (g->model->ssm_a_form == SG_SSM_A_NEG_EXP) ? 1u : 0u;
    uint32_t tiled = g->model->v_heads_tiled ? 1u : 0u;

    enc_op(E, g->mat_kernel, L->w_qkv, 0, g->b_h, 0, g->b_qkv, 0, nil, 0,
           PARAMS(g->conv_dim, c->hidden));
    enc_op(E, g->mat_kernel, L->w_z, 0, g->b_h, 0, L->zw, 0, nil, 0,
           PARAMS(g->value_dim, c->hidden));
    enc_op(E, g->mat_kernel, L->w_b, 0, g->b_h, 0, g->b_ab, c->n_v_heads, nil, 0,
           PARAMS(c->n_v_heads, c->hidden));
    enc_op(E, g->mat_kernel, L->w_a, 0, g->b_h, 0, g->b_ab, 0, nil, 0,
           PARAMS(c->n_v_heads, c->hidden));

    enc_op(E, KI_CONV1D, g->b_qkv, 0, L->conv_w, 0, L->conv_buf, 0, nil, 0,
           PARAMS(g->conv_dim, c->conv_kernel));
    enc_op(E, KI_SILU, L->conv_buf, 0, NULL, 0, L->conv_buf, 0, nil, 0,
           PARAMS(g->conv_dim));

    /* q and k are RMS-normed per key head with NO weight and a HARDCODED eps
     * of 1e-6 (qwen3_5.py, not the config's rms_norm_eps), then scaled: q by
     * 1/head_k_dim and k by 1/sqrt(head_k_dim). ref.c forms both scale
     * factors in double and rounds once, so the host passes the rounded
     * product and k_scale does a single multiply -- exact for the query
     * scale, one ulp for the key scale; see the note on k_scale. */
    enc_op(E, KI_RMSNORM_HEADS, L->conv_buf, 0, NULL, 0, L->conv_buf, 0, nil, 0,
           PARAMS(dk, c->n_k_heads, fbits(1e-6f), 0, dk));
    enc_op(E, KI_RMSNORM_HEADS, L->conv_buf, g->key_dim, NULL, 0,
           L->conv_buf, g->key_dim, nil, 0,
           PARAMS(dk, c->n_k_heads, fbits(1e-6f), 0, dk));
    enc_op(E, KI_SCALE, L->conv_buf, 0, NULL, 0, L->conv_buf, 0, nil, 0,
           PARAMS(g->key_dim, fbits((float)(inv * inv))));
    enc_op(E, KI_SCALE, L->conv_buf, g->key_dim, NULL, 0,
           L->conv_buf, g->key_dim, nil, 0,
           PARAMS(g->key_dim, fbits((float)inv)));

    enc_op(E, KI_DELTA_GATES, g->b_ab, 0, L->a_dt, 0, g->b_gates, 0, nil, 0,
           PARAMS(c->n_v_heads, neg_exp));
    enc_op(E, KI_DELTA_MULTI, L->ssm, 0, L->conv_buf, 0, g->b_y, 0,
           bufof(g->b_gates), offof(g->b_gates),
           PARAMS(dk, dv, c->n_v_heads, c->n_k_heads, g->key_dim, tiled));

    /* RMSNormGated: silu(z) * rms_norm(y, ssm_norm), the gate taken from the
     * UNNORMALIZED z. zw holds z then the layer's ssm_norm weight, which is
     * the single-buffer layout k_rmsnorm_gated wants. */
    enc_op(E, KI_RMSNORM_GATED, g->b_y, 0, L->zw, 0, g->b_y, 0, nil, 0,
           PARAMS(dv, c->n_v_heads, fbits(c->rms_eps)));
    enc_op(E, g->mat_kernel, L->w_out, 0, g->b_y, 0, g->b_r, 0, nil, 0,
           PARAMS(c->hidden, g->value_dim));
}

/* --------------------------------------------------------------------
 * Load / state / teardown
 * -------------------------------------------------------------------- */

static void gpu_free_state(sg_gpu *g) {
    if (g->ls) {
        for (uint32_t i = 0; i < g->cfg.n_layers; i++) {
            sg_gpu_layer *L = &g->ls[i];
            sg_gpu_buf_free(L->kv);       L->kv = NULL;
            sg_gpu_buf_free(L->conv_buf); L->conv_buf = NULL;
            sg_gpu_buf_free(L->ssm);      L->ssm = NULL;
        }
    }
    /* M5.2: the fp16 full-attention K/V (sg_kv-owned) and its f32 landing
     * scratch. sg_kv_free walks its own per-layer buffers through the
     * registered backend (sg_gpu_buf_free), same as the loop above does for
     * the f32 path's L->kv. */
    sg_kv_free(g->kv); g->kv = NULL;
    sg_gpu_buf_free(g->b_k32); g->b_k32 = NULL;
    sg_gpu_buf_free(g->b_v32); g->b_v32 = NULL;

    /* P2.3: the split-K partial triples, sized from max_ctx like everything
     * else in the state. */
    sg_gpu_buf_free(g->b_sk_m); g->b_sk_m = NULL;
    sg_gpu_buf_free(g->b_sk_s); g->b_sk_s = NULL;
    sg_gpu_buf_free(g->b_sk_acc); g->b_sk_acc = NULL;
    g->splitk_max_splits = 0;
    /* These three are a TEARDOWN RESET, not the defaults. Each resolves its real
     * value in sg_gpu_state_new from the env, and two of them (attn_splitk since
     * P2.3, attn_splitk_gqa since P4.0) resolve to TRUE there. false here means
     * "no state, so nothing is selected and nothing dispatches", which is what
     * sg_gpu_splitk_gqa_selected must answer on a torn-down gpu. */
    g->attn_splitk = false;
    g->attn_splitk_gqa = false;    /* P2.4: re-read from the env on the next state */
    g->attn_splitk_online = false; /* P2.8: same, SURGE_ATTN_SPLITK_ONLINE */
    g->splitk_gqa_cap = 0;         /* P2.6: 0 == "the measured default", see
                                    * splitk_gqa_cap_of; re-read on the next state */
    g->splitk_partial_dispatches = 0;
    g->splitk_gqa_dispatches = 0;
    g->splitk_online_dispatches = 0;

    void *shared[] = { g->b_x, g->b_h, g->b_r, g->b_qg, g->b_ctx, g->b_ffg,
                       g->b_ffu, g->b_qkv, g->b_ab, g->b_gates, g->b_y,
                       g->b_cs, g->b_logits };
    for (size_t i = 0; i < sizeof shared / sizeof *shared; i++) sg_gpu_buf_free(shared[i]);
    g->b_x = g->b_h = g->b_r = g->b_qg = g->b_ctx = NULL;
    g->b_ffg = g->b_ffu = g->b_qkv = g->b_ab = g->b_gates = NULL;
    g->b_y = g->b_cs = g->b_logits = NULL;
    g->h_x = g->h_cs = g->h_logits = NULL;
    g->max_ctx = g->used = 0;
    g->have_state = false;
    /* The attention score scratch is sized from max_ctx, so it belongs to the
     * state and not to the device. Holding a 262144-position row per head
     * alive after the state that needed it is gone would be a surprising
     * amount of private memory to keep around; sg_gpu_run_op regrows it on
     * demand, and sg_gpu_state_new re-ensures it. */
    [g->scratch release];
    g->scratch = nil;
    g->scratch_bytes = 0;
    /* P2.3: the split-K score scratch is now sized from max_ctx too (see
     * sg_gpu_state_new), so it belongs to the state for exactly the reason
     * stated above, and is released with it. splitk_scratch_ensure regrows it
     * on demand for the one-shot entry points. */
    [g->splitk_scratch release];
    g->splitk_scratch = nil;
    g->splitk_scratch_bytes = 0;
}

static void gpu_unload(sg_gpu *g) {
    if (!g) return;
    gpu_free_state(g);
    if (g->ls) {
        for (uint32_t i = 0; i < g->cfg.n_layers; i++) {
            sg_gpu_layer *L = &g->ls[i];
            void *hs[] = { L->w_q, L->w_k, L->w_v, L->w_o, L->w_qkv, L->w_z,
                           L->w_a, L->w_b, L->w_out, L->w_gate, L->w_up,
                           L->w_down, L->ln1, L->ln2, L->qk_norm, L->conv_w,
                           L->a_dt, L->zw };
            for (size_t k = 0; k < sizeof hs / sizeof *hs; k++) sg_gpu_buf_free(hs[k]);
        }
        free(g->ls);
        g->ls = NULL;
    }
    sg_gpu_buf_free(g->lm_head);  g->lm_head = NULL;
    sg_gpu_buf_free(g->out_norm); g->out_norm = NULL;
    g->model = NULL;
    memset(&g->cfg, 0, sizeof g->cfg);
}

/* Wrap a matmul weight of `rows x cols` elements in the model's weight
 * dtype. Both extents are bounded by the 1<<24 check in gpu_check_cfg (and
 * vocab is a checked uint32), so the byte count cannot wrap. For Q8_0 the
 * bytes are rows*(cols/32) blocks of 34 (f16 scale + 32 int8), exactly the
 * row stride sg_ref_matvec_q8 and k_matvec_q8 index; cols is a nonzero
 * multiple of 32 for every matmul (gpu_check_model enforces it on hidden,
 * and the derived widths are all multiples of head_dim, itself a multiple of
 * 32), but reject a stray non-multiple here rather than truncate the block
 * count and wrap a short buffer. */
static sg_err gpu_wrap_w(sg_gpu *g, const void *p, uint64_t rows, uint64_t cols,
                         void **out) {
    sg_tensor_type t = g->model->wtype;
    uint64_t nbytes;
    if (t == SG_T_Q8_0) {
        if (cols == 0 || cols % 32u != 0) {
            return (sg_err){"gpu: a Q8_0 matmul width is not a multiple of 32"};
        }
        nbytes = rows * (cols / 32u) * 34u;
    } else {
        uint64_t esz = (t == SG_T_BF16) ? 2 : 4;
        nbytes = rows * cols * esz;
    }
    return sg_gpu_wrap(g, p, nbytes, out);
}

static sg_err gpu_alloc_f32(sg_gpu *g, uint64_t n, void **buf, float **host) {
    void *h = NULL;
    sg_err e = sg_gpu_alloc(g, n * 4, buf, &h);
    if (host) *host = (float *)h;
    return e;
}

/* The same contract sg_ref_state_new enforces, restated for the GPU path so
 * this is usable without the reference state ever being built. Anything that
 * would make a kernel index outside a buffer is rejected here, once, rather
 * than becoming a device-side out-of-bounds read later (which is a GPU
 * fault, not a catchable one). */
static sg_err gpu_check_model(const sg_model *m) {
    if (!m || !m->layers || !m->tok_emb || !m->out_norm || !m->lm_head) {
        return (sg_err){"gpu: invalid model"};
    }
    const sg_cfg *c = &m->cfg;
    if (c->n_layers == 0 || c->hidden == 0 || c->ffn_hidden == 0 || c->vocab == 0
        || c->head_dim == 0 || c->n_heads == 0 || c->n_kv_heads == 0) {
        return (sg_err){"gpu: model config has a zero dimension"};
    }
    if (c->n_heads % c->n_kv_heads != 0) {
        return (sg_err){"gpu: n_heads is not a multiple of n_kv_heads"};
    }
    if (c->full_attn_interval == 0) return (sg_err){"gpu: full_attn_interval is zero"};
    if (m->wtype != SG_T_BF16 && m->wtype != SG_T_F32 && m->wtype != SG_T_Q8_0) {
        return (sg_err){"gpu: the Metal path needs bf16, f32 or Q8_0 matmul weights"};
    }
    /* Q8_0 rows are whole 32-element blocks, so the contraction width of every
     * matmul (hidden, and the head-derived widths) must be a multiple of 32.
     * hidden is the one that is not already a multiple of head_dim; check it
     * up front so a bad checkpoint fails here with a clear message rather than
     * inside gpu_wrap_w. */
    if (m->wtype == SG_T_Q8_0 && c->hidden % 32 != 0) {
        return (sg_err){"gpu: Q8_0 weights need a hidden size that is a multiple of 32"};
    }
    if ((m->dense_type != SG_T_BF16 && m->dense_type != SG_T_F32)
        || (m->ssm_a_type != SG_T_BF16 && m->ssm_a_type != SG_T_F32)
        || (m->ssm_norm_type != SG_T_BF16 && m->ssm_norm_type != SG_T_F32)) {
        return (sg_err){"gpu: unsupported small-tensor dtype (want bf16 or f32)"};
    }
    if (!(c->rms_eps >= 0.0f) || !isfinite(c->rms_eps)) {
        return (sg_err){"gpu: rms_eps must be finite and non-negative"};
    }

    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const sg_layer_w *w = &m->layers[i];
        bool is_attn = (w->q_proj != NULL);
        if (is_attn != (((i + 1) % c->full_attn_interval) == 0)) {
            return (sg_err){"gpu: layer kind disagrees with full_attention_interval"};
        }
        const void *shared[] = { w->ln1, w->ln2, w->gate_proj, w->up_proj, w->down_proj };
        for (size_t k = 0; k < sizeof shared / sizeof *shared; k++) {
            if (!shared[k]) return (sg_err){"gpu: a layer is missing an MLP or norm tensor"};
        }
        if (is_attn) {
            const void *a[] = { w->k_proj, w->v_proj, w->o_proj, w->q_norm, w->k_norm };
            for (size_t k = 0; k < sizeof a / sizeof *a; k++) {
                if (!a[k]) return (sg_err){"gpu: a full-attention layer is incomplete"};
            }
            n_attn++;
        } else {
            const void *d[] = { w->ssm_in_qkv, w->ssm_in_z, w->ssm_in_b, w->ssm_in_a,
                                w->ssm_a, w->ssm_dt_bias, w->ssm_conv1d, w->ssm_norm,
                                w->ssm_out };
            for (size_t k = 0; k < sizeof d / sizeof *d; k++) {
                if (!d[k]) return (sg_err){"gpu: a gated-DeltaNet layer is incomplete"};
            }
            n_gdn++;
        }
    }
    if (n_attn > 0) {
        if (c->rope_dim < 2 || c->rope_dim > c->head_dim || c->rope_dim % 2 != 0) {
            return (sg_err){"gpu: rope_dim must be even and in [2, head_dim]"};
        }
        if (!(c->rope_theta > 1.0f) || !isfinite(c->rope_theta)) {
            return (sg_err){"gpu: rope_theta must be finite and greater than 1"};
        }
    }
    if (n_gdn > 0) {
        if (c->n_k_heads == 0 || c->n_v_heads == 0 || c->head_k_dim == 0
            || c->head_v_dim == 0 || c->conv_kernel == 0) {
            return (sg_err){"gpu: model has gated-DeltaNet layers but no DeltaNet dims"};
        }
        if (c->n_v_heads % c->n_k_heads != 0) {
            return (sg_err){"gpu: n_v_heads is not a multiple of n_k_heads"};
        }
    }

    /* Same 2^24 ceiling as ref.c: every derived width below is a product of
     * two config u32s and a lying config must not be able to wrap one. */
    const uint64_t width_max = 1u << 24;
    uint64_t key_dim = (uint64_t)c->n_k_heads * c->head_k_dim;
    uint64_t value_dim = (uint64_t)c->n_v_heads * c->head_v_dim;
    if (key_dim > width_max || value_dim > width_max
        || 2 * key_dim + value_dim > width_max
        || 2 * (uint64_t)c->n_heads * c->head_dim > width_max
        || (uint64_t)c->n_kv_heads * c->head_dim > width_max
        || (uint64_t)c->hidden > width_max || (uint64_t)c->ffn_hidden > width_max
        || (uint64_t)c->conv_kernel > width_max) {
        return (sg_err){"gpu: a model dimension is implausibly large"};
    }
    return SG_OK;
}

sg_err sg_gpu_load_model(sg_gpu *g, const sg_model *m) {
    if (!g) return (sg_err){"gpu: sg_gpu_load_model got a NULL gpu"};
    sg_err e = gpu_check_model(m);
    if (sg_failed(e)) return e;

    gpu_unload(g);
    const sg_cfg *c = &m->cfg;
    g->model = m;
    g->cfg = *c;
    g->key_dim = c->n_k_heads * c->head_k_dim;
    g->value_dim = c->n_v_heads * c->head_v_dim;
    g->conv_dim = 2 * g->key_dim + g->value_dim;
    g->attn_width = c->n_heads * c->head_dim;
    /* Doubled only when the model folds an output gate into q_proj (Task P1;
     * cfg.attn_output_gate, surge.h). Dense qwen3 has none. */
    g->q_width = (c->attn_output_gate ? 2u : 1u) * g->attn_width;
    g->kv_width = c->n_kv_heads * c->head_dim;
    g->mat_kernel = matmul_kernel_for(m->wtype);

    g->ls = calloc(c->n_layers, sizeof *g->ls);
    if (!g->ls) { gpu_unload(g); return (sg_err){"gpu: out of memory"}; }

    /* mlx's sanitize adds 1.0 to ln1 / ln2 / q_norm / k_norm / out_norm and
     * to nothing else; ssm_norm is deliberately not in that list. */
    float shift = m->norms_are_residual ? 1.0f : 0.0f;
    float *host = NULL;

    e = gpu_alloc_f32(g, c->hidden, &g->out_norm, &host);
    if (sg_failed(e)) { gpu_unload(g); return e; }
    gpu_widen(m->out_norm, m->dense_type, host, c->hidden, shift);

    e = gpu_wrap_w(g, m->lm_head, c->vocab, c->hidden, &g->lm_head);
    if (sg_failed(e)) { gpu_unload(g); return e; }

    for (uint32_t i = 0; i < c->n_layers; i++) {
        const sg_layer_w *w = &m->layers[i];
        sg_gpu_layer *L = &g->ls[i];
        L->is_attn = (w->q_proj != NULL);

#define WRAP(field, src, rows, cols) do { \
            e = gpu_wrap_w(g, (src), (rows), (cols), &L->field); \
            if (sg_failed(e)) { gpu_unload(g); return e; } \
        } while (0)
#define SMALL(field, n) do { \
            e = gpu_alloc_f32(g, (n), &L->field, &host); \
            if (sg_failed(e)) { gpu_unload(g); return e; } \
        } while (0)

        WRAP(w_gate, w->gate_proj, c->ffn_hidden, c->hidden);
        WRAP(w_up,   w->up_proj,   c->ffn_hidden, c->hidden);
        WRAP(w_down, w->down_proj, c->hidden,     c->ffn_hidden);

        SMALL(ln1, c->hidden);
        gpu_widen(w->ln1, m->dense_type, host, c->hidden, shift);
        SMALL(ln2, c->hidden);
        gpu_widen(w->ln2, m->dense_type, host, c->hidden, shift);

        if (L->is_attn) {
            WRAP(w_q, w->q_proj, g->q_width,  c->hidden);
            WRAP(w_k, w->k_proj, g->kv_width, c->hidden);
            WRAP(w_v, w->v_proj, g->kv_width, c->hidden);
            WRAP(w_o, w->o_proj, c->hidden,   g->attn_width);
            /* q_norm then k_norm in one buffer: the two are bound as the
             * same argument with a head_dim offset apart. */
            SMALL(qk_norm, 2 * (uint64_t)c->head_dim);
            gpu_widen(w->q_norm, m->dense_type, host, c->head_dim, shift);
            gpu_widen(w->k_norm, m->dense_type, host + c->head_dim, c->head_dim, shift);
        } else {
            WRAP(w_qkv, w->ssm_in_qkv, g->conv_dim,   c->hidden);
            WRAP(w_z,   w->ssm_in_z,   g->value_dim,  c->hidden);
            WRAP(w_a,   w->ssm_in_a,   c->n_v_heads,  c->hidden);
            WRAP(w_b,   w->ssm_in_b,   c->n_v_heads,  c->hidden);
            WRAP(w_out, w->ssm_out,    c->hidden,     g->value_dim);

            SMALL(conv_w, (uint64_t)g->conv_dim * c->conv_kernel);
            gpu_widen(w->ssm_conv1d, m->dense_type, host,
                      (uint64_t)g->conv_dim * c->conv_kernel, 0.0f);
            /* ssm_a and dt_bias adjacent, in that order: k_delta_gates reads
             * both from one buffer. ssm_a carries its own recorded dtype
             * (A_log or -exp(A_log), f32 or bf16); dt_bias follows
             * dense_type. Neither is shifted. */
            SMALL(a_dt, 2 * (uint64_t)c->n_v_heads);
            gpu_widen(w->ssm_a, m->ssm_a_type, host, c->n_v_heads, 0.0f);
            gpu_widen(w->ssm_dt_bias, m->dense_type, host + c->n_v_heads,
                      c->n_v_heads, 0.0f);
            /* z scratch with the ssm_norm weight parked behind it. */
            SMALL(zw, (uint64_t)g->value_dim + c->head_v_dim);
            gpu_widen(w->ssm_norm, m->ssm_norm_type, host + g->value_dim,
                      c->head_v_dim, 0.0f);
        }
#undef WRAP
#undef SMALL
    }
    return SG_OK;
}

sg_err sg_gpu_state_new(sg_gpu *g, const sg_model *m, uint32_t max_ctx) {
    if (!g) return (sg_err){"gpu: sg_gpu_state_new got a NULL gpu"};
    if (!g->model) return (sg_err){"gpu: call sg_gpu_load_model first"};
    if (m != g->model) return (sg_err){"gpu: this gpu was loaded with a different sg_model"};
    if (max_ctx == 0) return (sg_err){"gpu: max_ctx must be at least 1"};

    gpu_free_state(g);
    const sg_cfg *c = &g->cfg;
    g->max_ctx = max_ctx;
    g->used = 0;

    /* SURGE_KV_DTYPE selects the full-attention K/V cache's element type:
     * f16 (default, M5.2) or f32 (the pre-M5.2 path, kept byte-for-bit so
     * the M2 gate's oracle comparison is untouched). An unrecognized value
     * logs a warning and falls back to the default rather than picking one
     * silently. */
    const char *dt_env = getenv("SURGE_KV_DTYPE");
    sg_tensor_type kv_dtype = SG_T_F16;
    if (dt_env && strcmp(dt_env, "f32") == 0) {
        kv_dtype = SG_T_F32;
    } else if (dt_env && dt_env[0] != '\0' && strcmp(dt_env, "f16") != 0) {
        fprintf(stderr, "gpu: SURGE_KV_DTYPE='%s' not recognized (want f16 or f32); "
                        "using f16\n", dt_env);
    }
    g->kv_dtype = kv_dtype;
    fprintf(stderr, "gpu: KV cache dtype = %s\n", kv_dtype == SG_T_F16 ? "f16" : "f32");

    /* Task P2.3: SURGE_ATTN_SPLITK selects the decode-step attention kernel.
     * Default ON (the P2.3a sweep measured 15.9x on the 27B decode shape and
     * 21.9x on the 4B dense shape at seq 262144), "0" pins the incumbent
     * k_attn_decode_f16 -- which is what makes the A/B measurable and is the
     * reason that kernel is kept reachable. Unrecognized values warn and keep
     * the default rather than picking one silently, matching SURGE_KV_DTYPE
     * just above. Forced off on the f32 KV path: the split-K kernels read
     * half-typed SEPARATE K and V buffers, which only the fp16 cache has, and
     * the f32 path is deliberately frozen at its pre-M5.2 shape anyway. */
    const char *sk_env = getenv("SURGE_ATTN_SPLITK");
    bool splitk_on = true;
    if (sk_env && strcmp(sk_env, "0") == 0) {
        splitk_on = false;
    } else if (sk_env && sk_env[0] != '\0' && strcmp(sk_env, "1") != 0) {
        fprintf(stderr, "gpu: SURGE_ATTN_SPLITK='%s' not recognized (want 0 or 1); "
                        "using 1\n", sk_env);
    }
    g->attn_splitk = splitk_on && kv_dtype == SG_T_F16;

    /* Task P2.4: SURGE_ATTN_SPLITK_GQA selects WHICH split-K partial the
     * decode path dispatches, once split-K itself is on. DEFAULT 1 SINCE TASK
     * P4.0 (2026-08-18, user-approved): the GQA-shared
     * k_attn_decode_splitk_partial_gqa, which reads each K/V element once per
     * GQA GROUP instead of once per query head. "0" pins the per-head
     * k_attn_decode_splitk_partial P2.3 measured and shipped, which is what
     * keeps the A/B runnable on ONE binary and is why that kernel stays
     * reachable rather than being deleted.
     *
     * THE FLIP WAS NOT FREE, IT WAS GATED. Three things had to be true, and each
     * has a gate (docs/17082026_splitk_gqa_threadgroups.md):
     *   - the two four-pass partials are BYTE-IDENTICAL at a fixed n_splits
     *     (P2.4), so this switch cannot change the answer wherever the two split
     *     policies agree, which is every seq below 65792 at the shipped cap;
     *   - where the GQA policy picks a DIFFERENT n_splits (P2.5), the greedy
     *     tokens were gated on a real model rather than argued from margins
     *     (P2.6: 321/321 positions differ numerically, argmax agrees 1600/1600);
     *   - the occupancy floor (P2.7) keeps the GQA kernel out of every shape and
     *     depth where the sweep MEASURED it losing, so the flip cannot regress
     *     short context. Measured 1.74x over the per-head partial at the 27B
     *     262144 decode shape.
     *
     * Same warn-and-keep-the-default rule for an unrecognized value as the two
     * env vars above; the one spelling that turns this off is exactly "0". */
    const char *gq_env = getenv("SURGE_ATTN_SPLITK_GQA");
    bool gqa_on = true;
    if (gq_env && strcmp(gq_env, "0") == 0) {
        gqa_on = false;
    } else if (gq_env && gq_env[0] != '\0' && strcmp(gq_env, "1") != 0) {
        fprintf(stderr, "gpu: SURGE_ATTN_SPLITK_GQA='%s' not recognized (want 0 or 1); "
                        "using 1\n", gq_env);
    }
    g->attn_splitk_gqa = gqa_on && g->attn_splitk;

    /* Task P2.8: SURGE_ATTN_SPLITK_ONLINE selects the ONLINE-SOFTMAX GQA
     * partial, k_attn_decode_splitk_partial_gqa_online, once split-K itself is
     * on. DEFAULT 0. It is a peer of SURGE_ATTN_SPLITK_GQA, not a modifier of
     * it: the online kernel is already GQA-shared, and if both are set the
     * online one runs (splitk_gqa_use yields to it).
     *
     * REJECTED, NOT IGNORED, on anything that is not exactly "0" or "1", which
     * is SURGE_SPLITK_GQA_CAP's rule below rather than the warn-and-default rule
     * the two switches above use, and for the same reason P2.6 gave: the gate
     * for this switch is an A/B, and the two arms of an A/B agree perfectly when
     * the "on" arm was silently never turned on. A typo ("on", "true", "yes",
     * "1 ") would make that gate pass vacuously, so it fails loudly here
     * instead. There is nothing lost by refusing: the only values that mean
     * anything are the two spellings accepted. */
    const char *on_env = getenv("SURGE_ATTN_SPLITK_ONLINE");
    bool online_on = false;
    if (on_env && on_env[0] != '\0') {
        if (strcmp(on_env, "1") == 0) {
            online_on = true;
        } else if (strcmp(on_env, "0") != 0) {
            gpu_free_state(g);
            return gpu_errf("gpu: SURGE_ATTN_SPLITK_ONLINE='%s' must be exactly 0 or 1 "
                            "(default 0; it selects the online-softmax split-K partial, "
                            "and a silently ignored value would make its A/B vacuous)",
                            on_env);
        }
    }
    g->attn_splitk_online = online_on && g->attn_splitk;
    if (g->attn_splitk_online) {
        fprintf(stderr, "gpu: SURGE_ATTN_SPLITK_ONLINE=1, decode attention uses the "
                        "online-softmax GQA split-K partial where the policy admits it "
                        "(head_dim <= %u, GQA group in [2, %u], >= %u threadgroups)\n",
                SG_TG, SG_SPLITK_GQA_MAX, SG_SPLITK_GQA_MIN_TG);
    }

    /* Task P2.6: SURGE_SPLITK_GQA_CAP overrides the GQA split policy's measured
     * saturation cap (SG_SPLITK_GQA_N_SPLITS_CAP == 256). Unset keeps the
     * measured value, which is the shipped policy and the only one anyone
     * running surge should use.
     *
     * IT EXISTS FOR THE P2.6 GATE AND FOR RETUNING, NOT FOR USERS. The GQA and
     * per-head policies only diverge from seq SG_TG * (cap + 1) on, so at the
     * shipped 256 the divergence starts at 65792 keys, which no `make check`
     * can reach. A lower cap reproduces the SAME mechanism at a reachable
     * length so the greedy-token gate can run in seconds.
     *
     * REJECTED, NOT IGNORED, on anything that is not a plain integer in
     * [SG_SPLITK_MIN, SG_SPLITK_MAX]. Every other env var in this function
     * warns and falls back, because falling back to the shipped default is the
     * safe answer there. Here it is not: a gate that sets this var and has it
     * silently ignored passes VACUOUSLY, with both arms picking the same
     * n_splits and nothing under test. Outside the band the value is also
     * meaningless (below SG_SPLITK_MIN the floor clamp eats it, above
     * SG_SPLITK_MAX the ceiling does), so there is no useful value being
     * refused. strtol with a leading-digit check AND a full-string end check, so
     * "4x" and "" are errors rather than 4 and 0, and so " 4" / "+4" are errors
     * too: strtol would silently accept both (it skips leading whitespace and a
     * sign), which would make this parser's behaviour wider than the "plain
     * integer" it documents. */
    const char *cap_env = getenv("SURGE_SPLITK_GQA_CAP");
    if (cap_env && cap_env[0] != '\0') {
        char *cap_end = NULL;
        errno = 0;
        long cap_v = strtol(cap_env, &cap_end, 10);
        if (errno != 0 || cap_env[0] < '0' || cap_env[0] > '9'
            || !cap_end || *cap_end != '\0'
            || cap_v < (long)SG_SPLITK_MIN || cap_v > (long)SG_SPLITK_MAX) {
            gpu_free_state(g);
            return gpu_errf("gpu: SURGE_SPLITK_GQA_CAP='%s' is not an integer in "
                            "[%u, %u] (default %u; this override exists for the "
                            "P2.6 greedy-token gate, not for tuning a run)",
                            cap_env, SG_SPLITK_MIN, SG_SPLITK_MAX,
                            SG_SPLITK_GQA_N_SPLITS_CAP);
        }
        g->splitk_gqa_cap = (uint32_t)cap_v;
        fprintf(stderr, "gpu: SURGE_SPLITK_GQA_CAP = %u (default %u); the GQA and "
                        "per-head split policies now diverge from seq %u\n",
                g->splitk_gqa_cap, SG_SPLITK_GQA_N_SPLITS_CAP,
                SG_TG * (g->splitk_gqa_cap + 1));
    }

    /* Zeroed here as well as in gpu_free_state (which ran just above), so a
     * gate can read them as "what this state did", not "what this process
     * did". */
    g->splitk_partial_dispatches = 0;
    g->splitk_gqa_dispatches = 0;
    g->splitk_online_dispatches = 0;
    g->splitk_max_splits = 0;

    uint64_t half = 0;
    if (kv_dtype == SG_T_F32) {
        /* k_attn_decode's v_cache offset is a uint32 param, so the K half of
         * one layer's cache has to be addressable in floats by a uint32.
         * That is 4 billion floats; the real ceiling here is memory, not
         * this check, but a silently truncated offset would read the wrong
         * half of the cache. Only the f32 path's combined buffer needs this
         * check: k_attn_decode_f16 has no v_cache offset at all (K and V
         * are separate buffers), and sg_kv_new enforces its own cap ceiling
         * (SG_KV_CAP_MAX) on that path instead. */
        half = (uint64_t)max_ctx * g->kv_width;
        if (half > UINT32_MAX) return (sg_err){"gpu: max_ctx is too large for this kv cache layout"};
    }

    sg_err e;
    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        if (g->ls[i].is_attn) n_attn++; else n_gdn++;
    }

#define SHARED(field, n) do { \
        e = gpu_alloc_f32(g, (n), &g->field, NULL); \
        if (sg_failed(e)) { gpu_free_state(g); return e; } \
    } while (0)

    e = gpu_alloc_f32(g, c->hidden, &g->b_x, &g->h_x);
    if (sg_failed(e)) { gpu_free_state(g); return e; }
    SHARED(b_h, c->hidden);
    SHARED(b_r, c->hidden);
    SHARED(b_ffg, c->ffn_hidden);
    SHARED(b_ffu, c->ffn_hidden);
    e = gpu_alloc_f32(g, c->vocab, &g->b_logits, &g->h_logits);
    if (sg_failed(e)) { gpu_free_state(g); return e; }

    if (n_attn > 0) {
        SHARED(b_qg, g->q_width);
        SHARED(b_ctx, g->attn_width);
        e = gpu_alloc_f32(g, c->rope_dim, &g->b_cs, &g->h_cs);
        if (sg_failed(e)) { gpu_free_state(g); return e; }
        /* One private score row per query head, the full context long. */
        e = scratch_ensure(g, (uint64_t)c->n_heads * max_ctx * 4);
        if (sg_failed(e)) { gpu_free_state(g); return e; }
        if (kv_dtype == SG_T_F16) {
            /* fp16 path: this token's freshly computed K and V land here in
             * f32 for q/k-norm and RoPE (same in-place ops the f32 path
             * uses), and k_kv_store_f16 then casts the result into g->kv's
             * per-layer half buffers. */
            SHARED(b_k32, g->kv_width);
            SHARED(b_v32, g->kv_width);

            /* Task P2.3: the split-K decode-attention buffers. Sized ONCE,
             * here, for the worst case this state can ever reach (n_splits is
             * nondecreasing in seq and seq <= max_ctx), so enc_attn_splitk can
             * never need to allocate or grow anything from inside an open
             * command buffer.
             *
             * The scratch bound: the partial's score rows are
             * n_heads * n_splits * ceil(seq/n_splits) floats, and
             * n_splits * ceil(seq/n_splits) <= seq + n_splits - 1 for every
             * seq, so n_heads * (max_ctx + splitk_max_splits) floats covers
             * every step at this max_ctx with room to spare. It is asked for
             * through splitk_scratch_ensure, the DEDICATED allocator (P2.2
             * review finding 1), never scratch_ensure. */
            g->splitk_max_splits = splitk_n_splits(max_ctx);
            SHARED(b_sk_m, (uint64_t)c->n_heads * g->splitk_max_splits);
            SHARED(b_sk_s, (uint64_t)c->n_heads * g->splitk_max_splits);
            SHARED(b_sk_acc, (uint64_t)c->n_heads * g->splitk_max_splits * c->head_dim);
            e = splitk_scratch_ensure(g, ((uint64_t)max_ctx + g->splitk_max_splits)
                                             * c->n_heads * 4);
            if (sg_failed(e)) { gpu_free_state(g); return e; }
        }
    }
    if (n_gdn > 0) {
        SHARED(b_qkv, g->conv_dim);
        SHARED(b_ab, 2 * (uint64_t)c->n_v_heads);
        SHARED(b_gates, 2 * (uint64_t)c->n_v_heads);
        SHARED(b_y, g->value_dim);
    }
#undef SHARED

    for (uint32_t i = 0; i < c->n_layers; i++) {
        sg_gpu_layer *L = &g->ls[i];
        if (L->is_attn) {
            if (kv_dtype == SG_T_F32) {
                e = gpu_alloc_f32(g, SG_KV_GROUPS * half, &L->kv, NULL);
                if (sg_failed(e)) { gpu_free_state(g); return e; }
            }
            /* kv_dtype == SG_T_F16: nothing per-layer here. g->kv (below)
             * owns every full-attention layer's K and V buffer at once. */
        } else {
            e = gpu_alloc_f32(g, (uint64_t)g->conv_dim * c->conv_kernel,
                              &L->conv_buf, NULL);
            if (sg_failed(e)) { gpu_free_state(g); return e; }
            e = gpu_alloc_f32(g, (uint64_t)c->n_v_heads * c->head_v_dim * c->head_k_dim,
                              &L->ssm, NULL);
            if (sg_failed(e)) { gpu_free_state(g); return e; }
        }
    }

    if (kv_dtype == SG_T_F16 && (n_attn > 0 || n_gdn > 0)) {
        /* sg_kv_new sizes and allocates every full-attention layer's
         * SEPARATE K and V buffer (the M5.1 layout) in one call, through the
         * backend sg_gpu_init registered. It also sizes DeltaNet conv/S
         * state internally (sg_kv models both layer kinds), which duplicates
         * the ad hoc L->conv_buf/L->ssm allocated just above; on the decode
         * path that state is untouched (decode reads L->conv_buf/L->ssm), but
         * M5.6's sg_gpu_prefill DOES thread the DeltaNet scan through sg_kv's
         * conv/S carriers and then bridges the final state back into
         * L->conv_buf/L->ssm, so those carriers must exist.
         *
         * M5.6 widened this from `n_attn > 0` to also allocate when the model
         * has ONLY DeltaNet layers: enc_gdn_prefill dereferences
         * sg_kv_conv/sg_kv_s, so a hybrid OR a DeltaNet-only model must have
         * g->kv non-NULL or prefill would fault on a NULL carrier. (With
         * layers > 0, `n_attn > 0 || n_gdn > 0` is always true on the f16
         * path, so this is effectively "always allocate on f16"; the explicit
         * disjunction documents WHY.) */
        e = sg_kv_new(g, c, max_ctx, SG_T_F16, &g->kv);
        if (sg_failed(e)) { gpu_free_state(g); return e; }
    }

    g->have_state = true;
    return SG_OK;
}

void sg_gpu_state_reset(sg_gpu *g) {
    if (!g || !g->have_state) return;
    g->used = 0;
    for (uint32_t i = 0; i < g->cfg.n_layers; i++) {
        sg_gpu_layer *L = &g->ls[i];
        /* The K/V caches are not cleared: nothing reads past `used`. The
         * DeltaNet state is, because it is read unconditionally. */
        float *h = (float *)sg_gpu_buf_host(L->conv_buf);
        if (h) memset(h, 0, (size_t)g->conv_dim * g->cfg.conv_kernel * sizeof *h);
        h = (float *)sg_gpu_buf_host(L->ssm);
        if (h) {
            memset(h, 0, (size_t)g->cfg.n_v_heads * g->cfg.head_v_dim
                             * g->cfg.head_k_dim * sizeof *h);
        }
    }
}

uint32_t sg_gpu_used(const sg_gpu *g) {
    if (!g || !g->have_state) return 0;
    return g->used;
}

/* --------------------------------------------------------------------
 * One token
 * -------------------------------------------------------------------- */

sg_err sg_gpu_forward(sg_gpu *g, const sg_model *m, int32_t token, uint32_t pos,
                      const float **logits) {
    if (!g || !m) return (sg_err){"gpu: sg_gpu_forward got a NULL argument"};
    if (!g->have_state) return (sg_err){"gpu: call sg_gpu_state_new first"};
    if (m != g->model) return (sg_err){"gpu: this gpu was loaded with a different sg_model"};
    const sg_cfg *c = &g->cfg;
    if (token < 0 || (uint32_t)token >= c->vocab) return (sg_err){"gpu: token id out of range"};
    if (pos >= g->max_ctx) return (sg_err){"gpu: position exceeds max_ctx"};
    /* The caches are append-only, so positions must arrive in order. */
    if (pos != g->used) return (sg_err){"gpu: positions must be presented in order"};

    /* No embedding scale: qwen3_5.py returns embed_tokens(inputs) untouched. */
    gpu_embed_row(m->tok_emb, m->wtype, (uint64_t)token, c->hidden, g->h_x);

    /* The RoPE angle and its sine/cosine in DOUBLE, uploaded as f32. See the
     * note on sg_ref_rope_partial: at this checkpoint's parameters the f32
     * rounding of the angle alone is worth 8e-3 at position 262143, which no
     * f32 kernel can undo. */
    if (g->h_cs) {
        uint32_t half = c->rope_dim / 2;
        for (uint32_t i = 0; i < half; i++) {
            double inv_freq = pow((double)c->rope_theta,
                                  -2.0 * (double)i / (double)c->rope_dim);
            double ang = (double)pos * inv_freq;
            g->h_cs[i] = (float)cos(ang);
            g->h_cs[half + i] = (float)sin(ang);
        }
    }

    __block sg_err rc = SG_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        /* Default dispatch type is MTLDispatchTypeSerial: dispatches run in
         * encode order with an implicit barrier between them, which is what
         * every read-after-write below depends on. */
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        if (!cb || !e) {
            rc = (sg_err){"gpu: could not open a compute encoder"};
        } else {
            sg_enc E = { g, e };
            uint32_t eps = fbits(c->rms_eps);

            for (uint32_t i = 0; i < c->n_layers; i++) {
                sg_gpu_layer *L = &g->ls[i];

                enc_op(&E, KI_RMSNORM, g->b_x, 0, L->ln1, 0, g->b_h, 0, nil, 0,
                       PARAMS(c->hidden, eps, 1));
                if (L->is_attn) enc_attn(&E, L, i, pos);
                else            enc_gdn(&E, L);
                enc_op(&E, KI_ADD, g->b_x, 0, g->b_r, 0, g->b_x, 0, nil, 0,
                       PARAMS(c->hidden));

                enc_op(&E, KI_RMSNORM, g->b_x, 0, L->ln2, 0, g->b_h, 0, nil, 0,
                       PARAMS(c->hidden, eps, 1));
                enc_op(&E, g->mat_kernel, L->w_gate, 0, g->b_h, 0, g->b_ffg, 0, nil, 0,
                       PARAMS(c->ffn_hidden, c->hidden));
                enc_op(&E, g->mat_kernel, L->w_up, 0, g->b_h, 0, g->b_ffu, 0, nil, 0,
                       PARAMS(c->ffn_hidden, c->hidden));
                enc_op(&E, KI_SWIGLU, g->b_ffg, 0, g->b_ffu, 0, g->b_ffg, 0, nil, 0,
                       PARAMS(c->ffn_hidden));
                enc_op(&E, g->mat_kernel, L->w_down, 0, g->b_ffg, 0, g->b_r, 0, nil, 0,
                       PARAMS(c->hidden, c->ffn_hidden));
                enc_op(&E, KI_ADD, g->b_x, 0, g->b_r, 0, g->b_x, 0, nil, 0,
                       PARAMS(c->hidden));
            }

            enc_op(&E, KI_RMSNORM, g->b_x, 0, g->out_norm, 0, g->b_h, 0, nil, 0,
                   PARAMS(c->hidden, eps, 1));
            /* lm_head aliases tok_emb when the embeddings are tied, which is
             * exactly mlx's embed_tokens.as_linear(out). */
            enc_op(&E, g->mat_kernel, g->lm_head, 0, g->b_h, 0, g->b_logits, 0, nil, 0,
                   PARAMS(c->vocab, c->hidden));

            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb error]) {
                rc = gpu_errf("gpu: decode step failed: %s",
                              [[[cb error] localizedDescription] UTF8String]);
            }
        }
    }
    if (sg_failed(rc)) return rc;

    g->used = pos + 1;
    if (logits) *logits = g->h_logits;
    return SG_OK;
}

/* --------------------------------------------------------------------
 * Chunked prompt prefill (Task M5.6)
 * --------------------------------------------------------------------
 *
 * sg_gpu_prefill wires the M5.4 full-attention prefill encoder
 * (enc_attn_prefill) and the M5.5 gated-DeltaNet prefill encoder
 * (enc_gdn_prefill) across ALL layers, one command buffer per chunk, and
 * closes the three M5.2/M5.4/M5.5 carry-forwards:
 *
 *   1. STATE BRIDGING. enc_gdn_prefill threads the DeltaNet recurrent state
 *      (conv tail + S) through sg_kv's per-layer conv/S carriers, but decode
 *      reads that state from the ad hoc L->conv_buf / L->ssm. After the last
 *      chunk this copies each DeltaNet layer's final conv tail and S out of the
 *      sg_kv carriers into L->conv_buf / L->ssm -- a small fixed-size per-layer
 *      host memcpy, safe because every chunk command buffer has been waited on
 *      and both are shared-storage unified memory. Full-attention KV is written
 *      straight into g->kv (the same buffers decode reads), and g->used is left
 *      at n_tokens with sg_kv advanced to match, so decode continues at
 *      pos == n_tokens with no further bridging.
 *   2. g->kv ALLOCATION. Fixed in sg_gpu_state_new (allocated for a hybrid OR a
 *      DeltaNet-only model on the f16 path, not only full-attn f16).
 *   3. WIRING. This is the first end-to-end drive of enc_attn_prefill /
 *      enc_gdn_prefill; tests/test_gpu_prefill.c gates it against the serial
 *      forward on the mini hybrid, which has both a full-attn and DeltaNet layer.
 *
 * The prefill encoders read the g->b_* scratch fields, which sg_gpu_state_new
 * sizes for ONE token. This swaps chunk-sized scratch into those fields for the
 * duration, then frees it and restores the one-token decode buffers untouched,
 * so a following sg_gpu_forward finds exactly the state it expects.
 */
typedef struct {
    void *b_x, *b_h, *b_r, *b_qg, *b_ctx, *b_ffg, *b_ffu,
         *b_qkv, *b_ab, *b_gates, *b_y, *b_cs, *b_k32, *b_v32;
    float *h_x, *h_cs;
} sg_prefill_bufs;

static void prefill_save(sg_gpu *g, sg_prefill_bufs *s) {
    s->b_x = g->b_x; s->b_h = g->b_h; s->b_r = g->b_r; s->b_qg = g->b_qg;
    s->b_ctx = g->b_ctx; s->b_ffg = g->b_ffg; s->b_ffu = g->b_ffu;
    s->b_qkv = g->b_qkv; s->b_ab = g->b_ab; s->b_gates = g->b_gates;
    s->b_y = g->b_y; s->b_cs = g->b_cs; s->b_k32 = g->b_k32; s->b_v32 = g->b_v32;
    s->h_x = g->h_x; s->h_cs = g->h_cs;
}

static void prefill_restore(sg_gpu *g, const sg_prefill_bufs *s) {
    g->b_x = s->b_x; g->b_h = s->b_h; g->b_r = s->b_r; g->b_qg = s->b_qg;
    g->b_ctx = s->b_ctx; g->b_ffg = s->b_ffg; g->b_ffu = s->b_ffu;
    g->b_qkv = s->b_qkv; g->b_ab = s->b_ab; g->b_gates = s->b_gates;
    g->b_y = s->b_y; g->b_cs = s->b_cs; g->b_k32 = s->b_k32; g->b_v32 = s->b_v32;
    g->h_x = s->h_x; g->h_cs = s->h_cs;
}

/* Frees whatever chunk buffers are currently in g's b_* fields and NULLs them,
 * so a partial-allocation failure and the normal teardown share one path. */
static void prefill_free_chunk(sg_gpu *g) {
    void *bufs[] = { g->b_x, g->b_h, g->b_r, g->b_qg, g->b_ctx, g->b_ffg,
                     g->b_ffu, g->b_qkv, g->b_ab, g->b_gates, g->b_y, g->b_cs,
                     g->b_k32, g->b_v32 };
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) sg_gpu_buf_free(bufs[i]);
    g->b_x = g->b_h = g->b_r = g->b_qg = g->b_ctx = g->b_ffg = g->b_ffu = NULL;
    g->b_qkv = g->b_ab = g->b_gates = g->b_y = g->b_cs = NULL;
    g->b_k32 = g->b_v32 = NULL;
    g->h_x = g->h_cs = NULL;
}

/* Monotonic seconds for the long-prefill progress log below and for the B8
 * duty-cycle's own GPU-busy accounting. */
static double pf_now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Task B8: sleeps ms milliseconds via nanosleep, retrying across EINTR (with
 * the remaining time nanosleep itself writes back into req) so a stray
 * signal cannot silently cut a rest short -- load-bearing for the real
 * 90000ms (90s) rests this feature exists for: a truncated rest could leave
 * the GPU idle for less than the firmware limiter's 60-120s recovery window,
 * quietly defeating the whole mitigation. Deliberately nanosleep rather than
 * usleep: nanosleep takes seconds+nanoseconds directly (no risk of a
 * uint32_t microsecond product overflowing) and its POSIX contract includes
 * writing the unslept remainder back on an EINTR return, which is exactly
 * what the retry loop needs. */
/*
 * Margin applied to the previous chunk's GPU time when predicting the next
 * chunk's. A heuristic, not a guarantee: chunk cost grows monotonically with
 * context because each chunk attends over a longer KV cache, so last-chunk time
 * always under-estimates next-chunk time, and the growth per chunk is small
 * once the context is large. Measured over the 2026-08-14 256K run, consecutive
 * chunk times grew by well under 25%. Callers who need a hard bound should set
 * the work budget below the watchdog window rather than rely on this.
 */
#define PF_EST_MARGIN 1.25

static void pf_sleep_ms(uint32_t ms) {
    struct timespec req = { .tv_sec = (time_t)(ms / 1000u),
                             .tv_nsec = (long)(ms % 1000u) * 1000000L };
    while (nanosleep(&req, &req) != 0 && errno == EINTR) {
        /* req now holds the remaining time; loop to finish it out. */
    }
}

/* Task B8: arms/disarms the prefill duty-cycle. Either argument 0 disables
 * it (the calloc default): no accumulator crosses work_budget_ms and
 * sg_gpu_prefill never sleeps, so its OUTPUT is byte-identical to before this
 * task (the chunk loop's two clock_gettime reads per chunk still run either
 * way, but feed only rest accounting). When both are nonzero, sg_gpu_prefill
 * sleeps rest_ms (GPU fully idle, no command buffer in flight) once the
 * accumulated GPU-busy wall time across chunks
 * reaches work_budget_ms, provided at least one chunk still remains; this is
 * pure timing and touches no buffer, state, or accumulator that feeds the
 * computed logits or KV state. Settings persist on g until changed again. */
void sg_gpu_set_prefill_rest(sg_gpu *g, uint32_t work_budget_ms, uint32_t rest_ms) {
    if (!g) return;
    g->prefill_work_budget_ms = work_budget_ms;
    g->prefill_rest_ms = rest_ms;
}

/*
 * Target ceiling, in ms, for how long any one prefill command buffer holds the
 * GPU. 0 (the calloc default) disables segmentation: one command buffer per
 * chunk, exactly as before.
 *
 * A target, not a guarantee, and specifically NOT a promise of a single
 * overrun. The check is reactive, so each overrun happens in full and the size
 * only halves afterwards: a 64-layer sweep can overrun at 64, then 32, then 16,
 * converging rather than stopping. A segment already at the 1-layer floor
 * cannot shrink further and will keep overrunning. The narrowed size does
 * persist for the rest of the call, so the sequence is bounded by log2(layers)
 * overruns, not by one.
 *
 * This is a different mechanism from the duty-cycle rest and solves a different
 * problem. The rest yields the GPU BETWEEN chunks; this bounds how long a
 * SINGLE submission holds it. Only the latter can protect the compositor once
 * one chunk runs longer than the watchdog window on its own.
 *
 * Unlike changing `chunk`, this cannot alter output: command buffer boundaries
 * carry no state, the same kernels are dispatched with the same arguments in
 * the same order, and buffers committed in sequence on one queue execute in
 * order. Settings persist on g until changed again.
 */
void sg_gpu_set_prefill_max_burst(sg_gpu *g, uint32_t max_burst_ms) {
    if (!g) return;
    g->prefill_max_burst_ms = max_burst_ms;
}

uint64_t sg_gpu_prefill_segments(const sg_gpu *g) {
    return g ? g->prefill_segments : 0;
}

/* Total time (ms) sg_gpu_prefill actually slept during its most recent call;
 * 0 if the feature was disabled or never triggered (e.g. every chunk's
 * GPU-busy time stayed under work_budget_ms). Reset to 0 at the start of
 * every sg_gpu_prefill call, so this always reflects the LAST call, not a
 * running total across calls. */
uint64_t sg_gpu_prefill_rest_ms(const sg_gpu *g) {
    return g ? g->prefill_rest_total_ms : 0;
}

sg_err sg_gpu_prefill(sg_gpu *g, const sg_model *m, const int32_t *tokens,
                      uint32_t n_tokens, uint32_t chunk_size,
                      const float **out_last_logits) {
    if (!g) return (sg_err){"gpu: sg_gpu_prefill got a NULL argument"};
    /* Task B8: reset the rest-time counter as the very first thing once g is
     * known non-NULL, before ANY other validation (including the m/tokens
     * NULL check right below) can return early. sg_gpu_prefill_rest_ms's
     * contract is "the most recent call", and a failed call (NULL m/tokens,
     * bad args, wrong KV dtype, oversized chunk, allocation failure) is still
     * a call: without this reset being unconditionally first, a call that
     * fails validation would otherwise still report whatever a PRIOR
     * successful call slept, which is stale and misleading. */
    g->prefill_rest_total_ms = 0;
    g->prefill_segments = 0;
    if (!m || !tokens) return (sg_err){"gpu: sg_gpu_prefill got a NULL argument"};
    if (!g->have_state) return (sg_err){"gpu: call sg_gpu_state_new first"};
    if (m != g->model) return (sg_err){"gpu: this gpu was loaded with a different sg_model"};
    const sg_cfg *c = &g->cfg;
    if (n_tokens == 0) return (sg_err){"gpu: sg_gpu_prefill needs at least one token"};
    if (n_tokens > g->max_ctx) return (sg_err){"gpu: prompt length exceeds max_ctx"};
    for (uint32_t t = 0; t < n_tokens; t++) {
        if (tokens[t] < 0 || (uint32_t)tokens[t] >= c->vocab) {
            return (sg_err){"gpu: a prompt token id is out of range"};
        }
    }
    /* enc_attn_prefill reads g->kv's fp16 K/V and enc_gdn_prefill reads its
     * conv/S carriers; the f32 KV path has no sg_kv object at all. */
    if (g->kv_dtype != SG_T_F16 || !g->kv) {
        return (sg_err){"gpu: sg_gpu_prefill needs the f16 KV path (SURGE_KV_DTYPE=f16)"};
    }

    uint32_t chunk = chunk_size ? chunk_size : SG_PREFILL_CHUNK_DEFAULT;
    uint32_t cn = (chunk < n_tokens) ? chunk : n_tokens;   /* longest chunk used */

    /* Several per-chunk kernel params are a chunk-length-times-width product
     * passed as a uint32 (KI_ADD's n*hidden, KI_SWIGLU's n*ffn_hidden,
     * KI_SILU's n*conv_dim in enc_gdn_prefill, and the smaller per-token slice
     * counts n*n_heads / n*n_kv_heads / n*n_v_heads). Those are formed in 32-bit
     * C arithmetic, so a chunk wide enough to overflow one would silently wrap
     * it (a WRONG result, not a device fault) even though every buffer is sized
     * in u64. n <= cn for every chunk, so bounding cn against the widest such
     * factor rejects the whole class up front. In practice the f16 path caps
     * n_tokens at SG_KV_CAP_MAX (262144), so this only bites an absurdly large
     * --chunk on a very wide model; reject with a clear message rather than
     * truncate. */
    {
        uint64_t widest = c->hidden;
        if (c->ffn_hidden > widest) widest = c->ffn_hidden;
        if (g->conv_dim > widest)   widest = g->conv_dim;
        if (g->value_dim > widest)  widest = g->value_dim;
        if (g->q_width > widest)    widest = g->q_width;
        uint64_t prod = 0;
        if (!mul_ck((uint64_t)cn, widest, &prod) || prod > UINT32_MAX) {
            return (sg_err){"gpu: chunk size too large for this model's widths "
                            "(a 32-bit kernel param would overflow); use a smaller --chunk"};
        }
    }

    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        if (g->ls[i].is_attn) n_attn++; else n_gdn++;
    }

    int gemm = gemm_kernel_for(m->wtype);
    uint32_t eps = fbits(c->rms_eps);
    uint32_t half = c->rope_dim / 2u;

    /* Swap the one-token decode scratch out and chunk-sized scratch in. */
    sg_prefill_bufs saved;
    prefill_save(g, &saved);
    g->b_x = g->b_h = g->b_r = g->b_qg = g->b_ctx = g->b_ffg = g->b_ffu = NULL;
    g->b_qkv = g->b_ab = g->b_gates = g->b_y = g->b_cs = NULL;
    g->b_k32 = g->b_v32 = NULL;
    g->h_x = g->h_cs = NULL;

    sg_err e = SG_OK;
#define PF_ALLOC(field, nelem, hostpp) do { \
        e = gpu_alloc_f32(g, (uint64_t)(nelem), &g->field, (hostpp)); \
        if (sg_failed(e)) goto cleanup; \
    } while (0)

    PF_ALLOC(b_x, (uint64_t)cn * c->hidden, &g->h_x);
    PF_ALLOC(b_h, (uint64_t)cn * c->hidden, NULL);
    PF_ALLOC(b_r, (uint64_t)cn * c->hidden, NULL);
    PF_ALLOC(b_ffg, (uint64_t)cn * c->ffn_hidden, NULL);
    PF_ALLOC(b_ffu, (uint64_t)cn * c->ffn_hidden, NULL);
    if (n_attn > 0) {
        PF_ALLOC(b_qg, (uint64_t)cn * g->q_width, NULL);
        PF_ALLOC(b_ctx, (uint64_t)cn * g->attn_width, NULL);
        PF_ALLOC(b_k32, (uint64_t)cn * g->kv_width, NULL);
        PF_ALLOC(b_v32, (uint64_t)cn * g->kv_width, NULL);
        PF_ALLOC(b_cs, (uint64_t)cn * c->rope_dim, &g->h_cs);
    }
    if (n_gdn > 0) {
        PF_ALLOC(b_qkv, (uint64_t)cn * g->conv_dim, NULL);
        PF_ALLOC(b_ab, 2ull * cn * c->n_v_heads, NULL);
        PF_ALLOC(b_gates, 2ull * cn * c->n_v_heads, NULL);
        PF_ALLOC(b_y, (uint64_t)cn * g->value_dim, NULL);
    }
#undef PF_ALLOC

    /* Start clean at position 0: reset used, zero the sg_kv DeltaNet carriers
     * (read unconditionally by the scan), and zero the ad hoc decode conv/S
     * (bridged from sg_kv at the very end, but zeroed now so the state is well
     * defined throughout). K/V caches are append-only and need no clearing. */
    g->used = 0;
    sg_kv_reset(g->kv);
    for (uint32_t i = 0; i < c->n_layers; i++) {
        sg_gpu_layer *L = &g->ls[i];
        if (L->is_attn) continue;
        float *h = (float *)sg_gpu_buf_host(L->conv_buf);
        if (h) memset(h, 0, (size_t)g->conv_dim * c->conv_kernel * sizeof *h);
        h = (float *)sg_gpu_buf_host(L->ssm);
        if (h) {
            memset(h, 0, (size_t)c->n_v_heads * c->head_v_dim
                             * c->head_k_dim * sizeof *h);
        }
    }

    /* Progress log for long prefills (a 256K ingest is minutes-to-hours on this
     * box). Auto-quiet for short prompts so `make check` stays silent; logs a
     * throttled line every 8th chunk plus the last. */
    /* Segmentation starts wide (one command buffer for the whole layer sweep,
     * i.e. today's behaviour) and narrows only if a submission overruns the
     * ceiling. Reset per call so one long-context run cannot leave a later,
     * shorter one permanently over-segmented. */
    g->prefill_seg_layers = c->n_layers;

    bool pf_log = n_tokens >= 8192u;
    double t_pf0 = pf_now_s();

    /* Task B8: prefill duty-cycle. rest_enabled false (either setting 0, the
     * calloc default) leaves every buffer, KV/accumulator write, and control
     * flow below untouched -- pf_work_acc_ms stays unread, no sleep ever
     * runs, and g->prefill_rest_total_ms (already reset to 0 above) stays 0,
     * so a disabled run's OUTPUT (gen_ids, logits, KV/decode state, g->used)
     * is byte-identical to before this task. The chunk-loop timing capture
     * below (t_gpu0/t_gpu1, two extra clock_gettime calls per chunk) does
     * run either way, but feeds only this accounting, never anything
     * output-affecting. */
    bool rest_enabled = (g->prefill_work_budget_ms > 0 && g->prefill_rest_ms > 0);
    double pf_work_acc_ms = 0.0;

    for (uint32_t base = 0; base < n_tokens; base += chunk) {
        uint32_t n = n_tokens - base;
        if (n > chunk) n = chunk;
        bool last = (base + n == n_tokens);

        /* Host embedding for the chunk (ref.c's wrow, exactly like decode). */
        for (uint32_t t = 0; t < n; t++) {
            gpu_embed_row(m->tok_emb, m->wtype, (uint64_t)tokens[base + t],
                          c->hidden, g->h_x + (size_t)t * c->hidden);
        }
        /* Per-token RoPE cos/sin table in double, uploaded f32: one row per
         * absolute position, [cos(rope_dim/2), sin(rope_dim/2)], byte-identical
         * to the one-token table sg_gpu_forward builds. */
        if (n_attn > 0 && g->h_cs) {
            for (uint32_t t = 0; t < n; t++) {
                float *row = g->h_cs + (size_t)t * c->rope_dim;
                uint32_t pos = base + t;
                for (uint32_t i = 0; i < half; i++) {
                    double inv_freq = pow((double)c->rope_theta,
                                          -2.0 * (double)i / (double)c->rope_dim);
                    double ang = (double)pos * inv_freq;
                    row[i] = (float)cos(ang);
                    row[half + i] = (float)sin(ang);
                }
            }
        }
        /* k_attn_prefill's per-threadgroup private score row is base+n long,
         * over n*n_heads threadgroups; ensure it before the command buffer
         * opens (never mid-encode). */
        if (n_attn > 0) {
            uint64_t need = 0, t0 = 0;
            if (!mul_ck((uint64_t)n * c->n_heads, (uint64_t)base + n, &t0)
                || !mul_ck(t0, 4, &need)) {
                e = (sg_err){"gpu: prefill score-scratch size overflows 64 bits"};
                goto cleanup;
            }
            e = scratch_ensure(g, need);
            if (sg_failed(e)) goto cleanup;
        }

        __block sg_err rc = SG_OK;
        /* GPU-busy wall time for this chunk, Task B8 duty-cycle accounting:
         * set around exactly the commit..waitUntilCompleted span below (the
         * interval the GPU is actually working), not the host-side encode
         * loop above. Captured unconditionally (two clock_gettime calls) so
         * the encode path itself never branches on rest_enabled; only the
         * accumulate-and-maybe-rest decision after the command buffer
         * completes does. */
        double chunk_gpu_s = 0.0;
        uint32_t seg = g->prefill_seg_layers;
        if (seg == 0u || seg > c->n_layers) seg = c->n_layers;

        /*
         * The cursor advances by the layers actually encoded (l1 - l0), NOT by
         * `seg`. `seg` can shrink at the bottom of this loop, and a `l0 += seg`
         * increment would then rewind over layers already applied to the
         * residual stream, running them a second time. That is silent: it
         * produces a plausible but wrong hidden state rather than an error.
         */
        for (uint32_t l0 = 0; l0 < c->n_layers && !sg_failed(rc); ) {
            uint32_t l1 = l0 + seg;
            if (l1 > c->n_layers) l1 = c->n_layers;

        double t_gpu0 = 0.0, t_gpu1 = 0.0;
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!cb || !enc) {
                rc = (sg_err){"gpu: could not open a compute encoder"};
            } else {
                sg_enc E = { g, enc };
                for (uint32_t i = l0; i < l1; i++) {
                    sg_gpu_layer *L = &g->ls[i];
                    /* ln1 per token: KI_RMSNORM_HEADS with heads=n is the
                     * per-slice form of decode's k_rmsnorm (identical scale
                     * formula and tg_sum tree), so ln1/ln2 are bit-identical to
                     * decode per token and the only prefill-vs-decode numeric
                     * gap is GEMM-vs-matvec reassociation. */
                    enc_op(&E, KI_RMSNORM_HEADS, g->b_x, 0, L->ln1, 0, g->b_h, 0,
                           nil, 0, PARAMS(c->hidden, n, eps, 1, c->hidden));
                    if (L->is_attn) enc_attn_prefill(&E, L, i, base, n);
                    else            enc_gdn_prefill(&E, L, i, base, n);
                    enc_op(&E, KI_ADD, g->b_x, 0, g->b_r, 0, g->b_x, 0, nil, 0,
                           PARAMS(n * c->hidden));

                    enc_op(&E, KI_RMSNORM_HEADS, g->b_x, 0, L->ln2, 0, g->b_h, 0,
                           nil, 0, PARAMS(c->hidden, n, eps, 1, c->hidden));
                    enc_matmul(&E, gemm, g->b_h, 0, L->w_gate, g->b_ffg, 0,
                               n, c->ffn_hidden, c->hidden);
                    enc_matmul(&E, gemm, g->b_h, 0, L->w_up, g->b_ffu, 0,
                               n, c->ffn_hidden, c->hidden);
                    enc_op(&E, KI_SWIGLU, g->b_ffg, 0, g->b_ffu, 0, g->b_ffg, 0,
                           nil, 0, PARAMS(n * c->ffn_hidden));
                    enc_matmul(&E, gemm, g->b_ffg, 0, L->w_down, g->b_r, 0,
                               n, c->hidden, c->ffn_hidden);
                    enc_op(&E, KI_ADD, g->b_x, 0, g->b_r, 0, g->b_x, 0, nil, 0,
                           PARAMS(n * c->hidden));
                }
                /* Logits belong to the FINAL segment of the final chunk, so
                 * that b_x already carries every layer's contribution. */
                if (last && l1 == c->n_layers) {
                    /* Only the last chunk's LAST row needs logits: out_norm on
                     * that row (KI_RMSNORM_HEADS, heads=1, reading b_x at the
                     * last token's offset) then lm_head via the matvec kernel,
                     * exactly decode's final two ops. */
                    enc_op(&E, KI_RMSNORM_HEADS, g->b_x,
                           (uint64_t)(n - 1) * c->hidden, g->out_norm, 0,
                           g->b_h, 0, nil, 0, PARAMS(c->hidden, 1, eps, 1, c->hidden));
                    enc_op(&E, g->mat_kernel, g->lm_head, 0, g->b_h, 0,
                           g->b_logits, 0, nil, 0, PARAMS(c->vocab, c->hidden));
                }
                [enc endEncoding];
                t_gpu0 = pf_now_s();
                [cb commit];
                [cb waitUntilCompleted];
                t_gpu1 = pf_now_s();
                if ([cb error]) {
                    rc = gpu_errf("gpu: prefill chunk failed: %s",
                                  [[[cb error] localizedDescription] UTF8String]);
                }
            }
        }
        chunk_gpu_s += t_gpu1 - t_gpu0;
        /* Counts SUBMITTED command buffers, so a segment whose encoder could
         * not be opened is not counted; t_gpu1/t_gpu0 are both 0 in that case
         * and rc is already set, ending the loop. */
        if (!sg_failed(rc)) g->prefill_segments++;
        l0 = l1;

        /* Adapt the segment size to the measured submission time.
         *
         * REACTIVE, not a hard cap: this segment has already run long by the
         * time we look. Overruns can therefore REPEAT while the size converges
         * (64 layers can overrun at 64, then 32, then 16), and a segment already
         * at the 1-layer floor cannot shrink further, so it can overrun forever.
         * What halving guarantees is only that each overrun makes the next
         * submission smaller until the floor. Choose a ceiling well under the
         * watchdog window so the overruns along the way still land inside it.
         *
         * Applies from the NEXT segment; the cursor above already advanced by
         * what was encoded, so shrinking here cannot rewind it. */
        if (g->prefill_max_burst_ms > 0u && seg > 1u &&
            (t_gpu1 - t_gpu0) * 1000.0 > (double)g->prefill_max_burst_ms) {
            seg /= 2u;
            g->prefill_seg_layers = seg;
            if (pf_log) {
                fprintf(stderr, "gpu: prefill segment -> %u layers "
                        "(command buffer ran %.0f ms, ceiling %u ms)\n",
                        seg, (t_gpu1 - t_gpu0) * 1000.0, g->prefill_max_burst_ms);
            }
        }
        }
        if (sg_failed(rc)) { e = rc; goto cleanup; }

        /* Advance both counters together: decode's position bookkeeping uses
         * g->used, and sg_kv's own used is kept consistent for its append-only
         * overflow guard (M5.2 note). */
        g->used += n;
        e = sg_kv_advance(g->kv, n);
        if (sg_failed(e)) goto cleanup;

        if (pf_log && (last || ((base / chunk) & 7u) == 0u)) {
            double el = pf_now_s() - t_pf0;
            fprintf(stderr, "gpu: prefill %u/%u tokens (%.1f%%), %.0f tok/s, "
                    "%.1fs elapsed\n", base + n, n_tokens,
                    100.0 * (double)(base + n) / (double)n_tokens,
                    el > 0.0 ? (double)(base + n) / el : 0.0, el);
        }

        /* Task B8 duty-cycle: between THIS chunk's waitUntilCompleted (just
         * above, GPU idle at that instant) and the NEXT chunk's encode (top of
         * the next iteration). Pure timing -- no buffer, state, or accumulator
         * that feeds the computed logits or KV state is touched here, only
         * pf_work_acc_ms (a local, loop-scoped counter) and g's rest-tracking
         * fields, so this cannot change gen_ids. No-op when disabled.
         *
         * The test is PREDICTIVE: it asks whether the NEXT chunk would carry
         * the accumulator past the budget, not whether this one already did.
         * Testing after the fact meant a burst always ran to budget + one
         * chunk, and one chunk grows with context: over the 2026-08-14 256K
         * run, 367 of 367 bursts overran a 150 s budget, median 199.5 s and
         * worst 332.9 s. Chunk cost grows monotonically with context (each
         * chunk attends over a longer KV cache), so the previous chunk is a
         * slight UNDER-estimate of the next; PF_EST_MARGIN covers that.
         *
         * The final chunk is now PROTECTED rather than special-cased. The old
         * rule skipped the budget test whenever the next chunk was the last
         * one, so the longest chunk of the run was the one most likely to be
         * submitted onto an already-exhausted budget. Here the test runs at the
         * end of every non-final chunk and accounts for the chunk that follows,
         * including when that chunk is the last. Resting AFTER the final chunk
         * is still pointless (nothing follows it), which is what !last means
         * now. */
        if (rest_enabled && !last) {
            double est_next_ms;

            pf_work_acc_ms += chunk_gpu_s * 1000.0;
            est_next_ms = chunk_gpu_s * 1000.0 * PF_EST_MARGIN;

            if (pf_work_acc_ms + est_next_ms >= (double)g->prefill_work_budget_ms) {
                if (pf_log) {
                    fprintf(stderr, "gpu: prefill duty-cycle resting %u ms "
                            "(worked %.0f ms since last rest)\n",
                            g->prefill_rest_ms, pf_work_acc_ms);
                }
                pf_sleep_ms(g->prefill_rest_ms);
                g->prefill_rest_total_ms += g->prefill_rest_ms;
                pf_work_acc_ms = 0.0;
            }
        }
    }

    /* Bridge the DeltaNet state from the sg_kv carriers into the decode
     * buffers. conv tail: sg_kv_conv is [conv_kernel-1, conv_dim] oldest-first;
     * L->conv_buf is [conv_kernel, conv_dim] (output row then the same tail), so
     * the tail lands at element offset conv_dim. S: identical [n_v, dv, dk]. */
    {
        uint64_t conv_tail = (uint64_t)(c->conv_kernel - 1u) * g->conv_dim;
        uint64_t s_elems = (uint64_t)c->n_v_heads * c->head_v_dim * c->head_k_dim;
        for (uint32_t i = 0; i < c->n_layers; i++) {
            sg_gpu_layer *L = &g->ls[i];
            if (L->is_attn) continue;
            float *cs = (float *)sg_gpu_buf_host(sg_kv_conv(g->kv, i));
            float *cd = (float *)sg_gpu_buf_host(L->conv_buf);
            if (cs && cd && conv_tail) {
                memcpy(cd + g->conv_dim, cs, (size_t)conv_tail * sizeof *cd);
            }
            float *ss = (float *)sg_gpu_buf_host(sg_kv_s(g->kv, i));
            float *sd = (float *)sg_gpu_buf_host(L->ssm);
            if (ss && sd) memcpy(sd, ss, (size_t)s_elems * sizeof *sd);
        }
    }

    if (out_last_logits) *out_last_logits = g->h_logits;

cleanup:
    prefill_free_chunk(g);
    prefill_restore(g, &saved);
    return e;
}
