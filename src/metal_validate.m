/* metal_validate.m - the per-dispatch validation and grid geometry of the
 * Metal host layer, split out of src/metal.m by task R3.
 *
 * WHAT IS HERE, and why this is the seam:
 *
 *   - check_sizes, the per-kernel buffer-size preconditions;
 *   - check_params, the per-kernel preconditions that are not about sizes;
 *   - gpu_grid, the kernel kind -> (threadgroups, threads) mapping.
 *
 * Every kernel reads its inputs from indices computed out of params[], so a
 * params/buffer mismatch is an out-of-bounds DEVICE read: not a crash the
 * process can catch, but a GPU fault that takes down the whole context (and
 * on a bad day the display driver). Hence a size precondition per kernel,
 * checked here, before anything is encoded. (That paragraph stood over
 * buf_big_enough in src/metal.m before this task; the predicate itself is now
 * `static inline` in src/metal_internal.h, because twenty-six of its
 * twenty-nine call sites stayed behind.)
 *
 * THE TRAFFIC IS ONE-WAY, AND IT RUNS THE OTHER WAY FROM TASK R2's. Nothing
 * here is reached from src/metal_prefill.m, and nothing here calls back into
 * src/metal.m except gpu_errf, which R2 had already promoted. What crosses is
 * the three functions BELOW: all six of their call sites (sg_gpu_run_op for
 * all three, the three split-K one-shots for check_params, enc_op for
 * gpu_grid) stayed in src/metal.m, so it is these three that lost `static`.
 *
 * THIS IS A MOVE, NOT A REWRITE. Every line below other than this header came
 * from src/metal.m verbatim, in its original order. There are exactly three
 * kinds of deviation and no others: the `static` dropped from the three
 * definitions, check_sizes' continuation line re-indented by the seven columns
 * that `static` used to occupy, and four comment blocks retargeted at the
 * files the names in them now live in. Behaviour, rule order, byte counts,
 * grid shapes and error text are unchanged.
 *
 * WHAT THE SPLIT COST, stated plainly because R1's byte-identity gate does NOT
 * transfer to Objective-C: check_sizes, check_params and gpu_grid have
 * external linkage now and can no longer be inlined into their callers. All
 * three are per-DISPATCH, never per-element: check_sizes and check_params run
 * once per one-shot sg_gpu_run_* call, which then commits a command buffer and
 * waits for the GPU, and gpu_grid runs once per encoded dispatch, next to the
 * half-dozen Objective-C message sends that bind that dispatch's buffers.
 *
 * The public contract for the kernels these rules validate lives in surge.h,
 * next to sg_gpu_run_op, and is not repeated.
 */
#include "metal_internal.h"

#include <stdint.h>
#include <string.h>

sg_err check_sizes(const char *kernel, const sg_gpu_buf *a, const sg_gpu_buf *b,
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
         * work. Their real byte counts go through splitk_sizes()
         * (src/metal.m), which the one-shots call and which guards every
         * product with mul_ck exactly as the rules above do. */
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
sg_err check_params(const char *kernel, const uint32_t *p) {
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
         * is skipped. A group WIDER than SG_SPLITK_GQA_MAX (src/metal.m) is
         * still answered correctly (the kernel's default arm walks it one head
         * at a time), so it is a policy question for splitk_gqa_use
         * (src/metal.m), not a validity one.
         *
         * P2.8's online partial shares it too, and adds NO head_dim ceiling
         * here on purpose: head_dim > SG_TG (src/metal_internal.h) makes that
         * kernel re-stream the split once per SG_TG-wide band of output dims,
         * which is slower than the four-pass kernel but still exactly correct,
         * so it is the same kind of policy question (splitk_online_use in
         * src/metal.m declines it) rather than a validity one. A rejection here
         * would instead break the one-shot for a shape the kernel answers
         * correctly. */
    }
    return SG_OK;
}

/* Grid geometry from the kernel's kind and its params. Split out of
 * sg_gpu_run_op (src/metal.m) because sg_gpu_forward's encoder there needs
 * exactly the same mapping and a second copy of it would be a silent way for
 * the batched path to dispatch a different shape than the one-shot path.
 * Products are u64: both factors are u32 params, so they cannot wrap.
 *
 * The SG_K_* kinds it switches on are declared in src/metal_internal.h (moved
 * there by task R3 with this function); the SG_KERNELS table that assigns a
 * kind to each kernel name stayed in src/metal.m. */
void gpu_grid(int kind, const uint32_t *p, uint64_t *groups, uint64_t *elems) {
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
     * both by hand instead, both in src/metal.m (sg_gpu_run_op's SG_K_TILES2D
     * case, and sg_gpu_run_attn_splitk_partial for SG_K_HEADS2D). Left at the
     * default *groups = 1 so a caller that ignored this comment gets an
     * obviously-wrong single threadgroup rather than a plausible-looking wrong
     * number. */
    default:           *groups = 1; break;
    }
}
