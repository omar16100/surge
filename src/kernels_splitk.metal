/* kernels_splitk.metal - the split-K decode-attention kernels (Tasks P2.2
 * through P2.9), a SECOND translation unit of the same shader library.
 *
 * WHY THIS FILE EXISTS. Tasks P2.3 to P4.0 grew src/kernels.metal to 2548
 * lines, past this project's ~2000-line guideline, entirely with the split-K
 * decode work below. Splitting it out is a PURE MOVE: the Makefile compiles
 * both .metal sources with the identical flags, each to its own .air, and
 * links the two into the one src/kernels.metallib, so metal.m still looks
 * every kernel up by name in a single library and no host code changed. The
 * move was gated on the emitted AIR being instruction-for-instruction
 * identical per kernel, not merely on the tests staying green.
 *
 * THE DETERMINISM MANDATE AT src/kernels.metal:7-27 GOVERNS EVERY KERNEL HERE.
 * That block is the authoritative statement and is deliberately not copied
 * (a second copy is a second thing to drift). In short, and non-authoritatively:
 * every reduction below has a shape fixed at compile time, the SG_TG-wide fold
 * trees with their fixed stride schedule; no atomics; no simd_sum / simd_max,
 * whose fold order the Metal spec does not pin down; no read-modify-write of a
 * shared accumulator by more than one thread. The precision rules carry over
 * too: transcendentals go through precise::, and the Makefile's -fno-fast-math
 * is not optional (the P2.4 notes below explain what an R-accumulator loop
 * would cost without it).
 *
 * SHARED HELPERS COME FROM kernels_common.metal.h, NEVER FROM A COPY. SG_TG,
 * tg_sum and tg_max have callers in both files, so they are defined once
 * there and included by both. Anything used only here belongs here; anything
 * used only by the kernels in kernels.metal belongs there; nothing belongs in
 * both.
 */
#include <metal_stdlib>
using namespace metal;

#include "kernels_common.metal.h"


/* =====================================================================
 * Split-K decode attention (Task P2.2)
 * =====================================================================
 *
 * k_attn_decode_f16 (src/kernels.metal, the sibling translation unit of this
 * one) dispatches exactly n_heads threadgroups, so at 32 heads only 32 of this
 * machine's 80 GPU cores have anything scheduled and ONE threadgroup walks the
 * entire 262,144-key sequence. These two kernels are the flash-decoding
 * answer: partition [0, seq) into n_splits contiguous ranges, give each
 * (query head, split) its own threadgroup, and fold the per-split partial
 * results afterwards by log-sum-exp rescaling.
 *
 * THEY ARE THE METAL TWIN OF THE ALREADY-GATED CPU ORACLE, statement for
 * statement:
 *   - the partition rule is sg_ref_attn_decode_splitk's exactly (src/ref.c,
 *     attn_decode_core): t0 = i*seq/n_splits, t1 = (i+1)*seq/n_splits, integer
 *     division with a 64-bit intermediate. It tiles [0, seq) exactly (split
 *     i's t1 IS split i+1's t0, the last t1 IS seq), leaves the ragged
 *     remainder in the high-index splits, and produces genuinely empty splits
 *     whenever n_splits > seq;
 *   - the per-split triple (m, s, acc) is sg_ref_attn_combine's contract
 *     (surge.h): m = max score over the split, s = sum exp(score - m),
 *     acc[d] = sum exp(score - m) * v[d], the UNNORMALIZED weighted-V sum;
 *   - an empty split (t0 >= t1) emits the documented m = -INFINITY, s = 0,
 *     acc = 0 encoding, which the combine's log-sum-exp handles with no
 *     special case at all (weight exp(-INFINITY - M) == 0);
 *   - the arithmetic, the GQA mapping (hk = h / repeat, falling back to kv
 *     head 0 when repeat == 0), the [seq, n_kv_heads, head_dim] KV indexing
 *     and the q_stride handling are k_attn_decode_f16's verbatim.
 *
 * DETERMINISM (the rule at the top of this file). No atomics, no simd_sum /
 * simd_max, no read-modify-write of a shared accumulator: the partial kernel
 * folds with the same fixed-shape tg_max/tg_sum trees k_attn_decode_f16 uses,
 * and the combine walks splits in strictly increasing index order. The split
 * count is a DISPATCH PARAMETER (params[6]), never derived from data, so the
 * partition boundaries and therefore every summation order are fixed before
 * the first byte is read.
 *
 * seq == 0: DELIBERATELY DIFFERENT FROM k_attn_decode_f16, which folds seq==0
 * into its early-return guard (src/kernels.metal:469) and leaves `out`
 * completely UNWRITTEN. These kernels instead match the CPU oracle: every
 * split is empty (t0 == t1 == 0 for every i), so every triple is the
 * -INFINITY/0/0 encoding and the combine's all-empty branch writes the
 * documented out[d] = 0.0. The divergence surge.h warns about under "KNOWN
 * DIVERGENCE FROM k_attn_decode_f16 AT seq==0" therefore does NOT apply to
 * this pair; they agree with sg_ref_attn_decode_splitk there. (metal.m accepts
 * seq == 0 for exactly this reason, unlike sg_gpu_run_attn_decode_f16, which
 * rejects it.)
 *
 * OCCUPANCY: n_splits BUYS THREADGROUPS AND SPENDS LANES (P2.2 review finding
 * 4). A threadgroup is SG_TG == 256 lanes and the partial hands out its
 * split's keys one per lane (thread `lid` takes keys lid, lid+256, ...), so a
 * split shorter than 256 keys leaves lanes idle: at seq 200 with n_splits 257
 * the per-split length is 1 and 255 of the 256 lanes do nothing at all. The
 * useful band is roughly
 *
 *     n_heads * n_splits >= GPU cores      (enough threadgroups to fill them)
 *     n_splits <= seq / SG_TG              (enough keys to fill each one)
 *
 * and the second bound is the one this task's motivation ignores: splitting
 * past seq/256 keeps adding threadgroups whose lanes are mostly idle, and past
 * seq it adds threadgroups that do nothing but write the empty encoding. At
 * seq 262,144 that upper bound is 1024 splits, so the interesting range is
 * wide, but it IS bounded.
 *
 * THE MEASUREMENT WAS TAKEN (P2.3a, `make bench-splitk`): the fastest n_splits
 * was seq / SG_TG in seven of the eight (seq, shape) cells swept, i.e. the TOP
 * of that band, giving each split exactly SG_TG keys. So the decode path uses
 * n_splits = clamp(seq / SG_TG, 4, 1024).
 *
 * THE EIGHTH CELL DISAGREES and the disagreement is reproduced: on the 27B
 * decode shape at seq 8192, n_splits 16 beats the closed form's 32 by about
 * 5 percent (P2.3's re-sweep 5.226x vs 4.965x; the P2.3 review's independent
 * one 5.447x vs 5.180x). The policy still ships the closed form on purpose: it
 * is the band's top rather than a fitted constant, the curve is shallow near
 * the optimum, and one outlier at one shape and depth does not earn a special
 * case. Read the closed form as a good default, not as a proven optimum at
 * every cell. metal.m's splitk_n_splits carries the full argument.
 *
 * WIRED INTO THE DECODE PATH (P2.3). enc_attn's fp16 branch dispatches this
 * pair through metal.m's enc_attn_splitk once seq reaches SG_TG * 4 == 1024,
 * and the incumbent k_attn_decode_f16 below that (where the clamp's floor
 * would force splits shorter than SG_TG keys), on the f32 KV path, and under
 * SURGE_ATTN_SPLITK=0. */

/* ref.c's attn_combine_weight in f32 (src/ref.c, above sg_ref_attn_combine),
 * including the reason for the equality test: mi == M means the weight is
 * exactly 1.0 by definition, and computing it as exp(mi - M) would give the
 * indeterminate INF - INF = NaN when both are +INFINITY even though the two
 * are genuinely equal. Every other combination evaluates correctly through
 * the subtraction, including an empty split's mi == -INFINITY against a
 * finite M (IEEE-754 makes that -INFINITY, not NaN, and exp(-INFINITY) is
 * exactly 0.0). A NaN mi is not caught by the equality (NaN == anything is
 * false) and falls through to exp(NaN - M) == NaN, which the combine
 * propagates rather than launders. */
static inline float attn_combine_weight(float mi, float M) {
    if (mi == M) return 1.0f;
    return precise::exp(mi - M);
}

/* One threadgroup per (query head, split), a 2D grid dispatched as
 * (x = split, y = head); the host's SG_K_HEADS2D class is what carries those
 * two group dimensions, since SG_K_ATTN's single *groups count cannot (see
 * sg_gpu_grid in metal_validate.m, and the SG_K_TILES2D precedent it follows;
 * both SG_K_ kinds are declared in metal_internal.h since task R3).
 *
 * Buffers: q f32 [n_heads, q_stride]; kc, vc f16 [seq, n_kv_heads, head_dim]
 * (the sg_kv layout); m, s f32 [n_heads, n_splits]; acc f32
 * [n_heads, n_splits, head_dim]; scores f32 scratch, one private row of
 * `span` floats per (head, split) where span = ceil(seq / n_splits), sized by
 * metal.m with the IDENTICAL formula.
 *
 * THE SCORES BUFFER IS THIS KERNEL'S OWN (P2.2 review finding 1). metal.m
 * binds sg_gpu.splitk_scratch here, NOT the process-wide sg_gpu.scratch that
 * k_attn_decode, k_attn_decode_f16, k_attn_prefill and the decode encoder all
 * bind at offset 0. That shared buffer is grown, never partitioned, so two
 * different users encoded into ONE command buffer would be writing the same
 * bytes, and this kernel exists precisely to be dispatched from the batched
 * encoder next to those. A separate allocation makes the overlap impossible
 * rather than merely unlikely.
 *
 * params: [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=seq [4]=q_stride
 * [5]=softmax scale bits [6]=n_splits (params[7] unused).
 *
 * `span` is the exact upper bound on a split's length: with the partition
 * rule above, t1 - t0 <= floor(seq/n_splits) + 1 <= ceil(seq/n_splits) for
 * every i, and the bound is TIGHT (some split always attains it, whether or
 * not n_splits divides seq), so it is neither too small nor wasteful. The
 * scores row therefore never overruns its slice and neighbouring splits
 * never share a byte. */
kernel void k_attn_decode_splitk_partial(device const float *q  [[buffer(0)]],
                                         device const half *kc   [[buffer(1)]],
                                         device const half *vc   [[buffer(2)]],
                                         device float *m_out     [[buffer(3)]],
                                         device float *s_out     [[buffer(4)]],
                                         device float *acc_out   [[buffer(5)]],
                                         constant uint *p        [[buffer(6)]],
                                         device float *scores    [[buffer(7)]],
                                         uint2 tg  [[threadgroup_position_in_grid]],
                                         uint2 tid [[thread_position_in_threadgroup]])
{
    uint n_heads = p[0], n_kv = p[1], hd = p[2], seq = p[3];
    uint q_stride = p[4];
    float scale = as_type<float>(p[5]);
    uint n_splits = p[6];

    uint part = tg.x, h = tg.y, lid = tid.x;
    /* Every term here is uniform across the threadgroup (tg.x/tg.y are, lid
     * is not used), so this return is uniform and cannot strand a thread on
     * one of the barriers below. */
    if (h >= n_heads || part >= n_splits || n_kv == 0u || hd == 0u) return;

    /* The partition rule, in 64-bit exactly like attn_decode_core's. */
    uint t0 = (uint)(((ulong)part * (ulong)seq) / (ulong)n_splits);
    uint t1 = (uint)((((ulong)part + 1ul) * (ulong)seq) / (ulong)n_splits);

    /* Flat (head, split) index into the m/s arrays. Built in 64-bit: n_heads
     * and n_splits are independent uint32 params, so h*n_splits alone can
     * exceed 32 bits even though metal.m's size checks (which guard the same
     * product in uint64) would reject a buffer that large. */
    size_t part_idx = (size_t)h * (size_t)n_splits + (size_t)part;
    device float *acc = acc_out + part_idx * (size_t)hd;

    /* Empty split, guaranteed for the high-index splits whenever
     * n_splits > seq (and for EVERY split when seq == 0): the documented
     * m = -INFINITY / s = 0 / acc = 0 encoding, which the combine consumes
     * with no special case. Uniform branch (t0, t1 are uniform). */
    if (t0 >= t1) {
        if (lid == 0u) {
            m_out[part_idx] = -INFINITY;
            s_out[part_idx] = 0.0f;
        }
        for (uint i = lid; i < hd; i += SG_TG) acc[i] = 0.0f;
        return;
    }

    uint repeat = n_heads / n_kv;         /* GQA: mlx repeats the kv-head axis */
    uint hk = repeat ? (h / repeat) : 0u;
    device const float *qh = q + (size_t)h * q_stride;

    /* This (head, split)'s private score row. span is recomputed here rather
     * than passed in so the kernel and metal.m cannot drift apart by a param
     * that one of them forgot to set; both compute ceil(seq/n_splits) in
     * 64-bit. */
    uint span = (uint)((((ulong)seq + (ulong)n_splits) - 1ul) / (ulong)n_splits);
    device float *sc = scores + part_idx * (size_t)span;
    uint len = t1 - t0;
    threadgroup float red[SG_TG];

    /* 1. scores over [t0, t1). Thread `lid` owns the split's keys lid,
     *    lid+256, ... (indexed RELATIVE to t0, so t0 + r cannot overflow the
     *    way t0 + lid could at a t0 near UINT32_MAX); the dot product is a
     *    serial loop over head_dim, so no reduction is needed here. */
    for (uint r = lid; r < len; r += SG_TG) {
        device const half *kt = kc + ((size_t)(t0 + r) * n_kv + hk) * hd;
        float dot = 0.0f;
        for (uint i = 0; i < hd; i++) dot += qh[i] * (float)kt[i];
        sc[r] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    /* 2. m = max score over THIS SPLIT only. Same fixed tree, same
     *    degenerate-branch-free shape as k_attn_decode_f16 (the host only
     *    ever passes keys at positions <= pos, so there is no mask value that
     *    could make a score non-finite). */
    float m = -INFINITY;
    for (uint r = lid; r < len; r += SG_TG) { float v = sc[r]; m = (v > m) ? v : m; }
    m = tg_max(red, lid, m);

    /* 3. s = sum exp(score - m) over the split, and the exponentials stay in
     *    the score row for step 4. NOTE THE ONE DELIBERATE DIFFERENCE from
     *    k_attn_decode_f16: there is NO division by the sum here. The split's
     *    weights must stay UNNORMALIZED, because normalizing per split is
     *    exactly what the combine's log-sum-exp rescaling has to undo. */
    float sacc = 0.0f;
    for (uint r = lid; r < len; r += SG_TG) {
        float e = precise::exp(sc[r] - m);
        sc[r] = e;
        sacc += e;
    }
    float s = tg_sum(red, lid, sacc);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    /* 4. acc[d] = sum_t exp(score_t - m) * v[t][d] over the split. Thread
     *    `lid` owns OUTPUT dims lid, lid+256, ... and walks t in increasing
     *    order, so there is no cross-thread reduction and the summation order
     *    is fixed, exactly as in k_attn_decode_f16's step 3. */
    for (uint i = lid; i < hd; i += SG_TG) {
        float a = 0.0f;
        for (uint r = 0; r < len; r++) {
            a += sc[r] * (float)vc[((size_t)(t0 + r) * n_kv + hk) * hd + i];
        }
        acc[i] = a;
    }

    /* tg_max/tg_sum hand the SAME value to every thread, so writing m and s
     * from one designated thread is not a reduction, just a store. */
    if (lid == 0u) {
        m_out[part_idx] = m;
        s_out[part_idx] = s;
    }
}

/* One threadgroup per query head, folding that head's n_splits partial
 * triples into the final attention output. sg_ref_attn_combine (src/ref.c)
 * pass for pass:
 *
 *   M = max_i m[i];  S = sum_i s[i]*w(m[i], M);
 *   out[d] = ( sum_i acc[i][d]*w(m[i], M) ) / S
 *
 * with w = attn_combine_weight above, and the oracle's degenerate cases in
 * the same order it checks them:
 *   - M is NaN: every out[d] is that NaN. surge.h's contract is "a NaN
 *     anywhere in m[] makes every out[d] NaN"; the scan below propagates a
 *     NaN from ANY index into M (the way tg_max does, and unlike the CPU
 *     loop, which only picks one up at i == 0 and catches the rest through
 *     its isnan(S) check instead). Both routes end at the same NaN-out
 *     answer the contract promises; only the intermediate differs.
 *   - M == -INFINITY, i.e. EVERY split empty (which is also the seq == 0
 *     case): out[d] = 0.0, the documented "attention over zero keys"
 *     convention, caught before it could become -INF - -INF = NaN.
 *   - M == +INFINITY needs no branch: attn_combine_weight gives every split
 *     tied at that +INFINITY weight exactly 1.0 and every other split 0.0.
 *   - S is NaN: propagate rather than let the S > 0.0 guard below launder it
 *     into a manufactured 0.0.
 *
 * NO tg_sum/tg_max FOLD HERE, unlike the partial kernel: there is no
 * cross-thread reduction to make. Every thread walks the same n_splits
 * triples in the same strictly increasing index order and independently
 * computes the identical M and S, which is both the determinism rule
 * satisfied a fortiori (no thread reads anything another thread wrote, so
 * there is no order to fix) and a closer match to sg_ref_attn_combine, whose
 * own passes are strictly-increasing serial sums rather than trees. The
 * weights are recomputed inside the pass-3 loop rather than cached, exactly
 * as the oracle does it: they are a pure function of already-fixed inputs, so
 * recomputing changes no bit. n_splits is a small dispatch parameter (tens to
 * low hundreds), not a sequence length, so the redundancy is cheap.
 *
 * Buffers: m, s f32 [n_heads, n_splits]; acc f32 [n_heads, n_splits,
 * head_dim]; out f32 [n_heads, head_dim]. params are the SAME array the
 * partial dispatch takes, so the host can fill it once; this kernel reads
 * only [0]=n_heads, [2]=head_dim and [6]=n_splits. */
kernel void k_attn_decode_splitk_combine(device const float *m_in   [[buffer(0)]],
                                         device const float *s_in   [[buffer(1)]],
                                         device const float *acc_in [[buffer(2)]],
                                         device float *out          [[buffer(3)]],
                                         constant uint *p           [[buffer(4)]],
                                         uint h   [[threadgroup_position_in_grid]],
                                         uint lid [[thread_position_in_threadgroup]])
{
    uint n_heads = p[0], hd = p[2], n_splits = p[6];
    if (h >= n_heads || hd == 0u || n_splits == 0u) return;

    size_t base = (size_t)h * (size_t)n_splits;
    device const float *mh = m_in + base;
    device const float *sh = s_in + base;
    device const float *ah = acc_in + base * (size_t)hd;
    device float *oh = out + (size_t)h * hd;

    /* Pass 1: M = max_i m[i], strictly increasing i, NaN-propagating (see the
     * header comment, and tg_max's own NaN rule in kernels_common.metal.h). */
    float M = -INFINITY;
    for (uint i = 0; i < n_splits; i++) {
        float v = mh[i];
        M = (v != v) ? v : ((v > M) ? v : M);
    }

    if (M != M) {                       /* NaN anywhere in m[]: propagate */
        for (uint d = lid; d < hd; d += SG_TG) oh[d] = M;
        return;
    }
    if (M == -INFINITY) {               /* every split empty (incl. seq == 0) */
        for (uint d = lid; d < hd; d += SG_TG) oh[d] = 0.0f;
        return;
    }

    /* Pass 2: S = sum_i s[i] * w(m[i], M), strictly increasing i. An empty
     * split contributes 0.0 * 0.0 with no special case. */
    float S = 0.0f;
    for (uint i = 0; i < n_splits; i++) S += sh[i] * attn_combine_weight(mh[i], M);

    if (S != S) {                       /* NaN: propagate, never launder */
        for (uint d = lid; d < hd; d += SG_TG) oh[d] = S;
        return;
    }

    /* Pass 3: out[d] = ( sum_i acc[i][d] * w(m[i], M) ) / S, strictly
     * increasing i for every d, one output dim per thread so nothing is
     * shared. S > 0.0 holds under the documented input contract (the split
     * achieving M is non-empty, so its own s[i] >= 1.0); the guard is
     * defensive only, for a caller that hands over a contradictory triple. */
    for (uint d = lid; d < hd; d += SG_TG) {
        float num = 0.0f;
        for (uint i = 0; i < n_splits; i++) {
            num += ah[(size_t)i * hd + d] * attn_combine_weight(mh[i], M);
        }
        oh[d] = (S > 0.0f) ? (num / S) : 0.0f;
    }
}

/* =====================================================================
 * GQA-shared split-K decode attention partial (Task P2.4)
 * =====================================================================
 *
 * SAME OUTPUT BYTES AS k_attn_decode_splitk_partial, LESS MEMORY TRAFFIC.
 * That kernel's grid is (n_splits, n_heads), and GQA maps repeat = n_heads /
 * n_kv_heads query heads onto ONE kv head, so for a given split the `repeat`
 * threadgroups h = hk*repeat .. hk*repeat+repeat-1 each stream the SAME K and
 * V slices out of memory independently. On the real 4B decode shape (32 heads,
 * 8 kv) that is 4x the unique bytes; on the 27B shape (24 heads, 4 kv), 6x.
 * Decode at depth is bandwidth-bound, so it is close to a `repeat`x tax on the
 * dominant cost. (P2.3a's harness reports achieved GB/s from ISSUED bytes,
 * which is why its headline number is `repeat` times the unique-bytes figure.)
 *
 * This kernel makes ONE threadgroup serve an entire GQA group: the grid is
 * (n_splits, n_kv_heads), each threadgroup reads a K (later V) element once
 * and applies it to all `repeat` query vectors that share it, holding `repeat`
 * independent running results.
 *
 * THE OUTPUT LAYOUT IS UNCHANGED, deliberately: m_out and s_out are still
 * indexed part_idx = h * n_splits + part and acc_out at part_idx * hd, for
 * EVERY h in the group, so k_attn_decode_splitk_combine, the score-scratch
 * sizing in metal.m and the [n_heads, n_splits] buffer shapes are all
 * untouched. Only which threadgroup writes a given triple changes.
 *
 * BYTE-IDENTICAL, NOT MERELY CLOSE. This is a data-reuse reorganization and
 * nothing about the arithmetic moves:
 *   - the dot product still accumulates qh[i] * (float)kt[i] over increasing
 *     i, per head, from 0.0f;
 *   - m and s are still folded by tg_max / tg_sum, one fixed-shape tree per
 *     head over the same per-lane partials (thread lid still owns the split's
 *     keys lid, lid+SG_TG, ...);
 *   - acc[d] still accumulates sc[r] * (float)v[t0+r][d] over increasing r,
 *     per head, from 0.0f.
 * So every output bit must match k_attn_decode_splitk_partial's, and any
 * difference is a bug here rather than a tolerance to widen. The one ordering
 * change is WHEN m and s are stored (right after each head's folds rather than
 * after the acc pass); those are plain stores of already-final values into
 * buffers no other threadgroup touches, so no bit depends on it.
 *
 * THAT CONTRACT RESTS ON -fno-fast-math, which the Makefile passes and calls
 * not optional. This kernel raises the stakes on it: an R-accumulator loop is
 * exactly the shape a fast-math optimizer would reassociate or contract
 * differently from the single-accumulator one, and the failure would be a
 * silent one-ulp drift rather than an error. Anyone building the metallib
 * outside the Makefile rule voids the guarantee.
 *
 * DETERMINISM (the rule at the top of this file) is unchanged: no atomics, no
 * simd_sum / simd_max, no read-modify-write of a shared accumulator. Holding
 * `repeat` SEPARATE per-thread accumulators is exactly what the rule allows;
 * the trees are still fixed-shape and still folded one head at a time.
 *
 * THE GROUP SIZE IS A COMPILE-TIME CONSTANT, by necessity: `repeat` private
 * accumulators indexed by a runtime value would be a stack array in device
 * memory rather than registers, which would cost more than the traffic it
 * saves. splitk_partial_group is therefore templated on it and the kernel
 * switches over the runtime `repeat`, with one instantiation per value up to
 * SG_SPLITK_GQA_MAX. A larger group still produces the right answer (the
 * default arm below runs the group one head at a time, which is bit for bit
 * what the per-head kernel does) but buys no reuse, so metal.m's policy keeps
 * the per-head kernel there instead.
 *
 * Q IS LEFT IN DEVICE MEMORY rather than staged into threadgroup memory. The
 * inner loop reads repeat*head_dim floats per key instead of head_dim, but
 * that is the SAME number of q loads the `repeat` separate threadgroups issued
 * before (it is only the K/V loads that collapse), a group's q slice is at
 * most 8 KB and is re-read every key, so it sits in cache; staging it would
 * add a fixed threadgroup allocation that cuts how many threadgroups a core
 * can hold. */

/* The largest GQA group this kernel keeps in registers. Mirrored in metal.m as
 * SG_SPLITK_GQA_MAX, the way SG_TG and SG_GEMM_TM already are; keep both in
 * sync. 8 covers every shape surge targets (27B: 24/4 = 6, 4B dense:
 * 32/8 = 4) with room for the 8 that 64-head/8-kv models use. */
constant uint SG_SPLITK_GQA_MAX = 8u;
/* The switch in the kernel below must have one arm per group size up to that
 * bound, and metal.m's policy uses the SAME bound to decide which shapes it
 * sends here; a bump that forgets either half would silently fall into the
 * no-reuse default arm. */
static_assert(SG_SPLITK_GQA_MAX == 8u,
              "SG_SPLITK_GQA_MAX changed: update the switch arms in "
              "k_attn_decode_splitk_partial_gqa and SG_SPLITK_GQA_MAX in metal.m");

/* Every loop that indexes one of the per-head accumulator arrays carries this.
 * The arrays must end up in REGISTERS: indexed by a runtime j they would be a
 * stack allocation in the thread address space, and the inner loop would then
 * do a load and a store per head per element, which is more traffic than the
 * K/V reads this kernel exists to remove. R is a compile-time constant, so the
 * trip count is always known and the hint always applies.
 *
 * WHAT THIS IS AND IS NOT EVIDENCE OF. `xcrun metal -c` emits AIR, and the AIR
 * carries the array as an `alloca [R x float]` with variable-index geps WITH OR
 * WITHOUT this pragma (a minimal standalone probe reproduces that), because the
 * unrolling and the register allocation happen in the driver's backend when the
 * pipeline is created, not in the offline compiler. So this is a hint whose
 * effect is only observable in a timing on hardware, and the compile-time check
 * cannot confirm it. */
#define SG_UNROLL_R _Pragma("clang loop unroll(full)")

/* One (kv head group, split) threadgroup's whole job, for a group of exactly R
 * query heads starting at h0. Written as a template so the `float dot[R]` and
 * `float a[R]` accumulators are register-resident (see SG_UNROLL_R above), and
 * called with R == 1 in a loop for group sizes past SG_SPLITK_GQA_MAX.
 *
 * Every argument is uniform across the threadgroup except lid, so both the
 * empty-split early return and the barriers below are reached by all threads
 * or none, whichever arm the caller took. */
template <uint R>
static void splitk_partial_group(device const float *q,
                                 device const half *kc,
                                 device const half *vc,
                                 device float *m_out,
                                 device float *s_out,
                                 device float *acc_out,
                                 device float *scores,
                                 threadgroup float *red,
                                 uint h0, uint hk, uint part, uint lid,
                                 uint n_kv, uint hd, uint seq, uint q_stride,
                                 float scale, uint n_splits)
{
    /* The partition rule, in 64-bit exactly like attn_decode_core's and
     * k_attn_decode_splitk_partial's. */
    uint t0 = (uint)(((ulong)part * (ulong)seq) / (ulong)n_splits);
    uint t1 = (uint)((((ulong)part + 1ul) * (ulong)seq) / (ulong)n_splits);

    /* The group's FIRST head's flat (head, split) index, plus the strides that
     * step to the next head in the group. Head h0+j sits at
     * part_idx + j*n_splits in m/s and (part_idx + j*n_splits)*hd in acc,
     * which is the per-query-head layout the combine already consumes. */
    size_t part_idx = (size_t)h0 * (size_t)n_splits + (size_t)part;
    size_t ms_stride = (size_t)n_splits;
    size_t acc_stride = (size_t)n_splits * (size_t)hd;
    device float *acc = acc_out + part_idx * (size_t)hd;

    /* Empty split: the documented m = -INFINITY / s = 0 / acc = 0 encoding for
     * EVERY head of the group. Uniform branch (t0, t1 are uniform). */
    if (t0 >= t1) {
        if (lid == 0u) {
            for (uint j = 0; j < R; j++) {
                m_out[part_idx + (size_t)j * ms_stride] = -INFINITY;
                s_out[part_idx + (size_t)j * ms_stride] = 0.0f;
            }
        }
        for (uint i = lid; i < hd; i += SG_TG) {
            for (uint j = 0; j < R; j++) acc[(size_t)j * acc_stride + i] = 0.0f;
        }
        return;
    }

    /* One private score row of `span` floats per (head, split), the same rows
     * and the same ceil(seq/n_splits) formula k_attn_decode_splitk_partial and
     * metal.m's splitk_sizes use, so the scratch buffer needs no resizing and
     * neighbouring rows still never share a byte. */
    uint span = (uint)((((ulong)seq + (ulong)n_splits) - 1ul) / (ulong)n_splits);
    device float *sc = scores + part_idx * (size_t)span;
    size_t sc_stride = (size_t)n_splits * (size_t)span;
    device const float *qh = q + (size_t)h0 * q_stride;
    uint len = t1 - t0;

    /* 1. scores over [t0, t1). Thread `lid` owns the split's keys lid,
     *    lid+256, ... exactly as before; the difference is that ONE read of
     *    kt[i] now feeds all R heads instead of R threadgroups each reading
     *    it. Per head the accumulation is still serial over increasing i from
     *    0.0f, so every dot product is bit for bit the per-head kernel's. */
    for (uint r = lid; r < len; r += SG_TG) {
        device const half *kt = kc + ((size_t)(t0 + r) * n_kv + hk) * hd;
        float dot[R];
        SG_UNROLL_R for (uint j = 0; j < R; j++) dot[j] = 0.0f;
        for (uint i = 0; i < hd; i++) {
            float kf = (float)kt[i];
            SG_UNROLL_R for (uint j = 0; j < R; j++) dot[j] += qh[(size_t)j * q_stride + i] * kf;
        }
        SG_UNROLL_R for (uint j = 0; j < R; j++) sc[(size_t)j * sc_stride + r] = dot[j] * scale;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    /* 2 + 3. Per head, in group order: m = max score over THIS split, then
     *    s = sum exp(score - m) with the exponentials left in the score row
     *    for step 4, and NO division by the sum (the split's weights must stay
     *    unnormalized for the combine's rescaling to undo). One fixed-shape
     *    tg_max and tg_sum per head over the same per-lane partials the
     *    per-head kernel folds, so both values are bit-identical to it.
     *
     *    tg_max / tg_sum are callable in a loop: each opens with a barrier
     *    that makes sure every thread has finished reading red[0] from the
     *    previous fold before this one overwrites red[lid]. Only this
     *    threadgroup writes these m/s slots, so storing them here rather than
     *    after step 4 changes no bit. */
    for (uint j = 0; j < R; j++) {
        device float *scj = sc + (size_t)j * sc_stride;
        float m = -INFINITY;
        for (uint r = lid; r < len; r += SG_TG) { float v = scj[r]; m = (v > m) ? v : m; }
        m = tg_max(red, lid, m);

        float sacc = 0.0f;
        for (uint r = lid; r < len; r += SG_TG) {
            float e = precise::exp(scj[r] - m);
            scj[r] = e;
            sacc += e;
        }
        float s = tg_sum(red, lid, sacc);

        /* tg_max/tg_sum hand the SAME value to every thread, so this is a
         * store from a designated thread, not a reduction. */
        if (lid == 0u) {
            m_out[part_idx + (size_t)j * ms_stride] = m;
            s_out[part_idx + (size_t)j * ms_stride] = s;
        }
    }
    /* The exponentials just written are read across threads in step 4. */
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    /* 4. acc[d] = sum_t exp(score_t - m) * v[t][d] over the split, thread
     *    `lid` owning OUTPUT dims lid, lid+256, ... and walking t in
     *    increasing order, so there is no cross-thread reduction and the
     *    summation order per head is fixed, exactly as before. ONE read of the
     *    V element now feeds all R heads. */
    for (uint i = lid; i < hd; i += SG_TG) {
        float a[R];
        SG_UNROLL_R for (uint j = 0; j < R; j++) a[j] = 0.0f;
        for (uint r = 0; r < len; r++) {
            float vf = (float)vc[((size_t)(t0 + r) * n_kv + hk) * hd + i];
            SG_UNROLL_R for (uint j = 0; j < R; j++) a[j] += sc[(size_t)j * sc_stride + r] * vf;
        }
        SG_UNROLL_R for (uint j = 0; j < R; j++) acc[(size_t)j * acc_stride + i] = a[j];
    }
}

/* One threadgroup per (KV HEAD, split), a 2D grid dispatched as
 * (x = split, y = kv head): metal.m's same SG_K_HEADS2D class as the per-head
 * partial, with the y extent changed from params[0] to params[1]. Buffers,
 * buffer indices and params are IDENTICAL to k_attn_decode_splitk_partial's,
 * so the two are drop-in alternatives for one another on the host side:
 *
 * params: [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=seq [4]=q_stride
 * [5]=softmax scale bits [6]=n_splits (params[7] unused).
 *
 * Like the per-head partial this binds metal.m's DEDICATED sg_gpu.splitk_
 * scratch at buffer 7, never the process-wide sg_gpu.scratch (P2.2 review
 * finding 1), and it writes the same rows of it. */
kernel void k_attn_decode_splitk_partial_gqa(device const float *q [[buffer(0)]],
                                             device const half *kc   [[buffer(1)]],
                                             device const half *vc   [[buffer(2)]],
                                             device float *m_out     [[buffer(3)]],
                                             device float *s_out     [[buffer(4)]],
                                             device float *acc_out   [[buffer(5)]],
                                             constant uint *p        [[buffer(6)]],
                                             device float *scores    [[buffer(7)]],
                                             uint2 tg  [[threadgroup_position_in_grid]],
                                             uint2 tid [[thread_position_in_threadgroup]])
{
    uint n_heads = p[0], n_kv = p[1], hd = p[2], seq = p[3];
    uint q_stride = p[4];
    float scale = as_type<float>(p[5]);
    uint n_splits = p[6];

    uint part = tg.x, hk = tg.y, lid = tid.x;
    /* Every term is uniform across the threadgroup (tg.x/tg.y are, lid is not
     * used), so these returns cannot strand a thread on a barrier below. */
    if (n_kv == 0u || hd == 0u || hk >= n_kv || part >= n_splits) return;

    uint repeat = n_heads / n_kv;
    /* UNREACHABLE through metal.m's entry points, which reject n_kv_heads == 0
     * and n_heads % n_kv_heads != 0 before encoding anything (sg_check_params).
     * Guarded anyway, and as a no-op rather than a best effort: with a group
     * size that does not tile n_heads exactly, the group starting at hk*repeat
     * would run off the end of the m/s/acc buffers, and a wild write is far
     * worse than an unwritten one. NOTE WHAT "UNWRITTEN" MEANS, though: the
     * m/s/acc buffers keep whatever they held, and the combine consumes those
     * stale bytes without complaint. This is a last resort against memory
     * corruption, NOT a diagnostic; sg_check_params is where a bad shape is
     * supposed to be caught, and it is. */
    if (repeat == 0u || n_kv * repeat != n_heads) return;

    threadgroup float red[SG_TG];
    uint h0 = hk * repeat;

    /* The switch is what makes R compile-time (see the header). Every thread
     * of the threadgroup takes the same arm, since `repeat` is uniform. */
    switch (repeat) {
    case 1u: splitk_partial_group<1>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 2u: splitk_partial_group<2>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 3u: splitk_partial_group<3>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 4u: splitk_partial_group<4>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 5u: splitk_partial_group<5>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 6u: splitk_partial_group<6>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 7u: splitk_partial_group<7>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    case 8u: splitk_partial_group<8>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                     h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                     scale, n_splits); break;
    default:
        /* A group larger than SG_SPLITK_GQA_MAX: still CORRECT, one head at a
         * time, which is bit for bit what k_attn_decode_splitk_partial does,
         * but with none of the reuse. metal.m's policy sends these shapes to
         * the per-head kernel instead, so this arm is the safety net for a
         * caller that reached this kernel anyway. */
        for (uint j = 0; j < repeat; j++) {
            splitk_partial_group<1>(q, kc, vc, m_out, s_out, acc_out, scores, red,
                                    h0 + j, hk, part, lid, n_kv, hd, seq, q_stride,
                                    scale, n_splits);
        }
        break;
    }
}

/* =====================================================================
 * ONLINE-SOFTMAX GQA-shared split-K decode partial (Task P2.8)
 * =====================================================================
 *
 * SAME OUTPUT LAYOUT AND SAME MATH AS k_attn_decode_splitk_partial_gqa, ONE
 * PASS INSTEAD OF FOUR, AND NO SCORE ROW IN DEVICE MEMORY AT ALL.
 *
 * The four-pass kernel above writes every score of the split into
 * `splitk_scratch` and then walks that row three more times (max, exponentiate,
 * accumulate). This kernel keeps a running `(m, s, acc)` per head and updates it
 * as the keys are visited, rescaling by exp(m_old - m_new) whenever the running
 * maximum moves. Consequences:
 *
 *   - the `scores` binding is GONE (seven buffers, not eight), so this kernel
 *     never touches g->splitk_scratch and never needs it to exist;
 *   - the score-row traffic (two writes and three reads of R*len floats per
 *     (group, split)) is replaced by threadgroup memory.
 *
 * BYTE-IDENTITY TO THE FOUR-PASS KERNEL IS NOT CLAIMED AND MUST NOT BE GATED
 * ON. Streaming changes the ORDER in which the exponentials are summed (per
 * tile, then across tiles with a rescale, instead of per lane then one tree),
 * so `s` and `acc` differ in the last bits whenever a split spans more than one
 * SG_TG-wide tile. `m` is still exactly the four-pass value (a maximum is
 * order-independent), and when the split fits in ONE tile the whole triple
 * happens to come out bit-identical (see the tile loop below). The bar for this
 * kernel is therefore accuracy against sg_ref_attn_decode_splitk plus
 * determinism, exactly as it was for P2.2's kernel, not memcmp against P2.4's.
 *
 * WHERE THE RUNNING ACCUMULATOR LIVES, WHICH IS THE WHOLE DESIGN PROBLEM.
 * A per-thread acc[R][head_dim] does not exist: at R = 8 and head_dim = 256
 * that is 2048 floats per thread, 2 MB per threadgroup. So the two roles are
 * split the way the four-pass kernel already splits them, and the streaming
 * state is attached to the role that makes it small:
 *
 *   - (m, s) are UNIFORM across a KEY GROUP (they come off a fold tree over that
 *     group's lanes), so every thread holds its own copy: 2*R registers.
 *   - acc[d] is owned by EXACTLY ONE thread per key group, the one that owns
 *     output dim d (thread lid owns d = dbase + lid % kw, see the dbase loop), so
 *     a thread needs ONE accumulator per head, not head_dim of them: R registers.
 *
 * Total streaming state is 4*R floats per thread (m, s, acc and the score of
 * the thread's own key), 32 at R = 8, which is a register file rather than a
 * memory allocation. That works because head_dim <= SG_TG, which holds for
 * every shape surge targets (27B 256, 4B dense 128) and is exactly SG_TG for
 * the widest of them. For head_dim > SG_TG the dbase loop runs more than once
 * and RECOMPUTES the split's scores per SG_TG-wide band of output dims: still
 * correct, but it re-reads K, so metal.m's policy declines those shapes and
 * sends them to the four-pass kernel instead (the same kind of policy decline
 * as a GQA group wider than SG_SPLITK_GQA_MAX, which this kernel also still
 * answers correctly one head at a time).
 *
 * THE PRICE IS A TRANSPOSE THROUGH THREADGROUP MEMORY, and it is unavoidable:
 * the score of key t is computed by the thread that owns key t and consumed by
 * every thread that owns an output dim, so the R x SG_TG block has to change
 * hands. Device memory is what the four-pass kernel uses for exactly this and
 * is what this kernel exists to avoid; threadgroup memory is the only other
 * shared store. The buffer is SG_SPLITK_GQA_MAX * SG_TG floats (8 KB, sized for
 * the worst-case group rather than the actual one, so nothing on the host has to
 * agree with the kernel about a threadgroup allocation length) and it does
 * double duty as the fold-tree scratch, which needs the same R x SG_TG shape.
 * 8 KB is well inside the 32 KB threadgroup limit, but it is 8x what the
 * four-pass kernel reserves and its effect on how many threadgroups a core can
 * hold is a TIMING question that has not been measured; if it costs occupancy,
 * the fix is a host-provided [[threadgroup(0)]] length of exactly R*SG_TG*4.
 *
 * KEY GROUPS, SO THE V PHASE HAS NO IDLE HALF (Task P2.9). One output dim per
 * thread means that at head_dim < SG_TG the threads with lid >= head_dim own
 * nothing: at head_dim 128 that is half the threadgroup sitting on a barrier
 * through the whole V phase of every tile, at head_dim 64 three quarters, and
 * P2.8's timing named it as a cause of the 0.690x-0.984x this kernel measured on
 * the 4B dense shape. The fix partitions the KEYS as well as the dims:
 *
 *   - kw, the KEY-GROUP WIDTH, is the smallest power of two >= head_dim, capped
 *     at SG_TG and floored at SG_SPLITK_ONLINE_KW_MIN. n_kgroups = SG_TG / kw.
 *   - thread lid belongs to key group lid / kw and owns output dim lid % kw, so
 *     EVERY thread owns a dim again (or, when head_dim is not a power of two,
 *     as few as possible do not: at head_dim 96, kw is 128 and 32 lanes per
 *     group still idle, which is strictly better than 128 of 256 idling).
 *   - a tile is still SG_TG keys and thread lid still scores key base + lid, so
 *     the score phase, its coalescing and its key-to-lane map are UNCHANGED.
 *     What changes is that the folds run per group over kw lanes instead of once
 *     over SG_TG, and the V phase walks only its own group's kw keys, so the
 *     serial length of the V phase drops by n_kgroups while the number of lanes
 *     issuing V loads rises by the same factor. The bytes are identical; the
 *     memory-level parallelism is not.
 *   - each group therefore ends the band with its OWN (m, s, acc[d]) over its own
 *     fixed subset of the split's keys, and those n_kgroups partials are merged
 *     by exactly the log-sum-exp merge k_attn_decode_splitk_combine already
 *     performs across splits (attn_combine_weight, shared, not re-derived),
 *     in FIXED group order 0, 1, ... so the result is data-independent. No
 *     division: the partial's contract is unnormalized acc, and the combine
 *     divides by S. An empty group (a split shorter than kw*n_kgroups leaves
 *     later groups no keys) carries m = -INFINITY, s = 0, acc = 0, whose weight
 *     against a finite M is exactly 0.0, so it contributes nothing without a
 *     special case; if EVERY group were empty the merge reproduces the
 *     -INFINITY/0/0 empty encoding, which is why the early return above is the
 *     only place that needs to write it.
 *   - THE MERGE COSTS NO THREADGROUP MEMORY. kw * n_kgroups == SG_TG, so the R
 *     group-local fold regions tile the SAME R x SG_TG buffer, and the acc
 *     exchange (R floats per thread) is exactly that buffer again. The 8 KB
 *     allocation above is unchanged, which matters because it is the OTHER
 *     suspect for the 4B loss and this task must not trade one for the other.
 *   - n_kgroups == 1 (head_dim >= SG_TG, the 27B's path) SKIPS the merge
 *     entirely and every expression above degenerates to P2.8's: kw == SG_TG
 *     makes lid % kw == lid, the fold's lane mask lid & (kw-1) == lid, and the
 *     group's key offset 0. That path was measured winning and must not move.
 *
 * DETERMINISM (the rule at the top of this file) is unchanged. Every thread
 * SCORES its OWN fixed subset of keys (lid, lid+SG_TG, ..., the same subset the
 * four-pass kernel gives it), accumulates V over its own key group's keys in
 * increasing order, the per-lane partials are combined only by fixed-shape trees
 * with a data-independent stride schedule, the key groups are merged in fixed
 * index order, and every acc slot has exactly one writer. No atomics, no
 * simd_sum/simd_max, no read-modify-write of a shared accumulator, and no
 * cross-thread update whose result depends on which thread got there first. The
 * rescale is a multiply by a value every thread computes identically from the
 * tree's output.
 *
 * P2.9 makes the fold WIDTH (and therefore the number of tree levels) a function
 * of head_dim, which is a shape parameter, uniform across the threadgroup and
 * known before the first barrier. That is the same kind of dependence the tile
 * loop already has on `len`: two dispatches of the SAME shape fold identically,
 * which is what the byte-exact gates and the 100x determinism check assert. What
 * the file's rule forbids, a shape that depends on the DATA or on the dispatched
 * threads_per_threadgroup, is still absent. */

/* The fold-tree, transpose and key-group-exchange scratch, shared (see the
 * header). One row of SG_TG floats per head of the widest supported group. */
constant uint SG_SPLITK_ONLINE_RED = SG_SPLITK_GQA_MAX * SG_TG;

/* P2.9: the narrowest key group the V phase will split into, which is what bounds
 * n_kgroups at SG_TG / SG_SPLITK_ONLINE_KW_MIN. 32 for two reasons:
 *
 *   - a simdgroup is 32 lanes on Apple silicon, so a fold narrower than 32 does
 *     not remove a single SIMD step from the tree, while every extra key group
 *     adds a log-sum-exp merge term and its rounding;
 *   - the merge publishes 2*n_kgroups floats per head into ONE SG_TG-wide row of
 *     the scratch (m then s), so n_kgroups must stay well under SG_TG/2. The
 *     static_assert below is that bound, not a style check.
 *
 * head_dim below 32 is not a shape surge dispatches (the smallest real one is
 * 128), so this floor only ever costs idle V lanes on toy shapes, never
 * correctness: a group whose lane owns no dim simply skips the V phase. */
constant uint SG_SPLITK_ONLINE_KW_MIN = 32u;
static_assert(2u * (SG_TG / SG_SPLITK_ONLINE_KW_MIN) <= SG_TG,
              "the key-group merge publishes 2*n_kgroups floats per head into one "
              "SG_TG-wide scratch row: raise SG_SPLITK_ONLINE_KW_MIN");
static_assert((SG_TG & (SG_TG - 1u)) == 0u
              && (SG_SPLITK_ONLINE_KW_MIN & (SG_SPLITK_ONLINE_KW_MIN - 1u)) == 0u,
              "the fold's lane mask (lid & (kw-1)) and n_kgroups == SG_TG/kw both "
              "need SG_TG and SG_SPLITK_ONLINE_KW_MIN to be powers of two");

/* R fixed-shape trees at once, one per head of the group, over the R x SG_TG
 * scratch. Folding R heads together rather than calling tg_max R times is what
 * keeps the barrier count independent of the group size: the online kernel folds
 * once per TILE, so an R-fold barrier multiplier would land on top of a
 * tiles-per-split multiplier.
 *
 * BIT-IDENTICAL TO R SEPARATE tg_max CALLS BY CONSTRUCTION: head j's row is
 * folded by the same stride schedule (kw/2, ..., 1), over the same per-lane
 * values, with the same NaN-wins comparison, and no row ever reads another
 * row's data. Only the barriers are shared.
 *
 * kw IS THE FOLD WIDTH (P2.9), a power of two dividing SG_TG. Each of the
 * SG_TG/kw KEY GROUPS folds its own kw-lane region [g*kw, g*kw+kw) of row j
 * independently, and (lid & (kw-1)) < stride keeps every read inside the region
 * that owns it ((lid & (kw-1)) + stride < 2*stride <= kw). At kw == SG_TG this is
 * P2.8's function operation for operation: lid < SG_TG makes lid & (kw-1) == lid,
 * so the mask, the schedule and the result slot are all the earlier ones.
 *
 * THE RESULT IS LEFT IN red[j*SG_TG + g*kw], not returned: the caller needs the R
 * results and an out-parameter array would be R more live registers in the
 * hottest loop in the kernel. Every thread may read its own group's slot from the
 * trailing barrier below until the next fold's LEADING barrier, which is the
 * same contract tg_max/tg_sum already have with their red[0]. */
template <uint R>
static inline void tg_max_group(threadgroup float *red, uint lid, thread const float *v,
                                uint kw) {
    uint lane = lid & (kw - 1u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SG_UNROLL_R for (uint j = 0; j < R; j++) red[j * SG_TG + lid] = v[j];
    for (uint stride = kw / 2u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < stride) {
            SG_UNROLL_R for (uint j = 0; j < R; j++) {
                float a = red[j * SG_TG + lid], b = red[j * SG_TG + lid + stride];
                red[j * SG_TG + lid] = (a != a) ? a : ((b != b) ? b : ((b > a) ? b : a));
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* Same tree, same contract, addition. tg_sum's own comment applies verbatim. */
template <uint R>
static inline void tg_sum_group(threadgroup float *red, uint lid, thread const float *v,
                                uint kw) {
    uint lane = lid & (kw - 1u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    SG_UNROLL_R for (uint j = 0; j < R; j++) red[j * SG_TG + lid] = v[j];
    for (uint stride = kw / 2u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < stride) {
            SG_UNROLL_R for (uint j = 0; j < R; j++) {
                red[j * SG_TG + lid] = red[j * SG_TG + lid] + red[j * SG_TG + lid + stride];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* One (kv head group, split) threadgroup's whole job, streamed. Same arguments
 * as splitk_partial_group above minus `scores`, same empty-split encoding, same
 * m/s/acc layout. Every argument except lid is uniform across the threadgroup,
 * so the early return and every barrier below is reached by all threads or
 * none. */
template <uint R>
static void splitk_partial_group_online(device const float *q,
                                        device const half *kc,
                                        device const half *vc,
                                        device float *m_out,
                                        device float *s_out,
                                        device float *acc_out,
                                        threadgroup float *red,
                                        uint h0, uint hk, uint part, uint lid,
                                        uint n_kv, uint hd, uint seq, uint q_stride,
                                        float scale, uint n_splits)
{
    /* EVERY red index below is j * SG_TG + something < SG_TG, so R rows must fit
     * the caller's allocation. The switch that instantiates this only reaches 8,
     * and SG_SPLITK_ONLINE_RED is sized from the same constant, but a future arm
     * added past the bound would silently write into another kernel's threadgroup
     * memory rather than fail to build. This is that failure, at compile time.
     * (P2.9 review finding: the bound was previously implicit.) */
    static_assert(R * SG_TG <= SG_SPLITK_ONLINE_RED,
                  "splitk_partial_group_online<R> indexes R rows of SG_TG floats: "
                  "R exceeds SG_SPLITK_ONLINE_RED / SG_TG");

    /* The partition rule, in 64-bit, IDENTICAL to the four-pass kernels' and to
     * sg_ref_attn_decode_splitk's. */
    uint t0 = (uint)(((ulong)part * (ulong)seq) / (ulong)n_splits);
    uint t1 = (uint)((((ulong)part + 1ul) * (ulong)seq) / (ulong)n_splits);

    size_t part_idx = (size_t)h0 * (size_t)n_splits + (size_t)part;
    size_t ms_stride = (size_t)n_splits;
    size_t acc_stride = (size_t)n_splits * (size_t)hd;
    device float *acc = acc_out + part_idx * (size_t)hd;

    /* Empty split: the documented m = -INFINITY / s = 0 / acc = 0 encoding for
     * every head of the group, byte for byte what the four-pass kernel writes. */
    if (t0 >= t1) {
        if (lid == 0u) {
            for (uint j = 0; j < R; j++) {
                m_out[part_idx + (size_t)j * ms_stride] = -INFINITY;
                s_out[part_idx + (size_t)j * ms_stride] = 0.0f;
            }
        }
        for (uint i = lid; i < hd; i += SG_TG) {
            for (uint j = 0; j < R; j++) acc[(size_t)j * acc_stride + i] = 0.0f;
        }
        return;
    }

    device const float *qh = q + (size_t)h0 * q_stride;
    uint len = t1 - t0;

    /* THE KEY-GROUP SPLIT (P2.9, see the header). kw is the smallest power of two
     * >= hd, capped at SG_TG and floored at SG_SPLITK_ONLINE_KW_MIN; n_kg is how
     * many groups of that width the threadgroup holds. Every term is uniform
     * across the threadgroup (hd is a param), so the loop trip count, every
     * barrier below and the n_kg > 1 merge are reached by all threads or none.
     *
     * hd >= SG_TG (the 27B's 256, and the multi-band path above it) gives
     * kw == SG_TG and n_kg == 1, which is P2.8's shape exactly: kgroup 0,
     * kg_off 0, dlane == lid. */
    uint kw = SG_TG;
    if (hd < SG_TG) {
        kw = SG_SPLITK_ONLINE_KW_MIN;
        /* Doubles from the floor, so it stops at the smallest power of two that
         * is >= hd and >= SG_SPLITK_ONLINE_KW_MIN. It cannot overrun SG_TG: hd is
         * < SG_TG here and SG_TG is itself a power of two, so the last doubling
         * that can happen lands exactly on SG_TG (n_kg 1, the P2.8 path). */
        while (kw < hd) kw <<= 1u;
    }
    uint n_kg = SG_TG / kw;
    uint kgroup = lid / kw;
    uint kg_off = kgroup * kw;          /* this group's key offset inside a tile */
    uint dlane = lid - kg_off;          /* lid % kw: this thread's slot in its group */

    /* Output dims are handed out kw at a time, and thread lid owns exactly ONE of
     * them per band (d = dbase + dlane), which is what makes acc[R] fit in
     * registers. head_dim <= kw (which n_kg > 1 guarantees by construction, and
     * which every shape metal.m sends here satisfies) runs this loop once; a
     * head_dim wider than SG_TG re-streams the split per band, which is correct
     * but re-reads K, and is why the policy declines it. */
    for (uint dbase = 0; dbase < hd; dbase += kw) {
        uint d = dbase + dlane;
        bool mine = (d < hd);

        /* The entire streaming state: R running maxima, R running sums, R
         * accumulators for THIS thread's output dim. */
        float m[R], s[R], a[R];
        SG_UNROLL_R for (uint j = 0; j < R; j++) { m[j] = -INFINITY; s[j] = 0.0f; a[j] = 0.0f; }

        /* One tile of at most SG_TG keys per pass, thread lid taking key
         * base+lid: over the whole split that is keys lid, lid+SG_TG, ..., the
         * SAME subset of the same split the four-pass kernel gives thread lid,
         * scored by the same operations in the same order. The tile width is
         * SG_TG whatever kw is, which is why P2.9's key groups leave the score
         * phase and its coalescing untouched: group g simply owns the tile's
         * keys [g*kw, g*kw+kw), and thread lid = g*kw + dlane owns key base+lid
         * within it.
         *
         * At n_kg == 1 a split short enough to be ONE tile comes out
         * bit-identical to the four-pass kernel: with one key per lane there is
         * no per-lane serial sum to reorder, the tile maximum IS the split
         * maximum, and the acc loop walks t in the same increasing order. At
         * n_kg > 1 that no longer holds even for one tile, because the exponential
         * sum is now per group and then merged; the bar for this kernel has always
         * been the CPU oracle and determinism, never memcmp (see the header). */
        for (uint base = 0; base < len; base += SG_TG) {
            uint tlen = len - base;
            if (tlen > SG_TG) tlen = SG_TG;

            /* 1. This thread's key against all R query vectors. ONE read of
             *    kt[i] feeds the whole group, and per head the accumulation is
             *    serial over increasing i from 0.0f, the four-pass kernel's
             *    dot product operand for operand. Lanes with no key sit out at
             *    -INFINITY so the maximum fold ignores them. */
            float sc[R];
            if (lid < tlen) {
                device const half *kt = kc + ((size_t)(t0 + base + lid) * n_kv + hk) * hd;
                SG_UNROLL_R for (uint j = 0; j < R; j++) sc[j] = 0.0f;
                for (uint i = 0; i < hd; i++) {
                    float kf = (float)kt[i];
                    SG_UNROLL_R for (uint j = 0; j < R; j++) sc[j] += qh[(size_t)j * q_stride + i] * kf;
                }
                SG_UNROLL_R for (uint j = 0; j < R; j++) sc[j] *= scale;
            } else {
                SG_UNROLL_R for (uint j = 0; j < R; j++) sc[j] = -INFINITY;
            }

            /* 2. This KEY GROUP's maximum over its slice of the tile, per head,
             *    one R-wide fixed tree per group over kw lanes. */
            tg_max_group<R>(red, lid, sc, kw);

            /* 3. The online update. Every thread computes the SAME m_new and
             *    the SAME rescale factor from the tree's output, so this is a
             *    private recomputation of a uniform value, not a shared
             *    accumulator being read-modify-written.
             *
             *    m_new == m_old gives an EXACT 1.0 rather than exp(0.0), which
             *    is not just cheaper: it makes the rescale a no-op bit for bit
             *    for every tile that does not move the maximum, so at n_kg == 1 a
             *    split whose maximum sits in its first tile keeps the four-pass
             *    kernel's acc exactly. It also defines the m_old == m_new ==
             *    -INFINITY case, where exp(-INF - -INF) would be a NaN
             *    manufactured out of two legitimate values, which is what a key
             *    group with NO keys in this tile hits on every one of them.
             *
             *    NaN PROPAGATES rather than being laundered: a NaN maximum
             *    fails the == test, so corr becomes NaN and carries into s and
             *    acc, which is the same NaN-out answer tg_max's rule gives the
             *    four-pass kernel. */
            SG_UNROLL_R for (uint j = 0; j < R; j++) {
                float mo = m[j], tmax = red[j * SG_TG + kg_off];
                float mn = (mo != mo) ? mo : ((tmax != tmax) ? tmax : ((tmax > mo) ? tmax : mo));
                float corr = (mn == mo) ? 1.0f : precise::exp(mo - mn);
                m[j] = mn;
                s[j] *= corr;
                /* Threads with no output dim in this band carry a[j] == 0.0
                 * forever and never store it, so rescaling it is harmless and
                 * cheaper than a branch. */
                a[j] *= corr;
                /* sc becomes the exponential in place; a lane with no key
                 * contributes an exact 0.0 to both the sum and the transpose. */
                sc[j] = (lid < tlen) ? precise::exp(sc[j] - mn) : 0.0f;
            }

            /* 4. This key group's sum over its slice of the tile, per head, same
             *    R-wide tree, added to the already-rescaled running sum. */
            tg_sum_group<R>(red, lid, sc, kw);
            SG_UNROLL_R for (uint j = 0; j < R; j++) s[j] += red[j * SG_TG + kg_off];

            /* 5. The transpose: the exponentials were computed by the thread
             *    that owns the KEY and are consumed by the threads that own the
             *    output DIMS, so the R x tlen block changes hands here. The
             *    leading barrier is what lets this overwrite the fold scratch
             *    (every thread has read its own group's slot above by then).
             *    Thread lid writes slot lid, which puts group g's exponentials in
             *    exactly the slots [g*kw, g*kw+kw) its own V phase reads. */
            threadgroup_barrier(mem_flags::mem_threadgroup);
            SG_UNROLL_R for (uint j = 0; j < R; j++) red[j * SG_TG + lid] = sc[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* 6. acc[d] += sum_t p_t * v[t][d] over THIS GROUP's slice of THIS
             *    tile, increasing t, one output dim per thread per group so there
             *    is no cross-thread reduction and the per-head summation order is
             *    fixed. ONE read of the V element feeds all R heads.
             *
             *    glen is how many of the group's kw keys the tile actually holds,
             *    which is kw for every full group, a remainder for at most one
             *    group, and ZERO for the groups past the end of a short tile. That
             *    is the only clamp needed for a split shorter than SG_TG: no
             *    thread reads a V row past the split, and a group with glen == 0
             *    keeps the -INFINITY/0/0 state the merge below weights out. */
            uint glen = (tlen > kg_off) ? (tlen - kg_off) : 0u;
            if (glen > kw) glen = kw;
            if (mine) {
                for (uint t = 0; t < glen; t++) {
                    float vf = (float)vc[((size_t)(t0 + base + kg_off + t) * n_kv + hk) * hd + d];
                    SG_UNROLL_R for (uint j = 0; j < R; j++) {
                        a[j] += red[j * SG_TG + kg_off + t] * vf;
                    }
                }
            }
        }

        /* 7. MERGE THE KEY GROUPS (P2.9). Each group now holds its own
         *    (m, s, acc[d]) over a disjoint, fixed subset of the split's keys, and
         *    combining those is exactly what k_attn_decode_splitk_combine does
         *    across splits, so this is that same log-sum-exp merge over
         *    attn_combine_weight (the shared helper, not a second copy):
         *
         *        M      = max_g m_g                (fixed order, NaN-propagating)
         *        s      = sum_g s_g   * w(m_g, M)  (fixed order)
         *        acc[d] = sum_g acc_g[d] * w(m_g, M)
         *
         *    with NO division: the partial's contract is the unnormalized acc and
         *    the combine kernel divides by S. The three degenerate cases need no
         *    special case, which is why there is none: an empty group's
         *    m_g == -INFINITY weights to exactly 0.0 against a finite M; a NaN
         *    anywhere makes M NaN and every weight NaN, which propagates to m/s/acc
         *    and then through the combine; and if every group were empty the result
         *    is the -INFINITY/0/0 empty encoding itself.
         *
         *    n_kg == 1 SKIPS this entirely (a uniform branch, so no thread waits on
         *    a barrier no other thread reaches), which is what keeps the
         *    head_dim >= SG_TG path bit for bit P2.8's. */
        if (n_kg > 1u) {
            /* 7a. Publish each group's (m, s): 2*n_kg floats per head, m in slots
             *     [0, n_kg) of row j and s in [n_kg, 2*n_kg), which fits because
             *     n_kg <= SG_TG / SG_SPLITK_ONLINE_KW_MIN (static_assert above).
             *     One designated thread per group, so no slot has two writers. The
             *     leading barrier is what lets this overwrite the transpose. */
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (dlane == 0u) {
                SG_UNROLL_R for (uint j = 0; j < R; j++) {
                    red[j * SG_TG + kgroup] = m[j];
                    red[j * SG_TG + n_kg + kgroup] = s[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            /* 7b. EVERY thread folds the same n_kg values in the same order into
             *     the same M and S, so this is a private recomputation of a
             *     uniform value rather than a shared reduction, exactly like the
             *     rescale in step 3. Each thread then scales its OWN group's
             *     accumulator, which is the multiplication that makes 7c a plain
             *     sum in group order. */
            SG_UNROLL_R for (uint j = 0; j < R; j++) {
                threadgroup const float *row = red + (size_t)j * SG_TG;
                float mm = -INFINITY;
                for (uint g = 0; g < n_kg; g++) {
                    float v = row[g];
                    mm = (v != v) ? v : ((v > mm) ? v : mm);
                }
                float ss = 0.0f;
                for (uint g = 0; g < n_kg; g++) {
                    ss += row[n_kg + g] * attn_combine_weight(row[g], mm);
                }
                a[j] *= attn_combine_weight(m[j], mm);
                m[j] = mm;
                s[j] = ss;
            }

            /* 7c. Exchange the weighted accumulators and sum them in group order.
             *     Thread lid publishes slot lid, so group g's dim dlane sits at
             *     g*kw + dlane, and group 0's thread for each dim collects the
             *     column. It re-reads its own contribution out of the scratch
             *     rather than using the register it already holds, so the summation
             *     order is g = 0, 1, ... for every dim with no special first term. */
            threadgroup_barrier(mem_flags::mem_threadgroup);
            SG_UNROLL_R for (uint j = 0; j < R; j++) red[j * SG_TG + lid] = a[j];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kgroup == 0u && mine) {
                SG_UNROLL_R for (uint j = 0; j < R; j++) {
                    float sum = 0.0f;
                    for (uint g = 0; g < n_kg; g++) sum += red[j * SG_TG + g * kw + dlane];
                    a[j] = sum;
                }
            }
        }

        /* m and s are uniform across the threadgroup once step 7 has run (before
         * it they are uniform only within a key group), so this is a store from a
         * designated thread, not a reduction. Thread 0 is in key group 0 either
         * way. Bands past the first recompute the identical values (same inputs,
         * same order), so only the first stores them; bands only exist at
         * n_kg == 1, where step 7 is skipped. */
        if (lid == 0u && dbase == 0u) {
            SG_UNROLL_R for (uint j = 0; j < R; j++) {
                m_out[part_idx + (size_t)j * ms_stride] = m[j];
                s_out[part_idx + (size_t)j * ms_stride] = s[j];
            }
        }
        /* Exactly one writer per acc slot: at n_kg == 1 every thread that owns a
         * dim, and at n_kg > 1 the group-0 thread that owns it, which is the one
         * step 7c left the merged value in. */
        if (kgroup == 0u && mine) {
            SG_UNROLL_R for (uint j = 0; j < R; j++) acc[(size_t)j * acc_stride + d] = a[j];
        }
    }
}

/* SEVEN BINDINGS, not eight: no score scratch. Otherwise a drop-in alternative
 * to k_attn_decode_splitk_partial_gqa, with the same (x = split, y = kv head)
 * SG_K_HEADS2D grid, the same params array and the same output layout, so
 * k_attn_decode_splitk_combine and every host-side size rule are untouched.
 *
 * params: [0]=n_heads [1]=n_kv_heads [2]=head_dim [3]=seq [4]=q_stride
 * [5]=softmax scale bits [6]=n_splits (params[7] unused). */
kernel void k_attn_decode_splitk_partial_gqa_online(device const float *q [[buffer(0)]],
                                                    device const half *kc  [[buffer(1)]],
                                                    device const half *vc  [[buffer(2)]],
                                                    device float *m_out    [[buffer(3)]],
                                                    device float *s_out    [[buffer(4)]],
                                                    device float *acc_out  [[buffer(5)]],
                                                    constant uint *p       [[buffer(6)]],
                                                    uint2 tg  [[threadgroup_position_in_grid]],
                                                    uint2 tid [[thread_position_in_threadgroup]])
{
    uint n_heads = p[0], n_kv = p[1], hd = p[2], seq = p[3];
    uint q_stride = p[4];
    float scale = as_type<float>(p[5]);
    uint n_splits = p[6];

    uint part = tg.x, hk = tg.y, lid = tid.x;
    /* Every term is uniform across the threadgroup, so these returns cannot
     * strand a thread on a barrier below. */
    if (n_kv == 0u || hd == 0u || hk >= n_kv || part >= n_splits) return;

    uint repeat = n_heads / n_kv;
    /* UNREACHABLE through metal.m's entry points (sg_check_params, in
     * metal_validate.m since task R3, rejects
     * n_kv_heads == 0 and n_heads % n_kv_heads != 0 first). Guarded as a no-op
     * for the reason the four-pass kernel states: with a group size that does
     * not tile n_heads, the group at hk*repeat would run off the end of the
     * m/s/acc buffers, and a wild write is worse than an unwritten one. */
    if (repeat == 0u || n_kv * repeat != n_heads) return;

    threadgroup float red[SG_SPLITK_ONLINE_RED];
    uint h0 = hk * repeat;

    /* The switch is what makes R compile-time, so m/s/acc/sc stay in registers
     * (see SG_UNROLL_R above). `repeat` is uniform, so every thread of the
     * threadgroup takes the same arm. */
    switch (repeat) {
    case 1u: splitk_partial_group_online<1>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 2u: splitk_partial_group_online<2>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 3u: splitk_partial_group_online<3>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 4u: splitk_partial_group_online<4>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 5u: splitk_partial_group_online<5>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 6u: splitk_partial_group_online<6>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 7u: splitk_partial_group_online<7>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    case 8u: splitk_partial_group_online<8>(q, kc, vc, m_out, s_out, acc_out, red,
                                            h0, hk, part, lid, n_kv, hd, seq, q_stride,
                                            scale, n_splits); break;
    default:
        /* A group wider than SG_SPLITK_GQA_MAX: still CORRECT, one head at a
         * time, but with none of the K/V reuse. metal.m's policy sends these
         * shapes to the four-pass per-head kernel instead; this arm is the
         * safety net for a caller that reached this kernel anyway. */
        for (uint j = 0; j < repeat; j++) {
            splitk_partial_group_online<1>(q, kc, vc, m_out, s_out, acc_out, red,
                                           h0 + j, hk, part, lid, n_kv, hd, seq, q_stride,
                                           scale, n_splits);
        }
        break;
    }
}
