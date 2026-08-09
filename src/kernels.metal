/* kernels.metal - the surge decode-step kernels (Task 9).
 *
 * Contract, buffer layouts and params[] are documented once, in surge.h,
 * next to sg_gpu_run_op. This file is the implementation and the notes here
 * are about HOW, not WHAT.
 *
 * DETERMINISM IS THE POINT. Every reduction in this file has a shape fixed
 * at compile time:
 *
 *   1. thread `lid` accumulates elements lid, lid+256, lid+512, ... into a
 *      private register, in increasing index order;
 *   2. the 256 partials go into threadgroup memory and are folded by a
 *      binary tree with a fixed stride schedule (128, 64, ... 1).
 *
 * The threadgroup width is the compile-time constant SG_TG, NOT the
 * dispatched `threads_per_threadgroup`, so the tree cannot change shape even
 * if a future caller dispatches a different width (metal.m asserts it
 * dispatches exactly SG_TG for these kernels). Consequences:
 *
 *   - no atomics anywhere;
 *   - no simd_sum / simd_max: those fold across a simdgroup in an order the
 *     Metal spec does not specify, and while Apple's implementation is in
 *     practice a fixed shuffle tree, "in practice" is not what Task 10's
 *     byte-exact gate can be built on;
 *   - no read-modify-write of a shared accumulator by more than one thread;
 *   - the elementwise kernels are trivially deterministic (one output
 *     element is written by exactly one thread, from inputs it alone reads).
 *
 * PRECISION. There is no double in Metal, so these cannot reproduce ref.c's
 * double accumulators; the measured gap is ~1e-7 relative at the sizes the
 * per-op tests use. Two deliberate choices keep it there:
 *
 *   - transcendentals go through the precise:: namespace (exp, sqrt), since
 *     the default fast-math versions are only ~2 ulp and, worse, differ from
 *     libm by more than the parity bar on the tails. metal.m's build also
 *     passes -fno-fast-math.
 *   - RoPE takes cos/sin as INPUTS. ref.c computes the angle
 *     pos * theta^(-2i/rope_dim) and its sine/cosine in double on purpose
 *     (at pos 262143 the f32-rounded angle is already 8e-3 off), and no f32
 *     kernel can reproduce that. So the host precomputes the table in double
 *     and uploads f32 cosines and sines, leaving the kernel with two
 *     multiplies and an add; parity is then ~6e-8 rather than ~1e-2.
 */
#include <metal_stdlib>
using namespace metal;

/* The one threadgroup width every reduction kernel is dispatched with. A
 * power of two, so the fold tree is exactly log2(SG_TG) levels. */
constant uint SG_TG = 256u;

/* bf16 -> f32: the bf16 bit pattern IS the top half of the f32 one, which
 * makes this exact and identical to ref.c's bf16_to_f32. */
static inline float bf16_to_f32(ushort h) {
    return as_type<float>((uint)h << 16);
}

/* Numerically stable sigmoid, same split as sg_ref_sigmoid: for x >= 0 use
 * 1/(1+e^-x) so exp() never overflows, and the mirrored form below zero. */
static inline float sg_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + precise::exp(-x));
    float e = precise::exp(x);
    return e / (1.0f + e);
}

static inline float sg_silu(float x) { return x * sg_sigmoid(x); }

/* Fixed-shape tree sum over a threadgroup. Callable repeatedly: the leading
 * barrier makes sure every thread has finished READING the previous result
 * out of red[0] before this call overwrites red[lid]. */
static inline float tg_sum(threadgroup float *red, uint lid, float v) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[lid] = v;
    for (uint stride = SG_TG / 2u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) red[lid] = red[lid] + red[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

/* Same tree, maximum. A NaN on EITHER side wins, so a NaN anywhere in the
 * row survives the fold; fmax() would swallow it and hand back a plausible
 * maximum computed from poisoned data. */
static inline float tg_max(threadgroup float *red, uint lid, float v) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[lid] = v;
    for (uint stride = SG_TG / 2u; stride > 0u; stride >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) {
            float a = red[lid], b = red[lid + stride];
            red[lid] = (a != a) ? a : ((b != b) ? b : ((b > a) ? b : a));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0];
}

/* =====================================================================
 * Normalization
 * ===================================================================== */

kernel void k_rmsnorm(device const float *x   [[buffer(0)]],
                      device const float *w   [[buffer(1)]],
                      device float *out       [[buffer(2)]],
                      constant uint *p        [[buffer(3)]],
                      uint lid [[thread_position_in_threadgroup]])
{
    uint n = p[0];
    float eps = as_type<float>(p[1]);
    bool has_w = p[2] != 0u;
    if (n == 0u) return;

    threadgroup float red[SG_TG];
    float acc = 0.0f;
    for (uint i = lid; i < n; i += SG_TG) acc += x[i] * x[i];
    float sumsq = tg_sum(red, lid, acc);

    /* Every thread recomputes the same scale from the same red[0]; no
     * broadcast step, no second reduction. */
    float scale = 1.0f / precise::sqrt(sumsq / (float)n + eps);
    for (uint i = lid; i < n; i += SG_TG) {
        out[i] = has_w ? (x[i] * scale * w[i]) : (x[i] * scale);
    }
}

/* RMSNormGated, one threadgroup per value head: the DeltaNet readout y is
 * RMS-normed with the layer's ssm_norm weight and then gated by silu of the
 * UNNORMALIZED z, exactly as gdn_layer composes sg_ref_rmsnorm + sg_ref_swiglu. */
kernel void k_rmsnorm_gated(device const float *y  [[buffer(0)]],
                            device const float *zw [[buffer(1)]],
                            device float *out      [[buffer(2)]],
                            constant uint *p       [[buffer(3)]],
                            uint h   [[threadgroup_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]])
{
    uint dv = p[0], heads = p[1];
    float eps = as_type<float>(p[2]);
    if (h >= heads || dv == 0u) return;

    device const float *yh = y + (size_t)h * dv;
    device const float *zh = zw + (size_t)h * dv;
    device const float *w  = zw + (size_t)heads * dv;   /* norm weight, shared */
    device float *oh = out + (size_t)h * dv;

    threadgroup float red[SG_TG];
    float acc = 0.0f;
    for (uint i = lid; i < dv; i += SG_TG) acc += yh[i] * yh[i];
    float sumsq = tg_sum(red, lid, acc);
    float scale = 1.0f / precise::sqrt(sumsq / (float)dv + eps);

    for (uint i = lid; i < dv; i += SG_TG) {
        oh[i] = sg_silu(zh[i]) * (yh[i] * scale * w[i]);
    }
}

/* =====================================================================
 * RoPE (partial, half-split). b holds cos[half] then sin[half].
 * ===================================================================== */

kernel void k_rope(device const float *x  [[buffer(0)]],
                   device const float *cs [[buffer(1)]],
                   device float *out      [[buffer(2)]],
                   constant uint *p       [[buffer(3)]],
                   uint i [[thread_position_in_grid]])
{
    uint head_dim = p[0], rope_dim = p[1];
    if (i >= head_dim) return;
    uint half_dim = rope_dim / 2u;

    if (i < half_dim) {
        /* Thread i owns the PAIR (i, i+half): mlx's traditional=false
         * pairing rotates element i against element i+rope_dim/2, so both
         * outputs are written here and threads in [half, rope_dim) do
         * nothing. Reading x and writing out is safe because out never
         * aliases x (surge.h's aliasing rule). */
        float c = cs[i], s = cs[half_dim + i];
        float lo = x[i], hi = x[i + half_dim];
        out[i] = lo * c - hi * s;
        out[i + half_dim] = lo * s + hi * c;
    } else if (i >= rope_dim) {
        /* The unrotated tail, [rope_dim, head_dim), copied verbatim. */
        out[i] = x[i];
    }
}

/* =====================================================================
 * Matrix-vector, y = W x, W row-major [rows, cols], one threadgroup per row
 * ===================================================================== */

kernel void k_matvec_bf16(device const ushort *w [[buffer(0)]],
                          device const float *x  [[buffer(1)]],
                          device float *y        [[buffer(2)]],
                          constant uint *p       [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1];
    if (row >= rows) return;

    threadgroup float red[SG_TG];
    device const ushort *wr = w + (size_t)row * cols;
    float acc = 0.0f;
    for (uint c = lid; c < cols; c += SG_TG) acc += bf16_to_f32(wr[c]) * x[c];
    float s = tg_sum(red, lid, acc);
    if (lid == 0u) y[row] = s;
}

kernel void k_matvec_f32(device const float *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y       [[buffer(2)]],
                         constant uint *p      [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]])
{
    uint rows = p[0], cols = p[1];
    if (row >= rows) return;

    threadgroup float red[SG_TG];
    device const float *wr = w + (size_t)row * cols;
    float acc = 0.0f;
    for (uint c = lid; c < cols; c += SG_TG) acc += wr[c] * x[c];
    float s = tg_sum(red, lid, acc);
    if (lid == 0u) y[row] = s;
}

/* =====================================================================
 * Softmax, single threadgroup, tree max then tree sum
 * ===================================================================== */

kernel void k_softmax(device const float *x [[buffer(0)]],
                      device const float *unused [[buffer(1)]],
                      device float *out     [[buffer(2)]],
                      constant uint *p      [[buffer(3)]],
                      uint lid [[thread_position_in_threadgroup]])
{
    (void)unused;
    uint n = p[0];
    if (n == 0u) return;

    threadgroup float red[SG_TG];

    /* Pass 1: max, plus a NaN count folded through the same tree. The
     * degenerate branches mirror sg_ref_softmax so that a masked or
     * saturated row behaves the same on both sides; ref decides "is there a
     * NaN" from a serial scan that only propagates a leading NaN, this
     * propagates any NaN, and the divergence is documented in the task
     * report (no reachable caller feeds NaN). */
    float m = -INFINITY;
    float nan_count = 0.0f;
    for (uint i = lid; i < n; i += SG_TG) {
        float v = x[i];
        if (v != v) nan_count += 1.0f;
        m = (v > m) ? v : m;
    }
    float nans = tg_sum(red, lid, nan_count);
    m = tg_max(red, lid, m);

    if (nans > 0.0f) {
        for (uint i = lid; i < n; i += SG_TG) out[i] = NAN;
        return;
    }
    if (m == -INFINITY) {                 /* every entry masked out */
        float u = 1.0f / (float)n;
        for (uint i = lid; i < n; i += SG_TG) out[i] = u;
        return;
    }
    if (m == INFINITY) {                  /* mass splits over the +inf entries */
        float hit = 0.0f;
        for (uint i = lid; i < n; i += SG_TG) if (x[i] == m) hit += 1.0f;
        float hits = tg_sum(red, lid, hit);
        float u = 1.0f / hits;
        for (uint i = lid; i < n; i += SG_TG) out[i] = (x[i] == m) ? u : 0.0f;
        return;
    }

    float s = 0.0f;
    for (uint i = lid; i < n; i += SG_TG) {
        float e = precise::exp(x[i] - m);
        out[i] = e;                       /* each thread reads back only its own writes */
        s += e;
    }
    float sum = tg_sum(red, lid, s);
    for (uint i = lid; i < n; i += SG_TG) out[i] = out[i] / sum;
}

/* =====================================================================
 * Elementwise
 * ===================================================================== */

kernel void k_swiglu(device const float *gate [[buffer(0)]],
                     device const float *up   [[buffer(1)]],
                     device float *out        [[buffer(2)]],
                     constant uint *p         [[buffer(3)]],
                     uint i [[thread_position_in_grid]])
{
    if (i >= p[0]) return;
    out[i] = sg_silu(gate[i]) * up[i];
}

kernel void k_silu(device const float *x [[buffer(0)]],
                   device const float *unused [[buffer(1)]],
                   device float *out     [[buffer(2)]],
                   constant uint *p      [[buffer(3)]],
                   uint i [[thread_position_in_grid]])
{
    (void)unused;
    if (i >= p[0]) return;
    out[i] = sg_silu(x[i]);
}

/* The attention output gate: sigmoid of the second half of q_proj, applied
 * to the attention context before o_proj. */
kernel void k_gate_sigmoid(device const float *x    [[buffer(0)]],
                           device const float *gate [[buffer(1)]],
                           device float *out        [[buffer(2)]],
                           constant uint *p         [[buffer(3)]],
                           uint i [[thread_position_in_grid]])
{
    if (i >= p[0]) return;
    out[i] = x[i] * sg_sigmoid(gate[i]);
}

/* =====================================================================
 * Attention decode: one threadgroup per QUERY head
 * ===================================================================== */

kernel void k_attn_decode(device const float *q  [[buffer(0)]],
                          device const float *kv [[buffer(1)]],
                          device float *out      [[buffer(2)]],
                          constant uint *p       [[buffer(3)]],
                          device float *scores   [[buffer(4)]],
                          uint h   [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]])
{
    uint n_heads = p[0], n_kv = p[1], hd = p[2], seq = p[3];
    uint q_stride = p[4], v_off = p[5];
    float scale = as_type<float>(p[6]);
    /* h is uniform across the threadgroup, so this return is uniform and
     * cannot strand a thread on a barrier below. */
    if (h >= n_heads || n_kv == 0u || hd == 0u || seq == 0u) return;

    uint repeat = n_heads / n_kv;         /* GQA: mlx repeats the kv-head axis */
    uint hk = repeat ? (h / repeat) : 0u;
    device const float *qh = q + (size_t)h * q_stride;
    device float *sc = scores + (size_t)h * seq;   /* this head's private slice */
    threadgroup float red[SG_TG];

    /* 1. scores. Thread `lid` owns keys lid, lid+256, ...; the dot product
     *    itself is a serial loop over head_dim, so no reduction is needed
     *    here at all. */
    for (uint t = lid; t < seq; t += SG_TG) {
        device const float *kt = kv + ((size_t)t * n_kv + hk) * hd;
        float dot = 0.0f;
        for (uint i = 0; i < hd; i++) dot += qh[i] * kt[i];
        sc[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    /* 2. softmax over 0..seq-1. The causal mask is implicit: the host only
     *    ever passes the keys at positions <= pos, so unlike k_softmax this
     *    one does NOT carry the all -inf / +inf / NaN branches -- there is no
     *    mask value to produce them, and every score here is a finite dot
     *    product of finite activations. If a masked variant is ever added,
     *    those branches come with it. */
    float m = -INFINITY;
    for (uint t = lid; t < seq; t += SG_TG) { float v = sc[t]; m = (v > m) ? v : m; }
    m = tg_max(red, lid, m);

    float s = 0.0f;
    for (uint t = lid; t < seq; t += SG_TG) { float e = precise::exp(sc[t] - m); sc[t] = e; s += e; }
    float sum = tg_sum(red, lid, s);
    /* Divide rather than multiply by a reciprocal, matching sg_ref_softmax. */
    for (uint t = lid; t < seq; t += SG_TG) sc[t] = sc[t] / sum;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    /* 3. context = sum_t p[t] * v[t]. Thread `lid` owns OUTPUT dims lid,
     *    lid+256, ... and walks t in increasing order, so again there is no
     *    cross-thread reduction and the summation order is fixed. */
    device float *oh = out + (size_t)h * hd;
    for (uint i = lid; i < hd; i += SG_TG) {
        float acc = 0.0f;
        for (uint t = 0; t < seq; t++) {
            acc += sc[t] * kv[v_off + ((size_t)t * n_kv + hk) * hd + i];
        }
        oh[i] = acc;
    }
}

/* =====================================================================
 * Gated DeltaNet decode step
 * ===================================================================== */

/* Per-channel causal depthwise conv with carried state, one thread per
 * channel. out[0 .. channels) is this token's conv output; out[channels ..)
 * is the carried state, read and rewritten in place (the ref op allows
 * state_in == state_out for exactly this reason). Thread c touches only
 * column c of the state, so the shift needs no synchronization. */
kernel void k_conv1d_step(device const float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device float *out     [[buffer(2)]],
                          constant uint *p      [[buffer(3)]],
                          uint c [[thread_position_in_grid]])
{
    uint channels = p[0], ksize = p[1];
    if (c >= channels || ksize == 0u) return;
    uint keep = ksize - 1u;

    device const float *wc = w + (size_t)c * ksize;
    device float *state = out + channels;
    float tok = x[c];

    /* Taps 0..keep-1 read the carried state oldest-first; tap `keep` is the
     * incoming token. */
    float acc = 0.0f;
    for (uint j = 0; j < keep; j++) acc += wc[j] * state[(size_t)j * channels + c];
    acc += wc[keep] * tok;

    /* Shift the tail one step older and append this token. Reads row j+1
     * before writing row j with j increasing, so nothing is clobbered early. */
    for (uint j = 0; j + 1u < keep; j++) {
        state[(size_t)j * channels + c] = state[(size_t)(j + 1u) * channels + c];
    }
    if (keep > 0u) state[(size_t)(keep - 1u) * channels + c] = tok;

    out[c] = acc;
}

/* One token of the gated delta rule for ONE value head, in place on S.
 * Thread `lid` owns rows lid, lid+256, ... of S, and a row is a closed
 * computation (decay, read, delta, write, readout), so there is no shared
 * state between threads and no reduction. The statement order is exactly
 * sg_ref_delta_step's, which is exactly mlx's _gated_delta_step_ops:
 * decay -> read from the DECAYED state -> delta = (v - kv) * beta -> write
 * -> read out from the POST-WRITE state. */
kernel void k_delta_step(device float *S         [[buffer(0)]],
                         device const float *qkv [[buffer(1)]],
                         device float *out       [[buffer(2)]],
                         constant uint *p        [[buffer(3)]],
                         uint lid [[thread_position_in_threadgroup]])
{
    uint dk = p[0], dv = p[1];
    float beta = as_type<float>(p[2]);
    float decay = as_type<float>(p[3]);
    if (dk == 0u || dv == 0u) return;

    device const float *q = qkv;
    device const float *k = qkv + dk;
    device const float *v = qkv + 2u * dk;

    for (uint j = lid; j < dv; j += SG_TG) {
        device float *row = S + (size_t)j * dk;

        float kv = 0.0f;
        for (uint i = 0; i < dk; i++) {
            float s = row[i] * decay;
            row[i] = s;
            kv += s * k[i];
        }
        float delta = (v[j] - kv) * beta;
        float y = 0.0f;
        for (uint i = 0; i < dk; i++) {
            float s = row[i] + k[i] * delta;
            row[i] = s;
            y += s * q[i];
        }
        out[j] = y;
    }
}
