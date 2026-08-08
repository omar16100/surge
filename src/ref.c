/* ref.c - scalar CPU reference implementations of every op the qwen3_5
 * hybrid forward pass needs. This file is the project's correctness
 * bedrock: the Metal kernels (Task 9) are validated op-by-op against it,
 * and it in turn is validated against fixtures produced by *running the
 * real mlx-lm implementation* (see tools/make_fixtures.py's
 * write_op_fixtures / write_hybrid_op_fixtures and tests/test_ref_ops.c).
 * Nothing here is written from a prose description of an op; every formula
 * below was confirmed against the mlx source and then pinned numerically.
 *
 * Ground truth read while writing this:
 *   .../mlx_lm/models/qwen3_5.py       GatedDeltaNet, DecoderLayer
 *   .../mlx_lm/models/qwen3_next.py    Qwen3NextAttention, Qwen3NextRMSNormGated
 *   .../mlx_lm/models/gated_delta.py   gated_delta_update / _gated_delta_step_ops
 *   .../mlx_lm/models/rope_utils.py    initialize_rope (qwen3_5 -> nn.RoPE)
 *
 * Conventions confirmed against mlx (each pinned by a fixture record):
 *
 * 1. RMSNorm is mx.fast.rms_norm: x * rsqrt(mean(x^2) + eps) * w, with the
 *    mean over the normalized axis only, and w optional. qwen3_5's DeltaNet
 *    normalizes q and k with weight=None and a hardcoded eps of 1e-6, not
 *    the config's rms_norm_eps.
 *
 * 2. RoPE for qwen3_5 resolves (rope_parameters type "default") to plain
 *    nn.RoPE(int(head_dim*partial_rotary_factor), traditional=False,
 *    base=rope_theta). traditional=False means the HALF-SPLIT pairing
 *    (element i pairs with i + rope_dim/2), not the interleaved
 *    even/odd pairing; and because the rotated width is smaller than
 *    head_dim, everything at [rope_dim, head_dim) passes through
 *    unrotated. Both checkpoints use partial_rotary_factor 0.25 over
 *    head_dim 256, i.e. rope_dim 64 (matching the GGUF's explicit
 *    qwen35.rope.dimension_count = 64).
 *
 * 3. The attention output gate is a SIGMOID (not silu, not a softmax):
 *    Qwen3NextAttention splits its double-width q_proj output per head into
 *    [queries | gate] and returns o_proj(attn_out * sigmoid(gate)). The
 *    split is per head over the reshaped [B, L, n_heads, 2*head_dim], so
 *    the gate for head h is contiguous right after head h's queries, NOT
 *    a single n_heads*head_dim block after all the queries.
 *
 * 4. The DeltaNet conv is a depthwise (groups == channels) cross-correlation
 *    with padding=0 over concat(carried state, new tokens); no kernel flip.
 *    So out[c] = sum_j w[c][j] * buf[j][c] with buf holding the last ksize
 *    tokens oldest-first: w[c][ksize-1] multiplies the newest token.
 *
 * 5. The delta rule: state decays first, then the readout uses the state
 *    AFTER the k/v write, not before (see sg_ref_delta_step). q and k are
 *    NOT L2-normalized: qwen3_5.py RMS-normalizes them (weight=None, eps
 *    1e-6) and then applies fixed scalars, q by 1/head_k_dim and k by
 *    1/sqrt(head_k_dim). That is the code, and the fixtures pin it.
 *
 * 6. The DeltaNet output norm is RMSNormGated: silu(z) * rms_norm(y, w, eps)
 *    -- the gate is applied to the *unnormalized* z with silu and multiplied
 *    against the normalized y (mlx's _precise_swiglu(h=y, gate=z, x=norm(y))).
 *
 * Precision policy: every reduction accumulates in double, and the
 * transcendentals are the double-precision libm ones. These ops define
 * "correct" for this project, so they buy accuracy with speed everywhere
 * it is a choice.
 */
#include "surge.h"

#include <math.h>
#include <string.h>

/* bf16 -> f32: the bf16 bit pattern is the top 16 bits of the f32 one.
 * Done through memcpy rather than a pointer cast to stay strict-aliasing
 * clean under -O2. */
static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* IEEE binary16 -> f32, for the Q8_0 block scale. Handles zero, subnormals,
 * and inf/NaN; Q8_0 scales are always finite and non-negative in practice,
 * but a corrupt file must not produce a garbage finite value. */
static float f16_to_f32(uint16_t h) {
    /* Named exp_bits, not exp: a local called `exp` shadows libm's exp() in
     * a file that calls it, which is a -Wshadow error on some builds. */
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp_bits = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;

    if (exp_bits == 0) {
        if (mant == 0) {
            bits = sign; /* +-0 */
        } else {
            /* Subnormal: renormalize into an f32 normal. */
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

/* ---------------------------------------------------------------------
 * Elementwise / normalization
 * --------------------------------------------------------------------- */

void sg_ref_rmsnorm(float *x, const float *w, uint32_t n, float eps) {
    if (!x || n == 0) return;

    double sumsq = 0.0;
    for (uint32_t i = 0; i < n; i++) sumsq += (double)x[i] * (double)x[i];

    double scale = 1.0 / sqrt(sumsq / (double)n + (double)eps);
    if (w) {
        for (uint32_t i = 0; i < n; i++) x[i] = (float)((double)x[i] * scale * (double)w[i]);
    } else {
        for (uint32_t i = 0; i < n; i++) x[i] = (float)((double)x[i] * scale);
    }
}

float sg_ref_sigmoid(float x) {
    /* Split on the sign so exp() never overflows: for x >= 0 use
     * 1/(1+e^-x), for x < 0 use e^x/(1+e^x). */
    double d = (double)x;
    if (d >= 0.0) return (float)(1.0 / (1.0 + exp(-d)));
    double e = exp(d);
    return (float)(e / (1.0 + e));
}

float sg_ref_softplus(float x) {
    /* mlx's nn.softplus is logaddexp(x, 0) == max(x,0) + log1p(e^-|x|).
     * That form is exact at both tails: it returns x for large positive x
     * (no overflow) and e^x for large negative x (no cancellation). */
    double d = (double)x;
    return (float)(fmax(d, 0.0) + log1p(exp(-fabs(d))));
}

float sg_ref_delta_decay(float a_log, float a, float dt_bias) {
    /* gated_delta.compute_g: exp(-exp(A_log) * softplus(a + dt_bias)).
     * A_log is kept in f32 by mlx's cast_predicate even in a quantized
     * checkpoint, so there is no precision subtlety to reproduce here. */
    double g = exp((double)a_log) * (double)sg_ref_softplus(a + dt_bias);
    return (float)exp(-g);
}

void sg_ref_silu(float *x, uint32_t n) {
    if (!x) return;
    for (uint32_t i = 0; i < n; i++) x[i] = (float)((double)x[i] * (double)sg_ref_sigmoid(x[i]));
}

void sg_ref_swiglu(float *gate, const float *up, uint32_t n) {
    if (!gate || !up) return;
    for (uint32_t i = 0; i < n; i++) {
        double s = (double)gate[i] * (double)sg_ref_sigmoid(gate[i]);
        gate[i] = (float)(s * (double)up[i]);
    }
}

void sg_ref_gate_sigmoid(float *x, const float *gate, uint32_t n) {
    if (!x || !gate) return;
    for (uint32_t i = 0; i < n; i++) x[i] = (float)((double)x[i] * (double)sg_ref_sigmoid(gate[i]));
}

void sg_ref_softmax(float *x, uint32_t n) {
    if (!x || n == 0) return;

    /* Subtract the max before exp: attention scores carry large-negative
     * mask values (-1e30 and below), and exp() of the raw score would
     * otherwise underflow to zero for every element and divide by zero. */
    float m = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > m) m = x[i];

    /* A non-finite max would make every (x[i] - m) a NaN: -inf - -inf and
     * +inf - +inf are both NaN. mlx masks with finfo(dtype).min rather than
     * -inf so it never hits this, but a caller here might, and returning a
     * row of NaNs that quietly poisons the whole forward pass is the worst
     * possible answer. Take the limit instead: an all -inf row is uniform,
     * and a row containing +inf puts all the mass on the +inf entries. */
    if (isnan(m)) {
        /* NaN somewhere in the input. There is no defensible distribution to
         * invent, so propagate rather than manufacture one. */
        for (uint32_t i = 0; i < n; i++) x[i] = m;
        return;
    }
    if (m == -INFINITY) { /* every entry is -inf */
        float u = 1.0f / (float)n;
        for (uint32_t i = 0; i < n; i++) x[i] = u;
        return;
    }
    if (m == INFINITY) {
        uint32_t hits = 0;
        for (uint32_t i = 0; i < n; i++) if (x[i] == m) hits++;
        float u = 1.0f / (float)hits; /* hits >= 1: m came from the array */
        for (uint32_t i = 0; i < n; i++) x[i] = (x[i] == m) ? u : 0.0f;
        return;
    }

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double e = exp((double)x[i] - (double)m);
        x[i] = (float)e;
        sum += e;
    }
    /* sum >= 1 always (the max element contributes exactly 1), so this
     * division is safe even when every other element underflowed. */
    for (uint32_t i = 0; i < n; i++) x[i] = (float)((double)x[i] / sum);
}

/* ---------------------------------------------------------------------
 * RoPE
 * --------------------------------------------------------------------- */

void sg_ref_rope_partial(float *q, uint32_t head_dim, uint32_t rope_dim,
                         uint32_t pos, float theta) {
    /* mlx rejects an odd rotated width outright ("[rope] dims must be even
     * but got 7"), so an odd rope_dim here is a caller bug, not a shape to
     * improvise a half-pair for. Refuse it rather than silently rotating
     * rope_dim-1 dims and leaving a stray one behind. */
    if (!q || rope_dim < 2 || rope_dim > head_dim || rope_dim % 2 != 0) return;

    uint32_t half = rope_dim / 2;
    for (uint32_t i = 0; i < half; i++) {
        /* inv_freq_i = theta^(-2i/rope_dim); angle = pos * inv_freq_i.
         *
         * Computed in double, which is a DELIBERATE and MEASURED divergence
         * from mlx at long context, not an accident. mlx rounds the angle to
         * f32 before the sine and cosine; at the real checkpoint's rope
         * parameters (head_dim 256, rope_dim 64, theta 1e7) the low-index
         * angles reach `pos` radians, so that rounding costs about
         * pos * 2^-24 radians and shows up directly in the output:
         *
         *   pos      4096   |this - mlx| = 1.5e-4
         *   pos     32768   |this - mlx| = 9.5e-4
         *   pos    262143   |this - mlx| = 8.1e-3   (max_position_embeddings)
         *
         * The error is mlx's, not this function's: mx.cos of the f32-rounded
         * angle agrees with the double cosine of that same rounded angle to
         * 6.6e-8, so mlx's transcendentals are fine and the whole gap is the
         * angle rounding. Matching it bit-for-bit is not portably reachable
         * (the exact inv_freq expression inside mx.fast.rope does not
         * reproduce from any of the obvious f32 formulations), and it would
         * mean deliberately importing someone else's rounding error into the
         * file that defines "correct" for this project. So: stay exact, and
         * pin the gap. tests/test_ref_ops.c's rope_real_* checks assert this
         * function against a float64 reference tightly AND assert the
         * measured mlx gap against the recorded per-position values, so the
         * number above cannot drift unnoticed. Task 8/9 note: this is a floor
         * on any surge-vs-mlx comparison at long context, independent of
         * anything else either implementation does. */
        double inv_freq = pow((double)theta, -2.0 * (double)i / (double)rope_dim);
        double ang = (double)pos * inv_freq;
        double c = cos(ang), s = sin(ang);
        double lo = (double)q[i], hi = (double)q[i + half];
        q[i] = (float)(lo * c - hi * s);
        q[i + half] = (float)(lo * s + hi * c);
    }
    /* q[rope_dim .. head_dim) is left exactly as it came in. */
}

void sg_ref_rope(float *q, uint32_t head_dim, uint32_t pos, float theta) {
    sg_ref_rope_partial(q, head_dim, head_dim, pos, theta);
}

/* ---------------------------------------------------------------------
 * Matrix-vector products, y = W x with W row-major [rows, cols]
 * --------------------------------------------------------------------- */

void sg_ref_matvec_f32(const float *w, const float *x, float *y,
                       uint32_t rows, uint32_t cols) {
    if (!w || !x || !y) return;
    for (uint32_t r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * cols;
        double acc = 0.0;
        for (uint32_t c = 0; c < cols; c++) acc += (double)row[c] * (double)x[c];
        y[r] = (float)acc;
    }
}

void sg_ref_matvec_bf16(const uint16_t *w, const float *x, float *y,
                        uint32_t rows, uint32_t cols) {
    if (!w || !x || !y) return;
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t *row = w + (size_t)r * cols;
        double acc = 0.0;
        for (uint32_t c = 0; c < cols; c++) acc += (double)bf16_to_f32(row[c]) * (double)x[c];
        y[r] = (float)acc;
    }
}

void sg_ref_matvec_q8(const void *w, const float *x, float *y,
                      uint32_t rows, uint32_t cols) {
    if (!w || !x || !y || cols % 32 != 0) return;

    const uint8_t *base = (const uint8_t *)w;
    uint32_t blocks_per_row = cols / 32;
    size_t row_bytes = (size_t)blocks_per_row * 34;

    for (uint32_t r = 0; r < rows; r++) {
        const uint8_t *row = base + (size_t)r * row_bytes;
        double acc = 0.0;
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            const uint8_t *blk = row + (size_t)b * 34;
            /* The f16 scale is stored little-endian and is NOT guaranteed
             * to be 2-byte aligned inside the tensor, so read it bytewise. */
            uint16_t sbits = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            double d = (double)f16_to_f32(sbits);
            const int8_t *q = (const int8_t *)(blk + 2);
            const float *xb = x + (size_t)b * 32;
            double blk_acc = 0.0;
            for (uint32_t i = 0; i < 32; i++) blk_acc += (double)q[i] * (double)xb[i];
            acc += d * blk_acc;
        }
        y[r] = (float)acc;
    }
}

/* ---------------------------------------------------------------------
 * Gated DeltaNet primitives
 * --------------------------------------------------------------------- */

void sg_ref_conv1d_causal(float *x, const float *w, const float *state_in,
                          float *state_out, uint32_t channels, uint32_t ksize) {
    if (!x || !w || ksize == 0 || channels == 0) return;

    uint32_t keep = ksize - 1; /* rows carried between tokens */

    /* One channel at a time, doing that channel's read, state shift and
     * in-place write before moving on. That ordering is what makes
     * state_in == state_out legal: within a channel the shift reads row
     * j+1 and writes row j with j increasing, so it never clobbers a value
     * it still has to read, and the incoming token is saved before x[c] is
     * overwritten with the output. */
    for (uint32_t c = 0; c < channels; c++) {
        const float *wc = w + (size_t)c * ksize;
        float tok = x[c];

        /* Taps 0..keep-1 read the carried state oldest-first; tap keep is
         * the token being processed. A NULL state_in means an all-zero
         * state (the start of a sequence). */
        double acc = 0.0;
        if (state_in) {
            for (uint32_t j = 0; j < keep; j++) {
                acc += (double)wc[j] * (double)state_in[(size_t)j * channels + c];
            }
        }
        acc += (double)wc[keep] * (double)tok;

        if (state_out) {
            for (uint32_t j = 0; j + 1 < keep; j++) {
                state_out[(size_t)j * channels + c] =
                    state_in ? state_in[(size_t)(j + 1) * channels + c] : 0.0f;
            }
            if (keep > 0) state_out[(size_t)(keep - 1) * channels + c] = tok;
        }

        x[c] = (float)acc;
    }
}

void sg_ref_delta_step(float *S, const float *q, const float *k, const float *v,
                       float beta, float decay, float *out,
                       uint32_t dk, uint32_t dv) {
    if (!S || !q || !k || !v || !out) return;

    for (uint32_t j = 0; j < dv; j++) {
        float *row = S + (size_t)j * dk;

        /* 1. decay, and read the memory the decayed state already holds for
         *    this key: kv[j] = sum_i (decay * S[j][i]) * k[i]. */
        double kv = 0.0;
        for (uint32_t i = 0; i < dk; i++) {
            double s = (double)row[i] * (double)decay;
            row[i] = (float)s;
            kv += s * (double)k[i];
        }

        /* 2. delta = (v[j] - kv[j]) * beta, then write it into the state
         *    along the key direction. */
        double delta = ((double)v[j] - kv) * (double)beta;

        /* 3. readout uses the state AFTER the write (mlx computes y from the
         *    updated state, not the decayed-only one). */
        double y = 0.0;
        for (uint32_t i = 0; i < dk; i++) {
            double s = (double)row[i] + (double)k[i] * delta;
            row[i] = (float)s;
            y += s * (double)q[i];
        }
        out[j] = (float)y;
    }
}
