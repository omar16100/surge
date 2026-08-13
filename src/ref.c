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
#include <stdio.h>
#include <stdlib.h>
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
     * This is the safetensors/mlx form, where the stored tensor really is
     * A_log. */
    double g = exp((double)a_log) * (double)sg_ref_softplus(a + dt_bias);
    return (float)exp(-g);
}

float sg_ref_delta_decay_neg_a(float neg_a, float a, float dt_bias) {
    /* The GGUF form. convert_hf_to_gguf's Qwen3Next path applies
     * -torch.exp() to A_log while writing blk.N.ssm_a, and llama.cpp then
     * multiplies that value straight into softplus with no exp and no
     * negation. So the exponentiation has already happened on disk and doing
     * it again here would compute exp(-exp(-exp(A_log)*sp)), which stays in
     * (0,1) and therefore fails silently rather than loudly.
     *
     * Verified elementwise rather than inferred: across every
     * linear-attention layer of Qwen3.6-27B, GGUF blk.L.ssm_a equals
     * -exp(A_log) of the matching HF checkpoint to 1.8e-6 once the value
     * heads are un-tiled, and all 2304 values are strictly negative (A_log
     * itself is mixed-sign in both HF checkpoints).
     *
     * neg_a is negative, so the product is negative and exp() of it lands in
     * (0,1] as a decay must. */
    double g = (double)neg_a * (double)sg_ref_softplus(a + dt_bias);
    return (float)exp(g);
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
 * Split-K attention combine (Task P2.0)
 * --------------------------------------------------------------------- */

/* The per-partition rescale weight exp(mi - M), with ONE guard: when mi and M
 * are the identical value (a tie for the max, checked by == not by
 * subtracting), the weight is exactly 1.0 by definition, computed without
 * subtracting at all. This is a no-op for an ordinary finite tie (ordinary
 * IEEE-754 subtraction already gives exp(0.0) == 1.0 exactly there), but it
 * is load-bearing for M == +INFINITY: mi == M == +INFINITY makes mi - M the
 * indeterminate form INF - INF (NaN), even though the two values are
 * genuinely equal and the weight they represent is unambiguously 1.0. Every
 * other combination mi can take relative to a finite-or-+INFINITY M (mi <
 * M, or mi == -INFINITY, i.e. an empty partition) already evaluates
 * correctly through the subtraction (mi - M is -INFINITY, and
 * exp(-INFINITY) == 0.0, no special case needed), including
 * -INFINITY - (+INFINITY), which IEEE-754 defines as -INFINITY, not NaN
 * (the indeterminate forms are only same-signed-infinity minus itself). A
 * NaN mi (not caught here, since NaN == anything is always false) still
 * correctly falls through to exp(NaN - M) == NaN, which is exactly the
 * signal sg_ref_attn_combine below propagates rather than launders. */
static double attn_combine_weight(double mi, double M) {
    if (mi == M) return 1.0;
    return exp(mi - M);
}

/* m/s/acc partial results into out. See surge.h for the full contract
 * (the math, the determinism rule, and every documented degenerate-input
 * case: n_parts == 0, all-empty, NaN, +/-INFINITY, and the NULL/contract-
 * violation convention). */
void sg_ref_attn_combine(const float *m, const float *s, const float *acc,
                         uint32_t n_parts, uint32_t head_dim, float *out) {
    if (!out || head_dim == 0) return;
    /* NULL m/s/acc with n_parts > 0 is a caller contract violation, not the
     * documented all-empty case: leave out untouched, matching the matvec
     * functions' NULL convention above. */
    if (n_parts > 0 && (!m || !s || !acc)) return;

    /* n_parts == 0: no partitions at all. m/s/acc are never indexed below,
     * so they may legitimately be NULL for this call. Same "attention over
     * zero keys" convention as the all-empty branch further down: write a
     * defined zero rather than leaving out whatever the caller's buffer
     * previously held. */
    if (n_parts == 0) {
        for (uint32_t d = 0; d < head_dim; d++) out[d] = 0.0f;
        return;
    }

    /* Pass 1: M = max_i m[i]. Strictly increasing i (determinism rule, see
     * surge.h / src/kernels.metal:7-27) -- not that max-of-a-sequence can
     * differ by iteration order anyway, but every pass here follows the same
     * fixed order on principle, since a future Metal port copies this shape
     * verbatim. A NaN m[i] at i > 0 is never picked up here (NaN > M is
     * always false), matching sg_ref_softmax's own max-scan above; that is
     * handled below by the isnan(S) check instead, not here. */
    double M = (double)m[0];
    for (uint32_t i = 1; i < n_parts; i++) {
        if ((double)m[i] > M) M = (double)m[i];
    }

    /* m[0] itself NaN (the only way M can end up NaN, by the same reasoning
     * sg_ref_softmax's isnan(m) check above relies on): mirror softmax's
     * choice exactly -- "there is no defensible distribution to invent, so
     * propagate rather than manufacture one." Every output dimension gets
     * the same NaN value M carries, not a silently-manufactured 0.0. */
    if (isnan(M)) {
        for (uint32_t d = 0; d < head_dim; d++) out[d] = (float)M;
        return;
    }

    /* All partitions empty (every m[i] == -INFINITY): M is -INFINITY here,
     * and M - M below would be -INFINITY - -INFINITY = NaN. Caught up front
     * and defined instead: out[d] = 0.0 for every d, the same documented
     * convention as n_parts == 0 above. */
    if (M == -INFINITY) {
        for (uint32_t d = 0; d < head_dim; d++) out[d] = 0.0f;
        return;
    }

    /* M == +INFINITY (at least one partition reports +INFINITY as its own
     * max) needs no branch of its own: attn_combine_weight gives every
     * partition tied at that +INFINITY weight exactly 1.0 and every other
     * partition weight exactly 0.0 (see its comment), which is precisely
     * "the partition(s) achieving the max dominate completely" -- the same
     * log-sum-exp limit sg_ref_softmax's own m == +INFINITY branch computes
     * by counting hits, just expressed through the general formula instead
     * of a separate hit-count loop. A single +INFINITY partition reduces
     * exactly to that partition alone (out[d] = acc_k[d]/s_k, an implicit
     * K == 1); several tied +INFINITY partitions combine among only
     * themselves via their own (s_i, acc_i), exactly as if the others did
     * not exist.
     *
     * Pass 2: S = sum_i s[i] * attn_combine_weight(m[i], M), strictly
     * increasing i. An empty partition has m[i] == -INFINITY here (M is
     * finite or +INFINITY because of the checks above), so its weight is
     * 0.0 and its s[i] (0.0 by contract) times that is 0.0, no special case
     * needed. */
    double S = 0.0;
    for (uint32_t i = 0; i < n_parts; i++) {
        S += (double)s[i] * attn_combine_weight((double)m[i], M);
    }

    /* A NaN m[i] at i > 0 was not caught by isnan(M) above (see pass 1's
     * comment); it poisons this sum instead, exactly like sg_ref_softmax's
     * own sum gets poisoned by a stray NaN score. Catch it HERE rather than
     * only guarding the final division: S > 0.0 is false for a NaN S (every
     * comparison with NaN is false), so without this check the division
     * guard below would silently turn that NaN into a manufactured 0.0 --
     * the exact failure mode sg_ref_softmax's own header comment calls out
     * as "the worst possible answer." Propagate instead. */
    if (isnan(S)) {
        for (uint32_t d = 0; d < head_dim; d++) out[d] = (float)S;
        return;
    }

    /* Pass 3: out[d] = ( sum_i acc[i][d] * attn_combine_weight(m[i], M) ) /
     * S, strictly increasing i for every d. attn_combine_weight(m[i], M)
     * does not depend on d and is recomputed here rather than cached in a
     * scratch array: this file's policy is accuracy and obvious correctness
     * over speed (the Metal kernel, a later task, is where performance is
     * actually earned), and recomputing a pure function of already-fixed
     * inputs changes no bit of the result.
     *
     * S > 0.0 whenever this line is reached under the documented input
     * contract: the partition achieving M is non-empty (M came from a real
     * m[i]), and a non-empty partition's s[i] sums at least one
     * exp(score - m[i]) term that equals exactly 1.0 for the key achieving
     * that partition's own max, so s[i] >= 1.0 and S >= 1.0 * exp(0) = 1.0.
     * The guard is defensive only, for a caller that violates that contract
     * (e.g. a "non-empty" m[i] paired with s[i] == 0); S == NaN is ruled out
     * above, so this comparison is never silently swallowing one. */
    for (uint32_t d = 0; d < head_dim; d++) {
        double num = 0.0;
        for (uint32_t i = 0; i < n_parts; i++) {
            num += (double)acc[(size_t)i * head_dim + d] * attn_combine_weight((double)m[i], M);
        }
        out[d] = (S > 0.0) ? (float)(num / S) : 0.0f;
    }
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

/* =====================================================================
 * The reference forward pass
 * =====================================================================
 *
 * Structure, one token at position `pos`, mirroring qwen3_5.py's
 * Qwen3_5TextModel.__call__ / DecoderLayer.__call__ statement for statement:
 *
 *   x = embed_tokens[token]                       (NO scaling: qwen3_5 has none)
 *   for each layer L:
 *       r = (L is full attention ? attention : gated_deltanet)(rms_norm(x, ln1))
 *       x = x + r
 *       x = x + down_proj(silu(gate_proj(rms_norm(x, ln2))) * up_proj(...))
 *   logits = lm_head(rms_norm(x, out_norm))       (lm_head == embed_tokens when tied)
 *
 * A layer is a full-attention layer iff (L + 1) % full_attention_interval
 * == 0 (qwen3_5.py's DecoderLayer.is_linear, negated). sg_ref_state_new
 * checks that rule against which tensor group the loader actually found and
 * refuses a model where the two disagree, so the dispatch below can read
 * either one.
 *
 * Everything above the ops is f32. Weights are widened from bf16 or
 * dequantized from Q8_0 at the point of use for the big matmuls (so the
 * model is never materialized in f32), and ONCE into owned f32 buffers for
 * the small per-layer tensors -- the norms, the conv kernel, dt_bias and
 * ssm_a. Those owned copies are also where sg_model.norms_are_residual is
 * applied: the +1.0 mlx's sanitize adds to ln1/ln2/q_norm/k_norm/out_norm
 * (and never to ssm_norm) cannot be done in place on a read-only mmap.
 *
 * NOT modelled, deliberately: batching, left padding, and therefore the
 * DeltaNet mask path (create_ssm_mask returns None for the single-sequence
 * ArraysCache this project targets, and create_attention_mask's causal mask
 * is exactly "attend to positions 0..pos", which is what the loop below
 * does by construction).
 */

/* ---------------------------------------------------------------------
 * Typed weight access
 * --------------------------------------------------------------------- */

static uint64_t q8_row_bytes(uint32_t cols) { return (uint64_t)(cols / 32) * 34; }

/* y = W x for a weight of any of the three dtypes the two checkpoint
 * families use. sg_ref_state_new has already rejected anything else, and
 * (for Q8_0) any cols that is not a multiple of 32. */
static void wmatvec(const void *w, sg_tensor_type t, const float *x, float *y,
                    uint32_t rows, uint32_t cols) {
    switch (t) {
    case SG_T_BF16: sg_ref_matvec_bf16((const uint16_t *)w, x, y, rows, cols); break;
    case SG_T_Q8_0: sg_ref_matvec_q8(w, x, y, rows, cols); break;
    default:        sg_ref_matvec_f32((const float *)w, x, y, rows, cols); break;
    }
}

/* Widen n elements of a small (non-matmul) tensor into an owned f32 buffer.
 *
 * `shift` is the +1.0 that a residual RMSNorm weight needs, and WHERE IT IS
 * APPLIED IS A DELIBERATE DIVERGENCE FROM mlx-lm. mlx runs
 * TextModel.sanitize on the raw checkpoint arrays, so its `v + 1.0` happens
 * in bf16 and the sum is re-rounded; here the stored bf16 is widened to f32
 * FIRST and 1.0 is added in f32, which is what the model's own reference
 * implementation does:
 *
 *   transformers/models/qwen3_5/modeling_qwen3_5.py, Qwen3_5RMSNorm.forward
 *       output = output * (1.0 + self.weight.float())
 *
 * The difference is not academic: measured over 16 x 64 teacher-forced
 * positions of Qwen3.5-2B it moves mlx's own logits by up to 1.79e-01 and
 * flips one top-1 token. tools/tf_compare.py therefore measures surge
 * against BOTH mlx-lm as shipped and mlx-lm with this one rounding
 * corrected, and prints both. Same policy as sg_ref_rope_partial's angle
 * precision: ref.c defines correct, and the gap is pinned rather than
 * hidden. */
static void wwiden(const void *w, sg_tensor_type t, float *out, uint64_t n, float shift) {
    if (!w || !out) return;
    if (t == SG_T_BF16) {
        const uint16_t *b = (const uint16_t *)w;
        for (uint64_t i = 0; i < n; i++) out[i] = bf16_to_f32(b[i]) + shift;
    } else {
        const float *f = (const float *)w;
        for (uint64_t i = 0; i < n; i++) out[i] = f[i] + shift;
    }
}

/* One row of a [rows, cols] weight, widened/dequantized into out[cols].
 * Only used for the embedding lookup, where `row` is the token id. */
static void wrow(const void *w, sg_tensor_type t, uint64_t row, uint32_t cols, float *out) {
    if (!w || !out || cols == 0) return;
    /* Q8_0 rows are whole 32-element blocks. sg_ref_state_new already
     * rejects a model whose hidden size is not a multiple of 32, but a
     * partial write here would leave the tail of the embedding buffer
     * holding the PREVIOUS token's values, which no sanitizer can see and
     * which looks like a subtly wrong model rather than a bug. */
    if (t == SG_T_Q8_0 && cols % 32 != 0) {
        for (uint32_t i = 0; i < cols; i++) out[i] = 0.0f;
        return;
    }
    if (t == SG_T_BF16) {
        const uint16_t *b = (const uint16_t *)w + row * cols;
        for (uint32_t i = 0; i < cols; i++) out[i] = bf16_to_f32(b[i]);
        return;
    }
    if (t == SG_T_Q8_0) {
        const uint8_t *p = (const uint8_t *)w + row * q8_row_bytes(cols);
        for (uint32_t b = 0; b < cols / 32; b++) {
            const uint8_t *blk = p + (size_t)b * 34;
            uint16_t sbits = (uint16_t)((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            float d = f16_to_f32(sbits);
            const int8_t *q = (const int8_t *)(blk + 2);
            for (uint32_t i = 0; i < 32; i++) out[b * 32 + i] = d * (float)q[i];
        }
        return;
    }
    memcpy(out, (const float *)w + row * cols, (size_t)cols * sizeof *out);
}

/* ---------------------------------------------------------------------
 * State
 * --------------------------------------------------------------------- */

/* Per-layer union state. Exactly one of the two groups is populated,
 * decided by the same tensor-presence rule the loader used. */
typedef struct {
    bool is_attn;
    /* full attention: append-only f32 caches, [max_ctx, n_kv_heads, head_dim] */
    float *k_cache, *v_cache;
    /* gated DeltaNet: a fixed-size pair that does not grow with context */
    float *conv_state;   /* [conv_kernel-1, conv_dim], oldest token first */
    float *ssm_state;    /* [n_v_heads, head_v_dim, head_k_dim] */
    /* owned f32 copies of the small tensors (norm shift already applied) */
    float *ln1, *ln2;    /* [hidden] */
    float *q_norm_w;     /* [head_dim] */
    float *k_norm_w;     /* [head_dim] */
    float *ssm_norm_w;   /* [head_v_dim] */
    float *conv_w;       /* [conv_dim, conv_kernel] */
    float *dt_bias;      /* [n_v_heads] */
    float *ssm_a;        /* [n_v_heads]; A_log or -exp(A_log) per m->ssm_a_form */
} sg_ref_layer;

struct sg_ref_state {
    /* The model this state was built for. sg_ref_forward takes the model
     * again (the state deliberately does not own it), and every buffer size
     * and every semantic flag below was fixed at sg_ref_state_new time from
     * THAT model, so being handed a different one later is a silent
     * buffer-overrun waiting to happen. Checked, not documented away. */
    const sg_model *owner;
    sg_cfg cfg;
    uint32_t max_ctx;
    uint32_t used;             /* number of positions already consumed */

    /* derived widths */
    uint32_t key_dim, value_dim, conv_dim;
    uint32_t q_width;          /* attn_width, doubled when cfg.attn_output_gate
                                 * (queries + folded output gate) */
    uint32_t kv_width;         /* n_kv_heads * head_dim */
    uint32_t attn_width;       /* n_heads * head_dim */

    /* copied from sg_model so the forward never re-derives semantics */
    sg_tensor_type mat_type;
    sg_ssm_a_form a_form;
    bool v_tiled;

    sg_ref_layer *ls;          /* cfg.n_layers */
    float *out_norm_w;         /* [hidden] */

    /* scratch */
    float *x, *h, *r;                       /* [hidden] */
    float *qg, *kbuf, *vbuf, *scores, *ctx, *gate;
    float *qkv, *zbuf, *abuf, *bbuf, *ybuf;
    float *ff_gate, *ff_up;
    double *dacc;                           /* [head_dim] */
    float *logits;                          /* [vocab] */
};

static void *xzalloc(size_t n, size_t sz, bool *ok) {
    if (!*ok) return NULL;
    if (n == 0) return NULL;
    void *p = calloc(n, sz);
    if (!p) *ok = false;
    return p;
}

void sg_ref_state_free(sg_ref_state *st) {
    if (!st) return;
    if (st->ls) {
        for (uint32_t i = 0; i < st->cfg.n_layers; i++) {
            sg_ref_layer *L = &st->ls[i];
            free(L->k_cache); free(L->v_cache);
            free(L->conv_state); free(L->ssm_state);
            free(L->ln1); free(L->ln2);
            free(L->q_norm_w); free(L->k_norm_w);
            free(L->ssm_norm_w); free(L->conv_w);
            free(L->dt_bias); free(L->ssm_a);
        }
        free(st->ls);
    }
    free(st->out_norm_w);
    free(st->x); free(st->h); free(st->r);
    free(st->qg); free(st->kbuf); free(st->vbuf);
    free(st->scores); free(st->ctx); free(st->gate);
    free(st->qkv); free(st->zbuf); free(st->abuf); free(st->bbuf); free(st->ybuf);
    free(st->ff_gate); free(st->ff_up);
    free(st->dacc);
    free(st->logits);
    free(st);
}

void sg_ref_state_reset(sg_ref_state *st) {
    if (!st) return;
    st->used = 0;
    for (uint32_t i = 0; i < st->cfg.n_layers; i++) {
        sg_ref_layer *L = &st->ls[i];
        if (L->conv_state) {
            memset(L->conv_state, 0,
                   (size_t)(st->cfg.conv_kernel - 1) * st->conv_dim * sizeof(float));
        }
        if (L->ssm_state) {
            memset(L->ssm_state, 0,
                   (size_t)st->cfg.n_v_heads * st->cfg.head_v_dim * st->cfg.head_k_dim
                       * sizeof(float));
        }
        /* The K/V caches are not cleared: nothing reads past `used`. */
    }
}

sg_err sg_ref_state_new(const sg_model *m, uint32_t max_ctx, sg_ref_state **out) {
    if (!out) return (sg_err){"ref: invalid arguments"};
    *out = NULL;
    if (!m || !m->layers || !m->tok_emb || !m->out_norm || !m->lm_head) {
        return (sg_err){"ref: invalid arguments"};
    }
    const sg_cfg *c = &m->cfg;
    if (max_ctx == 0) return (sg_err){"ref: max_ctx must be at least 1"};
    if (c->n_layers == 0 || c->hidden == 0 || c->ffn_hidden == 0 || c->vocab == 0
        || c->head_dim == 0 || c->n_heads == 0 || c->n_kv_heads == 0) {
        return (sg_err){"ref: model config has a zero dimension"};
    }
    if (c->n_heads % c->n_kv_heads != 0) {
        return (sg_err){"ref: n_heads is not a multiple of n_kv_heads"};
    }
    if (c->full_attn_interval == 0) {
        return (sg_err){"ref: full_attn_interval is zero"};
    }
    if (m->wtype != SG_T_BF16 && m->wtype != SG_T_Q8_0 && m->wtype != SG_T_F32) {
        return (sg_err){"ref: unsupported matmul weight dtype (want bf16, q8_0 or f32)"};
    }
    /* wwiden reads anything that is not BF16 as F32, so an F16 recorded in
     * any of these three would be read at twice the width. */
    if ((m->dense_type != SG_T_BF16 && m->dense_type != SG_T_F32)
        || (m->ssm_a_type != SG_T_BF16 && m->ssm_a_type != SG_T_F32)
        || (m->ssm_norm_type != SG_T_BF16 && m->ssm_norm_type != SG_T_F32)) {
        return (sg_err){"ref: unsupported small-tensor dtype (want bf16 or f32)"};
    }

    /* Layer-kind census, and the cross-check between the two sources of
     * truth for it. The loader decided by tensor presence; qwen3_5.py
     * decides by (L+1) % full_attention_interval. Disagreement means the
     * checkpoint is not the architecture the config claims, and guessing
     * which one to believe would produce a model that runs and is wrong. */
    uint32_t n_attn = 0, n_gdn = 0;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        bool is_attn = (m->layers[i].q_proj != NULL);
        bool want_attn = ((i + 1) % c->full_attn_interval) == 0;
        if (is_attn != want_attn) {
            fprintf(stderr,
                    "ref: layer %u carries %s tensors but full_attention_interval %u "
                    "says it should be %s\n", i,
                    is_attn ? "full-attention" : "gated-DeltaNet",
                    c->full_attn_interval, want_attn ? "full-attention" : "gated-DeltaNet");
            return (sg_err){"ref: layer kind disagrees with full_attention_interval"};
        }
        if (is_attn) n_attn++; else n_gdn++;
    }

    if (n_attn > 0) {
        if (c->rope_dim < 2 || c->rope_dim > c->head_dim || c->rope_dim % 2 != 0) {
            return (sg_err){"ref: rope_dim must be even and in [2, head_dim]"};
        }
        /* pow(theta, -2i/rope_dim) with theta <= 0 is a NaN generator, and a
         * NaN here poisons every logit with no other symptom. */
        if (!(c->rope_theta > 1.0f) || !isfinite(c->rope_theta)) {
            return (sg_err){"ref: rope_theta must be finite and greater than 1"};
        }
    }
    if (!(c->rms_eps >= 0.0f) || !isfinite(c->rms_eps)) {
        return (sg_err){"ref: rms_eps must be finite and non-negative"};
    }

    /* Every per-layer pointer the forward will dereference. The loaders
     * guarantee complete groups, but this is a public entry point that
     * already checks the three model-level pointers, and wwiden/wmatvec on a
     * NULL is a crash rather than an error return. */
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const sg_layer_w *w = &m->layers[i];
        const void *shared[] = { w->ln1, w->ln2, w->gate_proj, w->up_proj, w->down_proj };
        for (size_t k = 0; k < sizeof shared / sizeof *shared; k++) {
            if (!shared[k]) return (sg_err){"ref: a layer is missing an MLP or norm tensor"};
        }
        if (w->q_proj) {
            const void *a[] = { w->k_proj, w->v_proj, w->o_proj, w->q_norm, w->k_norm };
            for (size_t k = 0; k < sizeof a / sizeof *a; k++) {
                if (!a[k]) return (sg_err){"ref: a full-attention layer is incomplete"};
            }
        } else {
            const void *d[] = { w->ssm_in_qkv, w->ssm_in_z, w->ssm_in_b, w->ssm_in_a,
                                w->ssm_a, w->ssm_dt_bias, w->ssm_conv1d, w->ssm_norm,
                                w->ssm_out };
            for (size_t k = 0; k < sizeof d / sizeof *d; k++) {
                if (!d[k]) return (sg_err){"ref: a gated-DeltaNet layer is incomplete"};
            }
        }
    }
    if (n_gdn > 0) {
        if (c->n_k_heads == 0 || c->n_v_heads == 0 || c->head_k_dim == 0
            || c->head_v_dim == 0 || c->conv_kernel == 0) {
            return (sg_err){"ref: model has gated-DeltaNet layers but no DeltaNet dims"};
        }
        if (c->n_v_heads % c->n_k_heads != 0) {
            return (sg_err){"ref: n_v_heads is not a multiple of n_k_heads"};
        }
    }

    /* Every derived width below is a product of two config u32s, so compute
     * them in u64 and refuse anything implausible BEFORE narrowing. Without
     * this a checkpoint claiming n_v_heads = 1e5 and head_v_dim = 1e5 wraps
     * value_dim to a small number, every allocation is sized from the wrapped
     * value, and the forward writes past all of them. 1<<24 is far above any
     * real model (the 27B's largest is conv_dim 10240) and far below the wrap
     * point. */
    const uint64_t width_max = 1u << 24;
    uint64_t key_dim = (uint64_t)c->n_k_heads * c->head_k_dim;
    uint64_t value_dim = (uint64_t)c->n_v_heads * c->head_v_dim;
    uint64_t conv_dim = 2 * key_dim + value_dim;
    uint64_t attn_width = (uint64_t)c->n_heads * c->head_dim;
    /* Doubled only when the model actually folds an output gate into q_proj
     * (Task P1; see cfg.attn_output_gate in surge.h). Dense qwen3 has none. */
    uint64_t q_width = c->attn_output_gate ? attn_width * 2 : attn_width;
    uint64_t kv_width = (uint64_t)c->n_kv_heads * c->head_dim;
    if (key_dim > width_max || value_dim > width_max || conv_dim > width_max
        || q_width > width_max || kv_width > width_max
        || (uint64_t)c->hidden > width_max || (uint64_t)c->ffn_hidden > width_max
        || (uint64_t)c->conv_kernel > width_max) {
        return (sg_err){"ref: a model dimension is implausibly large"};
    }

    bool ok = true;
    sg_ref_state *st = calloc(1, sizeof *st);
    if (!st) return (sg_err){"ref: out of memory"};
    st->owner = m;
    st->cfg = *c;
    st->max_ctx = max_ctx;
    st->mat_type = m->wtype;
    st->a_form = m->ssm_a_form;
    st->v_tiled = m->v_heads_tiled;
    st->key_dim = (uint32_t)key_dim;
    st->value_dim = (uint32_t)value_dim;
    st->conv_dim = (uint32_t)conv_dim;
    st->q_width = (uint32_t)q_width;
    st->kv_width = (uint32_t)kv_width;
    st->attn_width = (uint32_t)attn_width;

    /* Q8_0 rows are whole 32-element blocks, so every matmul's `cols` must
     * be a multiple of 32. sg_ref_matvec_q8 silently returns without
     * touching y otherwise (Task 7 concern 7), which would be a
     * uninitialized-output footgun several layers deep; check it once here
     * instead. */
    if (st->mat_type == SG_T_Q8_0) {
        const uint32_t cols[] = { c->hidden, c->ffn_hidden, st->attn_width, st->value_dim };
        for (size_t i = 0; i < sizeof cols / sizeof *cols; i++) {
            if (cols[i] != 0 && cols[i] % 32 != 0) {
                sg_ref_state_free(st);
                return (sg_err){"ref: a Q8_0 matmul width is not a multiple of 32"};
            }
        }
    }

    st->ls = xzalloc(c->n_layers, sizeof *st->ls, &ok);
    st->out_norm_w = xzalloc(c->hidden, sizeof(float), &ok);
    st->x = xzalloc(c->hidden, sizeof(float), &ok);
    st->h = xzalloc(c->hidden, sizeof(float), &ok);
    st->r = xzalloc(c->hidden, sizeof(float), &ok);
    st->ff_gate = xzalloc(c->ffn_hidden, sizeof(float), &ok);
    st->ff_up = xzalloc(c->ffn_hidden, sizeof(float), &ok);
    st->logits = xzalloc(c->vocab, sizeof(float), &ok);
    st->dacc = xzalloc(c->head_dim, sizeof(double), &ok);
    if (n_attn > 0) {
        st->qg = xzalloc(st->q_width, sizeof(float), &ok);
        st->kbuf = xzalloc(st->kv_width, sizeof(float), &ok);
        st->vbuf = xzalloc(st->kv_width, sizeof(float), &ok);
        st->scores = xzalloc(max_ctx, sizeof(float), &ok);
        st->ctx = xzalloc(st->attn_width, sizeof(float), &ok);
        st->gate = xzalloc(st->attn_width, sizeof(float), &ok);
    }
    if (n_gdn > 0) {
        st->qkv = xzalloc(st->conv_dim, sizeof(float), &ok);
        st->zbuf = xzalloc(st->value_dim, sizeof(float), &ok);
        st->abuf = xzalloc(c->n_v_heads, sizeof(float), &ok);
        st->bbuf = xzalloc(c->n_v_heads, sizeof(float), &ok);
        st->ybuf = xzalloc(st->value_dim, sizeof(float), &ok);
    }
    if (!ok) { sg_ref_state_free(st); return (sg_err){"ref: out of memory"}; }

    /* mlx's sanitize adds 1.0 to input_layernorm / post_attention_layernorm /
     * model.norm / q_norm / k_norm, and to nothing else. ssm_norm is
     * deliberately NOT in that list. */
    float shift = m->norms_are_residual ? 1.0f : 0.0f;
    wwiden(m->out_norm, m->dense_type, st->out_norm_w, c->hidden, shift);

    for (uint32_t i = 0; i < c->n_layers; i++) {
        const sg_layer_w *w = &m->layers[i];
        sg_ref_layer *L = &st->ls[i];
        L->is_attn = (w->q_proj != NULL);

        L->ln1 = xzalloc(c->hidden, sizeof(float), &ok);
        L->ln2 = xzalloc(c->hidden, sizeof(float), &ok);
        if (!ok) { sg_ref_state_free(st); return (sg_err){"ref: out of memory"}; }
        wwiden(w->ln1, m->dense_type, L->ln1, c->hidden, shift);
        wwiden(w->ln2, m->dense_type, L->ln2, c->hidden, shift);

        if (L->is_attn) {
            L->q_norm_w = xzalloc(c->head_dim, sizeof(float), &ok);
            L->k_norm_w = xzalloc(c->head_dim, sizeof(float), &ok);
            L->k_cache = xzalloc((size_t)max_ctx * st->kv_width, sizeof(float), &ok);
            L->v_cache = xzalloc((size_t)max_ctx * st->kv_width, sizeof(float), &ok);
            if (!ok) { sg_ref_state_free(st); return (sg_err){"ref: out of memory"}; }
            wwiden(w->q_norm, m->dense_type, L->q_norm_w, c->head_dim, shift);
            wwiden(w->k_norm, m->dense_type, L->k_norm_w, c->head_dim, shift);
        } else {
            L->ssm_norm_w = xzalloc(c->head_v_dim, sizeof(float), &ok);
            L->conv_w = xzalloc((size_t)st->conv_dim * c->conv_kernel, sizeof(float), &ok);
            L->dt_bias = xzalloc(c->n_v_heads, sizeof(float), &ok);
            L->ssm_a = xzalloc(c->n_v_heads, sizeof(float), &ok);
            L->conv_state = xzalloc((size_t)(c->conv_kernel - 1) * st->conv_dim,
                                    sizeof(float), &ok);
            L->ssm_state = xzalloc((size_t)c->n_v_heads * c->head_v_dim * c->head_k_dim,
                                   sizeof(float), &ok);
            if (!ok) { sg_ref_state_free(st); return (sg_err){"ref: out of memory"}; }
            /* ssm_norm and ssm_a have their own recorded dtypes; conv1d and
             * dt_bias follow dense_type. None of the four is shifted. */
            wwiden(w->ssm_norm, m->ssm_norm_type, L->ssm_norm_w, c->head_v_dim, 0.0f);
            wwiden(w->ssm_a, m->ssm_a_type, L->ssm_a, c->n_v_heads, 0.0f);
            wwiden(w->ssm_conv1d, m->dense_type, L->conv_w,
                   (uint64_t)st->conv_dim * c->conv_kernel, 0.0f);
            wwiden(w->ssm_dt_bias, m->dense_type, L->dt_bias, c->n_v_heads, 0.0f);
        }
    }

    *out = st;
    return SG_OK;
}

/* ---------------------------------------------------------------------
 * The two layer kinds
 * --------------------------------------------------------------------- */

/* Qwen3NextAttention for one token. `in` is the already-normed hidden state,
 * `out` receives the layer's residual contribution. */
static void attn_layer(sg_ref_state *st, const sg_layer_w *w, sg_ref_layer *L,
                       const float *in, float *out, uint32_t pos) {
    const sg_cfg *c = &st->cfg;
    uint32_t hd = c->head_dim;
    /* Task P1: per-head stride into st->qg. Double width (queries + folded
     * gate) on the hybrid, single width on dense qwen3 -- see
     * cfg.attn_output_gate (surge.h) and st->q_width above, which this must
     * stay consistent with (st->q_width == c->n_heads * q_stride always). */
    uint32_t q_stride = c->attn_output_gate ? 2 * hd : hd;

    /* q_proj is DOUBLE width on the hybrid: mlx reshapes to [n_heads,
     * 2*head_dim] and splits on the LAST axis, so head h's gate sits
     * immediately after head h's queries rather than in one block after all
     * of them. Dense qwen3 has no gate at all: q_proj is single width and
     * q_stride collapses to hd, so every access below reads exactly the
     * queries and nothing past them. */
    wmatvec(w->q_proj, st->mat_type, in, st->qg, st->q_width, c->hidden);
    wmatvec(w->k_proj, st->mat_type, in, st->kbuf, st->kv_width, c->hidden);
    wmatvec(w->v_proj, st->mat_type, in, st->vbuf, st->kv_width, c->hidden);

    for (uint32_t h = 0; h < c->n_heads; h++) {
        float *qh = st->qg + (size_t)h * q_stride;
        /* q_norm applies to the queries only, never to the gate. */
        sg_ref_rmsnorm(qh, L->q_norm_w, hd, c->rms_eps);
        sg_ref_rope_partial(qh, hd, c->rope_dim, pos, c->rope_theta);
        if (c->attn_output_gate) {
            memcpy(st->gate + (size_t)h * hd, qh + hd, (size_t)hd * sizeof(float));
        }
    }
    for (uint32_t h = 0; h < c->n_kv_heads; h++) {
        float *kh = st->kbuf + (size_t)h * hd;
        sg_ref_rmsnorm(kh, L->k_norm_w, hd, c->rms_eps);
        sg_ref_rope_partial(kh, hd, c->rope_dim, pos, c->rope_theta);
    }

    memcpy(L->k_cache + (size_t)pos * st->kv_width, st->kbuf,
           (size_t)st->kv_width * sizeof(float));
    memcpy(L->v_cache + (size_t)pos * st->kv_width, st->vbuf,
           (size_t)st->kv_width * sizeof(float));
    uint32_t used = pos + 1;

    double scale = 1.0 / sqrt((double)hd);
    uint32_t repeat = c->n_heads / c->n_kv_heads;
    for (uint32_t h = 0; h < c->n_heads; h++) {
        uint32_t hk = h / repeat;      /* GQA: mlx repeats the kv head axis */
        const float *qh = st->qg + (size_t)h * q_stride;
        for (uint32_t t = 0; t < used; t++) {
            const float *kt = L->k_cache + ((size_t)t * c->n_kv_heads + hk) * hd;
            double dot = 0.0;
            for (uint32_t i = 0; i < hd; i++) dot += (double)qh[i] * (double)kt[i];
            st->scores[t] = (float)(dot * scale);
        }
        /* The causal mask is implicit: only 0..pos are ever scored. */
        sg_ref_softmax(st->scores, used);

        for (uint32_t i = 0; i < hd; i++) st->dacc[i] = 0.0;
        for (uint32_t t = 0; t < used; t++) {
            const float *vt = L->v_cache + ((size_t)t * c->n_kv_heads + hk) * hd;
            double p = (double)st->scores[t];
            for (uint32_t i = 0; i < hd; i++) st->dacc[i] += p * (double)vt[i];
        }
        float *ch = st->ctx + (size_t)h * hd;
        for (uint32_t i = 0; i < hd; i++) ch[i] = (float)st->dacc[i];
    }

    /* The output gate is a SIGMOID applied before o_proj -- only when the
     * model actually has one (Task P1). Dense qwen3 feeds the raw attention
     * output straight to o_proj. */
    if (c->attn_output_gate) {
        sg_ref_gate_sigmoid(st->ctx, st->gate, st->attn_width);
    }
    wmatvec(w->o_proj, st->mat_type, st->ctx, out, c->hidden, st->attn_width);
}

/* GatedDeltaNet for one token, with the conv tail and the delta-rule state
 * carried in L. */
static void gdn_layer(sg_ref_state *st, const sg_layer_w *w, sg_ref_layer *L,
                      const float *in, float *out) {
    const sg_cfg *c = &st->cfg;
    uint32_t dk = c->head_k_dim, dv = c->head_v_dim;

    wmatvec(w->ssm_in_qkv, st->mat_type, in, st->qkv, st->conv_dim, c->hidden);
    wmatvec(w->ssm_in_z, st->mat_type, in, st->zbuf, st->value_dim, c->hidden);
    wmatvec(w->ssm_in_b, st->mat_type, in, st->bbuf, c->n_v_heads, c->hidden);
    wmatvec(w->ssm_in_a, st->mat_type, in, st->abuf, c->n_v_heads, c->hidden);

    sg_ref_conv1d_causal(st->qkv, L->conv_w, L->conv_state, L->conv_state,
                         st->conv_dim, c->conv_kernel);
    sg_ref_silu(st->qkv, st->conv_dim);

    float *q = st->qkv;
    float *k = st->qkv + st->key_dim;
    float *v = st->qkv + 2 * st->key_dim;

    /* q and k are RMS-normed per key head with NO weight and a HARDCODED
     * eps of 1e-6 (qwen3_5.py line 180-181, not the config's rms_norm_eps),
     * then scaled: q by 1/head_k_dim and k by 1/sqrt(head_k_dim). They are
     * not L2-normalized. */
    double inv = 1.0 / sqrt((double)dk);
    for (uint32_t h = 0; h < c->n_k_heads; h++) {
        float *qh = q + (size_t)h * dk;
        float *kh = k + (size_t)h * dk;
        sg_ref_rmsnorm(qh, NULL, dk, 1e-6f);
        sg_ref_rmsnorm(kh, NULL, dk, 1e-6f);
        for (uint32_t i = 0; i < dk; i++) {
            qh[i] = (float)((double)qh[i] * inv * inv);
            kh[i] = (float)((double)kh[i] * inv);
        }
    }

    for (uint32_t h = 0; h < c->n_v_heads; h++) {
        uint32_t hk = sg_ssm_k_head(h, c->n_k_heads, c->n_v_heads, st->v_tiled);
        float beta = sg_ref_sigmoid(st->bbuf[h]);
        /* Dispatch on the recorded form: the GGUF stores -exp(A_log) and the
         * safetensors store A_log, and reading one as the other stays in
         * (0,1) and fails silently. */
        float decay = (st->a_form == SG_SSM_A_NEG_EXP)
            ? sg_ref_delta_decay_neg_a(L->ssm_a[h], st->abuf[h], L->dt_bias[h])
            : sg_ref_delta_decay(L->ssm_a[h], st->abuf[h], L->dt_bias[h]);
        sg_ref_delta_step(L->ssm_state + (size_t)h * dv * dk,
                          q + (size_t)hk * dk, k + (size_t)hk * dk,
                          v + (size_t)h * dv, beta, decay,
                          st->ybuf + (size_t)h * dv, dk, dv);
    }

    /* RMSNormGated: silu(z) * rms_norm(y, w, eps), per value head, with the
     * gate taken from the UNNORMALIZED z. */
    for (uint32_t h = 0; h < c->n_v_heads; h++) {
        float *yh = st->ybuf + (size_t)h * dv;
        float *zh = st->zbuf + (size_t)h * dv;
        sg_ref_rmsnorm(yh, L->ssm_norm_w, dv, c->rms_eps);
        sg_ref_swiglu(zh, yh, dv);   /* zh = silu(zh) * yh */
        memcpy(yh, zh, (size_t)dv * sizeof(float));
    }

    wmatvec(w->ssm_out, st->mat_type, st->ybuf, out, c->hidden, st->value_dim);
}

/* ---------------------------------------------------------------------
 * One token
 * --------------------------------------------------------------------- */

sg_err sg_ref_forward(sg_ref_state *st, const sg_model *m, int32_t token,
                      uint32_t pos, const float **logits) {
    if (!st || !m || !m->layers) return (sg_err){"ref: invalid arguments"};
    if (m != st->owner) {
        return (sg_err){"ref: this state was built for a different sg_model"};
    }
    const sg_cfg *c = &st->cfg;
    if (token < 0 || (uint32_t)token >= c->vocab) {
        return (sg_err){"ref: token id out of range"};
    }
    if (pos >= st->max_ctx) return (sg_err){"ref: position exceeds max_ctx"};
    /* The caches are append-only, so positions must arrive in order. A
     * caller that restarts a sequence calls sg_ref_state_reset. */
    if (pos != st->used) return (sg_err){"ref: positions must be presented in order"};

    /* No embedding scale: qwen3_5.py's Qwen3_5TextModel returns
     * embed_tokens(inputs) untouched. */
    wrow(m->tok_emb, st->mat_type, (uint64_t)token, c->hidden, st->x);

    for (uint32_t i = 0; i < c->n_layers; i++) {
        const sg_layer_w *w = &m->layers[i];
        sg_ref_layer *L = &st->ls[i];

        memcpy(st->h, st->x, (size_t)c->hidden * sizeof(float));
        sg_ref_rmsnorm(st->h, L->ln1, c->hidden, c->rms_eps);
        if (L->is_attn) attn_layer(st, w, L, st->h, st->r, pos);
        else            gdn_layer(st, w, L, st->h, st->r);
        for (uint32_t j = 0; j < c->hidden; j++) st->x[j] += st->r[j];

        memcpy(st->h, st->x, (size_t)c->hidden * sizeof(float));
        sg_ref_rmsnorm(st->h, L->ln2, c->hidden, c->rms_eps);
        wmatvec(w->gate_proj, st->mat_type, st->h, st->ff_gate, c->ffn_hidden, c->hidden);
        wmatvec(w->up_proj, st->mat_type, st->h, st->ff_up, c->ffn_hidden, c->hidden);
        sg_ref_swiglu(st->ff_gate, st->ff_up, c->ffn_hidden);
        wmatvec(w->down_proj, st->mat_type, st->ff_gate, st->r, c->hidden, c->ffn_hidden);
        for (uint32_t j = 0; j < c->hidden; j++) st->x[j] += st->r[j];
    }

    memcpy(st->h, st->x, (size_t)c->hidden * sizeof(float));
    sg_ref_rmsnorm(st->h, st->out_norm_w, c->hidden, c->rms_eps);
    /* m->lm_head aliases m->tok_emb when the embeddings are tied, which is
     * exactly mlx's embed_tokens.as_linear(out). */
    wmatvec(m->lm_head, st->mat_type, st->h, st->logits, c->vocab, c->hidden);

    st->used = pos + 1;
    if (logits) *logits = st->logits;
    return SG_OK;
}

