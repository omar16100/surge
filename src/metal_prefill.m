/* metal_prefill.m - the chunked prompt prefill half of the Metal host layer,
 * split out of src/metal.m by task R2 when that file passed 4600 lines.
 *
 * WHAT IS HERE, and why this is the seam:
 *
 *   - enc_attn_prefill (Task M5.4) and enc_gdn_prefill (Task M5.5), the two
 *     per-layer CHUNK encoders;
 *   - sg_gpu_prefill (Task M5.6) and its duty-cycle / segmentation controls
 *     (Task B8), which drive those two across every layer.
 *
 * NOTHING IN THE DECODE PATH CALLS ANY OF IT. sg_gpu_forward, enc_attn and
 * enc_gdn are in src/metal.m and reach nothing in this file, so the seam runs
 * between two halves that already did not talk to each other; the traffic is
 * one-way, from here into the shared helpers of src/metal_internal.h.
 *
 * THIS IS A MOVE, NOT A REWRITE. Every line below other than this header came
 * from src/metal.m verbatim, in its original order (the two encoders first,
 * then the M5.6 orchestration, which is the order they had there). Behaviour,
 * dispatch order, buffer sizes and error text are unchanged.
 *
 * WHAT THE SPLIT DID COST, stated plainly because R1's byte-identity gate does
 * NOT transfer to Objective-C: enc_op, enc_matmul, enc_kv_store, gpu_errf,
 * scratch_ensure, gpu_elem_width, gemm_kernel_for, gpu_embed_row and
 * gpu_alloc_f32 were `static` in src/metal.m and are now declared in
 * src/metal_internal.h, so they have external linkage and can no longer be
 * inlined into the call sites below. They are all per-dispatch or
 * per-allocation, never per-element. The one-line accessors that ARE on the
 * per-element path (bufof, offof, fbits) went into that header as
 * `static inline` instead, precisely so they keep inlining in both files.
 *
 * The public contract for the entry points here lives in surge.h, next to
 * their declarations, and is not repeated.
 */
#include "metal_internal.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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
 * public one-shots (sg_gpu_run_conv1d_chunk, sg_gpu_run_delta_gates_chunk,
 * sg_gpu_run_delta_chunk, sg_gpu_run_rmsnorm_gated_chunk), which stayed in
 * src/metal.m when task R2 moved this file out of it.
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
