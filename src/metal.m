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

#include <mach-o/dyld.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    SG_K_GROUPS2  /* params[2] threadgroups of SG_TG */
};

#define SG_TG 256u

typedef struct {
    const char *name;
    int kind;
} sg_kernel_desc;

/* sg_gpu.pipes is indexed by position here, so the enum and the table must
 * stay in lockstep; the static assert below is what enforces that. The first
 * twelve are Task 9's per-op kernels, reachable through sg_gpu_run_op; the
 * rest are Task 10's fused/strided variants, which take buffer layouts
 * sg_gpu_run_op has no size rule for and are therefore encoded only by
 * sg_gpu_forward. */
enum {
    KI_RMSNORM = 0, KI_RMSNORM_GATED, KI_ROPE, KI_MATVEC_BF16, KI_MATVEC_F32,
    KI_SOFTMAX, KI_SWIGLU, KI_SILU, KI_GATE_SIGMOID, KI_ATTN, KI_CONV1D,
    KI_DELTA, KI_RMSNORM_HEADS, KI_ROPE_HEADS, KI_GATE_STRIDED, KI_SCALE,
    KI_ADD, KI_DELTA_GATES, KI_DELTA_MULTI, KI_COUNT
};

static const sg_kernel_desc SG_KERNELS[] = {
    { "k_rmsnorm",             SG_K_TG1     },
    { "k_rmsnorm_gated",       SG_K_GATED   },
    { "k_rope",                SG_K_ELEM    },
    { "k_matvec_bf16",         SG_K_ROWS    },
    { "k_matvec_f32",          SG_K_ROWS    },
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
    void *kv;         /* [2, max_ctx, n_kv_heads, head_dim]: K then V */
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
     * put a ceiling of ~8k tokens on the context length. */
    id<MTLBuffer> scratch;
    uint64_t scratch_bytes;

    /* --- the loaded model (sg_gpu_load_model) --- */
    const sg_model *model;
    sg_cfg cfg;
    uint32_t key_dim, value_dim, conv_dim, q_width, kv_width, attn_width;
    int mat_kernel;            /* KI_MATVEC_BF16 or KI_MATVEC_F32 */
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
    [g->lib release];
    [g->queue release];
    [g->dev release];
    g->scratch = nil;
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
    uint64_t t0 = 0, t1 = 0;
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
    } else if (strcmp(kernel, "k_matvec_bf16") == 0) {
        ok = mul_ck(p[0], p[1], &t0) && mul_ck(t0, 2, &need_a) &&
             mul_ck(p[1], f, &need_b) && mul_ck(p[0], f, &need_o);
    } else if (strcmp(kernel, "k_matvec_f32") == 0) {
        ok = mul_ck(p[0], p[1], &t0) && mul_ck(t0, f, &need_a) &&
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
    if (strcmp(kernel, "k_rope") == 0) {
        if (p[1] < 2 || p[1] > p[0] || p[1] % 2 != 0) {
            return gpu_errf("gpu: k_rope rope_dim %u must be even and in [2, head_dim %u]",
                            p[1], p[0]);
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
    gpu_grid(kind, params, &groups, &elems);
    if (kind == SG_K_ELEM || kind == SG_K_ELEM01 || kind == SG_K_ELEM02) {
        if (elems == 0) return gpu_errf("gpu: %s dispatched with zero elements", kernel);
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

/* ref.c's wrow for the two dtypes this path accepts. Q8_0 is rejected at
 * load, so there is no dequantizing branch here. */
static void gpu_embed_row(const void *w, sg_tensor_type t, uint64_t row,
                          uint32_t cols, float *out) {
    if (t == SG_T_BF16) {
        const uint16_t *b = (const uint16_t *)w + row * cols;
        for (uint32_t i = 0; i < cols; i++) out[i] = gpu_bf16_to_f32(b[i]);
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

/* Qwen3NextAttention for one token, mirroring ref.c's attn_layer statement
 * for statement. b_h holds rms_norm(x, ln1) on entry; b_r receives the
 * layer's residual contribution.
 *
 * k_proj and v_proj write STRAIGHT INTO the cache slot for this position and
 * the qk-norm and RoPE then run in place there, which is the same values
 * ref.c computes into kbuf/vbuf and memcpy's afterwards, with one fewer
 * copy. The queries keep their interleaved [head, 2*head_dim] layout the
 * whole way through: q_norm and RoPE touch only the first head_dim of each
 * head (k_rmsnorm_heads / k_rope_heads take the stride), so the attention
 * output gate in the second half arrives at k_gate_sigmoid_strided exactly
 * as q_proj produced it. */
static void enc_attn(sg_enc *E, sg_gpu_layer *L, uint32_t pos) {
    sg_gpu *g = E->g;
    const sg_cfg *c = &g->cfg;
    uint32_t hd = c->head_dim;
    uint32_t used = pos + 1;
    uint64_t koff = (uint64_t)pos * g->kv_width;
    uint64_t vbase = (uint64_t)g->max_ctx * g->kv_width;
    float scale = (float)(1.0 / sqrt((double)hd));

    enc_op(E, g->mat_kernel, L->w_q, 0, g->b_h, 0, g->b_qg, 0, nil, 0,
           PARAMS(g->q_width, c->hidden));
    enc_op(E, g->mat_kernel, L->w_k, 0, g->b_h, 0, L->kv, koff, nil, 0,
           PARAMS(g->kv_width, c->hidden));
    enc_op(E, g->mat_kernel, L->w_v, 0, g->b_h, 0, L->kv, vbase + koff, nil, 0,
           PARAMS(g->kv_width, c->hidden));

    /* q_norm applies to the queries only, never to the gate. */
    enc_op(E, KI_RMSNORM_HEADS, g->b_qg, 0, L->qk_norm, 0, g->b_qg, 0, nil, 0,
           PARAMS(hd, c->n_heads, fbits(c->rms_eps), 1, 2 * hd));
    enc_op(E, KI_ROPE_HEADS, g->b_qg, 0, g->b_cs, 0, g->b_qg, 0, nil, 0,
           PARAMS(hd, c->rope_dim, c->n_heads, 2 * hd));
    enc_op(E, KI_RMSNORM_HEADS, L->kv, koff, L->qk_norm, hd, L->kv, koff, nil, 0,
           PARAMS(hd, c->n_kv_heads, fbits(c->rms_eps), 1, hd));
    enc_op(E, KI_ROPE_HEADS, L->kv, koff, g->b_cs, 0, L->kv, koff, nil, 0,
           PARAMS(hd, c->rope_dim, c->n_kv_heads, hd));

    enc_op(E, KI_ATTN, g->b_qg, 0, L->kv, 0, g->b_ctx, 0, g->scratch, 0,
           PARAMS(c->n_heads, c->n_kv_heads, hd, used, 2 * hd,
                  (uint32_t)vbase, fbits(scale)));

    /* The output gate is a sigmoid applied before o_proj. */
    enc_op(E, KI_GATE_STRIDED, g->b_ctx, 0, g->b_qg, 0, g->b_ctx, 0, nil, 0,
           PARAMS(hd, c->n_heads, 2 * hd, hd));
    enc_op(E, g->mat_kernel, L->w_o, 0, g->b_ctx, 0, g->b_r, 0, nil, 0,
           PARAMS(c->hidden, g->attn_width));
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
 * dtype. Both extents are bounded by the 1<<24 check in gpu_check_cfg, so
 * the byte count cannot wrap. */
static sg_err gpu_wrap_w(sg_gpu *g, const void *p, uint64_t rows, uint64_t cols,
                         void **out) {
    uint64_t esz = (g->model->wtype == SG_T_BF16) ? 2 : 4;
    return sg_gpu_wrap(g, p, rows * cols * esz, out);
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
    if (m->wtype != SG_T_BF16 && m->wtype != SG_T_F32) {
        return (sg_err){"gpu: the Metal path needs bf16 or f32 matmul weights "
                        "(Q8_0 arrives with M3)"};
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
    g->q_width = 2 * g->attn_width;
    g->kv_width = c->n_kv_heads * c->head_dim;
    g->mat_kernel = (m->wtype == SG_T_BF16) ? KI_MATVEC_BF16 : KI_MATVEC_F32;

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

    /* k_attn_decode's v_cache offset is a uint32 param, so the K half of one
     * layer's cache has to be addressable in floats by a uint32. That is 4
     * billion floats; the real ceiling here is memory, not this check, but a
     * silently truncated offset would read the wrong half of the cache. */
    uint64_t half = (uint64_t)max_ctx * g->kv_width;
    if (half > UINT32_MAX) return (sg_err){"gpu: max_ctx is too large for this kv cache layout"};

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
            e = gpu_alloc_f32(g, SG_KV_GROUPS * half, &L->kv, NULL);
            if (sg_failed(e)) { gpu_free_state(g); return e; }
        } else {
            e = gpu_alloc_f32(g, (uint64_t)g->conv_dim * c->conv_kernel,
                              &L->conv_buf, NULL);
            if (sg_failed(e)) { gpu_free_state(g); return e; }
            e = gpu_alloc_f32(g, (uint64_t)c->n_v_heads * c->head_v_dim * c->head_k_dim,
                              &L->ssm, NULL);
            if (sg_failed(e)) { gpu_free_state(g); return e; }
        }
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
                if (L->is_attn) enc_attn(&E, L, pos);
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
