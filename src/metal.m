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
    SG_K_ELEM,   /* params[0] threads, non-uniform threadgroups */
    SG_K_ROWS,   /* params[0] threadgroups of SG_TG (one per output row) */
    SG_K_TG1,    /* exactly one threadgroup of SG_TG */
    SG_K_ATTN,   /* params[0] threadgroups of SG_TG, plus a scores scratch */
    SG_K_GATED   /* params[1] threadgroups of SG_TG (one per value head) */
};

#define SG_TG 256u

typedef struct {
    const char *name;
    int kind;
} sg_kernel_desc;

/* Order is not significant; sg_gpu.pipes is indexed by position here. */
static const sg_kernel_desc SG_KERNELS[] = {
    { "k_rmsnorm",       SG_K_TG1  },
    { "k_rmsnorm_gated", SG_K_GATED },
    { "k_rope",          SG_K_ELEM },
    { "k_matvec_bf16",   SG_K_ROWS },
    { "k_matvec_f32",    SG_K_ROWS },
    { "k_softmax",       SG_K_TG1  },
    { "k_swiglu",        SG_K_ELEM },
    { "k_silu",          SG_K_ELEM },
    { "k_gate_sigmoid",  SG_K_ELEM },
    { "k_attn_decode",   SG_K_ATTN },
    { "k_conv1d_step",   SG_K_ELEM },
    { "k_delta_step",    SG_K_TG1  },
};
#define SG_N_KERNELS ((int)(sizeof SG_KERNELS / sizeof SG_KERNELS[0]))

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

void sg_gpu_free(sg_gpu *g) {
    if (!g) return;
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

/* Two handles collide when they name the same MTLBuffer and their
 * [offset, offset+nbytes) ranges intersect. Comparing the handles alone would
 * miss the case that actually happens with sg_gpu_wrap: two wraps of the same
 * mapping are two different handles over one buffer. */
static bool bufs_overlap(const sg_gpu_buf *x, const sg_gpu_buf *y) {
    if (!x || !y || x->buf != y->buf) return false;
    return x->offset < y->offset + y->nbytes && y->offset < x->offset + x->nbytes;
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
    switch (kind) {
    case SG_K_ELEM:  elems = params[0]; break;
    case SG_K_ROWS:  groups = params[0]; break;
    case SG_K_ATTN:  groups = params[0]; break;
    case SG_K_GATED: groups = params[1]; break;
    default:         groups = 1; break;
    }
    if (kind == SG_K_ELEM) {
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

            if (kind == SG_K_ELEM) {
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
