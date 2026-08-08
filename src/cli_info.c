/* cli_info.c - GGUF metadata dump CLI.
 *
 * surge-info <path>
 *
 * Prints GGUF version, alignment, all metadata keys and values, tensor count,
 * and first 8 + last 2 tensor names with types and dimensions.
 */
#include "surge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char *kv_type_name(sg_gguf_kv_type t) {
    switch (t) {
        case SG_GGUF_U8: return "U8";
        case SG_GGUF_I8: return "I8";
        case SG_GGUF_U16: return "U16";
        case SG_GGUF_I16: return "I16";
        case SG_GGUF_U32: return "U32";
        case SG_GGUF_I32: return "I32";
        case SG_GGUF_F32: return "F32";
        case SG_GGUF_BOOL: return "BOOL";
        case SG_GGUF_STR: return "STR";
        case SG_GGUF_ARR: return "ARR";
        case SG_GGUF_U64: return "U64";
        case SG_GGUF_I64: return "I64";
        case SG_GGUF_F64: return "F64";
        default: return "?";
    }
}

static const char *tensor_type_name(sg_tensor_type t) {
    switch (t) {
        case SG_T_F32: return "F32";
        case SG_T_F16: return "F16";
        case SG_T_Q8_0: return "Q8_0";
        case SG_T_BF16: return "BF16";
        default: return "?";
    }
}

static void print_kv_value(const sg_gguf *g, const char *key, sg_gguf_kv_type type) {
    switch (type) {
        case SG_GGUF_U8: {
            printf("    %s (U8)\n", key);
            break;
        }
        case SG_GGUF_I8: {
            printf("    %s (I8)\n", key);
            break;
        }
        case SG_GGUF_U16: {
            printf("    %s (U16)\n", key);
            break;
        }
        case SG_GGUF_I16: {
            printf("    %s (I16)\n", key);
            break;
        }
        case SG_GGUF_U32: {
            uint32_t v;
            if (sg_gguf_get_u32(g, key, &v)) {
                printf("    %s = %u\n", key, v);
            } else {
                printf("    %s (unavailable)\n", key);
            }
            break;
        }
        case SG_GGUF_I32: {
            printf("    %s (I32)\n", key);
            break;
        }
        case SG_GGUF_F32: {
            float v;
            if (sg_gguf_get_f32(g, key, &v)) {
                printf("    %s = %g\n", key, (double)v);
            } else {
                printf("    %s (unavailable)\n", key);
            }
            break;
        }
        case SG_GGUF_BOOL: {
            printf("    %s (BOOL)\n", key);
            break;
        }
        case SG_GGUF_STR: {
            const char *v;
            if (sg_gguf_get_str(g, key, &v)) {
                printf("    %s = \"%s\"\n", key, v);
            } else {
                printf("    %s (unavailable)\n", key);
            }
            break;
        }
        case SG_GGUF_ARR: {
            sg_gguf_kv_type elem_type;
            uint64_t count;
            if (sg_gguf_get_arr(g, key, &elem_type, NULL, &count)) {
                printf("    %s = [%llu x %s]\n", key, (unsigned long long)count, kv_type_name(elem_type));
            } else {
                printf("    %s (unavailable)\n", key);
            }
            break;
        }
        case SG_GGUF_U64: {
            printf("    %s (U64)\n", key);
            break;
        }
        case SG_GGUF_I64: {
            printf("    %s (I64)\n", key);
            break;
        }
        case SG_GGUF_F64: {
            printf("    %s (F64)\n", key);
            break;
        }
        default: {
            printf("    %s (unknown type)\n", key);
            break;
        }
    }
}

static void print_dims(uint32_t n_dims, const uint64_t *dims) {
    printf("[");
    for (uint32_t i = 0; i < n_dims; i++) {
        if (i > 0) printf(", ");
        printf("%llu", (unsigned long long)dims[i]);
    }
    printf("]");
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: surge-info <path>\n");
        return 1;
    }

    sg_gguf *g = NULL;
    sg_err e = sg_gguf_open(argv[1], &g);
    if (sg_failed(e)) {
        fprintf(stderr, "surge-info: failed to open %s: %s\n", argv[1], e.msg);
        return 1;
    }

    printf("GGUF Version: %u\n", sg_gguf_version(g));

    uint32_t alignment = 32;
    if (sg_gguf_get_u32(g, "general.alignment", &alignment)) {
        printf("Alignment: %u\n", alignment);
    } else {
        printf("Alignment: %u (default)\n", alignment);
    }

    printf("\nMetadata (%llu keys):\n", (unsigned long long)sg_gguf_kv_count(g));
    for (uint64_t i = 0; i < sg_gguf_kv_count(g); i++) {
        const char *key;
        sg_gguf_kv_type type;
        if (sg_gguf_kv_at(g, i, &key, &type)) {
            print_kv_value(g, key, type);
        }
    }

    uint64_t n_tensors = sg_gguf_tensor_count(g);
    printf("\nTensors: %llu total\n", (unsigned long long)n_tensors);

    if (n_tensors > 0) {
        uint64_t to_show_first = (n_tensors < 8) ? n_tensors : 8;
        printf("\nFirst %llu tensors:\n", (unsigned long long)to_show_first);
        for (uint64_t i = 0; i < to_show_first; i++) {
            const sg_tensor *t = sg_gguf_tensor_at(g, i);
            if (t) {
                printf("  [%llu] %s: %s ", (unsigned long long)i, t->name, tensor_type_name(t->type));
                print_dims(t->n_dims, t->dims);
                printf(" (%llu bytes)\n", (unsigned long long)t->nbytes);
            }
        }

        if (n_tensors > 8) {
            uint64_t to_show_last = (n_tensors - 8 < 2) ? (n_tensors - 8) : 2;
            printf("\nLast %llu tensors:\n", (unsigned long long)to_show_last);
            for (uint64_t i = n_tensors - to_show_last; i < n_tensors; i++) {
                const sg_tensor *t = sg_gguf_tensor_at(g, i);
                if (t) {
                    printf("  [%llu] %s: %s ", (unsigned long long)i, t->name, tensor_type_name(t->type));
                    print_dims(t->n_dims, t->dims);
                    printf(" (%llu bytes)\n", (unsigned long long)t->nbytes);
                }
            }
        }
    }

    sg_gguf_close(g);
    return 0;
}
