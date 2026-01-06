#ifndef SF_KERNEL_UTILS_H
#define SF_KERNEL_UTILS_H

#include <sionflow/isa/sf_instruction.h>
#include <sionflow/base/sf_math.h>
#include "sf_ops_internal.h"
#include <math.h>
#include <string.h>
#include <sionflow/isa/sf_exec_ctx.h>

// --- Stride Inference ---

#define SF_GET_STRIDE_D(inst)  (ctx->reg_strides[(inst)->dest_idx])
#define SF_GET_STRIDE_S1(inst) (ctx->reg_strides[(inst)->src1_idx])
#define SF_GET_STRIDE_S2(inst) (ctx->reg_strides[(inst)->src2_idx])
#define SF_GET_STRIDE_S3(inst) (ctx->reg_strides[(inst)->src3_idx])
#define SF_GET_STRIDE_S4(inst) (ctx->reg_strides[(inst)->src4_idx])

// --- Macros: Optimized Kernel Definitions ---

#define SF_SAFE_F32(x) (isfinite((float)(x)) ? (f32)(x) : 0.0f)

#define SF_KERNEL_AUTO(NAME, EXPR, ARITY) \
void op_##NAME(sf_exec_ctx* ctx, const struct sf_instruction* inst) { \
    const size_t sz = ctx->batch_size; \
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx]; \
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx]; \
    u8* b_ptr = (ARITY >= 2) ? (u8*)ctx->reg_ptrs[inst->src2_idx] : NULL; \
    u8* c_ptr = (ARITY >= 3) ? (u8*)ctx->reg_ptrs[inst->src3_idx] : NULL; \
    const i32 st0 = SF_GET_STRIDE_D(inst); \
    const i32 st1 = SF_GET_STRIDE_S1(inst); \
    const i32 st2 = (ARITY >= 2) ? SF_GET_STRIDE_S2(inst) : 0; \
    const i32 st3 = (ARITY >= 3) ? SF_GET_STRIDE_S3(inst) : 0; \
    for(size_t i=0; i<sz; ++i) { \
        const f32 va = *(f32*)a_ptr; \
        const f32 vb = (ARITY >= 2) ? *(f32*)b_ptr : 0.0f; \
        const f32 vc = (ARITY >= 3) ? *(f32*)c_ptr : 0.0f; \
        *(f32*)d_ptr = SF_SAFE_F32(EXPR); \
        a_ptr += st1; \
        if (ARITY >= 2) b_ptr += st2; \
        if (ARITY >= 3) c_ptr += st3; \
        d_ptr += st0; \
    } \
}

#endif // SF_KERNEL_UTILS_H