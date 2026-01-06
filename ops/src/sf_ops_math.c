#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/base/sf_math.h>
#include <sionflow/isa/sf_exec_ctx.h>
#include "sf_ops_internal.h"
#include <math.h>

/**
 * SionFlow Atomic Kernels
 * Note: Automatic kernels are now in sf_ops_auto.c
 */

// --- Vector Math (Custom Kernels) ---

static inline f32 _vec_dot_impl(f32* a_ptr, f32* b_ptr, size_t len) {
    f32 sum = 0;
    for (size_t j = 0; j < len; ++j) {
        sum += a_ptr[j] * b_ptr[j];
    }
    return sum;
}

static inline f32 _vec_len_sq_impl(f32* a_ptr, size_t len) {
    f32 sum = 0;
    for (size_t j = 0; j < len; ++j) {
        f32 v = a_ptr[j];
        sum += v * v;
    }
    return sum;
}

void op_DOT(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const sf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* b_ptr = (u8*)ctx->reg_ptrs[inst->src2_idx];
    
    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);
    i32 st2 = SF_GET_STRIDE_S2(inst);
    
    for (size_t i = 0; i < sz; ++i) {
        *(f32*)d_ptr = SF_SAFE_F32(_vec_dot_impl((f32*)a_ptr, (f32*)b_ptr, vec_len));
        a_ptr += st1;
        b_ptr += st2;
        d_ptr += st0;
    }
}

void op_LENGTH(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const sf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        *(f32*)d_ptr = SF_SAFE_F32(sqrtf(_vec_len_sq_impl((f32*)a_ptr, vec_len)));
        a_ptr += st1;
        d_ptr += st0;
    }
}

void op_NORMALIZE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const sf_type_info* a_info = &ctx->reg_info[inst->src1_idx];
    size_t vec_len = a_info->shape[a_info->ndim - 1];
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* a_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 len = sqrtf(_vec_len_sq_impl((f32*)a_ptr, vec_len));
        f32 inv_len = (len > 1e-6f) ? (1.0f / len) : 0.0f;

        f32* d_f32 = (f32*)d_ptr;
        f32* a_f32 = (f32*)a_ptr;
        for (size_t j = 0; j < vec_len; ++j) {
            d_f32[j] = a_f32[j] * inv_len;
        }
        a_ptr += st1;
        d_ptr += st0;
    }
}

void op_SMOOTHSTEP(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    size_t sz = ctx->batch_size;
    
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* x_ptr = (u8*)ctx->reg_ptrs[inst->src2_idx];
    u8* e_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    
    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);
    i32 st2 = SF_GET_STRIDE_S2(inst);

    for (size_t i = 0; i < sz; ++i) {
        f32 e0 = ((f32*)e_ptr)[0];
        f32 e1 = ((f32*)e_ptr)[1];
        f32 val = *(f32*)x_ptr;
        
        f32 span = e1 - e0;
        if (fabsf(span) < 1e-6f) span = (span < 0) ? -1e-6f : 1e-6f;

        f32 t = (val - e0) / span;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        
        *(f32*)d_ptr = SF_SAFE_F32(t * t * (3.0f - 2.0f * t));
        
        x_ptr += st2;
        e_ptr += st1;
        d_ptr += st0;
    }
}

// --- Reduction ---

void op_SUM(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const sf_type_info* src_info = &ctx->reg_info[inst->src1_idx];
    size_t sz = ctx->batch_size;
    
    f32 sum = 0;
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    i32 st1 = SF_GET_STRIDE_S1(inst);

    for (size_t i = 0; i < sz; ++i) {
        sum += *(f32*)s_ptr;
        s_ptr += st1;
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr = sum;
}

void op_SIZE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const sf_type_info* src_info = &ctx->reg_info[inst->src1_idx];
    size_t count = 1;
    for (int i = 0; i < src_info->ndim; ++i) {
        count *= (src_info->shape[i] > 0 ? (size_t)src_info->shape[i] : 1);
    }
    
    f32* d_ptr = (f32*)ctx->reg_ptrs[inst->dest_idx];
    *d_ptr = (f32)count;
}