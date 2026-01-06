#include "sf_ops_internal.h"
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sionflow/base/sf_log.h>

// --- Op: CumSum (Prefix Sum) ---
void op_CUMSUM(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    size_t count = ctx->batch_size;
    u8* src_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* dst_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];

    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);

    if (ctx->sync_data && ctx->sync_pass == 0) {
        // Pass 0: Local Scan + Reporting chunk total
        f32 sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum += *(f32*)src_ptr;
            *(f32*)dst_ptr = sum;
            src_ptr += st1;
            dst_ptr += st0;
        }
        ((f32*)ctx->sync_data)[ctx->job_idx] = sum;
    } 
    else if (ctx->sync_data && ctx->sync_pass == 1) {
        // Pass 1: Adding global offset from previous chunks
        f32 offset = ((f32*)ctx->sync_data)[ctx->job_idx];
        if (offset == 0.0f) return; // Nothing to add

        for (size_t i = 0; i < count; ++i) {
            *(f32*)dst_ptr += offset;
            dst_ptr += st0;
        }
    }
    else {
        // Fallback: Sequential (if no sync context)
        f32 sum = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            sum += *(f32*)src_ptr;
            *(f32*)dst_ptr = sum;
            src_ptr += st1;
            dst_ptr += st0;
        }
    }
}

// --- Op: Compress (Filter) ---
void op_COMPRESS(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    (void)ctx; (void)inst;
    SF_LOG_WARN("OpCompress is temporarily disabled in the new Flat Execution model.");
}

// --- Op: Gather (Random Access) ---
void op_GATHER(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    u8* dst_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];
    u8* idx_ptr = (u8*)ctx->reg_ptrs[inst->src2_idx];
    u8* data_base = (u8*)ctx->reg_ptrs[inst->src1_idx];

    const sf_type_info* idx_info = &ctx->reg_info[inst->src2_idx];
    const sf_type_info* data_info = &ctx->reg_info[inst->src1_idx];

    size_t data_count = 1;
    for(int i=0; i<data_info->ndim; ++i) data_count *= (size_t)data_info->shape[i];
    
    size_t out_count = ctx->batch_size;
    size_t elem_size = sf_dtype_size(data_info->dtype);
    
    i32 st_dst = SF_GET_STRIDE_D(inst);
    i32 st_idx = SF_GET_STRIDE_S2(inst);

    bool is_contiguous = true;
    if (data_info->ndim > 0) {
        int32_t expected_stride = 1;
        for (int i = data_info->ndim - 1; i >= 0; --i) {
            if (data_info->strides[i] != expected_stride) { is_contiguous = false; break; }
            expected_stride *= data_info->shape[i];
        }
    }

    for (size_t i = 0; i < out_count; ++i) {
        int idx = -1;
        if (idx_info->dtype == SF_DTYPE_F32) {
            idx = (int)(*(f32*)idx_ptr);
        } else {
            idx = *(i32*)idx_ptr;
        }

        u8* target = dst_ptr;
        if (idx >= 0 && (size_t)idx < data_count) {
            void* src_item_ptr = NULL;
            if (is_contiguous) {
                src_item_ptr = data_base + (size_t)idx * elem_size;
            } else {
                size_t offset = 0;
                size_t temp_idx = (size_t)idx;
                for (int d = data_info->ndim - 1; d >= 0; --d) {
                    offset += (temp_idx % data_info->shape[d]) * data_info->strides[d];
                    temp_idx /= data_info->shape[d];
                }
                src_item_ptr = data_base + offset * elem_size;
            }
            memcpy(target, src_item_ptr, elem_size);
        } else {
            memset(target, 0, elem_size);
            if (_sf_should_log_error(ctx)) {
                ctx->error = SF_ERROR_OUT_OF_BOUNDS;
                ctx->error_idx = (u32)i;
                SF_LOG_ERROR("Gather OOB: Index %d at batch element %zu. Data size: %zu. Using 0.", 
                             idx, i, data_count);
            }
        }
        dst_ptr += st_dst;
        idx_ptr += st_idx;
    }
}
