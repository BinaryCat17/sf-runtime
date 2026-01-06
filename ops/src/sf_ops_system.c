#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>

/**
 * SionFlow System Kernels (Index Generation, etc.)
 */

static void sf_generate_index_chunk(void* out_raw, sf_dtype dtype, u32 count, u32 job_offset, u8 axis, bool is_vector, u8 domain_ndim, const u32* domain_shape) {
    u32 current_coords[SF_MAX_DIMS];
    u32 temp_idx = job_offset;
    for (int i = (int)domain_ndim - 1; i >= 0; --i) {
        current_coords[i] = temp_idx % domain_shape[i];
        temp_idx /= domain_shape[i];
    }
    for (u32 e = 0; e < count; ++e) {
        if (is_vector) {
            for (u32 d = 0; d < domain_ndim; ++d) {
                float val = (f32)current_coords[d];
                if (dtype == SF_DTYPE_F32) ((f32*)out_raw)[e * domain_ndim + d] = val;
                else if (dtype == SF_DTYPE_I32) ((i32*)out_raw)[e * domain_ndim + d] = (i32)current_coords[d];
            }
        } else {
            float val = (axis < domain_ndim) ? (f32)current_coords[axis] : 0.0f;
            if (dtype == SF_DTYPE_F32) ((f32*)out_raw)[e] = val;
            else if (dtype == SF_DTYPE_I32) ((i32*)out_raw)[e] = (axis < domain_ndim) ? (i32)current_coords[axis] : 0;
        }
        for (int d = (int)domain_ndim - 1; d >= 0; --d) {
            current_coords[d]++;
            if (current_coords[d] < domain_shape[d] || d == 0) break;
            current_coords[d] = 0;
        }
    }
}

void op_INDEX_X(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    u16 r_out = inst->dest_idx;
    sf_generate_index_chunk(ctx->reg_ptrs[r_out], ctx->reg_info[r_out].dtype, ctx->batch_size, ctx->linear_offset, 0, false, ctx->ndim, ctx->domain_shape);
}

void op_INDEX_Y(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    u16 r_out = inst->dest_idx;
    sf_generate_index_chunk(ctx->reg_ptrs[r_out], ctx->reg_info[r_out].dtype, ctx->batch_size, ctx->linear_offset, 1, false, ctx->ndim, ctx->domain_shape);
}

void op_INDEX_Z(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    u16 r_out = inst->dest_idx;
    sf_generate_index_chunk(ctx->reg_ptrs[r_out], ctx->reg_info[r_out].dtype, ctx->batch_size, ctx->linear_offset, 2, false, ctx->ndim, ctx->domain_shape);
}
