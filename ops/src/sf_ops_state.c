#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/base/sf_math.h>
#include "sf_ops_internal.h"
#include <string.h>

void op_COPY(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    const sf_type_info* info = &ctx->reg_info[inst->src1_idx];
    size_t sz = ctx->batch_size;
    size_t esize = sf_dtype_size(info->dtype);
    
    u8* s_ptr = (u8*)ctx->reg_ptrs[inst->src1_idx];
    u8* d_ptr = (u8*)ctx->reg_ptrs[inst->dest_idx];

    i32 st0 = SF_GET_STRIDE_D(inst);
    i32 st1 = SF_GET_STRIDE_S1(inst);

    for(size_t i=0; i<sz; ++i) {
        memcpy(d_ptr, s_ptr, esize);
        s_ptr += st1;
        d_ptr += st0;
    }
}

void op_SLICE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    op_COPY(ctx, inst);
}

void op_RESHAPE(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    op_COPY(ctx, inst);
}