#ifndef SF_OPS_INTERNAL_H
#define SF_OPS_INTERNAL_H

#include <sionflow/ops/sf_ops_core.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_platform.h>
#include <sionflow/isa/sf_tensor.h>
#include <sionflow/isa/sf_exec_ctx.h>

/**
 * Internal helpers for SionFlow kernels.
 */

static inline bool _sf_should_log_error(sf_exec_ctx* ctx) {
    if (ctx->error != SF_ERROR_NONE) return false;
    if (ctx->global_error_ptr && sf_atomic_load(ctx->global_error_ptr) != 0) return false;
    return true;
}

// Generic Pointer Check
#define SF_CHECK_PTR(CTX, PTR) \
    do { \
        if (!(PTR)) { \
            if (_sf_should_log_error(CTX)) { \
                SF_LOG_ERROR("Runtime Error: Internal pointer is NULL. Op execution aborted."); \
            } \
            (CTX)->error = SF_ERROR_RUNTIME; \
            return; \
        } \
    } while(0)

#endif // SF_OPS_INTERNAL_H