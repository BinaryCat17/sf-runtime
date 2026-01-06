#ifndef SF_OPS_CORE_H
#define SF_OPS_CORE_H

#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/isa/sf_instruction.h>
#include <sionflow/base/sf_types.h>

#include <sionflow/isa/sf_exec_ctx.h>

/**
 * @brief Function signature for a SionFlow Operation Kernel (CPU Interpreter).
 */
typedef void (*sf_op_func)(struct sf_exec_ctx* ctx, const struct sf_instruction* inst);

// Registers all available operations to the table.
void sf_ops_fill_table(sf_op_func* table);

#endif // SF_OPS_CORE_H