#include <sionflow/ops/sf_ops_core.h>
#include "sf_ops_internal.h"
#include <sionflow/isa/sf_opcodes.h>
#include <string.h>

/**
 * SionFlow Kernel Registration Hub
 * Uses X-Macros to automatically fill the opcode dispatch table.
 */

// No-operation kernel
void op_NOOP(sf_exec_ctx* ctx, const struct sf_instruction* inst) {
    (void)ctx; (void)inst;
}

// Forward declarations for all kernels defined in other modules
#define SF_OPCODE(suffix, value) extern void op_##suffix(sf_exec_ctx* ctx, const struct sf_instruction* inst);
SF_OPCODE_LIST
#undef SF_OPCODE

void sf_ops_fill_table(sf_op_func* table) {
    if (!table) return;
    memset(table, 0, sizeof(sf_op_func) * SF_OP_LIMIT);
    
#define SF_OPCODE(suffix, value) table[SF_OP_##suffix] = op_##suffix;
    SF_OPCODE_LIST
#undef SF_OPCODE
}