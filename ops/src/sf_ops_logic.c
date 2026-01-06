#include <sionflow/ops/sf_ops_core.h>
#include "sf_kernel_utils.h"
#include <sionflow/isa/sf_opcodes.h>
#include "sf_ops_internal.h"
#include <string.h>

/**
 * SionFlow Logic Kernels
 * Automatically generated from sf_ops_db.inc
 */

#define SF_GEN_AUTO(_op, _ke, _ar) SF_KERNEL_AUTO(_op, _ke, _ar)
#define SF_GEN_MANUAL(...)

#define SF_OP(_s, _n, _op, _cat, _strat, _in, _out, _tr, _sr, _ar, _p1, _p2, _p3, _p4, _kt, _ke, _arity) \
    SF_GEN_##_kt(_op, _ke, _arity)

SF_OP_LIST

#undef SF_OP
#undef SF_GEN_AUTO
#undef SF_GEN_MANUAL
