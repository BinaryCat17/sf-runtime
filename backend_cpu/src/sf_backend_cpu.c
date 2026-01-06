#include <sionflow/backend_cpu/sf_backend_cpu.h>
#include <sionflow/ops/sf_ops_core.h>
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/isa/sf_state.h>
#include <sionflow/isa/sf_exec_ctx.h>
#include <sionflow/base/sf_thread_pool.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_platform.h>
#include <sionflow/base/sf_shape.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdalign.h>

// --- Constants ---

#define SF_CPU_JOB_SIZE         4096         // Elements per job (Linear)
#define SF_CPU_INLINE_THRESHOLD 1024         // If total elements < this, run inline
#define SF_CPU_WORKER_HEAP_SZ   (64*1024*1024) // 64MB per worker

// --- Internal Structures ---

typedef struct {
    const sf_program* program;
    
    // Pre-allocated scratchpads
    f32* reduction_scratch;
    u32 reduction_scratch_size;

    f32* sync_scratch;
    u32 sync_scratch_size; 
} sf_cpu_baked_kernel;

typedef struct {
    sf_thread_pool* pool;
    sf_op_func op_table[SF_OP_LIMIT];
} sf_backend_cpu_state;

typedef struct {
    int thread_idx;
    sf_exec_ctx ctx;
    sf_arena temp_arena; 
    void* heap_mem;
    size_t heap_size;
} sf_backend_cpu_worker_state;

typedef struct {
    const sf_program* program;
    sf_state* main_state;
    sf_op_func* op_table;
    
    const sf_task* current_task;
    uint32_t start_inst;
    uint32_t inst_count;
    
    size_t total_elements;
    u8 ndim;
    u32 domain_shape[SF_MAX_DIMS];
    
    // Parallel Sync Support
    int sync_pass;
    void* sync_data;

    // Parallel Reduction Support
    f32* reduction_scratch; // [num_threads * num_registers]
    u32 reduction_scratch_per_thread;
    int num_threads;
} sf_cpu_parallel_batch;

// --- Worker Lifecycle ---

static void* worker_init(int thread_idx, void* user_data) {
    (void)user_data;
    sf_backend_cpu_worker_state* state = malloc(sizeof(sf_backend_cpu_worker_state));
    if (!state) return NULL;
    state->thread_idx = thread_idx;
    
#ifdef _WIN32
    state->heap_mem = _aligned_malloc(SF_CPU_WORKER_HEAP_SZ, 16);
#else
    state->heap_mem = aligned_alloc(16, SF_CPU_WORKER_HEAP_SZ);
#endif

    if (!state->heap_mem) {
        free(state);
        return NULL;
    }
    state->heap_size = SF_CPU_WORKER_HEAP_SZ;
    sf_arena_init(&state->temp_arena, state->heap_mem, state->heap_size);
    return state;
}

static void worker_cleanup(void* thread_local_data, void* user_data) {
    (void)user_data;
    sf_backend_cpu_worker_state* state = (sf_backend_cpu_worker_state*)thread_local_data;
    if (!state) return;
#ifdef _WIN32
    _aligned_free(state->heap_mem);
#else
    free(state->heap_mem);
#endif
    free(state);
}

// --- Execution Logic ---

static const char* _dtype_to_str(sf_dtype type) {
    switch(type) {
        case SF_DTYPE_F32: return "F32";
        case SF_DTYPE_I32: return "I32";
        case SF_DTYPE_U8:  return "U8";
        default: return "UNK";
    }
}

static const char* find_reg_name(const sf_program* prog, u32 reg_idx) {
    if (!prog || !prog->symbols) return "temp";
    for (u32 i = 0; i < prog->meta.symbol_count; ++i) {
        if (prog->symbols[i].register_idx == reg_idx) return prog->symbols[i].name;
    }
    return "temp";
}

static void format_tensor_debug(char* buf, const sf_exec_ctx* ctx, int reg_idx, const sf_program* prog, const char* port_name) {
    if (reg_idx < 0 || reg_idx >= SF_MAX_REGISTERS) {
        sprintf(buf, "Reg %-2d (INVALID)", reg_idx);
        return;
    }
    
    const char* name = find_reg_name(prog, reg_idx);
    const sf_type_info* info = &ctx->reg_info[reg_idx];
    void* data = ctx->reg_ptrs[reg_idx];

    char shape_str[64] = {0};
    int pos = 0;
    if (info->ndim == 0) strcpy(shape_str, "Scalar");
    else {
        for (int i = 0; i < info->ndim; ++i) pos += sprintf(shape_str + pos, "%d%s", info->shape[i], (i < info->ndim - 1) ? "," : "");
    }

    char tag[128];
    if (port_name) sprintf(tag, "Reg %-2d (%s) [%s]", reg_idx, name, port_name);
    else sprintf(tag, "Reg %-2d (%s)", reg_idx, name);

    if (!data) {
        sprintf(buf, "%-30s : <NULL PTR> [%s] Shape: [%s]", tag, _dtype_to_str(info->dtype), shape_str);
        return;
    }

    if (info->ndim == 0 || (info->ndim == 1 && info->shape[0] == 1)) {
        float val = 0;
        if (info->dtype == SF_DTYPE_F32) val = *(f32*)data;
        else if (info->dtype == SF_DTYPE_I32) val = (f32)*(int32_t*)data;
        else if (info->dtype == SF_DTYPE_U8) val = (f32)*(u8*)data;
        sprintf(buf, "%-30s : Value: %-10.3f (%s)", tag, val, _dtype_to_str(info->dtype));
    } else {
        sprintf(buf, "%-30s : Tensor[%-10s] (%s) Ptr: %p", tag, shape_str, _dtype_to_str(info->dtype), data);
    }
}

static void report_crash(sf_exec_ctx* ctx, const sf_cpu_parallel_batch* batch, u32 inst_idx) {
    const sf_instruction* inst = &batch->program->code[inst_idx];
    const sf_runtime_op_metadata* meta = sf_get_op_metadata(inst->opcode);

    char coords[128] = {0};
    int pos = 0;
    
    u32 exact_linear = ctx->linear_offset + ctx->error_idx;
    u32 temp_idx = exact_linear;
    u32 exact_coords[SF_MAX_DIMS];
    for (int i = (int)ctx->ndim - 1; i >= 0; --i) {
        exact_coords[i] = temp_idx % ctx->domain_shape[i];
        temp_idx /= ctx->domain_shape[i];
    }
    for (int d = 0; d < ctx->ndim; ++d) pos += sprintf(coords + pos, "%u%s", exact_coords[d], (d < ctx->ndim - 1) ? ", " : "");

    char s1_info[128], s2_info[128], s3_info[128], s4_info[128], d_info[128];
    format_tensor_debug(d_info,  ctx, inst->dest_idx, batch->program, "out");
    format_tensor_debug(s1_info, ctx, inst->src1_idx, batch->program, meta ? meta->ports[0] : "src1");
    format_tensor_debug(s2_info, ctx, inst->src2_idx, batch->program, meta ? meta->ports[1] : "src2");
    format_tensor_debug(s3_info, ctx, inst->src3_idx, batch->program, meta ? meta->ports[2] : "src3");
    format_tensor_debug(s4_info, ctx, inst->src4_idx, batch->program, meta ? meta->ports[3] : "src4");

    SF_LOG_FATAL("\nKERNEL CRASH #%u Opcode: %s\nDest: %s\nSrc1: %s\nSrc2: %s\nSrc3: %s\nSrc4: %s\nCoord: [%s] Error: %s\n",
                 inst_idx, sf_opcode_to_str(inst->opcode), d_info, s1_info, s2_info, s3_info, s4_info,
                 coords, sf_exec_error_to_str(ctx->error));
}

static inline void sf_cpu_exec(sf_exec_ctx* ctx, const sf_cpu_parallel_batch* batch, u32 count) {
    for (uint32_t i = 0; i < count; ++i) {
        if (ctx->error != SF_ERROR_NONE) break;
        if (batch->main_state && sf_atomic_load((sf_atomic_i32*)&batch->main_state->error_code) != 0) break;

        u32 inst_idx = batch->start_inst + i;
        const sf_instruction* inst = &batch->program->code[inst_idx];
        
        sf_op_func op = batch->op_table[inst->opcode];
        if (op) {
            op(ctx, inst);
            if (ctx->error != SF_ERROR_NONE) { report_crash(ctx, batch, inst_idx); break; }
        }
    }
}

static void prepare_registers(sf_backend_cpu_worker_state* state, const sf_cpu_parallel_batch* batch, size_t start_idx, size_t count) {
    sf_exec_ctx* ctx = &state->ctx;
    int tid = state->thread_idx;
    const sf_task* task = batch->current_task;
    const sf_program* prog = batch->program;

    for (u32 b = 0; b < task->binding_count; ++b) {
        const sf_bin_task_binding* bind = &prog->bindings[task->binding_offset + b];
        u16 i = bind->reg_idx;
        
        ctx->reg_info[i] = prog->tensor_infos[i];
        
        // For dynamic resources (aliased), we must update the info to match the bound resource
        sf_tensor* t = &batch->main_state->registers[i];
        uint8_t flags = prog->tensor_flags[i];
        
        if (flags & SF_TENSOR_FLAG_ALIAS) {
            ctx->reg_info[i] = t->info;
        }

        // DYNAMIC STRIDE CALCULATION
        // The baked bind->byte_stride is only valid if shapes haven't changed.
        // We recalculate it based on current info and batch total elements.
        size_t reg_elements = sf_shape_calc_count(ctx->reg_info[i].shape, ctx->reg_info[i].ndim);
        i32 elem_stride = sf_shape_calc_linear_stride(reg_elements, batch->total_elements);
        ctx->reg_strides[i] = elem_stride * (i32)sf_dtype_size(ctx->reg_info[i].dtype);

        if (batch->reduction_scratch && (bind->flags & SF_BINDING_FLAG_REDUCTION)) {
            ctx->reg_ptrs[i] = &batch->reduction_scratch[tid * batch->reduction_scratch_per_thread + i];
            ctx->reg_strides[i] = 0;
            continue;
        }

        // Buffer-based (Symbol, Constant, or Scratch)
        if (t->buffer && t->buffer->data) {
            ctx->reg_ptrs[i] = (u8*)t->buffer->data + t->byte_offset + (start_idx * ctx->reg_strides[i]);
        } else {
            ctx->reg_ptrs[i] = NULL;
            if (ctx->error == SF_ERROR_NONE) {
                SF_LOG_ERROR("Backend: Reg %u (%s) has NULL buffer data (Flags: 0x%X)", i, find_reg_name(prog, i), flags);
                ctx->error = SF_ERROR_RUNTIME;
            }
        }
    }
}

static void cpu_worker_job(u32 job_idx, void* thread_local_data, void* user_data) {
    sf_backend_cpu_worker_state* state = (sf_backend_cpu_worker_state*)thread_local_data;
    sf_cpu_parallel_batch* batch = (sf_cpu_parallel_batch*)user_data;
    size_t start_idx = (size_t)job_idx * SF_CPU_JOB_SIZE;
    size_t count = SF_CPU_JOB_SIZE;
    if (start_idx + count > batch->total_elements) count = batch->total_elements - start_idx;
    if (count == 0) return;
    
    sf_arena_reset(&state->temp_arena);
    sf_exec_ctx_init(&state->ctx, (sf_allocator*)&state->temp_arena);
    
    state->ctx.batch_size = (u32)count;
    state->ctx.ndim = batch->ndim; 
    if (batch->main_state) state->ctx.global_error_ptr = batch->main_state->global_error_ptr ? batch->main_state->global_error_ptr : &batch->main_state->error_code;
    state->ctx.linear_offset = (u32)start_idx;
    state->ctx.job_idx = job_idx;
    state->ctx.sync_pass = batch->sync_pass;
    state->ctx.sync_data = batch->sync_data;

    // Coordinate decomposition
    if (batch->ndim > 1) {
        size_t temp_idx = start_idx;
        for (int i = batch->ndim - 1; i >= 0; --i) {
            state->ctx.tile_offset[i] = (u32)(temp_idx % batch->domain_shape[i]);
            temp_idx /= batch->domain_shape[i];
        }
    } else {
        state->ctx.tile_offset[0] = (u32)start_idx;
        for (int i = 1; i < SF_MAX_DIMS; ++i) state->ctx.tile_offset[i] = 0;
    }
    
    for(int d=0; d<batch->ndim; ++d) state->ctx.domain_shape[d] = batch->domain_shape[d];
    
    prepare_registers(state, batch, start_idx, count);
    sf_cpu_exec(&state->ctx, batch, batch->inst_count);
    
    if (state->ctx.error != SF_ERROR_NONE && batch->main_state) {
        sf_atomic_store(&batch->main_state->error_code, (int32_t)state->ctx.error);
    }
}

static void sf_backend_cpu_dispatch_batch(sf_backend_cpu_state* state, sf_cpu_parallel_batch* batch, const sf_task* task) {
    if (task->inst_count == 0) return;
    batch->current_task = task;
    batch->start_inst = task->start_inst;
    batch->inst_count = task->inst_count;
    u32 total_jobs = (u32)((batch->total_elements + SF_CPU_JOB_SIZE - 1) / SF_CPU_JOB_SIZE);
    if (batch->total_elements <= SF_CPU_INLINE_THRESHOLD || total_jobs == 1) {
        sf_backend_cpu_worker_state local_worker;
        _Alignas(16) u8 local_heap[SF_MB(4)]; 
        local_worker.thread_idx = 0; local_worker.heap_mem = local_heap; local_worker.heap_size = sizeof(local_heap);
        sf_arena_init(&local_worker.temp_arena, local_worker.heap_mem, local_worker.heap_size);
        cpu_worker_job(0, &local_worker, batch);
    } else if (state->pool) sf_thread_pool_run(state->pool, total_jobs, cpu_worker_job, batch);
}

static void* sf_backend_cpu_bake(void* backend_state, const struct sf_program* program) {
    sf_backend_cpu_state* state = (sf_backend_cpu_state*)backend_state;
    sf_cpu_baked_kernel* baked = calloc(1, sizeof(sf_cpu_baked_kernel));
    baked->program = program;

    // Scratchpad allocation
    int num_threads = state->pool ? sf_thread_pool_get_thread_count(state->pool) : 1;
    if (program->meta.reduction_scratch_size > 0 && num_threads > 1) {
        baked->reduction_scratch_size = num_threads * program->meta.reduction_scratch_size;
        baked->reduction_scratch = calloc(baked->reduction_scratch_size, sizeof(f32));
    }

    if (program->meta.sync_scratch_size > 0) {
        baked->sync_scratch_size = program->meta.sync_scratch_size;
        baked->sync_scratch = calloc(baked->sync_scratch_size, sizeof(f32));
    }

    return baked;
}

static void sf_backend_cpu_free_baked(void* backend_state, void* baked_data) {
    (void)backend_state;
    sf_cpu_baked_kernel* baked = (sf_cpu_baked_kernel*)baked_data;
    if (baked) {
        if (baked->reduction_scratch) free(baked->reduction_scratch);
        if (baked->sync_scratch) free(baked->sync_scratch);
        free(baked);
    }
}

static void sf_backend_cpu_dispatch(void* backend_state, const struct sf_program* program, struct sf_state* main_state, const sf_tensor* domain, uint32_t start_inst, uint32_t inst_count) {
    sf_backend_cpu_state* state = (sf_backend_cpu_state*)backend_state;
    sf_cpu_baked_kernel* baked = (sf_cpu_baked_kernel*)main_state->baked_data;
    if (!domain || !baked) return;
    size_t total_elements = sf_tensor_count(domain);
    if (total_elements == 0) return;
    int num_threads = state->pool ? sf_thread_pool_get_thread_count(state->pool) : 1;
    sf_cpu_parallel_batch batch = {
        .program = program, .main_state = main_state, .op_table = state->op_table,
        .total_elements = total_elements, .ndim = domain->info.ndim, .num_threads = num_threads,
        .reduction_scratch = baked->reduction_scratch, .reduction_scratch_per_thread = program->meta.reduction_scratch_size
    };
    memcpy(batch.domain_shape, domain->info.shape, sizeof(u32) * SF_MAX_DIMS);
    
    // Find the task that matches this instruction range
    const sf_task* target_task = NULL;
    for (u32 s = 0; s < program->meta.task_count; ++s) {
        if (program->tasks[s].start_inst == start_inst) {
            target_task = &program->tasks[s];
            break;
        }
    }

    if (!target_task) {
        SF_LOG_ERROR("Backend: Could not find task starting at %u", start_inst);
        return;
    }

    if (batch.reduction_scratch && (target_task->strategy == SF_STRATEGY_REDUCTION)) {
        memset(batch.reduction_scratch, 0, baked->reduction_scratch_size * sizeof(f32));
    }

    if (target_task->strategy == SF_STRATEGY_TWO_PASS_SYNC) {
        u32 total_jobs = (u32)((batch.total_elements + SF_CPU_JOB_SIZE - 1) / SF_CPU_JOB_SIZE);
        f32* sync_ptr = baked->sync_scratch;
        if (total_jobs > baked->sync_scratch_size) sync_ptr = calloc(total_jobs, sizeof(f32));
        batch.sync_pass = 0; batch.sync_data = sync_ptr;
        sf_backend_cpu_dispatch_batch(state, &batch, target_task);
        f32 global_acc = 0;
        for (u32 j = 0; j < total_jobs; ++j) { f32 chunk_total = sync_ptr[j]; sync_ptr[j] = global_acc; global_acc += chunk_total; }
        batch.sync_pass = 1;
        sf_backend_cpu_dispatch_batch(state, &batch, target_task);
        if (sync_ptr != baked->sync_scratch) free(sync_ptr);
    } else {
        sf_backend_cpu_dispatch_batch(state, &batch, target_task);
    }

    if (batch.reduction_scratch && (target_task->strategy == SF_STRATEGY_REDUCTION)) {
        for (u32 i = 0; i < program->meta.tensor_count; ++i) {
            if (program->tensor_flags[i] & SF_TENSOR_FLAG_REDUCTION) {
                f32 final_val = 0;
                for (int t = 0; t < num_threads; ++t) final_val += batch.reduction_scratch[t * batch.reduction_scratch_per_thread + i];
                sf_tensor* main_t = &main_state->registers[i];
                if (main_t->buffer && main_t->buffer->data) {
                    *((f32*)main_t->buffer->data + main_t->byte_offset / sizeof(f32)) = final_val;
                }
            }
        }
    }
}

static void sf_backend_cpu_shutdown(void* backend_state) {
    sf_backend_cpu_state* state = (sf_backend_cpu_state*)backend_state;
    if (!state) return;
    if (state->pool) sf_thread_pool_destroy(state->pool);
    free(state);
}

void sf_backend_cpu_init(sf_backend* backend, int num_threads) {
    memset(backend, 0, sizeof(sf_backend));
    sf_backend_cpu_state* state = calloc(1, sizeof(sf_backend_cpu_state));
    sf_thread_pool_desc pool_desc = { .num_threads = num_threads, .init_fn = worker_init, .cleanup_fn = worker_cleanup };
    state->pool = sf_thread_pool_create(&pool_desc);
    sf_ops_fill_table(state->op_table);
    backend->state = state; backend->bake = sf_backend_cpu_bake;
    backend->free_baked = sf_backend_cpu_free_baked; backend->shutdown = sf_backend_cpu_shutdown;
    backend->dispatch = sf_backend_cpu_dispatch;
}