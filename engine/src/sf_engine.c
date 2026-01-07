#include <sionflow/engine/sf_engine.h>
#include "sf_engine_internal.h"
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_shape.h>
#include <sionflow/base/sf_utils.h>
#include <sionflow/isa/sf_exec_ctx.h>
#include <string.h>
#include <stdio.h>

// --- Internal State Management ---

void sf_state_reset(sf_state* state, const sf_program* prog, sf_arena* arena, sf_backend* backend) {
    if (!prog) return;
    
    state->register_count = prog->meta.tensor_count;
    
    // Consolidate allocations: registers + ownership flags + task_strides
    size_t regs_sz = sizeof(sf_tensor) * state->register_count;
    size_t flags_sz = sizeof(uint8_t) * state->register_count;
    size_t strides_sz = sizeof(int32_t) * state->register_count;
    u8* block = SF_ARENA_PUSH(arena, u8, regs_sz + flags_sz + strides_sz);
    
    if (!block) {
        SF_LOG_ERROR("Engine: Failed to allocate registers for kernel state. Arena OOM.");
        state->register_count = 0;
        state->registers = NULL;
        state->ownership_flags = NULL;
        state->task_strides = NULL;
        return;
    }
    
    state->registers = (sf_tensor*)block;
    state->ownership_flags = (uint8_t*)(block + regs_sz);
    state->task_strides = (int32_t*)(block + regs_sz + flags_sz);
    
    memset(state->ownership_flags, 0, state->register_count);
    memset(state->task_strides, 0, strides_sz);

    for (u32 i = 0; i < state->register_count; ++i) {
        sf_type_info* info_prog = &prog->tensor_infos[i];
        void* data_prog = prog->tensor_data[i];
        sf_tensor* t_reg = &state->registers[i];
        uint8_t flags = prog->tensor_flags[i];
        
        t_reg->info = *info_prog;
        t_reg->byte_offset = 0;
        t_reg->buffer = NULL;

        if (data_prog) {
            t_reg->buffer = state->allocator->alloc(state->allocator, sizeof(sf_buffer));
            sf_buffer_init_view(t_reg->buffer, data_prog, sf_shape_calc_bytes(info_prog->dtype, info_prog->shape, info_prog->ndim));
            state->ownership_flags[i] = 1; // Mark for cleanup
        } else {
            // Pre-allocate only non-alias, non-generator static tensors
            if (!(flags & SF_TENSOR_FLAG_ALIAS) && !(flags & SF_TENSOR_FLAG_GENERATOR)) {
                bool is_static = true;
                for (int d = 0; d < t_reg->info.ndim; ++d) if (t_reg->info.shape[d] < 0) { is_static = false; break; }
                if (is_static) {
                    t_reg->buffer = state->allocator->alloc(state->allocator, sizeof(sf_buffer));
                    if (sf_tensor_alloc(t_reg, state->allocator, &t_reg->info)) {
                        state->ownership_flags[i] = 1;
                    } else {
                        state->allocator->free(state->allocator, t_reg->buffer);
                        t_reg->buffer = NULL;
                    }
                }
            }
        }
    }

    // --- BAKING PHASE ---
    if (backend && backend->bake) {
        state->baked_data = backend->bake(backend->state, prog);
    }
}

static void sf_state_shutdown(sf_state* state, sf_backend* backend) {
    if (!state->registers || !state->allocator) return;
    
    if (backend && backend->free_baked && state->baked_data) {
        backend->free_baked(backend->state, state->baked_data);
        state->baked_data = NULL;
    }

    for (u32 i = 0; i < state->register_count; ++i) {
        if (state->ownership_flags && state->ownership_flags[i]) {
            sf_tensor* t = &state->registers[i];
            if (t->buffer) {
                sf_buffer_free(t->buffer); 
                state->allocator->free(state->allocator, t->buffer);
                t->buffer = NULL;
            }
        }
    }
}

// --- Engine API ---

sf_engine* sf_engine_create(const sf_engine_desc* desc) {
    SF_LOG_INFO("Creating Engine...");
    sf_engine* engine = calloc(1, sizeof(sf_engine));
    if (!engine) return NULL;

    size_t arena_size = (desc && desc->arena_size > 0) ? desc->arena_size : SF_MB(8);
    size_t heap_size = (desc && desc->heap_size > 0) ? desc->heap_size : SF_MB(64);

    engine->arena_buffer = malloc(arena_size);
    if (!engine->arena_buffer) { free(engine); return NULL; }
    sf_arena_init(&engine->arena, engine->arena_buffer, arena_size);

    engine->heap_buffer = malloc(heap_size);
    if (!engine->heap_buffer) { free(engine->arena_buffer); free(engine); return NULL; }
    sf_heap_init(&engine->heap, engine->heap_buffer, heap_size);

    if (desc) engine->backend = desc->backend;

    engine->front_idx = 0;
    engine->back_idx = 1;
    sf_atomic_store(&engine->error_code, 0);

    return engine;
}

void sf_engine_destroy(sf_engine* engine) {
    if (!engine) return;
    sf_engine_reset(engine);
    if (engine->heap_buffer) free(engine->heap_buffer);
    if (engine->arena_buffer) free(engine->arena_buffer);
    free(engine);
}

void sf_engine_reset(sf_engine* engine) {

    if (!engine) return;

    for (u32 i = 0; i < engine->kernel_count; ++i) {

        sf_state_shutdown(&engine->kernels[i].state, &engine->backend);

    }

    for (u32 i = 0; i < engine->resource_count; ++i) {




        if (engine->resources[i].buffers[0]) sf_buffer_free(engine->resources[i].buffers[0]);
        if (engine->resources[i].buffers[1] && engine->resources[i].buffers[1] != engine->resources[i].buffers[0]) {
            sf_buffer_free(engine->resources[i].buffers[1]);
        }
    }
    sf_arena_reset(&engine->arena);
    if (engine->heap_buffer) sf_heap_init(&engine->heap, engine->heap_buffer, engine->heap.size);
    engine->kernel_count = 0;
    engine->resource_count = 0;
    sf_atomic_store(&engine->error_code, 0);
}

sf_arena* sf_engine_get_arena(sf_engine* engine) {
    return engine ? &engine->arena : NULL;
}

void sf_engine_dispatch(sf_engine* engine) {
    if (!engine || sf_atomic_load(&engine->error_code) != 0) return;

    u8 front = engine->front_idx;
    u8 back  = engine->back_idx;

    for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
        sf_kernel_inst* ker = &engine->kernels[k_idx];
        if (sf_atomic_load(&engine->error_code) != 0) break;
        
        // 1. Resource Binding
        for (u32 b = 0; b < ker->binding_count; ++b) {
            sf_kernel_binding* bind = &ker->bindings[b];
            sf_resource_inst* res = &engine->resources[bind->global_res];
            sf_tensor* t = &ker->state.registers[bind->local_reg];
            *t = res->desc;
            t->buffer = (bind->flags & SF_SYMBOL_FLAG_OUTPUT) ? res->buffers[back] : res->buffers[front];
            t->byte_offset = 0;
        }
        
        // 3. Execution
        for (u32 f = 0; f < ker->frequency; ++f) {
            if (engine->backend.dispatch) {
                ker->state.global_error_ptr = &engine->error_code;
                for (u32 t = 0; t < ker->program->meta.task_count; ++t) {
                    const sf_task* task = &ker->program->tasks[t];
                    const sf_tensor* task_domain = &ker->state.registers[task->domain_reg];
                    size_t domain_elements = sf_tensor_count(task_domain);

                    // Pre-calculate strides for this task
                    for (u32 b = 0; b < task->binding_count; ++b) {
                        u16 reg_idx = ker->program->bindings[task->binding_offset + b].reg_idx;
                        sf_tensor* reg = &ker->state.registers[reg_idx];
                        size_t reg_elements = sf_tensor_count(reg);
                        i32 elem_stride = sf_shape_calc_linear_stride(reg_elements, domain_elements);
                        ker->state.task_strides[reg_idx] = elem_stride * (i32)sf_dtype_size(reg->info.dtype);
                    }

                    engine->backend.dispatch(engine->backend.state, ker->program, &ker->state, task_domain, task);
                    if (sf_atomic_load(&engine->error_code) != 0) goto end_dispatch;
                }
            }
        }
    }
    
end_dispatch:
    engine->frame_index++;
    engine->front_idx = 1 - engine->front_idx;
    engine->back_idx  = 1 - engine->back_idx;
}

sf_tensor* sf_engine_map_resource(sf_engine* engine, const char* name) {
    if (!engine) return NULL;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (strcmp(engine->resources[i].name, name) == 0) {
            sf_resource_inst* res = &engine->resources[i];
            res->desc.buffer = res->buffers[engine->front_idx];
            res->desc.byte_offset = 0;
            return &res->desc;
        }
    }
    return NULL;
}

bool sf_engine_resize_resource(sf_engine* engine, const char* name, const int32_t* new_shape, uint8_t new_ndim) {
    if (!engine || !name) return false;
    u32 hash = sf_fnv1a_hash(name);
    int32_t res_idx = find_resource_idx(engine, hash);
    if (res_idx == -1) {
        SF_LOG_ERROR("Engine: Cannot resize resource '%s' - not found.", name);
        return false;
    }

    sf_resource_inst* res = &engine->resources[res_idx];
    sf_allocator* alloc = (sf_allocator*)&engine->heap;
    
    sf_type_info new_info;
    sf_type_info_init_contiguous(&new_info, res->desc.info.dtype, new_shape, new_ndim);
    size_t new_bytes = sf_shape_calc_count(new_shape, new_ndim) * sf_dtype_size(new_info.dtype);
    
    if (res->size_bytes != new_bytes) {
        bool is_transient = (res->buffers[0] == res->buffers[1]);
        if (res->buffers[0] && res->buffers[0]->data) sf_buffer_free(res->buffers[0]);
        if (!sf_buffer_alloc(res->buffers[0], alloc, new_bytes)) return false;
        
        if (is_transient) res->buffers[1] = res->buffers[0];
        else {
            if (res->buffers[1] && res->buffers[1]->data) sf_buffer_free(res->buffers[1]);
            if (!sf_buffer_alloc(res->buffers[1], alloc, new_bytes)) return false;
        }
        res->size_bytes = new_bytes;
    }
    res->desc.info = new_info;
    return true;
}

void sf_engine_sync_resource(sf_engine* engine, const char* name) {
    if (!engine || !name) return;
    u32 hash = sf_fnv1a_hash(name);
    int32_t idx = find_resource_idx(engine, hash);
    if (idx == -1) return;
    sf_resource_inst* res = &engine->resources[idx];
    if (res->buffers[0] && res->buffers[1] && res->buffers[0] != res->buffers[1]) {
        if (res->buffers[0]->data && res->buffers[1]->data) {
            memcpy(res->buffers[1 - engine->front_idx]->data, res->buffers[engine->front_idx]->data, res->size_bytes);
        }
    }
}

sf_engine_error sf_engine_get_error(sf_engine* engine) {
    if (!engine) return SF_ENGINE_ERR_NONE;
    int32_t err = sf_atomic_load(&engine->error_code);
    if (err == 0) return SF_ENGINE_ERR_NONE;
    if (err == SF_ERROR_OOM) return SF_ENGINE_ERR_OOM;
    if (err == SF_ERROR_SHAPE_MISMATCH) return SF_ENGINE_ERR_SHAPE;
    if (err == SF_ERROR_INVALID_OP) return SF_ENGINE_ERR_INVALID_OP;
    return SF_ENGINE_ERR_RUNTIME;
}

void sf_engine_iterate_resources(sf_engine* engine, sf_engine_resource_cb cb, void* user_data) {

    if (!engine || !cb) return;

    for (u32 i = 0; i < engine->resource_count; ++i) {

        sf_resource_inst* res = &engine->resources[i];

        res->desc.buffer = res->buffers[engine->front_idx];

        res->desc.byte_offset = 0;

        cb(res->name, &res->desc, user_data);

    }

}


