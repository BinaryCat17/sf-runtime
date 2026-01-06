#include <sionflow/engine/sf_engine.h>
#include "sf_engine_internal.h"
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_utils.h>
#include <sionflow/base/sf_shape.h>
#include <string.h>

// --- Helpers ---

int32_t find_resource_idx(sf_engine* engine, u32 name_hash) {
    for (u32 i = 0; i < engine->resource_count; ++i) {
        if (engine->resources[i].name_hash == name_hash) return (int32_t)i;
    }
    return -1;
}

int32_t find_symbol_idx(const sf_program* prog, u32 name_hash) {
    if (!prog || !prog->symbols) return -1;
    for (u32 i = 0; i < prog->meta.symbol_count; ++i) {
        if (prog->symbols[i].name_hash == name_hash) return (int32_t)i;
    }
    return -1;
}

// --- Internal Engine Logic ---

static void _setup_resource_inst(sf_resource_inst* res, const char* name, const char* provider, sf_dtype dtype, const int32_t* shape, uint8_t ndim, uint8_t flags, sf_arena* arena) {
    res->name = sf_arena_strdup(arena, name);
    res->name_hash = sf_fnv1a_hash(res->name);
    res->provider = provider ? sf_arena_strdup(arena, provider) : NULL;
    res->flags = flags;
    
    memset(&res->desc, 0, sizeof(sf_tensor));
    res->desc.info.dtype = dtype;
    res->desc.info.ndim = ndim;
    if (ndim > 0 && shape) memcpy(res->desc.info.shape, shape, sizeof(int32_t) * ndim);
    sf_shape_calc_strides(&res->desc.info);
    
    res->size_bytes = sf_tensor_size_bytes(&res->desc);
    res->buffers[0] = res->buffers[1] = NULL;
}

static void analyze_transience(sf_engine* engine) {
    for (u32 r_idx = 0; r_idx < engine->resource_count; ++r_idx) {
        sf_resource_inst* res = &engine->resources[r_idx];
        if (res->flags & (SF_RESOURCE_FLAG_PERSISTENT | SF_RESOURCE_FLAG_TRANSIENT)) continue;
        
        bool read_before_write = false, write_happened = false;
        for (u32 k_idx = 0; k_idx < engine->kernel_count; ++k_idx) {
            sf_kernel_inst* ker = &engine->kernels[k_idx];
            bool k_reads = false, k_writes = false;
            
            for (u32 b = 0; b < ker->binding_count; ++b) {
                if (ker->bindings[b].global_res == r_idx) {
                    if (ker->bindings[b].flags & SF_SYMBOL_FLAG_INPUT) k_reads = true;
                    if (ker->bindings[b].flags & SF_SYMBOL_FLAG_OUTPUT) k_writes = true;
                }
            }
            if (k_reads && !write_happened) { read_before_write = true; break; }
            if (k_writes) write_happened = true;
        }
        if (!read_before_write && write_happened) res->flags |= SF_RESOURCE_FLAG_TRANSIENT;
    }
}

static void allocate_resources(sf_engine* engine) {
    sf_allocator* alloc = (sf_allocator*)&engine->heap;
    for (u32 i = 0; i < engine->resource_count; ++i) {
        sf_resource_inst* res = &engine->resources[i];
        bool trans = (res->flags & SF_RESOURCE_FLAG_TRANSIENT) != 0;
        
        if (res->size_bytes == 0 && res->desc.info.ndim > 0) {
            res->size_bytes = sf_tensor_size_bytes(&res->desc);
        }

        res->buffers[0] = SF_ARENA_PUSH(&engine->arena, sf_buffer, 1);
        if (res->size_bytes > 0) {
            sf_buffer_alloc(res->buffers[0], alloc, res->size_bytes);
        } else {
            memset(res->buffers[0], 0, sizeof(sf_buffer));
        }

        if (trans) {
            res->buffers[1] = res->buffers[0];
        } else {
            res->buffers[1] = SF_ARENA_PUSH(&engine->arena, sf_buffer, 1);
            if (res->size_bytes > 0) {
                sf_buffer_alloc(res->buffers[1], alloc, res->size_bytes);
            } else {
                memset(res->buffers[1], 0, sizeof(sf_buffer));
            }
        }
    }
}

static void apply_initial_data(sf_engine* engine) {
    for (u32 k = 0; k < engine->kernel_count; ++k) {
        sf_kernel_inst* ker = &engine->kernels[k];
        for (u32 b = 0; b < ker->binding_count; ++b) {
            sf_kernel_binding* bind = &ker->bindings[b];
            void* data = ker->program->tensor_data[bind->local_reg];
            if (data) {
                sf_resource_inst* res = &engine->resources[bind->global_res];
                if (res->size_bytes > 0 && res->buffers[0]->data) {
                    memcpy(res->buffers[0]->data, data, res->size_bytes);
                    if (res->buffers[1] != res->buffers[0]) {
                        memcpy(res->buffers[1]->data, data, res->size_bytes);
                    }
                }
            }
        }
    }
}

static void sf_engine_finalize_setup(sf_engine* engine) {
    analyze_transience(engine);
    allocate_resources(engine);
    apply_initial_data(engine);

    for (u32 k = 0; k < engine->kernel_count; ++k) {
        sf_state_reset(&engine->kernels[k].state, engine->kernels[k].program, &engine->arena, &engine->backend);
    }
}

// --- Public API ---

void sf_engine_bind_cartridge(sf_engine* engine, sf_program** programs, const char** names, uint32_t count) {
    if (!engine || !programs || count == 0) return;

    // 1. Gather unique resources
    u32 total_syms = 0;
    for (u32 k = 0; k < count; ++k) total_syms += programs[k]->meta.symbol_count;
    
    engine->resources = (total_syms > 0) ? SF_ARENA_PUSH(&engine->arena, sf_resource_inst, total_syms) : NULL;
    engine->resource_count = 0;

    for (u32 k = 0; k < count; ++k) {
        sf_program* prog = programs[k];
        for (u32 s = 0; s < prog->meta.symbol_count; ++s) {
            sf_bin_symbol* sym = &prog->symbols[s];
            if (!(sym->flags & (SF_SYMBOL_FLAG_INPUT | SF_SYMBOL_FLAG_OUTPUT))) continue;

            int32_t r_idx = find_resource_idx(engine, sym->name_hash);
            if (r_idx != -1) {
                engine->resources[r_idx].flags |= sym->flags;
                continue;
            }

            sf_type_info* t = &prog->tensor_infos[sym->register_idx];
            _setup_resource_inst(&engine->resources[engine->resource_count++], sym->name, sym->provider[0] ? sym->provider : NULL, t->dtype, t->shape, t->ndim, sym->flags, &engine->arena);
        }
    }

    // 2. Init Kernels
    engine->kernels = SF_ARENA_PUSH(&engine->arena, sf_kernel_inst, count);
    engine->kernel_count = count;
    for (u32 k = 0; k < count; ++k) {
        sf_program* prog = programs[k];
        sf_kernel_inst* inst = &engine->kernels[k];
        inst->id = sf_arena_strdup(&engine->arena, (names && names[k]) ? names[k] : "kernel");
        inst->id_hash = sf_fnv1a_hash(inst->id);
        inst->program = prog;
        inst->frequency = 1;
        inst->state.allocator = (sf_allocator*)&engine->heap;
        
        inst->bindings = (prog->meta.symbol_count > 0) ? SF_ARENA_PUSH(&engine->arena, sf_kernel_binding, prog->meta.symbol_count) : NULL;
        inst->binding_count = 0;
        for (u32 s = 0; s < prog->meta.symbol_count; ++s) {
            sf_bin_symbol* sym = &prog->symbols[s];
            if (!(sym->flags & (SF_SYMBOL_FLAG_INPUT | SF_SYMBOL_FLAG_OUTPUT))) continue;
            int32_t r_idx = find_resource_idx(engine, sym->name_hash);
            if (r_idx != -1) {
                sf_kernel_binding* kb = &inst->bindings[inst->binding_count++];
                kb->local_reg = (u16)sym->register_idx;
                kb->global_res = (u16)r_idx;
                kb->flags = sym->flags;
            }
        }
    }

    sf_engine_finalize_setup(engine);
}

void sf_engine_bind_pipeline(sf_engine* engine, const sf_pipeline_desc* pipe, sf_program** programs) {
    if (!engine || !pipe) return;

    // 1. Init Resources from Desc
    engine->resources = (pipe->resource_count > 0) ? SF_ARENA_PUSH(&engine->arena, sf_resource_inst, pipe->resource_count) : NULL;
    engine->resource_count = pipe->resource_count;
    for (u32 i = 0; i < pipe->resource_count; ++i) {
        sf_pipeline_resource* d = &pipe->resources[i];
        _setup_resource_inst(&engine->resources[i], d->name, d->provider, d->dtype, d->shape, d->ndim, d->flags, &engine->arena);
    }

    // 2. Init Kernels
    engine->kernels = SF_ARENA_PUSH(&engine->arena, sf_kernel_inst, pipe->kernel_count);
    engine->kernel_count = pipe->kernel_count;
    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        sf_pipeline_kernel* d = &pipe->kernels[i];
        sf_kernel_inst* k = &engine->kernels[i];
        k->id = sf_arena_strdup(&engine->arena, d->id);
        k->id_hash = sf_fnv1a_hash(k->id);
        k->program = programs[i];
        k->frequency = d->frequency;
        k->state.allocator = (sf_allocator*)&engine->heap;

        k->bindings = SF_ARENA_PUSH(&engine->arena, sf_kernel_binding, d->binding_count + k->program->meta.symbol_count);
        k->binding_count = 0;
        for (u32 b = 0; b < d->binding_count; ++b) {
            int32_t s_idx = find_symbol_idx(k->program, sf_fnv1a_hash(d->bindings[b].kernel_port));
            int32_t r_idx = find_resource_idx(engine, sf_fnv1a_hash(d->bindings[b].global_resource));
            if (s_idx != -1 && r_idx != -1) {
                sf_kernel_binding* kb = &k->bindings[k->binding_count++];
                kb->local_reg = (u16)k->program->symbols[s_idx].register_idx;
                kb->global_res = (u16)r_idx;
                kb->flags = k->program->symbols[s_idx].flags;
            }
        }
        for (u32 s = 0; s < k->program->meta.symbol_count; ++s) {
            sf_bin_symbol* sym = &k->program->symbols[s];
            if (!(sym->flags & (SF_SYMBOL_FLAG_INPUT | SF_SYMBOL_FLAG_OUTPUT))) continue;
            bool bound = false;
            for (u32 b = 0; b < k->binding_count; ++b) if (k->bindings[b].local_reg == sym->register_idx) bound = true;
            if (bound) continue;
            int32_t r_idx = find_resource_idx(engine, sym->name_hash);
            if (r_idx != -1) {
                sf_kernel_binding* kb = &k->bindings[k->binding_count++];
                kb->local_reg = (u16)sym->register_idx;
                kb->global_res = (u16)r_idx;
                kb->flags = sym->flags;
            }
        }
    }

    sf_engine_finalize_setup(engine);
}
