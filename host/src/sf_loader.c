#include "sf_loader.h"
#include <sionflow/compiler/sf_compiler.h>
#include <sionflow/engine/sf_engine.h>
#include <sionflow/backend_cpu/sf_backend_cpu.h>
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/host/sf_host_desc.h>
#include <sionflow/base/sf_json.h>
#include <sionflow/base/sf_shape.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_utils.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void sf_loader_init_backend(sf_backend* backend, int num_threads) {
    if (!backend) return;
    sf_backend_cpu_init(backend, num_threads);
}

static sf_program* _load_program_from_mem(const u8* data, size_t len, sf_arena* arena) {
    if (len < sizeof(sf_bin_header)) return NULL;
    
    sf_bin_header* head = (sf_bin_header*)data;
    sf_program* prog = SF_ARENA_PUSH(arena, sf_program, 1);
    prog->meta = *head;
    size_t offset = sizeof(sf_bin_header);
    
    // 1. Code
    prog->code = SF_ARENA_PUSH(arena, sf_instruction, head->instruction_count);
    memcpy(prog->code, data + offset, sizeof(sf_instruction) * head->instruction_count);
    offset += sizeof(sf_instruction) * head->instruction_count;

    // 2. Symbols
    if (head->symbol_count > 0) {
        prog->symbols = SF_ARENA_PUSH(arena, sf_bin_symbol, head->symbol_count);
        memcpy(prog->symbols, data + offset, sizeof(sf_bin_symbol) * head->symbol_count);
        offset += sizeof(sf_bin_symbol) * head->symbol_count;
    } else prog->symbols = NULL;

    // 3. Tasks
    if (head->task_count > 0) {
        prog->tasks = SF_ARENA_PUSH(arena, sf_task, head->task_count);
        memcpy(prog->tasks, data + offset, sizeof(sf_task) * head->task_count);
        offset += sizeof(sf_task) * head->task_count;
    } else prog->tasks = NULL;

    // 4. Task Bindings
    if (head->binding_count > 0) {
        prog->bindings = SF_ARENA_PUSH(arena, sf_bin_task_binding, head->binding_count);
        memcpy(prog->bindings, data + offset, sizeof(sf_bin_task_binding) * head->binding_count);
        offset += sizeof(sf_bin_task_binding) * head->binding_count;
    } else prog->bindings = NULL;

    // 5. Tensor Descriptors
    sf_bin_tensor_desc* descs = (sf_bin_tensor_desc*)(data + offset);
    offset += sizeof(sf_bin_tensor_desc) * head->tensor_count;

    size_t n = head->tensor_count;
    size_t sz_info  = sizeof(sf_type_info) * n;
    size_t sz_data  = sizeof(void*) * n;
    size_t sz_bid   = sizeof(uint8_t) * n;
    size_t sz_axis  = sizeof(uint8_t) * n;
    size_t sz_flags = sizeof(uint8_t) * n;
    
    u8* block = SF_ARENA_PUSH(arena, u8, sz_info + sz_data + sz_bid + sz_axis + sz_flags);
    
    prog->tensor_infos = (sf_type_info*)block;
    prog->tensor_data  = (void**)(block + sz_info);
    prog->builtin_ids  = (uint8_t*)(block + sz_info + sz_data);
    prog->builtin_axes = (uint8_t*)(block + sz_info + sz_data + sz_bid);
    prog->tensor_flags = (uint8_t*)(block + sz_info + sz_data + sz_bid + sz_axis);
    
    for (u32 i = 0; i < n; ++i) {
        sf_bin_tensor_desc* d = &descs[i];
        sf_type_info_init_contiguous(&prog->tensor_infos[i], (sf_dtype)d->dtype, d->shape, d->ndim);
        prog->builtin_ids[i] = d->builtin_id;
        prog->builtin_axes[i] = d->builtin_axis;
        prog->tensor_flags[i] = d->flags;
    }

    // 6. Constant Data
    for (u32 i = 0; i < head->tensor_count; ++i) {
        if (descs[i].is_constant) {
            size_t bytes = sf_shape_calc_bytes(prog->tensor_infos[i].dtype, prog->tensor_infos[i].shape, prog->tensor_infos[i].ndim);
            void* mem = SF_ARENA_PUSH(arena, u8, bytes);
            memcpy(mem, data + offset, bytes);
            prog->tensor_data[i] = mem;
            offset += bytes;
        } else prog->tensor_data[i] = NULL;
    }

    return prog;
}

sf_cartridge* sf_cartridge_open(const char* path) {
    size_t size = 0;
    void* data = sf_file_read_bin(path, &size);
    if (!data) return NULL;

    if (size < sizeof(sf_cartridge_header)) {
        free(data);
        return NULL;
    }

    sf_cartridge_header* head = (sf_cartridge_header*)data;
    if (head->magic != SF_BINARY_MAGIC) {
        free(data);
        return NULL;
    }

    sf_cartridge* cart = malloc(sizeof(sf_cartridge));
    cart->data = data;
    cart->size = size;
    cart->header = *head;
    return cart;
}

void sf_cartridge_close(sf_cartridge* cart) {
    if (!cart) return;
    free(cart->data);
    free(cart);
}

void* sf_cartridge_get_section(sf_cartridge* cart, const char* name, sf_section_type type, size_t* out_size) {
    if (!cart) return NULL;

    for (u32 i = 0; i < cart->header.section_count; ++i) {
        sf_section_header* s = &cart->header.sections[i];
        if (s->type == (u32)type && strcmp(s->name, name) == 0) {
            if (s->offset + s->size > cart->size) return NULL;
            if (out_size) *out_size = s->size;
            return (u8*)cart->data + s->offset;
        }
    }
    return NULL;
}

int sf_app_load_config(const char* path, sf_host_desc* out_desc) {
    if (!path || !out_desc) return -1;
    
    // Initialize arena for descriptor strings and arrays
    out_desc->arena_backing = malloc(SF_KB(128));
    sf_arena_init(&out_desc->arena, out_desc->arena_backing, SF_KB(128));
    sf_arena* arena = &out_desc->arena;

    const char* ext = sf_path_get_ext(path);
    bool is_bin = (strcmp(ext, "bin") == 0 || strcmp(ext, "sfc") == 0);

    if (is_bin) {
        sf_cartridge* cart = sf_cartridge_open(path);
        if (!cart) { sf_host_desc_cleanup(out_desc); return -1; }

        out_desc->window_title = sf_arena_strdup(arena, cart->header.app_title[0] ? cart->header.app_title : "SionFlow App");
        out_desc->width = cart->header.window_width ? (int)cart->header.window_width : 800;
        out_desc->height = cart->header.window_height ? (int)cart->header.window_height : 600;
        out_desc->resizable = cart->header.resizable;
        out_desc->vsync = cart->header.vsync;
        out_desc->fullscreen = cart->header.fullscreen;
        out_desc->num_threads = (int)cart->header.num_threads;
        out_desc->has_pipeline = true;
        
        u32 prog_count = 0;
        for (u32 i = 0; i < cart->header.section_count; ++i) {
            if (cart->header.sections[i].type == SF_SECTION_PROGRAM) prog_count++;
        }

        out_desc->pipeline.kernel_count = prog_count;
        out_desc->pipeline.kernels = SF_ARENA_PUSH(arena, sf_pipeline_kernel, prog_count);
        u32 current_prog = 0;
        for (u32 i = 0; i < cart->header.section_count; ++i) {
            if (cart->header.sections[i].type == SF_SECTION_PROGRAM) {
                out_desc->pipeline.kernels[current_prog].id = sf_arena_strdup(arena, cart->header.sections[i].name);
                out_desc->pipeline.kernels[current_prog].graph_path = sf_arena_strdup(arena, path); 
                out_desc->pipeline.kernels[current_prog].frequency = 1;
                current_prog++;
            }
        }

        sf_cartridge_close(cart);
        return 0;
    }
    
    // Load as JSON manifest
    u8 temp_backing[1024 * 64]; 
    sf_arena temp_arena;
    sf_arena_init(&temp_arena, temp_backing, sizeof(temp_backing));

    sf_compiler_manifest manifest;
    if (sf_compiler_load_manifest(path, &manifest, &temp_arena)) {
        out_desc->window_title = sf_arena_strdup(arena, manifest.app_ir.app_title[0] ? manifest.app_ir.app_title : "SionFlow App");
        out_desc->width = manifest.app_ir.window_width ? manifest.app_ir.window_width : 800;
        out_desc->height = manifest.app_ir.window_height ? manifest.app_ir.window_height : 600;
        out_desc->resizable = manifest.app_ir.resizable;
        out_desc->vsync = manifest.app_ir.vsync;
        out_desc->fullscreen = manifest.app_ir.fullscreen;
        out_desc->num_threads = manifest.app_ir.num_threads;
        out_desc->has_pipeline = true;
        
        out_desc->pipeline.kernel_count = manifest.kernel_count;
        out_desc->pipeline.kernels = SF_ARENA_PUSH(arena, sf_pipeline_kernel, manifest.kernel_count);
        for (u32 i = 0; i < manifest.kernel_count; ++i) {
            out_desc->pipeline.kernels[i].id = sf_arena_strdup(arena, manifest.kernels[i].id);
            out_desc->pipeline.kernels[i].graph_path = sf_arena_strdup(arena, manifest.kernels[i].path);
            out_desc->pipeline.kernels[i].frequency = 1;
        }

        // Assets
        out_desc->asset_count = manifest.asset_count;
        out_desc->assets = SF_ARENA_PUSH(arena, sf_host_asset, manifest.asset_count);
        for (u32 i = 0; i < manifest.asset_count; ++i) {
            out_desc->assets[i].resource_name = sf_arena_strdup(arena, manifest.assets[i].name);
            out_desc->assets[i].path = sf_arena_strdup(arena, manifest.assets[i].path);
            out_desc->assets[i].type = (manifest.assets[i].type == SF_SECTION_IMAGE) ? SF_ASSET_IMAGE : SF_ASSET_FONT;
            out_desc->assets[i].font_size = 32.0f;
        }

        return 0;
    }

    sf_host_desc_cleanup(out_desc);
    return -3;
}

bool sf_loader_load_pipeline(sf_engine* engine, const sf_pipeline_desc* pipe) {
    if (!engine || !pipe) return false;
    sf_engine_reset(engine);
    sf_arena* arena = sf_engine_get_arena(engine);
    sf_program** programs = malloc(sizeof(sf_program*) * pipe->kernel_count);

    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        const char* path = pipe->kernels[i].graph_path;
        const char* ext = sf_path_get_ext(path);
        if (strcmp(ext, "json") == 0) {
            sf_compiler_diag diag; sf_compiler_diag_init(&diag, arena);
            sf_graph_ir ir = {0};
            if (!sf_compile_load_json(path, &ir, arena, &diag)) { free(programs); return false; }
            programs[i] = sf_compile(&ir, arena, &diag);
        } else {
            sf_cartridge* cart = sf_cartridge_open(path);
            if (!cart) { free(programs); return false; }
            
            size_t sec_size = 0;
            void* sec_data = sf_cartridge_get_section(cart, pipe->kernels[i].id, SF_SECTION_PROGRAM, &sec_size);
            if (sec_data) {
                programs[i] = _load_program_from_mem(sec_data, sec_size, arena);
            } else {
                for (u32 s = 0; s < cart->header.section_count; ++s) {
                    if (cart->header.sections[s].type == SF_SECTION_PROGRAM) {
                        programs[i] = _load_program_from_mem((u8*)cart->data + cart->header.sections[s].offset, cart->header.sections[s].size, arena);
                        break;
                    }
                }
            }
            sf_cartridge_close(cart);
        }
        if (!programs[i]) { free(programs); return false; }
    }

    if (pipe->resource_count == 0) {
        const char** names = malloc(sizeof(char*) * pipe->kernel_count);
        for (u32 i = 0; i < pipe->kernel_count; ++i) names[i] = pipe->kernels[i].id;
        sf_engine_bind_cartridge(engine, programs, names, pipe->kernel_count);
        free(names);
    } else {
        sf_engine_bind_pipeline(engine, pipe, programs);
    }
    
    free(programs); 
    return true;
}
