#include "sf_loader.h"
#include <sionflow/engine/sf_engine.h>
#include <sionflow/isa/sf_opcodes.h>
#include <sionflow/host/sf_host_desc.h>
#include <sionflow/base/sf_shape.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_utils.h>
#include <sionflow/base/sf_json.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static sf_program* _load_program_from_mem(const u8* data, size_t len, sf_arena* arena) {
    sf_program* prog = SF_ARENA_PUSH(arena, sf_program, 1);
    if (!prog) return NULL;
    memset(prog, 0, sizeof(sf_program));

    if (!sf_program_load_from_buffer(prog, data, len, arena)) {
        SF_LOG_ERROR("Failed to load program from buffer (SFC 2.0)");
        return NULL;
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

    sf_cartridge* cart = sf_cartridge_open(path);
    if (!cart) {
        SF_LOG_ERROR("Loader: Only binary .sfc/.bin cartridges are supported in production runtime.");
        sf_host_desc_cleanup(out_desc); 
        return -1; 
    }

    out_desc->window_title = sf_arena_strdup(arena, cart->header.app_title[0] ? cart->header.app_title : "SionFlow App");
    out_desc->width = cart->header.window_width ? (int)cart->header.window_width : 800;
    out_desc->height = cart->header.window_height ? (int)cart->header.window_height : 600;
    out_desc->resizable = cart->header.resizable;
    out_desc->vsync = cart->header.vsync;
    out_desc->fullscreen = cart->header.fullscreen;
    out_desc->num_threads = (int)cart->header.num_threads;
    out_desc->has_pipeline = true;

    // Load full pipeline definition if available
    size_t pipe_json_size = 0;
    const char* pipe_json = (const char*)sf_cartridge_get_section(cart, "pipeline", SF_SECTION_PIPELINE, &pipe_json_size);
    if (pipe_json) {
        sf_json_value* root = sf_json_parse(pipe_json, arena);
        if (root && root->type == SF_JSON_VAL_OBJECT) {
            const sf_json_value* pipe = sf_json_get_field(root, "pipeline");
            if (pipe && pipe->type == SF_JSON_VAL_OBJECT) {
                // Parse Resources
                const sf_json_value* res_arr = sf_json_get_field(pipe, "resources");
                if (res_arr && res_arr->type == SF_JSON_VAL_ARRAY) {
                    out_desc->pipeline.resource_count = res_arr->as.array.count;
                    out_desc->pipeline.resources = SF_ARENA_PUSH(arena, sf_pipeline_resource, out_desc->pipeline.resource_count);
                    for (u32 i = 0; i < res_arr->as.array.count; ++i) {
                        const sf_json_value* r = &res_arr->as.array.items[i];
                        sf_pipeline_resource* dst = &out_desc->pipeline.resources[i];
                        const sf_json_value* v_name = sf_json_get_field(r, "name");
                        const sf_json_value* v_dtype = sf_json_get_field(r, "dtype");
                        const sf_json_value* v_shape = sf_json_get_field(r, "shape");
                        const sf_json_value* v_pers = sf_json_get_field(r, "persistent");
                        const sf_json_value* v_ro = sf_json_get_field(r, "readonly");
                        const sf_json_value* v_ss = sf_json_get_field(r, "screen_size");
                        const sf_json_value* v_out = sf_json_get_field(r, "output");

                        dst->name = v_name ? sf_arena_strdup(arena, v_name->as.s) : "unknown";
                        dst->dtype = v_dtype ? sf_dtype_from_str(v_dtype->as.s) : SF_DTYPE_F32;
                        dst->flags = 0;
                        if (v_pers && v_pers->as.b) dst->flags |= SF_RESOURCE_FLAG_PERSISTENT;
                        if (v_ro && v_ro->as.b) dst->flags |= SF_RESOURCE_FLAG_READONLY;
                        if (v_ss && v_ss->as.b) dst->flags |= SF_RESOURCE_FLAG_SCREEN_SIZE;
                        if (v_out && v_out->as.b) dst->flags |= SF_RESOURCE_FLAG_OUTPUT;

                        dst->ndim = 0;
                        if (v_shape && v_shape->type == SF_JSON_VAL_ARRAY) {
                            dst->ndim = (u8)v_shape->as.array.count;
                            for (u32 k = 0; k < dst->ndim; ++k) dst->shape[k] = (int32_t)v_shape->as.array.items[k].as.n;
                        }
                    }
                }

                // Parse Kernels
                const sf_json_value* ker_arr = sf_json_get_field(pipe, "kernels");
                if (ker_arr && ker_arr->type == SF_JSON_VAL_ARRAY) {
                    out_desc->pipeline.kernel_count = ker_arr->as.array.count;
                    out_desc->pipeline.kernels = SF_ARENA_PUSH(arena, sf_pipeline_kernel, out_desc->pipeline.kernel_count);
                    for (u32 i = 0; i < ker_arr->as.array.count; ++i) {
                        const sf_json_value* k = &ker_arr->as.array.items[i];
                        sf_pipeline_kernel* dst = &out_desc->pipeline.kernels[i];
                        const sf_json_value* v_id = sf_json_get_field(k, "id");
                        const sf_json_value* v_freq = sf_json_get_field(k, "frequency");
                        const sf_json_value* v_binds = sf_json_get_field(k, "bindings");

                        dst->id = v_id ? sf_arena_strdup(arena, v_id->as.s) : "kernel";
                        dst->graph_path = sf_arena_strdup(arena, path);
                        dst->frequency = v_freq ? (u32)v_freq->as.n : 1;
                        
                        if (v_binds && v_binds->type == SF_JSON_VAL_ARRAY) {
                            dst->binding_count = v_binds->as.array.count;
                            dst->bindings = SF_ARENA_PUSH(arena, sf_pipeline_binding, dst->binding_count);
                            for (u32 b = 0; b < dst->binding_count; ++b) {
                                const sf_json_value* bind = &v_binds->as.array.items[b];
                                const sf_json_value* v_port = sf_json_get_field(bind, "port");
                                const sf_json_value* v_res = sf_json_get_field(bind, "resource");
                                dst->bindings[b].kernel_port = v_port ? sf_arena_strdup(arena, v_port->as.s) : "unknown";
                                dst->bindings[b].global_resource = v_res ? sf_arena_strdup(arena, v_res->as.s) : "unknown";
                            }
                        } else {
                            dst->binding_count = 0;
                            dst->bindings = NULL;
                        }
                    }
                }
            }
        }
    } else {
        // Fallback: Default pipeline if no section found
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
                out_desc->pipeline.kernels[current_prog].binding_count = 0;
                out_desc->pipeline.kernels[current_prog].bindings = NULL;
                current_prog++;
            }
        }
    }

    // Load assets from cartridge sections
    u32 asset_count = 0;
    for (u32 i = 0; i < cart->header.section_count; ++i) {
        u32 type = cart->header.sections[i].type;
        if (type == SF_SECTION_IMAGE || type == SF_SECTION_FONT) asset_count++;
    }

    out_desc->asset_count = asset_count;
    out_desc->assets = SF_ARENA_PUSH(arena, sf_host_asset, asset_count);
    u32 cur_asset = 0;
    for (u32 i = 0; i < cart->header.section_count; ++i) {
        u32 type = cart->header.sections[i].type;
        if (type == SF_SECTION_IMAGE || type == SF_SECTION_FONT) {
            out_desc->assets[cur_asset].resource_name = sf_arena_strdup(arena, cart->header.sections[i].name);
            out_desc->assets[cur_asset].path = sf_arena_strdup(arena, path); // Use cartridge as source
            out_desc->assets[cur_asset].type = (type == SF_SECTION_IMAGE) ? SF_ASSET_IMAGE : SF_ASSET_FONT;
            out_desc->assets[cur_asset].font_size = 32.0f;
            cur_asset++;
        }
    }

    sf_cartridge_close(cart);
    return 0;
}

bool sf_loader_load_pipeline(sf_engine* engine, const sf_pipeline_desc* pipe) {
    if (!engine || !pipe) return false;
    sf_engine_reset(engine);
    sf_arena* arena = sf_engine_get_arena(engine);
    sf_program** programs = malloc(sizeof(sf_program*) * pipe->kernel_count);

    for (u32 i = 0; i < pipe->kernel_count; ++i) {
        const char* path = pipe->kernels[i].graph_path;
        sf_cartridge* cart = sf_cartridge_open(path);
        if (!cart) { free(programs); return false; }
        
        size_t sec_size = 0;
        void* sec_data = sf_cartridge_get_section(cart, pipe->kernels[i].id, SF_SECTION_PROGRAM, &sec_size);
        if (sec_data) {
            programs[i] = _load_program_from_mem(sec_data, sec_size, arena);
        } else {
            // Fallback: load first program found
            for (u32 s = 0; s < cart->header.section_count; ++s) {
                if (cart->header.sections[s].type == SF_SECTION_PROGRAM) {
                    programs[i] = _load_program_from_mem((u8*)cart->data + cart->header.sections[s].offset, cart->header.sections[s].size, arena);
                    break;
                }
            }
        }
        sf_cartridge_close(cart);
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