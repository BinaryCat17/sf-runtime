#include <sionflow/host/sf_host_desc.h>
#include <sionflow/engine/sf_engine.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_platform.h>
#include "sf_host_internal.h"
#include "sf_loader.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

void sf_host_init_logger(void) {
    if (sf_fs_mkdir("logs")) {
        sf_fs_clear_dir("logs"); 
    }

    sf_log_init();

    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char log_path[256];
    if (t) {
        strftime(log_path, sizeof(log_path), "logs/log_%Y-%m-%d_%H-%M-%S.txt", t);
    } else {
        strcpy(log_path, "logs/latest_log.txt");
    }
    
    sf_log_add_file_sink(log_path, SF_LOG_LEVEL_TRACE);
}

void sf_host_desc_cleanup(sf_host_desc* desc) {
    if (!desc) return;
    if (desc->arena_backing) free(desc->arena_backing);
    memset(desc, 0, sizeof(sf_host_desc));
}

static void sf_host_app_bind_resources(sf_host_app* app) {
    app->resources.time = sf_engine_map_resource(app->engine, "u_Time");
    app->resources.mouse = sf_engine_map_resource(app->engine, "u_Mouse");
    app->resources.resolution = sf_engine_map_resource(app->engine, "u_Resolution");
    app->resources.res_x = sf_engine_map_resource(app->engine, "u_ResX");
    app->resources.res_y = sf_engine_map_resource(app->engine, "u_ResY");
    app->resources.aspect = sf_engine_map_resource(app->engine, "u_Aspect");

    app->resources.output = NULL;
    for (u32 i = 0; i < app->desc.pipeline.resource_count; ++i) {
        if (app->desc.pipeline.resources[i].flags & SF_RESOURCE_FLAG_OUTPUT) {
            app->resources.output = sf_engine_map_resource(app->engine, app->desc.pipeline.resources[i].name);
            if (app->resources.output) break;
        }
    }
    if (!app->resources.output) {
        app->resources.output = sf_engine_map_resource(app->engine, "out_Color");
    }
}

static void _on_resource_resize(const char* name, sf_tensor* tensor, void* user_data) {
    sf_host_app* app = (sf_host_app*)user_data;
    for (u32 i = 0; i < app->desc.pipeline.resource_count; ++i) {
        if (strcmp(app->desc.pipeline.resources[i].name, name) == 0) {
            if (app->desc.pipeline.resources[i].flags & SF_RESOURCE_FLAG_SCREEN_SIZE) {
                int32_t new_shape[3] = { app->inputs.height, app->inputs.width, 4 };
                if (tensor->info.ndim >= 1) new_shape[2] = tensor->info.shape[tensor->info.ndim-1];
                sf_engine_resize_resource(app->engine, name, new_shape, tensor->info.ndim);
            }
            break;
        }
    }
}

void sf_host_app_update_inputs(sf_host_app* app, const sf_host_inputs* inputs) {
    if (!app || !app->is_initialized || !inputs) return;

    bool res_changed = (inputs->width != app->inputs.width || inputs->height != app->inputs.height);
    app->inputs = *inputs;

    if (res_changed) {
        sf_engine_iterate_resources(app->engine, _on_resource_resize, app);
        if (app->resources.resolution) {
            f32* d = sf_tensor_data(app->resources.resolution);
            if (d) { d[0] = (f32)inputs->width; d[1] = (f32)inputs->height; }
            sf_engine_sync_resource(app->engine, "u_Resolution");
        }
        if (app->resources.res_x) {
            f32* d = sf_tensor_data(app->resources.res_x);
            if (d) *d = (f32)inputs->width;
            sf_engine_sync_resource(app->engine, "u_ResX");
        }
        if (app->resources.res_y) {
            f32* d = sf_tensor_data(app->resources.res_y);
            if (d) *d = (f32)inputs->height;
            sf_engine_sync_resource(app->engine, "u_ResY");
        }
        if (app->resources.aspect) {
            f32* d = sf_tensor_data(app->resources.aspect);
            if (d) *d = (f32)inputs->width / (f32)inputs->height;
            sf_engine_sync_resource(app->engine, "u_Aspect");
        }
    }

    if (app->resources.time) {
        f32* d = (f32*)sf_tensor_data(app->resources.time);
        if (d) { *d = inputs->time; sf_engine_sync_resource(app->engine, "u_Time"); }
    }

    if (app->resources.mouse) {
        f32* d = (f32*)sf_tensor_data(app->resources.mouse);
        if (d) {
            d[0] = inputs->mouse_x; d[1] = inputs->mouse_y;
            d[2] = inputs->mouse_lmb ? 1.0f : 0.0f;
            d[3] = inputs->mouse_rmb ? 1.0f : 0.0f;
            sf_engine_sync_resource(app->engine, "u_Mouse");
        }
    }
}

int sf_host_app_init(sf_host_app* app, const sf_host_desc* desc, sf_backend backend) {
    if (!app || !desc) return -1;
    memset(app, 0, sizeof(sf_host_app));
    app->desc = *desc; 

    sf_engine_desc engine_desc = { 
        .arena_size = SF_MB(64), 
        .heap_size = SF_MB(256),
        .backend = backend
    };

    app->engine = sf_engine_create(&engine_desc);
    if (!app->engine) return -2;

    if (!sf_loader_load_pipeline(app->engine, &desc->pipeline)) {
        SF_LOG_ERROR("Host: Failed to load pipeline");
        sf_engine_destroy(app->engine);
        return -3;
    }

    // Load Assets
    for (int i = 0; i < desc->asset_count; ++i) {
        sf_host_asset* asset = &desc->assets[i];
        if (asset->type == SF_ASSET_IMAGE) {
            sf_loader_load_image(app->engine, asset->resource_name, asset->path);
        } else if (asset->type == SF_ASSET_FONT) {
            sf_loader_load_font(app->engine, asset->resource_name, asset->path, asset->font_size);
        }
    }

    sf_host_app_bind_resources(app);
    
    // Initial sync
    sf_host_inputs initial_inputs = {
        .time = 0, .width = desc->width, .height = desc->height,
        .mouse_x = 0, .mouse_y = 0, .mouse_lmb = false, .mouse_rmb = false
    };
    sf_host_app_update_inputs(app, &initial_inputs);

    app->is_initialized = true;
    return 0;
}

sf_engine_error sf_host_app_step(sf_host_app* app) {
    if (!app || !app->engine) return SF_ENGINE_ERR_NONE;
    sf_engine_dispatch(app->engine);
    return sf_engine_get_error(app->engine);
}

void sf_host_app_cleanup(sf_host_app* app) {
    if (!app) return;
    if (app->engine) sf_engine_destroy(app->engine);
    memset(app, 0, sizeof(sf_host_app));
}
