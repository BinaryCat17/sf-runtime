#include <sionflow/host/sf_host_headless.h>
#include <sionflow/engine/sf_engine.h>
#include <sionflow/isa/sf_tensor.h>
#include <sionflow/base/sf_log.h>
#include "sf_host_internal.h"
#include "sf_loader.h"
#include <stdio.h>
#include <string.h>

static void debug_print_resource_callback(const char* name, sf_tensor* t, void* user_data) {
    (void)user_data;
    sf_tensor_print(name, t);
}

int sf_host_run_headless(const sf_host_desc* desc, sf_backend backend, int frames) {
    if (!desc) return 1;

    sf_host_app app;
    if (sf_host_app_init(&app, desc, backend) != 0) {
        SF_LOG_ERROR("Failed to initialize Host App");
        return 1;
    }

    SF_LOG_INFO("Running for %d frames...\n", frames);
    for (int f = 0; f < frames; ++f) {
        sf_host_inputs inputs = {
            .time = (f32)f * 0.016f,
            .width = desc->width,
            .height = desc->height
        };
        sf_host_app_update_inputs(&app, &inputs);

        sf_engine_error err = sf_host_app_step(&app);
        if (err != SF_ENGINE_ERR_NONE) {
            SF_LOG_ERROR("Engine failure: %s", sf_engine_error_to_str(err));
            break;
        }
        
        if (f < 3) {
             SF_LOG_INFO("--- Frame %d ---\n", f);
             sf_engine_iterate_resources(app.engine, debug_print_resource_callback, NULL);
        }
    }
    
    SF_LOG_INFO("--- Final State ---\n");
    sf_engine_iterate_resources(app.engine, debug_print_resource_callback, NULL);

    sf_host_app_cleanup(&app);
    return 0;
}