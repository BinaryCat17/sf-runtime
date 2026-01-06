#ifndef SF_HOST_INTERNAL_H
#define SF_HOST_INTERNAL_H

#include <sionflow/host/sf_host_desc.h>
#include <sionflow/engine/sf_engine.h>

typedef struct {
    float time;
    float mouse_x;
    float mouse_y;
    bool  mouse_lmb;
    bool  mouse_rmb;
    int   width;
    int   height;
} sf_host_inputs;

/**
 * @brief Shared context for a running SionFlow application.
 * Internal to the host module.
 */
typedef struct {
    sf_host_desc desc;
    sf_engine* engine;
    
    struct {
        sf_tensor* time;
        sf_tensor* mouse;
        sf_tensor* resolution;
        sf_tensor* res_x;
        sf_tensor* res_y;
        sf_tensor* aspect;
        sf_tensor* output;
    } resources;

    sf_host_inputs inputs;
    bool is_initialized;
} sf_host_app;

/**
 * @brief Initializes the host application context.
 */
int sf_host_app_init(sf_host_app* app, const sf_host_desc* desc, sf_backend backend);

/**
 * @brief Updates all system resources (Time, Mouse, Res) in one go.
 */
void sf_host_app_update_inputs(sf_host_app* app, const sf_host_inputs* inputs);

/**
 * @brief Executes a single frame of the application.
 * Updates state, runs kernels, and checks for errors.
 */
sf_engine_error sf_host_app_step(sf_host_app* app);

/**
 * @brief Shuts down the application context.
 */
void sf_host_app_cleanup(sf_host_app* app);

#endif // SF_HOST_INTERNAL_H
