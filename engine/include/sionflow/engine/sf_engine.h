#ifndef SF_ENGINE_H
#define SF_ENGINE_H

#include <sionflow/isa/sf_tensor.h>
#include <sionflow/isa/sf_backend.h>
#include <sionflow/base/sf_types.h>
#include <sionflow/engine/sf_pipeline.h>

// Forward declarations
typedef struct sf_program sf_program;
typedef struct sf_arena sf_arena;

// Opaque Engine Handle
typedef struct sf_engine sf_engine;

/**
 * @brief Configuration for initializing the engine.
 */
typedef struct sf_engine_desc {
    size_t arena_size;      // Static arena for Code/Metadata (default: 8MB)
    size_t heap_size;       // Dynamic heap for Tensors (default: 64MB)
    sf_backend backend;     // Backend implementation
} sf_engine_desc;

/**
 * @brief Engine status codes.
 */
typedef enum {
    SF_ENGINE_ERR_NONE = 0,
    SF_ENGINE_ERR_OOM,
    SF_ENGINE_ERR_SHAPE,
    SF_ENGINE_ERR_INVALID_OP,
    SF_ENGINE_ERR_RUNTIME
} sf_engine_error;

static inline const char* sf_engine_error_to_str(sf_engine_error err) {
    switch (err) {
        case SF_ENGINE_ERR_NONE:       return "NONE";
        case SF_ENGINE_ERR_OOM:        return "OUT_OF_MEMORY";
        case SF_ENGINE_ERR_SHAPE:      return "SHAPE_MISMATCH";
        case SF_ENGINE_ERR_INVALID_OP: return "INVALID_OPCODE";
        case SF_ENGINE_ERR_RUNTIME:    return "RUNTIME_KERNEL_FAILURE";
        default:                       return "UNKNOWN_ENGINE_ERROR";
    }
}

// --- Lifecycle ---

sf_engine*      sf_engine_create(const sf_engine_desc* desc);
void            sf_engine_destroy(sf_engine* engine);
void            sf_engine_reset(sf_engine* engine);
sf_arena*       sf_engine_get_arena(sf_engine* engine);

// --- Setup ---

/**
 * @brief Binds a pipeline and allocates resources.
 */
void            sf_engine_bind_pipeline(sf_engine* engine, const sf_pipeline_desc* pipe, sf_program** programs);

/**
 * @brief Binds one or more programs as a cartridge, automatically discovering resources from symbol templates.
 */
void            sf_engine_bind_cartridge(sf_engine* engine, sf_program** programs, const char** names, uint32_t program_count);

// --- Execution ---

/**
 * @brief Dispatches the current frame.
 */
void            sf_engine_dispatch(sf_engine* engine);

// --- State & Resource Access ---

/**
 * @brief Returns the current view of a global resource.
 */
sf_tensor*      sf_engine_map_resource(sf_engine* engine, const char* name);

/**
 * @brief Force resize a global resource.
 */
bool            sf_engine_resize_resource(sf_engine* engine, const char* name, const int32_t* new_shape, uint8_t new_ndim);

/**
 * @brief Synchronizes front and back buffers for a resource (for static data loading).
 */
void            sf_engine_sync_resource(sf_engine* engine, const char* name);

/**
 * @brief Returns the last error status.
 */
sf_engine_error sf_engine_get_error(sf_engine* engine);

/**
 * @brief Callback for resource iteration.
 */
typedef void (*sf_engine_resource_cb)(const char* name, sf_tensor* tensor, void* user_data);

/**
 * @brief Iterates over all active global resources.
 */
void            sf_engine_iterate_resources(sf_engine* engine, sf_engine_resource_cb cb, void* user_data);

#endif // SF_ENGINE_H