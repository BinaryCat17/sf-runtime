#ifndef SF_ENGINE_INTERNAL_H
#define SF_ENGINE_INTERNAL_H

#include <sionflow/engine/sf_engine.h>
#include <sionflow/isa/sf_state.h>
#include <sionflow/isa/sf_program.h>
#include <sionflow/isa/sf_backend.h>
#include <sionflow/engine/sf_pipeline.h>
#include <sionflow/base/sf_buffer.h>

/**
 * @brief Mapping between a Local Register in a Kernel and a Global Resource.
 */
typedef struct {
    u16 local_reg;   // Register index in the compiled program
    u16 global_res;  // Resource index in the engine's registry
    u8  flags;       // Symbol flags (Input, Output, etc.)
} sf_kernel_binding;

/**
 * @brief Runtime instance of a Kernel (Program + State).
 */
typedef struct {
    const char* id;
    u32         id_hash;
    sf_program* program;
    sf_state    state;       // Local registers and memory
    uint32_t    frequency;   // Execution frequency per frame
    
    sf_kernel_binding* bindings;
    u32                binding_count;
} sf_kernel_inst;

/**
 * @brief Concrete instance of a Global Resource (Double Buffered).
 */
typedef struct {
    const char* name;
    const char* provider;
    u32         name_hash;
    sf_buffer*  buffers[2];   // [0] Front, [1] Back
    size_t      size_bytes;
    sf_tensor   desc;         // Metadata and current view
    u8          flags;        // SF_RESOURCE_FLAG_*
} sf_resource_inst;

/**
 * @brief The Core Engine Structure.
 */
struct sf_engine {
    // Memory Management
    sf_arena arena;           // Static memory (Code, Metadata)
    void*    arena_buffer;
    sf_heap  heap;            // Dynamic memory (Tensors, Data)
    void*    heap_buffer;

    // Backend Implementation
    sf_backend backend;

    // Pipeline State
    sf_resource_inst* resources;
    u32               resource_count;
    sf_kernel_inst*   kernels;
    u32               kernel_count;

    // Buffer Synchronization
    u8 front_idx;             // Index for Read
    u8 back_idx;              // Index for Write
    
    // Status
    sf_atomic_i32 error_code; // Global Kill Switch (Atomic)

    // Stats
    uint64_t frame_index;
};

// --- Internal Utilities (Shared across module files) ---

/**
 * @brief Resets/Initializes the internal state for a kernel program.
 * Defined in sf_engine.c, used in sf_pipeline.c.
 */
void sf_state_reset(sf_state* state, const sf_program* prog, sf_arena* arena, sf_backend* backend);

/**
 * @brief Finds resource index by its name hash.
 */
int32_t find_resource_idx(sf_engine* engine, u32 name_hash);

/**
 * @brief Finds symbol index in a program by its name hash.
 */
int32_t find_symbol_idx(const sf_program* prog, u32 name_hash);

#endif // SF_ENGINE_INTERNAL_H