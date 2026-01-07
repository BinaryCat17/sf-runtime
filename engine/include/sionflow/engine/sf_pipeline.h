#ifndef SF_PIPELINE_H
#define SF_PIPELINE_H

#include <sionflow/base/sf_types.h>
#include <sionflow/isa/sf_tensor.h>

// Description of a Global Resource (Blackboard Buffer)
typedef struct {
    const char* name;
    sf_dtype dtype;
    int32_t shape[SF_MAX_DIMS];
    uint8_t ndim;
    uint8_t flags;
} sf_pipeline_resource;

// Mapping between a Kernel's internal Symbol and a Global Resource
typedef struct {
    const char* kernel_port;     // Symbol name in the .json/.bin
    const char* global_resource; // Resource name defined in sf_pipeline_desc
} sf_pipeline_binding;

// Description of a single execution unit (Shader/Kernel)
typedef struct {
    const char* id;
    const char* graph_path; // Path to .json or .bin
    uint32_t frequency;     // 1 = every frame, N = N times per frame
    
    sf_pipeline_binding* bindings;
    uint32_t binding_count;
} sf_pipeline_kernel;

// Complete Pipeline Configuration (from .mfapp)
typedef struct {
    sf_pipeline_resource* resources;
    uint32_t resource_count;
    
    sf_pipeline_kernel* kernels;
    uint32_t kernel_count;
} sf_pipeline_desc;

#endif // SF_PIPELINE_H
