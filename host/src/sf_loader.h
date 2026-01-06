#ifndef SF_LOADER_H
#define SF_LOADER_H

#include <sionflow/base/sf_types.h>
#include <sionflow/engine/sf_engine.h>
#include <sionflow/isa/sf_program.h>
#include <sionflow/isa/sf_backend.h>
#include <sionflow/host/sf_host_desc.h>

/**
 * SionFlow Unified Loader
 * 
 * Responsibilities:
 * - Parsing .mfapp manifests
 * - Compiling/Loading kernel programs
 * - Loading assets (Images, Fonts) into Engine resources
 */

// --- Backend Setup ---
void sf_loader_init_backend(sf_backend* backend, int num_threads);

// --- Manifest Parsing ---
int sf_app_load_config(const char* mfapp_path, sf_host_desc* out_desc);

// --- Pipeline Loading ---
bool            sf_loader_load_pipeline(sf_engine* engine, const sf_pipeline_desc* pipe);

/**
 * @brief Simple view over a loaded cartridge file.
 */
typedef struct {
    void* data;
    size_t size;
    sf_cartridge_header header;
} sf_cartridge;

/**
 * @brief Opens a cartridge file and maps it into memory.
 */
sf_cartridge*   sf_cartridge_open(const char* path);

/**
 * @brief Closes a cartridge and frees memory.
 */
void            sf_cartridge_close(sf_cartridge* cart);

/**
 * @brief Extracts a specific section from a cartridge.
 */
void*           sf_cartridge_get_section(sf_cartridge* cart, const char* name, sf_section_type type, size_t* out_size);

bool            sf_loader_load_image(sf_engine* engine, const char* name, const char* path);
bool            sf_loader_load_font(sf_engine* engine, const char* resource_name, const char* path, float font_size);

#endif // SF_LOADER_H
