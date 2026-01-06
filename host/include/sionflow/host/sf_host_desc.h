#ifndef SF_HOST_DESC_H
#define SF_HOST_DESC_H

#include <stdbool.h>
#include <sionflow/engine/sf_pipeline.h>
#include <sionflow/base/sf_memory.h>

typedef enum {
    SF_ASSET_IMAGE,
    SF_ASSET_FONT
} sf_asset_type;

typedef struct {
    sf_asset_type type;
    const char* resource_name;
    const char* path;
    float font_size; // only for fonts
} sf_host_asset;

// Configuration for the Host Application
typedef struct sf_host_desc {
    sf_arena arena;
    void*    arena_backing;

    const char* window_title;
    int width;
    int height;
    
    // Pipeline configuration (All programs run through this)
    sf_pipeline_desc pipeline;
    bool has_pipeline;

    // Assets to load into resources
    sf_host_asset* assets;
    int asset_count;
    
    // Optional: Number of worker threads (0 = Auto)
    int num_threads;

    // Logging Interval (in seconds) for TRACE logs and screenshots. 0 = Disable periodic logging.
    float log_interval;
    
    // Window Options
    bool fullscreen;
    bool vsync;
    bool resizable;
} sf_host_desc;

/**
 * @brief Initializes the unified logging system for the host application.
 * Creates the 'logs/' directory and sets up both console and file output.
 */
void sf_host_init_logger(void);

/**
 * @brief Cleans up memory allocated within sf_host_desc (e.g. by manifest loader).
 */
void sf_host_desc_cleanup(sf_host_desc* desc);

/**
 * @brief Loads an application manifest (.mfapp) and populates the host descriptor.
 * 
 * This function parses the JSON manifest, resolving relative paths (e.g. for the graph entry)
 * against the manifest's location.
 * 
 * @param mfapp_path Path to the .mfapp file.
 * @param out_desc Pointer to the descriptor to populate.
 * @return int 0 on success, non-zero on error.
 */
int sf_app_load_config(const char* mfapp_path, sf_host_desc* out_desc);

#endif // SF_HOST_DESC_H
