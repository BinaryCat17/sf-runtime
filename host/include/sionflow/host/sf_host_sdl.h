#ifndef SF_HOST_SDL_H
#define SF_HOST_SDL_H

#include <sionflow/host/sf_host_desc.h>
#include <sionflow/isa/sf_backend.h>

/**
 * @brief Runs the standard SionFlow Host Loop using SDL2.
 * 
 * This function initializes SDL, creates a window, loads the graph,
 * and runs the loop until the user closes the window.
 * 
 * @param desc Configuration descriptor.
 * @param backend Pre-initialized backend implementation.
 * @return int 0 on success, non-zero on error.
 */
int sf_host_run(const sf_host_desc* desc, sf_backend backend);

#endif // SF_HOST_SDL_H