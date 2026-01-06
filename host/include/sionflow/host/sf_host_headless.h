#ifndef SF_HOST_HEADLESS_H
#define SF_HOST_HEADLESS_H

#include <sionflow/host/sf_host_desc.h>

/**
 * @brief Runs the engine in headless mode (CLI).
 * Initializes the engine, loads the graph specified in the descriptor,
 * executes for a specified number of frames, and prints output.
 * 
 * @param desc Configuration descriptor (graph path, settings).
 * @param frames Number of frames to simulate.
 * @return int Exit code (0 on success).
 */
int sf_host_run_headless(const sf_host_desc* desc, int frames);

#endif // SF_HOST_HEADLESS_H
