#ifndef SF_BACKEND_CPU_H
#define SF_BACKEND_CPU_H

#include <sionflow/isa/sf_backend.h>

/**
 * @brief Initializes the CPU backend.
 * Creates an internal thread pool and fills the dispatch table.
 * 
 * @param backend Pointer to the backend structure to fill.
 * @param num_threads Number of threads (0 = auto).
 */
void sf_backend_cpu_init(sf_backend* backend, int num_threads);

#endif // SF_BACKEND_CPU_H
