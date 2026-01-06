#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sionflow/host/sf_host_headless.h>
#include <sionflow/host/sf_host_desc.h>
#include <sionflow/base/sf_log.h>

static void print_help(const char* prog) {
    printf("Usage: %s <app.mfapp> [options]\n", prog);
    printf("Options:\n");
    printf("  --frames <n>   Number of frames to execute (default: 1)\n");
    printf("  --trace        Enable trace logging\n");
}

int main(int argc, char** argv) {
    sf_host_init_logger();

    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    const char* mfapp_path = argv[1];
    int frames = 1;
    
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--trace") == 0) {
            sf_log_set_global_level(SF_LOG_LEVEL_TRACE);
        }
    }

    sf_host_desc app_desc = {0};
    if (sf_app_load_config(mfapp_path, &app_desc) != 0) {
        SF_LOG_ERROR("Failed to load application from %s", mfapp_path);
        return 1;
    }

    int result = sf_host_run_headless(&app_desc, frames);
    sf_host_desc_cleanup(&app_desc);

    return result;
}