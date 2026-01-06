#include <sionflow/host/sf_host_sdl.h>
#include <sionflow/host/sf_host_desc.h>
#include <sionflow/base/sf_log.h>
#include <sionflow/base/sf_platform.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    sf_host_init_logger();
    sf_log_set_global_level(SF_LOG_LEVEL_INFO);

    if (argc < 2) {
        printf("Usage: sf-window <cartridge.sfc> [--log-interval <seconds>] [--trace] [--debug]\n");
        return 1;
    }

    const char* mfapp_path = argv[1];
    sf_host_desc desc = {0};
    desc.log_interval = 5.0f;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--log-interval") == 0 && i + 1 < argc) {
            desc.log_interval = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--trace") == 0) {
            sf_log_set_global_level(SF_LOG_LEVEL_TRACE);
        } else if (strcmp(argv[i], "--debug") == 0) {
            sf_log_set_global_level(SF_LOG_LEVEL_DEBUG);
        }
    }

    if (sf_app_load_config(mfapp_path, &desc) != 0) {
        SF_LOG_ERROR("Failed to load application from %s", mfapp_path);
        return 1;
    }

    int result = sf_host_run(&desc);
    sf_host_desc_cleanup(&desc);

    return result;
}
