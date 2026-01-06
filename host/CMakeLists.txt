find_package(SDL2 CONFIG QUIET)
find_package(Stb REQUIRED)

# --- Host Core (Manifest, Config, Loader, Headless) ---
add_library(host_core STATIC
    src/sf_host_headless.c
    src/sf_host_common.c
    src/sf_loader.c
    src/sf_assets.c
)
add_library(SionFlow::host_core ALIAS host_core)

target_include_directories(host_core 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        src
        ${Stb_INCLUDE_DIR}
)
target_link_libraries(host_core 
    PUBLIC
        SionFlow::engine
    PRIVATE 
        SionFlow::backend_cpu
        SionFlow::base
)

# --- Host SDL (Window & Input) ---
if(TARGET SDL2::SDL2)
    add_library(host_sdl STATIC
        src/sf_host_sdl.c
    )
    add_library(SionFlow::host_sdl ALIAS host_sdl)

    target_include_directories(host_sdl 
        PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE src
    )

    target_link_libraries(host_sdl 
        PUBLIC 
            SionFlow::host_core
        PRIVATE
            SionFlow::engine
            SDL2::SDL2 
            SDL2::SDL2main
    )
    message(STATUS "Building sf_host_sdl (SDL2 found)")
else()
    message(WARNING "SDL2 not found. sf_host_sdl (SDL backend) will not be built. sf_host_core is available.")
endif()
