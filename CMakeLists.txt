cmake_minimum_required(VERSION 3.25)
project(sf-runtime LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Direct Linking: Search in build root if available
if(DEFINED ENV{SIONFLOW_BUILD_ROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{SIONFLOW_BUILD_ROOT}/sf-spec/x64-linux")
    message(STATUS "Direct Linking: Added sf-spec build path to CMAKE_PREFIX_PATH")
endif()

# External dependencies
find_package(SDL2 CONFIG REQUIRED)
find_package(Stb REQUIRED)

# Dependencies
find_package(sf-spec REQUIRED)

# Modules
add_subdirectory(engine)
add_subdirectory(host)

# Apps

# --- Installation & Export ---

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install targets
set(INSTALL_TARGETS engine host_core)
if(TARGET host_sdl)
    list(APPEND INSTALL_TARGETS host_sdl)
endif()

install(TARGETS ${INSTALL_TARGETS}
    EXPORT SionFlowRuntimeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for use from the build directory
export(EXPORT SionFlowRuntimeTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/SionFlowRuntimeTargets.cmake"
    NAMESPACE SionFlow::
)

# Install headers
install(DIRECTORY engine/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY host/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install the export file
install(EXPORT SionFlowRuntimeTargets
    FILE SionFlowRuntimeTargets.cmake
    NAMESPACE SionFlow::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-runtime
)

# Create the config file
configure_package_config_file(
    cmake/sf-runtime-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sf-runtime-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-runtime
)

# Install the config file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sf-runtime-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-runtime
)
