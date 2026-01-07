cmake_minimum_required(VERSION 3.25)
project(sf-runtime LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Dependencies
if(NOT TARGET isa)
    find_package(sf-spec REQUIRED)
endif()

if(NOT TARGET compiler)
    find_package(sf-compiler REQUIRED)
endif()

# Modules
add_subdirectory(engine)
add_subdirectory(host)

# --- Installation & Export ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS engine host_core
    EXPORT SionFlowRuntimeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

export(EXPORT SionFlowRuntimeTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/SionFlowRuntimeTargets.cmake"
    NAMESPACE SionFlow::
)

install(DIRECTORY engine/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY host/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SionFlowRuntimeTargets
    FILE SionFlowRuntimeTargets.cmake
    NAMESPACE SionFlow::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-runtime
)

configure_package_config_file(
    cmake/sf-runtime-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sf-runtime-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-runtime
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sf-runtime-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-runtime
)