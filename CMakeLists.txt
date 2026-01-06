cmake_minimum_required(VERSION 3.25)
project(sf-runtime LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# External dependencies
find_package(SDL2 CONFIG REQUIRED)
find_package(Stb REQUIRED)

# Temporary: linking to sf-spec
if (NOT TARGET sf_base)
    add_subdirectory(../sf-spec/base sf-spec-base)
endif()
if (NOT TARGET sf_isa)
    add_subdirectory(../sf-spec/isa sf-spec-isa)
endif()

# Modules
add_subdirectory(ops)
add_subdirectory(backend_cpu)
add_subdirectory(engine)
add_subdirectory(host)

# Apps
add_subdirectory(sf-runner)
add_subdirectory(sf-window)
