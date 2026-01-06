cmake_minimum_required(VERSION 3.25)

project(sf-window)

if(TARGET SionFlow::host_sdl)
    add_executable(sf-window src/main.c)

    target_link_libraries(sf-window 
        PRIVATE 
            SionFlow::host_sdl
            SionFlow::base
    )
else()
    message(STATUS "Skipping sf-window (SionFlow::host_sdl not available)")
endif()