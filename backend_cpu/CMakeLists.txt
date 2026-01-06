add_library(backend_cpu STATIC
    src/sf_backend_cpu.c
)
add_library(SionFlow::backend_cpu ALIAS backend_cpu)

target_include_directories(backend_cpu 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE src
)

target_link_libraries(backend_cpu 
    PUBLIC 
        SionFlow::base
        SionFlow::isa 
        SionFlow::ops
)
