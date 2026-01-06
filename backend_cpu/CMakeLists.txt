add_library(sf_backend_cpu STATIC
    src/sf_backend_cpu.c
)
add_library(SionFlow::backend_cpu ALIAS sf_backend_cpu)

target_include_directories(sf_backend_cpu 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(sf_backend_cpu 
    PUBLIC 
    SionFlow::base
    SionFlow::isa 
        SionFlow::ops
)
