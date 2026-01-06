add_library(sf_ops STATIC
    src/sf_ops_array.c
    src/sf_ops_core.c
    src/sf_ops_math.c
    src/sf_ops_logic.c
    src/sf_ops_matrix.c
    src/sf_ops_state.c
)
add_library(SionFlow::ops ALIAS sf_ops)

target_include_directories(sf_ops 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(sf_ops 
    PUBLIC 
        SionFlow::isa
        m
)
