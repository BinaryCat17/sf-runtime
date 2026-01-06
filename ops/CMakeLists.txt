add_library(ops STATIC
    src/sf_ops_array.c
    src/sf_ops_core.c
    src/sf_ops_math.c
    src/sf_ops_logic.c
    src/sf_ops_matrix.c
    src/sf_ops_state.c
    src/sf_ops_system.c
)
add_library(SionFlow::ops ALIAS ops)

target_include_directories(ops 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE src
)

target_link_libraries(ops 
    PUBLIC 
        SionFlow::isa
        m
)
