add_library(engine STATIC
    src/sf_engine.c
    src/sf_pipeline.c
)
add_library(SionFlow::engine ALIAS engine)

target_include_directories(engine 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE src
)

target_link_libraries(engine 
    PUBLIC 
        SionFlow::base
        SionFlow::isa
)

