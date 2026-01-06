add_library(sf_engine STATIC
    src/sf_engine.c
    src/sf_pipeline.c
)
add_library(SionFlow::engine ALIAS sf_engine)

target_include_directories(sf_engine 
    PUBLIC include
    PRIVATE src
)

target_link_libraries(sf_engine 
    PUBLIC 
    SionFlow::base
    SionFlow::isa
)
