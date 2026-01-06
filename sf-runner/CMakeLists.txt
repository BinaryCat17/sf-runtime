add_executable(sf-runner 
    src/main.c
)

target_include_directories(sf-runner PRIVATE 
    # No extra include needed, it comes from SionFlow::host_core interface
)

target_link_libraries(sf-runner 
    PRIVATE 
        SionFlow::host_core
        SionFlow::base
)
