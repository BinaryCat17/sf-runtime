# --- Kernel Generation ---
find_package(Python3 REQUIRED)
set(ISA_GEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/../../sf-spec/tools/isa_gen.py")
set(ISA_JSON "${CMAKE_CURRENT_SOURCE_DIR}/../../sf-spec/tools/metadata/isa.json")
set(CPU_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/../tools/metadata/cpu_spec.json")
set(KERNELS_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/../tools/templates/sf_ops_cpu_auto.c.j2")
set(KERNELS_C "${CMAKE_CURRENT_SOURCE_DIR}/../tools/generated/sf_ops_auto.c")
set(DISPATCH_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/../tools/templates/sf_ops_auto_dispatch.c.j2")
set(DISPATCH_C "${CMAKE_CURRENT_SOURCE_DIR}/../tools/generated/sf_ops_core.c")

add_custom_command(
    OUTPUT "${KERNELS_C}" "${DISPATCH_C}"
    COMMAND Python3::Interpreter "${ISA_GEN_PY}" --isa "${ISA_JSON}" --backend "${CPU_SPEC}"
            --render "${KERNELS_TMPL}:${KERNELS_C}"
                     "${DISPATCH_TMPL}:${DISPATCH_C}"
    DEPENDS "${ISA_JSON}" "${CPU_SPEC}" "${ISA_GEN_PY}" "${KERNELS_TMPL}" "${DISPATCH_TMPL}"
    COMMENT "Generating CPU Kernels and Dispatch Table"
    VERBATIM
)

add_library(ops STATIC
    src/sf_ops_array.c
    "${DISPATCH_C}"
    src/sf_ops_math.c
    src/sf_ops_logic.c
    src/sf_ops_matrix.c
    src/sf_ops_state.c
    src/sf_ops_system.c
    "${KERNELS_C}"
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
