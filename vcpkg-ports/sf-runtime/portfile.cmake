set(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../sf-runtime")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    GENERATOR "Ninja"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME sf-runtime CONFIG_PATH lib/cmake/sf-runtime)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

if(EXISTS "${SOURCE_PATH}/LICENSE")
    file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/sf-runtime" RENAME copyright)
else()
    file(WRITE "${CURRENT_PACKAGES_DIR}/share/sf-runtime/copyright" "Copyright (c) SionFlow")
endif()
