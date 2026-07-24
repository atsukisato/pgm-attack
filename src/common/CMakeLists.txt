add_library(pgm_common INTERFACE)

target_include_directories(pgm_common INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_SOURCE_DIR}/src/common
    ${PGM_INDEX_INCLUDE_DIR}
)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(pgm_common INTERFACE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(pgm_common INTERFACE
    nlohmann_json::nlohmann_json
    sdsl::sdsl
)
