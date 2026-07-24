add_executable(benchmark_fiting_tree benchmark_fiting_tree.cpp)
target_link_libraries(benchmark_fiting_tree PRIVATE pgm_common)
target_include_directories(benchmark_fiting_tree PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pgm
    ${CMAKE_SOURCE_DIR}/third_party/FITing-Tree/include
    ${CMAKE_SOURCE_DIR}/third_party/FITing-Tree/lib/stx-btree-0.9/include
)
