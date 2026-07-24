# benchmark_radix_spline
add_executable(benchmark_radix_spline benchmark_radix_spline.cpp)
target_include_directories(benchmark_radix_spline PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pgm
    ${CMAKE_SOURCE_DIR}/third_party/RadixSpline/include
)
target_link_libraries(benchmark_radix_spline PRIVATE pgm_common)
