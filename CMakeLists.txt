cmake_minimum_required(VERSION 3.20)
project(pgm_attack_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenMP)
find_package(nlohmann_json REQUIRED)

# ---- SDSL (installed to /usr/local by Dockerfile) ----
find_path(SDSL_INCLUDE_DIR sdsl/bit_vectors.hpp PATHS /usr/local/include)
find_library(SDSL_LIBRARY sdsl PATHS /usr/local/lib)

if(NOT SDSL_INCLUDE_DIR OR NOT SDSL_LIBRARY)
    message(FATAL_ERROR "SDSL not found. Did you install sdsl-lite to /usr/local?")
endif()

add_library(sdsl::sdsl UNKNOWN IMPORTED)
set_target_properties(sdsl::sdsl PROPERTIES
    IMPORTED_LOCATION "${SDSL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SDSL_INCLUDE_DIR}"
)

# ---- PGM-Index (included as a subdirectory) ----
set(PGM_INDEX_INCLUDE_DIR
    ${CMAKE_SOURCE_DIR}/third_party/pgm-index/include
)

add_subdirectory(src)
