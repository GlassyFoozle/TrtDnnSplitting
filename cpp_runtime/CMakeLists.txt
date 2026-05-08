cmake_minimum_required(VERSION 3.18)
project(table4_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── TensorRT ──────────────────────────────────────────────────────────────────
# On Jetson (JetPack 6 / aarch64) TRT headers and libs land in standard paths.
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include
    REQUIRED
)
find_library(TRT_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib
    REQUIRED
)
find_library(TRT_RUNTIME_LIB nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib
)

message(STATUS "TRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TRT lib:     ${TRT_LIB}")

# ── nlohmann/json (header-only, fetched from GitHub) ─────────────────────────
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(table4_runner
    src/trt_engine.cpp
    src/buffer_pool.cpp
    src/chunk_pipeline.cpp
    src/main_table4.cpp
)

target_include_directories(table4_runner PRIVATE
    include
    ${TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(table4_runner PRIVATE
    ${TRT_LIB}
    CUDA::cudart
    nlohmann_json::nlohmann_json
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS table4_runner DESTINATION bin)
