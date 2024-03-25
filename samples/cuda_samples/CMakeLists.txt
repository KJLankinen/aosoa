cmake_minimum_required(VERSION 3.17)

enable_language(CUDA)
set(cuda_archs 70;72)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_samples)
target_sources(
    cuda_samples
    PRIVATE
    cuda_sample.cu
    )

include_directories(PRIVATE "${PROJECT_SOURCE_DIR}/include")

set_target_properties(cuda_samples PROPERTIES CUDA_ARCHITECTURES "${cuda_archs}")
target_link_libraries(cuda_samples ${PROJECT_NAME})
