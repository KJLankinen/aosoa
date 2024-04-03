cmake_minimum_required(VERSION 3.17)

include_directories(SYSTEM
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_subdirectory(cpu_samples)
add_subdirectory(cuda_samples)
add_subdirectory(hip_samples)
add_subdirectory(sycl_samples)
