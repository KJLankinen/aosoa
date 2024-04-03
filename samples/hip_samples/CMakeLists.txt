cmake_minimum_required(VERSION 3.21)

enable_language(HIP)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set(hip_archs gfx90a;gfx90c)

if(WIN32)
    set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

add_executable(hip_samples)
target_sources(
    hip_samples
    PRIVATE
    pixels_sample.cpp
    )

set_source_files_properties(pixels_sample.cpp PROPERTIES LANGUAGE HIP)

target_include_directories(hip_samples PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/samples>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${install_dir}/${PROJECT_NAME}>
)

set_target_properties(hip_samples PROPERTIES HIP_ARCHITECTURES "${hip_archs}")
target_link_libraries(hip_samples ${PROJECT_NAME})
