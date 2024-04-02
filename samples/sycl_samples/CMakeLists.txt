cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_COMPILER_ID IntelLLVM)

find_package(IntelSYCL REQUIRED)

add_executable(sycl_samples)
target_sources(
    sycl_samples
    PRIVATE
    pixels_sample.cpp
    )

add_sycl_to_target(TARGET sycl_samples SOURCES pixels_sample.cpp)

target_include_directories(sycl_samples PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/samples>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${install_dir}/${PROJECT_NAME}>
)

target_link_libraries(sycl_samples ${PROJECT_NAME})
