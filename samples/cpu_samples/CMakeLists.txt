cmake_minimum_required(VERSION 3.17)

add_executable(cpu_sample)
target_sources(
    cpu_sample
    PRIVATE
    pixels_sample.cpp
    )

target_include_directories(cpu_sample PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/samples>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${install_dir}/${PROJECT_NAME}>
)

target_link_libraries(cpu_sample ${PROJECT_NAME})
