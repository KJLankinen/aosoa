cmake_minimum_required(VERSION 3.17)

set(headers
    accessor.h
    aligned_pointers.h
    aosoa.h
    compile_time_string.h
    cuda_memory_operations.h
    c_memory_operations.h
    definitions.h
    memory_operations.h
    row.h
    type_operations.h
    variable.h
    )

set(install_dir "include/${PROJECT_NAME}")

add_library(${PROJECT_NAME} INTERFACE ${headers})

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${install_dir}>
)

install(TARGETS ${PROJECT_NAME})
install(FILES ${headers}
    DESTINATION ${install_dir}
)
