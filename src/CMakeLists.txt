cmake_minimum_required(VERSION 3.17)

set(headers
    aosoa.h
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
