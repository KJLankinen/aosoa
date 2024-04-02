cmake_minimum_required(VERSION 3.17)

set(testname ${PROJECT_NAME}_tests)

add_executable(
    ${testname}
    tests.cpp
    row_tests.cpp
)

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/googletest-1.14.0/googletest/include"
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
  ${PROJECT_NAME}
)

add_subdirectory(googletest-1.14.0)

include(GoogleTest)
gtest_discover_tests(${testname})

