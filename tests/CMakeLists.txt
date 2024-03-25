cmake_minimum_required(VERSION 3.17)

set(testname ${PROJECT_NAME}_tests)
# create the testing file and list of tests
create_test_sourcelist(Tests
  ${testname}.cpp
  tests.cpp
  )

# add the executable
add_executable(${testname} ${Tests})

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tabulate/include"
    )

target_link_libraries(${testname} PUBLIC ${PROJECT_NAME})

# remove the test driver source file
set(TestsToRun ${Tests})
remove(TestsToRun ${testname}.cpp)

# Add all the ADD_TEST for each test
foreach(test ${TestsToRun})
  get_filename_component(TName ${test} NAME_WE)
  add_test(NAME ${TName} COMMAND ${testname} ${TName})
endforeach()
