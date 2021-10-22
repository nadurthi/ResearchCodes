cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)


#---------------------------------------------------------------------------------------
# compiler config
#---------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++1y ${CMAKE_CXX_FLAGS}")
endif()




#-------------------------------------------------------
# header and myproject_SOURCES
#-------------------------------------------------------
set(HEADER_BASE "${CMAKE_CURRENT_SOURCE_DIR}/include")


#---------------------------------------------------------------------------------------
# essential libraries and variables
#---------------------------------------------------------------------------------------
#find_package (Threads)
#find_package (Boost)
#find_package(Eigen3 REQUIRED)

# find_package(spdlog REQUIRED)
# 



#-------------------------------------------------------------------------------------------
# Add pybind11 project
#-------------------------------------------------------------------------------------------
add_subdirectory(pybind11)





pybind11_add_module(cmake_example src/main.cpp)
target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(cmake_example PUBLIC ${HEADER_BASE} ${Boost_INCLUDE_DIRS})
#target_link_libraries(cmake_example ${Boost_LIBRARIES})

set_target_properties(cmake_example 
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)