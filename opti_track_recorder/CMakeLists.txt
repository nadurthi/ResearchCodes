#-------------------------------------------------------------------------------------------
# Copyright (C) Electronic Arts Inc.  All rights reserved.
#-------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.1)
project(recorder C CXX)

#---------------------------------------------------------------------------------------
# compiler config
#---------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++1y ${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#-------------------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------------------
option(recorder_BUILD_BENCHMARK "Enable generation of build files for benchmark" ON)
option(recorder_BUILD_TESTS "Enable generation of build files for tests" ON)
option(recorder_BUILD_MAINS "Enable generation of build files for main scripts" ON)

#-------------------------------------------------------
# header and recorder_SOURCES
#-------------------------------------------------------
set(HEADER_BASE "${CMAKE_CURRENT_SOURCE_DIR}/include")

#---------------------------------------------------------------------------------------
# essential libraries and variables
#---------------------------------------------------------------------------------------
find_package (Threads)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# find_package (Boost)
# 
# find_library(UQ
#     NAMES UQ libUQ
# )

# find_package(Eigen3 REQUIRED)
# find_package(spdlog REQUIRED)
# 
#-------------------------------------------------------------------------------------------
# Sub-projects
#-------------------------------------------------------------------------------------------
# add_subdirectory(test/packages/UQBase)

# if(UQ_BUILD_TESTS OR UQ_BUILD_BENCHMARK)
#     add_subdirectory(test/packages/UQAssert)
# endif()

# if(recorder_BUILD_BENCHMARK)
#     add_subdirectory(benchmark)
# endif()

# if(recorder_BUILD_MAINS)
#     add_subdirectory(mains)
# endif()

# if(recorder_BUILD_TESTS)
#     add_subdirectory(test)
# endif()

#-------------------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------------------
# add_definitions(-D_RELEASE)

#-------------------------------------------------------------------------------------------
# Library definition
#-------------------------------------------------------------------------------------------
file(GLOB recorder_SOURCES "src/*.cpp" "include/*.h")
add_library(recorder ${recorder_SOURCES})
include_directories("include")
# target_link_libraries(recorder ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
# target_link_libraries(recorder ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})

#--------------------------
# Add executable
#----------------------------
# file(GLOB recorder_SOURCES "src/*.cpp" "include/recorder/*.h")
# add_library(recorder ${recorder_SOURCES})

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

set_target_properties(recorder PROPERTIES LINKER_LANGUAGE CXX)

add_executable(smplclt mains/SampleClient.cpp ${recorder_SOURCES})
target_link_libraries(smplclt recorder NatNetLibShared Eigen3::Eigen)

add_executable(optrkrecord mains/optitrack_recorder.cpp ${recorder_SOURCES})
target_link_libraries(optrkrecord recorder NatNetLibShared Eigen3::Eigen)



#-------------------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------------------
install(TARGETS recorder DESTINATION lib)
install(DIRECTORY include/recorder DESTINATION include)

