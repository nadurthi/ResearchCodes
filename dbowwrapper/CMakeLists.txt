cmake_minimum_required(VERSION 2.8.12)
project(dbowwrapper)

#find_package(PCL 1.10 REQUIRED)
#find_package(VTK 8.0 REQUIRED)
find_package(DBoW3 REQUIRED)
#find_package(pybind11 REQUIRED)




#include(${VTK_USE_FILE})
include_directories(${DBoW3_INCLUDE_DIRS})
link_directories(${DBoW3_LIBRARY_DIRS})
add_definitions(${DBoW3_DEFINITIONS})

add_subdirectory(pybind11)
pybind11_add_module(dbowwrapper src/main.cpp)
target_link_libraries (dbowwrapper PRIVATE ${DBoW3_LIBRARIES})
