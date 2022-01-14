cmake_minimum_required(VERSION 3.4...3.18)
project(slam)


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
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB SRC_FILES ${SOURCE_DIR}/*.cpp)

#---------------------------------------------------------------------------------------
# essential libraries and variables
#---------------------------------------------------------------------------------------
#find_package (Threads)
find_package (Boost)
find_package(Eigen3 REQUIRED)
find_package(PCL)
find_package(OpenMP REQUIRED)
# find_package(spdlog REQUIRED)
#

find_package(nlohmann_json)

#-------------------------------------------------------------------------------------------
# Add pybind11 project
#-------------------------------------------------------------------------------------------
add_subdirectory(pybind11)
add_definitions(${PCL_DEFINITIONS})




pybind11_add_module(${CMAKE_PROJECT_NAME} "${SOURCE_DIR}/main.cpp" ${SRC_FILES} )
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} )
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${HEADER_BASE} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES}  nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)

#target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} ${PCL_DEFINITIONS})
#target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${HEADER_BASE} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVA ${PCL_LIBRARIES} ${Boost_LIBRARIES})

set_target_properties(${CMAKE_PROJECT_NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
