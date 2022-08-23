cmake_minimum_required(VERSION 3.4...3.18)
project(slam)


#---------------------------------------------------------------------------------------
# compiler config
#---------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++17 ${CMAKE_CXX_FLAGS}")
endif()




#-------------------------------------------------------
# header and myproject_SOURCES
#-------------------------------------------------------
set(HEADER_BASE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SCRIPT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
file(GLOB SRC_FILES ${SOURCE_DIR}/*.cpp)


#---------------------------------------------------------------------------------------
# essential libraries and variables
#---------------------------------------------------------------------------------------
#find_package (Threads)
find_package (Boost)
find_package(Eigen3 3.4 REQUIRED)
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
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})



pybind11_add_module(slam "${SCRIPT_DIR}/main.cpp" ${SRC_FILES} )
target_compile_definitions(slam PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} )
target_precompile_headers(slam PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/base.h")
target_include_directories(slam PRIVATE ${HEADER_BASE} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(slam PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES}  nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX Eigen3::Eigen)

#pybind11_add_module(binmatch "${SCRIPT_DIR}/binmatchwrap.cpp" ${SRC_FILES} )
#target_compile_definitions(binmatch PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} )
#target_precompile_headers(binmatch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/base.h")
#target_include_directories(binmatch PRIVATE ${HEADER_BASE} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
#target_link_libraries(binmatch PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES}  nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX Eigen3::Eigen)

pybind11_add_module(kittilocal "${SCRIPT_DIR}/kittiLocalize.cpp" ${SRC_FILES} )
target_compile_definitions(kittilocal PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} )
target_precompile_headers(kittilocal PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/base.h")
target_include_directories(kittilocal PRIVATE ${HEADER_BASE} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(kittilocal PRIVATE ${PCL_LIBRARIES} ${Boost_LIBRARIES}  nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX Eigen3::Eigen)

add_executable (pclvisualizer "${SCRIPT_DIR}/pclvisualizer.cpp")
target_link_libraries (pclvisualizer ${PCL_LIBRARIES})

add_executable (pclviewer "${SCRIPT_DIR}/pclviewer.cpp")
target_link_libraries (pclviewer ${PCL_LIBRARIES})

#target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} ${PCL_DEFINITIONS})
#target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${HEADER_BASE} ${Boost_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
#target_link_libraries(${CMAKE_PROJECT_NAME} PRIVA ${PCL_LIBRARIES} ${Boost_LIBRARIES})

set_target_properties(slam
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

#set_target_properties(binmatch
#    PROPERTIES
#    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#)

set_target_properties(kittilocal
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set_target_properties(pclvisualizer
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(pclviewer
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
