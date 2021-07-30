cmake_minimum_required(VERSION 3.1)
project(facefeature-tnn CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC  -std=c++11")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -O3 -fPIC -pthread")

option(TNN_OPENVINO_ENABLE "with openvino lib" ON)
option(LOG_ENABLE "with log print" OFF)

message(${TNN_LIB_DIR})
message(${TNN_INCLUDE_DIR})
message(${OpenCV_DIR})
message(${GFLAGS_DIR})

if (${TNN_OPENVINO_ENABLE})
    add_definitions(-D_OPENVINO_)
endif()

if (${LOG_ENABLE})
    add_definitions(-D_LOG_)
endif()

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${TNN_INCLUDE_DIR})
link_directories(${TNN_LIB_DIR})
link_libraries(TNN)

add_subdirectory(${GFLAGS_DIR} ${GFLAGS_DIR})
get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
link_libraries(gflags)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/)
file(GLOB_RECURSE
    ALL_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/core/**.cpp
)

add_library(FaceFeature SHARED ${ALL_SOURCE_FILES})
add_executable(test test/test.cpp)
target_link_libraries(test FaceFeature ${OpenCV_LIBS})

