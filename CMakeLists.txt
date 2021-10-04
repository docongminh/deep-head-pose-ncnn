#1.cmake verson
cmake_minimum_required(VERSION 2.8)

#2.project name
project(HeadPose)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

message(${SRC_FILES})

# list(APPEND SRCS ${LAYER_ARCH_SRC})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -std=c++11 -pthread -fopenmp")
add_library(HeadPose STATIC ${SRC_FILES})

find_package(OpenCV)
message("Opencv Version: ${OPENCV_VERSION}" )
include_directories(${CMAKE_SOURCE_DIR}/ncnn/include/ncnn)
link_directories(${CMAKE_SOURCE_DIR}/ncnn/lib)
target_link_libraries(HeadPose PUBLIC ${OpenCV_LIBS} ncnn)
#
target_include_directories(HeadPose
    PUBLIC
        # include for detect face
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
)
#
add_executable(main ${CMAKE_SOURCE_DIR}/src/main.cpp)

# 
target_link_libraries(main PUBLIC HeadPose)