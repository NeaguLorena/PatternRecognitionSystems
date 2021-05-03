cmake_minimum_required(VERSION 3.15)
project(ImageProcessing)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(ImageProcessing
        OpenCVApplication.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ImageProcessing ${OpenCV_LIBS})