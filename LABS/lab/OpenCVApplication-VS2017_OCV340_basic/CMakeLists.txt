cmake_minimum_required(VERSION 3.15)
project(Lab_prs)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Lab_prs
          covariance.cpp)

add_executable(Lab_prs1
        exam.cpp)

add_executable(Lab_prs2
        OpenCVApplication.cpp)

add_executable(Lab_prs3
        OpenCVApplication-8.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Lab_prs ${OpenCV_LIBS})