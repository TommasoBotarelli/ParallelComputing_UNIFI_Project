cmake_minimum_required(VERSION 3.5)
project(
        kernel_image_processing
        VERSION 1.0
        DESCRIPTION "1st laboratory of parallel programming course"
        LANGUAGES CXX
        )

find_package(OpenCV REQUIRED)

add_executable(
               kernel_image_processing 
               src/main.cpp
              )

target_link_libraries(kernel_image_processing ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
