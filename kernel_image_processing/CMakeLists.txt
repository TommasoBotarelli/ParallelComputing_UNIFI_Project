cmake_minimum_required(VERSION 3.5)
project(
        kernel_image_processing
        VERSION 1.0
        DESCRIPTION "2nd laboratory of parallel programming course"
        LANGUAGES CUDA CXX
        )

set(OpenCV_DIR ~/opencv_build/install/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(
               kernel_image_processing 
               src/main.cpp
              )

target_link_libraries(kernel_image_processing ${OpenCV_LIBS})
target_link_libraries(kernel_image_processing OpenMP::OpenMP_CXX)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86")

target_compile_options(kernel_image_processing PRIVATE -fopenmp -Wall -O3 -march=native -fopt-info-vec-optimized)
