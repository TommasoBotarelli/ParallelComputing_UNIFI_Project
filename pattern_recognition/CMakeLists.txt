cmake_minimum_required(VERSION 3.5)

project(
        patternRecognition
        VERSION 1.0
        DESCRIPTION "1st laboratory of parallel programming course"
        LANGUAGES CXX
        )

find_package(OpenMP REQUIRED)
set({OMP_PROC_BIND} spread)

add_executable(
               patternRecognition
               src/main.cpp
               src/algo.cpp
               src/sequentialAlgo.cpp
               src/parallelStaticAlgo.cpp
               src/parallelStaticAlgoNoFirstTouch.cpp
               src/parallelSmartStaticAlgo.cpp
               src/parallelManualAlgo.cpp
               src/parallelManualAlgoNoFirstTouch.cpp
               src/parallelDynamicAlgo.cpp
               src/parallelDynamicAlgoNoFirstTouch.cpp
               )

target_link_libraries(patternRecognition PUBLIC OpenMP::OpenMP_CXX)

target_compile_features(patternRecognition PRIVATE cxx_std_20)
target_compile_options(patternRecognition PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native -fopt-info-vec-optimized)