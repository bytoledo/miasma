cmake_minimum_required(VERSION 3.20)

project(miasma_parallel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_library(miasma_core STATIC
    src/cuda_engine.cu
    src/cpu_engine.cpp
)

target_include_directories(miasma_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(miasma_core PUBLIC
    CUDA::cudart
    CUDA::curand
    OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(miasma_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor /O2 /arch:AVX2 /W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=/Zc:preprocessor>
    )
else()
    target_compile_options(miasma_core PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -mavx2 -Wall -Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
endif()

if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(miasma_parallel src/main.cpp)
    target_link_libraries(miasma_parallel PRIVATE miasma_core opengl32 user32 gdi32 comctl32)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_invariants.cpp")
    enable_testing()
    add_executable(miasma_tests tests/test_invariants.cpp)
    target_link_libraries(miasma_tests PRIVATE miasma_core)
    add_test(NAME miasma_invariants COMMAND miasma_tests)
endif()
