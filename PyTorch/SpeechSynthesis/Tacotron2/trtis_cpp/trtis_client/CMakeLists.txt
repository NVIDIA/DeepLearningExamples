cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tacotron2_inference)

if (DEFINED DEVEL AND NOT DEVEL EQUAL 0) 
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Wall")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Werror")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Wpedantic")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Weffc++")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -Wextra")
    set(CPP_DEVEL_FLAGS "${CPP_DEVEL_FLAGS} -DDEVEL=1")

    # nvcc warnings
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler -Wall")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler -Werror")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler -Weffc++")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler -Wextra")
    set(CUDA_DEVEL_FLAGS "${CUDA_DEVEL_FLAGS} -Xcompiler -DDEVEL=1")
  endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

separate_arguments(CPP_DEVEL_FLAGS)
separate_arguments(CUDA_DEVEL_FLAGS)

add_subdirectory("src")
