// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

#ifndef COMMON_H_
#define COMMON_H_

using ULLInt = unsigned long long int;

// Use to compute things like number of blocks
#define CEIL_DIV_INT(a, b) ((a + b - 1) / b)

#define CUDA_CHECK(cmd)                                                                     \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)


#endif  // COMMON_H_
