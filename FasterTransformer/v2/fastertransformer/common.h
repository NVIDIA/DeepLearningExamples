/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace fastertransformer{

  enum class OperationType{FP32, FP16};
  enum class AllocatorType{CUDA, TF};

#define PRINT_FUNC_NAME_() do{\
  std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
} while (0)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}


template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + \
        (_cudaGetErrorEnum(result)) + " " + file +  \
        ":" + std::to_string(line) + " \n");\
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void print_to_file(T* result, const int size, char* file)
{
  FILE* fd = fopen(file, "w");
  float* tmp = (float*)malloc(sizeof(float) * size);
  check_cuda_error(cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost));
  for(int i = 0; i < size; ++i)
    fprintf(fd, "%f\n", (float)tmp[i]);
  free(tmp);
  fclose(fd);
}

template <typename T>
void print_to_screen(T* result, const int size)
{
  float* tmp = (float*)malloc(sizeof(float) * size);
  check_cuda_error(cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost));
  for(int i = 0; i < size; ++i)
    printf("%d, %f\n", i, (float)tmp[i]);
  free(tmp);
}

template<typename T>
void check_max_val(const T* result, const int size){
  T* tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float max_val = -100000;
  for(int i = 0 ; i < size; i++){
    float val = (float)(tmp[i]);
    if(val > max_val) max_val = val;
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

}//namespace fastertransformer
