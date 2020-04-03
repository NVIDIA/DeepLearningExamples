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

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <ctime>
#include <sys/time.h>
#include "common.h"

using namespace std;

template<typename T>
void generate_decoding_gemm_config(int batch_size,
                                  int beam_width,
                                  int head_number,
                                  int size_per_head,
                                  int vocab_size,
                                  int seq_len,
                                  int memory_hidden_units)
{
  FILE* fd = fopen("decoding_gemm_config.in", "w");
  if(fd == NULL)
  {
    printf("[ERROR] Cannot write to file decoding_gemm_config.in\n");
    return;
  }

  const int hidden_units = head_number * size_per_head;
  const int gemm_num = 5;
  int M[gemm_num];
  int N[gemm_num];
  int K[gemm_num];
  char mess[gemm_num][256];
  
  //gemm1 
  M[0] = batch_size * beam_width;
  K[0] = hidden_units;
  N[0] = vocab_size;
  strcpy(mess[0], "decoder_output * embedding_kernel -> embedding_output");

  //gemm2
  M[1] = batch_size * beam_width;
  K[1] = hidden_units;
  N[1] = hidden_units;
  strcpy(mess[1], "from_tensor * weightQ/K/V in masked attention");

  //gemm3
  M[2] = M[0] * seq_len;
  K[2] = memory_hidden_units;
  N[2] = hidden_units;
  strcpy(mess[2], "from_tensor * weightK/V in cross attention");

  M[3] = batch_size * beam_width;
  K[3] = hidden_units;
  N[3] = hidden_units * 4;
  strcpy(mess[3], "ffn gemm1 ");

  M[4] = batch_size * beam_width;
  K[4] = hidden_units * 4;
  N[4] = hidden_units; 
  strcpy(mess[4], "ffn gemm2");

  cublasHandle_t cublas_handle;
  check_cuda_error(cublasCreate(&cublas_handle));

  cudaDataType_t AType;
  cudaDataType_t BType;
  cudaDataType_t CType;
  cudaDataType_t computeType;
  int startAlgo, endAlgo;
  const int ites = 100;
  struct timeval start, end;
  
  if(sizeof(T) == sizeof(float)){
    AType = CUDA_R_32F;
    BType = CUDA_R_32F;
    CType = CUDA_R_32F;
    computeType = CUDA_R_32F;
    startAlgo = (int)CUBLAS_GEMM_DEFAULT;
    endAlgo = (int)CUBLAS_GEMM_ALGO23;
  }
  else{
    AType = CUDA_R_16F;
    BType = CUDA_R_16F;
    CType = CUDA_R_16F;
    computeType = CUDA_R_16F;
    startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  }
  T alpha = (T)1.0f;
  T beta = (T)0.0f;

  printf("***Decoding Gemm Testing***\n");
  for(int i = 0; i < gemm_num; ++i)
  {
    int m = M[i], n = N[i], k = K[i];
    printf("\n-----------------------------\n");
    printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
    T* d_A;
    T* d_B;
    T* d_C;
    check_cuda_error(cudaMalloc((void**)&d_A, sizeof(T) * m * k));
    check_cuda_error(cudaMalloc((void**)&d_B, sizeof(T) * k * n));
    check_cuda_error(cudaMalloc((void**)&d_C, sizeof(T) * m * n));

    float exec_time = 99999.0f;
    int fast_algo = 0;
    for(int algo = startAlgo; algo <= endAlgo; algo++)
    {
      cublasStatus_t status;
      cudaDeviceSynchronize();
      gettimeofday(&start, NULL);
      for(int ite = 0; ite < ites; ++ite)
      {
        status = cublasGemmEx(cublas_handle, 
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k, 
                              &alpha, 
                              d_B, BType, n, 
                              d_A, AType, k, 
                              &beta, 
                              d_C, CType, n, 
                              computeType, 
                              static_cast<cublasGemmAlgo_t>(algo));
      }
      cudaDeviceSynchronize();
      gettimeofday(&end, NULL);
      if(status == CUBLAS_STATUS_SUCCESS)
      {
        printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
        if(diffTime(start, end) / ites < exec_time)
        {
          exec_time = diffTime(start, end) / ites;
          fast_algo = algo;
        }
      }
    }
    printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
    fprintf(fd, "%d\n", fast_algo);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }
}

