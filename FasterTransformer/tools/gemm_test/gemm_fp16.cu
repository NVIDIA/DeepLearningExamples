/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
#include <unistd.h>
#include <sys/time.h>
#include "common.h"
using namespace std;
double diffTime(timeval start, timeval end)
{
  return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}
int main(int argc, char* argv[])
{
  FILE* fd = fopen("gemm_config.in", "w");
  if(fd == NULL)
  {
    printf("Cannot write to file gemm_config.in\n");
    return 0;
  }
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device %s\n", prop.name);

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int head_num = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);

  const int gemm_num = 5;
  int M[gemm_num];
  int N[gemm_num];
  int K[gemm_num];
  int batchCount[gemm_num] = {1,1,1,1,1};
  char mess[gemm_num][256];
  
  //gemm1 
  M[0] = batch_size * seq_len;
  K[0] = head_num * size_per_head;
  N[0] = K[0];
  strcpy(mess[0], "from_tensor * weightQ/K/V, attr * output_kernel");

  //gemm2
  M[1] = M[0];
  K[1] = K[0];
  N[1] = 4 * N[0];
  strcpy(mess[1], "attr_output * inter_kernel");

  //gemm3
  M[2] = M[0];
  K[2] = 4 * K[0];
  N[2] = N[0];
  strcpy(mess[2], "inter_matmul * output_kernel");

  M[3] = seq_len;
  N[3] = seq_len;
  K[3] = size_per_head;
  batchCount[3] = batch_size * head_num;
  strcpy(mess[3], "attention batched Gemm1");

  M[4] = seq_len;
  N[4] = size_per_head; 
  K[4] = seq_len;
  batchCount[4] = batch_size * head_num;
  strcpy(mess[4], "attention batched Gemm2");

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  typedef __half T;
  cudaDataType_t AType = CUDA_R_16F;
  cudaDataType_t BType = CUDA_R_16F;
  cudaDataType_t CType = CUDA_R_16F;
  cudaDataType_t computeType = CUDA_R_16F;
  const int ites = 100;
  struct timeval start, end;
  int startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  int endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
  T alpha = (T)1.0f;
  T beta = (T)0.0f;

  printf("***FP16 Gemm Testing***\n");
  for(int i = 0; i < gemm_num; ++i)
  {
    int m = M[i], n = N[i], k = K[i];
    printf("\n-----------------------------\n");
    printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
    T* d_A;
    T* d_B;
    T* d_C;
    check_cuda_error(cudaMalloc((void**)&d_A, sizeof(T) * m * k * batchCount[i]));
    check_cuda_error(cudaMalloc((void**)&d_B, sizeof(T) * k * n * batchCount[i]));
    check_cuda_error(cudaMalloc((void**)&d_C, sizeof(T) * m * n * batchCount[i]));

    float exec_time = 99999.0f;
    int fast_algo = 0;
    for(int algo = startAlgo; algo <= endAlgo; algo++)
    {
      cudaDeviceSynchronize();
      gettimeofday(&start, NULL);
      for(int ite = 0; ite < ites; ++ite)
      {
        if(i < 3)
        {
          check_cuda_error(cublasGemmEx(cublas_handle, 
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, 
                &alpha, 
                d_B, BType, n, 
                d_A, AType, k, 
                &beta, 
                d_C, CType, n, 
                computeType, 
                static_cast<cublasGemmAlgo_t>(algo)));
        }
        else if(i == 3)
        {
          check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, size_per_head,
                &alpha,
                d_B, BType, size_per_head, seq_len * size_per_head,
                d_A, AType, size_per_head, seq_len * size_per_head,
                &beta,
                d_C, CType, seq_len, seq_len * seq_len,
                batch_size * head_num,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo)));
        }
        else
        {
          check_cuda_error(cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                size_per_head, seq_len, seq_len,
                &alpha,
                d_B, BType, size_per_head, seq_len * size_per_head,
                d_A, AType, seq_len, seq_len * seq_len,
                &beta,
                d_C, CType, size_per_head, seq_len * size_per_head,
                batch_size * head_num,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo)));
        }
      }
      cudaDeviceSynchronize();
      gettimeofday(&end, NULL);
      printf("algo_%d costs %.3fms \n", algo, diffTime(start, end) / ites);
      if(diffTime(start, end) / ites < exec_time)
      {
        exec_time = diffTime(start, end) / ites;
        fast_algo = algo;
      }
    }
    printf("fast_algo %d costs %.3f ms\n", fast_algo, exec_time);
    fprintf(fd, "%d\n", fast_algo);
  }

}

