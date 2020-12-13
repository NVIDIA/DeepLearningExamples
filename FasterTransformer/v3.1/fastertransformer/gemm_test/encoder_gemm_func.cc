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

#include "encoder_gemm_func.h"
#include "fastertransformer/common.h"
#include <vector>


namespace fastertransformer{

double diffTime(timeval start, timeval end)
{
  return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

template<typename T>
void generate_encoder_gemm_config(int batch_size,
                                    int seq_len,
                                    int head_num,
                                    int size_per_head,
                                    void *buffer, 
                                    bool isAppend)
{

  
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);
  
  //check config 
  FILE *fd;
  if (!isAppend)
  {
    fd = fopen(GEMM_CONFIG, "w+");
  }
  else
  {
    fd = fopen(GEMM_CONFIG, "a+");
    std::vector<std::string> config;
    char line[1024];
    while (fgets(line, 1024, fd) != NULL)
    {
      config.push_back(std::string(line));
    }
    if (config.size() >= MAX_CONFIG_NUM*GEMM_NUM)
    {
      int startIdx = config.size() - (MAX_CONFIG_NUM - 1)*GEMM_NUM;
      fclose(fd);
      fd = fopen(GEMM_CONFIG, "w+");
      for (int i = startIdx ; i < config.size() ; i++)
      {
        fprintf(fd, "%s", config[i].c_str());
      }
    }
  }

  const int gemm_num = 6;
  int M[gemm_num];
  int N[gemm_num];
  int K[gemm_num];
  int batchCount[gemm_num] = {1,1,1,1,1,1};
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

  M[5] = batch_size * seq_len;
  N[5] = head_num * size_per_head; 
  K[5] = N[5];
  batchCount[5] = 3;
  strcpy(mess[5], "from_tensor * weight_QKV in BatchGemm");

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

  printf("***Encoder Gemm Testing Begin***\n");
  for(int i = 0; i < gemm_num; ++i)
  {
    // if(i != 0 && i != 5) continue; 

    int m = M[i], n = N[i], k = K[i];
    printf("\n-----------------------------\n");
    printf("GEMM test %d: [M: %d, K: %d, N: %d] %s\n", i, m, k, n, mess[i]);
    T* d_A = (T*)buffer;
    T* d_B = d_A + m * k * batchCount[i];
    T* d_C = d_B + k * n * batchCount[i];

    // array of pointer for batchedGemm
    T* harray[9];
    harray[0] = (T*)buffer;
    harray[1] = (T*)(buffer + sizeof(T) * m * k);
    harray[2] = (T*)(buffer + 2 * sizeof(T) * m * k);
    harray[3] = (T*)(buffer + 3 * sizeof(T) * m * k);
    harray[4] = (T*)(buffer + 3 * sizeof(T) * m * k + sizeof(T) * k * n);
    harray[5] = (T*)(buffer + 3 * sizeof(T) * m * k + 2 * sizeof(T) * k * n);
    harray[6] = (T*)(buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n);
    harray[7] = (T*)(buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n + sizeof(T) * m * n);
    harray[8] = (T*)(buffer + 3 * sizeof(T) * m * k + 3 * sizeof(T) * k * n + 2 * sizeof(T) * m * n);

    T** darray = 0;
    check_cuda_error(cudaMalloc((void**)&darray, sizeof(T*) * 9));
    cudaMemcpy((void*)darray, (void*)harray, sizeof(T*) * 9, cudaMemcpyHostToDevice);
    T** dAarray = darray;
    T** dBarray = darray + 3;
    T** dCarray = darray + 6;

    float exec_time = 99999.0f;
    int fast_algo = 0;
    for(int algo = startAlgo; algo <= endAlgo; algo++)
    {
      cublasStatus_t status;
      cudaDeviceSynchronize();
      gettimeofday(&start, NULL);
      for(int ite = 0; ite < ites; ++ite)
      {
        if(i < 3)
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
        else if(i == 3)
        {
          status = cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, size_per_head,
                &alpha,
                d_B, BType, size_per_head, seq_len * size_per_head,
                d_A, AType, size_per_head, seq_len * size_per_head,
                &beta,
                d_C, CType, seq_len, seq_len * seq_len,
                batch_size * head_num,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
        }
        else if(i == 4)
        {
          status = cublasGemmStridedBatchedEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                size_per_head, seq_len, seq_len,
                &alpha,
                d_B, BType, size_per_head, seq_len * size_per_head,
                d_A, AType, seq_len, seq_len * seq_len,
                &beta,
                d_C, CType, size_per_head, seq_len * size_per_head,
                batch_size * head_num,
                computeType,
                static_cast<cublasGemmAlgo_t>(algo));
        }
        else if(i == 5)
        {
          status = cublasGemmBatchedEx(cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N, 
                            n, m, k, 
                            &alpha, 
                            (const void* const*) dBarray, BType, n,
                            (const void* const*) dAarray, AType, k,
                            &beta,
                            (void* const*)dCarray, CType, n,
                            3, 
                            computeType,
                            static_cast<cublasGemmAlgo_t>(algo));
        }
        if(status != CUBLAS_STATUS_SUCCESS) break;
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
    int is_fp16 = 0;
    if (sizeof(T) == sizeof(half))
        is_fp16 = 1;
    fprintf(fd, "%d %d %d %d ### %d %d %d %d %d %d %f\n", batch_size, seq_len, head_num, size_per_head, batchCount[i], m, n, k, is_fp16, fast_algo, exec_time);
    cudaFree(darray);
  }
 
  fclose(fd); 
  printf("***Encoder Gemm Testing End***\n");
  return;
}

template void generate_encoder_gemm_config<float>(int batch_size, int seq_len, int head_num, int size_per_head, void *buffer, bool isAppend);
template void generate_encoder_gemm_config<half>(int batch_size, int seq_len, int head_num, int size_per_head, void *buffer, bool isAppend);


size_t calGemmTestBufSizeInByte(int batch_size, int seq_len, int head_num, int size_per_head, int int8_mode, int is_fp16)
{
    size_t buf_size_in_byte;
    if (int8_mode > 0)
    {	
      int m = batch_size*seq_len;
      int n = head_num*size_per_head;
      int k = n;
      int batchCount;

      size_t size1 = 3*(m*k*sizeof(int8_t) + k*n*sizeof(int8_t) + m*n*sizeof(int));
      size_t size2 = batch_size*head_num*(seq_len*size_per_head*sizeof(int8_t) + size_per_head*seq_len*sizeof(int8_t) + seq_len*seq_len*sizeof(int));
      size_t size3 = batch_size*head_num*(seq_len*seq_len*sizeof(int8_t) + seq_len*size_per_head*sizeof(int8_t) + seq_len*size_per_head*sizeof(int));
      size_t size4 = m*k*sizeof(int8_t) + k*4*n*sizeof(int8_t) + 4*m*n*sizeof(int);
      buf_size_in_byte = size1 > size2 ? size1 : size2;
      buf_size_in_byte = buf_size_in_byte > size3 ? buf_size_in_byte : size3;
      buf_size_in_byte = buf_size_in_byte > size4 ? buf_size_in_byte : size4;
    }
    else
    {
      int m = batch_size*seq_len;
      int n = head_num*size_per_head;
      int k = n;
      int wordSize = (is_fp16 == 1 ? sizeof(half) : sizeof(float));
      size_t size1 = 3*(m*k + k*n + m*n)*wordSize;
      size_t size2 = batch_size*head_num*(seq_len*seq_len + seq_len*size_per_head + seq_len*size_per_head)*wordSize;
      size_t size3 = (m*k + k*4*n + m*4*n)*wordSize;
      buf_size_in_byte = size1 > size2 ? size1 : size2;
      buf_size_in_byte = buf_size_in_byte > size3 ? buf_size_in_byte : size3;
    }
    return buf_size_in_byte;
}

}
