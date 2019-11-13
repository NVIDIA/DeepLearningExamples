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
#include "fastertransformer/faster_transformer.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>
using namespace fastertransformer;

typedef __half T;
double diffTime(timeval start, timeval end)
{
  return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

void host_malloc(T** ptr, int size)
{
  (*ptr) = (T*)malloc(sizeof(T) * size);
}

void device_malloc(T** ptr, int size)
{
  cudaMalloc((void**)ptr, sizeof(T) * size);
}

void copy_to_device(T** d_ptr, T** h_ptr, int size)
{
  cudaMemcpy((*d_ptr), (*h_ptr), sizeof(T) * size, cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[])
{
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if(argc != 6)
  {
    printf("./transformer_fp16 batch_size num_layers seq_len head_num size_per_head\n");
    printf("e.g., ./transformer_fp16 1 12 128 12 64\n");
    return 0;
  }

  printf("Device %s\n", prop.name);
  int batch_size = atoi(argv[1]);
  int num_layers = atoi(argv[2]);
  int seq_len = atoi(argv[3]);
  int head_num = atoi(argv[4]);
  int size_per_head = atoi(argv[5]);

  int from_seq_len = seq_len;
  int to_seq_len = seq_len;
  int hidden_dim = head_num * size_per_head;

  T *d_from_tensor = NULL, *d_transformer_out = NULL;
  T *d_attr_kernel_Q = NULL, *d_attr_kernel_K = NULL, *d_attr_kernel_V = NULL;
  T *d_attr_bias_Q = NULL, *d_attr_bias_K = NULL, *d_attr_bias_V = NULL;
  T *d_attr_mask = NULL, *d_attr_output_kernel = NULL, *d_attr_output_bias = NULL;
  T *d_attr_output_layernorm_beta = NULL;
  T *d_attr_output_layernorm_gamma = NULL;
  T *d_inter_kernel = NULL, *d_inter_bias = NULL;
  T *d_output_kernel = NULL, *d_output_bias = NULL, *d_output_layernorm_beta = NULL, *d_output_layernorm_gamma = NULL;

  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  float free = (float)(free_bytes)/ 1024.0 / 1024.0 / 1024.0;
  float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  printf("before allocate free %.2f GB total %.2f GB\n", free, total);

  device_malloc(&d_from_tensor, batch_size * seq_len * hidden_dim);
  device_malloc(&d_transformer_out, batch_size * seq_len * hidden_dim);
  device_malloc(&d_attr_kernel_Q, hidden_dim * hidden_dim);
  device_malloc(&d_attr_kernel_K, hidden_dim * hidden_dim);
  device_malloc(&d_attr_kernel_V, hidden_dim * hidden_dim);
  device_malloc(&d_attr_bias_Q, hidden_dim);
  device_malloc(&d_attr_bias_K, hidden_dim);
  device_malloc(&d_attr_bias_V, hidden_dim);
  device_malloc(&d_attr_mask, batch_size * seq_len * seq_len);
  device_malloc(&d_attr_output_kernel, hidden_dim * hidden_dim);
  device_malloc(&d_attr_output_bias, hidden_dim);
  device_malloc(&d_attr_output_layernorm_beta, hidden_dim);
  device_malloc(&d_attr_output_layernorm_gamma, hidden_dim);
  device_malloc(&d_inter_kernel, hidden_dim * hidden_dim * 4);
  device_malloc(&d_inter_bias, hidden_dim * 4);
  device_malloc(&d_output_kernel, hidden_dim * hidden_dim * 4);
  device_malloc(&d_output_bias, hidden_dim);
  device_malloc(&d_output_layernorm_beta, hidden_dim);
  device_malloc(&d_output_layernorm_gamma, hidden_dim);

  cudaMemGetInfo(&free_bytes, &total_bytes);
  free = (float)(free_bytes)/ 1024.0 / 1024.0 / 1024.0;
  total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  printf("After allocate free %.2f GB used %.2f GB total %.2f GB\n", free, total - free, total);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasSetStream(cublasHandle, stream);

  typedef BertEncoderTransformerTraits<OperationType::HALF,  cuda::OpenMultiHeadAttention> EncoderTraits_;
  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  EncoderInitParam<__half> encoder_param; //init param here

  encoder_param.from_tensor = d_from_tensor;
  encoder_param.to_tensor = d_from_tensor;
  encoder_param.attr_kernel_Q = d_attr_kernel_Q;
  encoder_param.attr_kernel_K = d_attr_kernel_K;
  encoder_param.attr_kernel_V = d_attr_kernel_V;
  encoder_param.attr_bias_Q = d_attr_bias_Q;
  encoder_param.attr_bias_K = d_attr_bias_K;
  encoder_param.attr_bias_V = d_attr_bias_V;
  encoder_param.attr_mask = d_attr_mask;
  encoder_param.attr_output_kernel = d_attr_output_kernel;
  encoder_param.attr_output_bias = d_attr_output_bias;
  encoder_param.attr_output_layernorm_beta = d_attr_output_layernorm_beta;
  encoder_param.attr_output_layernorm_gamma = d_attr_output_layernorm_gamma;
  encoder_param.inter_kernel = d_inter_kernel;
  encoder_param.inter_bias = d_inter_bias;
  encoder_param.output_kernel = d_output_kernel;
  encoder_param.output_bias = d_output_bias;
  encoder_param.output_layernorm_beta = d_output_layernorm_beta;
  encoder_param.output_layernorm_gamma = d_output_layernorm_gamma;
  encoder_param.transformer_out = d_transformer_out;
  encoder_param.cublas_handle = cublasHandle;
  encoder_param.stream = stream;

  BertEncoderTransformer<EncoderTraits_> *encoder_transformer_ = new 
    BertEncoderTransformer<EncoderTraits_>(allocator, batch_size, from_seq_len, to_seq_len, head_num, size_per_head);
  encoder_transformer_->initialize(encoder_param);
  
  int ite = 200;
  //warp up
  for(int i = 0; i < ite; ++i)
    encoder_transformer_->forward();

  struct timeval ss, ee;
  cudaDeviceSynchronize();
  gettimeofday(&ss, NULL);
  for(int i = 0; i < ite; ++i)
  {
    for(int j = 0; j < num_layers; ++j)
      encoder_transformer_->forward();
  }

  cudaDeviceSynchronize();
  gettimeofday(&ee, NULL);
  printf("[batch_size %d seq_len %d %d transformer layers] costs %.2f ms\n", batch_size, seq_len, num_layers,
      diffTime(ss, ee) / ite);

  delete encoder_transformer_;
  return 0;
}


