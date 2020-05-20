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

#include "fastertransformer/faster_transformer.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>

using namespace fastertransformer;

template <typename T>
void device_malloc(T **ptr, int size);

template <typename T>
void encoder_sample(int batch_size,
                    int num_layers,
                    int seq_len,
                    int head_num,
                    int size_per_head);

int main(int argc, char* argv[])
{
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  if(argc != 7)
  {
    printf("[ERROR] encoder_sample batch_size num_layers seq_len head_num size_per_head is_fp16\n");
    printf("e.g., ./bin/encoder_sample 1 12 128 12 64 0\n");
    return 0;
  }

  printf("Device %s\n", prop.name);
  int batch_size = atoi(argv[1]);
  int num_layers = atoi(argv[2]);
  int seq_len = atoi(argv[3]);
  int head_num = atoi(argv[4]);
  int size_per_head = atoi(argv[5]);

  if(atoi(argv[6]) == 0)
    encoder_sample<float>(batch_size, num_layers, seq_len, head_num, size_per_head);
  else if(atoi(argv[6]) == 1)
    encoder_sample<half>(batch_size, num_layers, seq_len, head_num, size_per_head);
  else
  {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }
  
  return 0;
}

template <typename T>
void device_malloc(T **ptr, int size)
{
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
}

template <typename T>
void encoder_sample(int batch_size,
                    int num_layers,
                    int seq_len,
                    int head_num,
                    int size_per_head)
{
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
  check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
  float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
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

  check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
  free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
  total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
  printf("After allocate free %.2f GB used %.2f GB total %.2f GB\n", free, total - free, total);

  cublasHandle_t cublasHandle;
  check_cuda_error(cublasCreate(&cublasHandle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  const fastertransformer::OperationType type = sizeof(T) == sizeof(float) ? OperationType::FP32 : OperationType::FP16;
  typedef BertEncoderTransformerTraits<type, cuda::OpenMultiHeadAttention> EncoderTraits_;
  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  EncoderInitParam<T> encoder_param; //init param here

  encoder_param.from_tensor = d_from_tensor;
  encoder_param.to_tensor = d_from_tensor;
  encoder_param.self_attention.query_weight.kernel = d_attr_kernel_Q;
  encoder_param.self_attention.key_weight.kernel = d_attr_kernel_K;
  encoder_param.self_attention.value_weight.kernel = d_attr_kernel_V;
  encoder_param.self_attention.query_weight.bias = d_attr_bias_Q;
  encoder_param.self_attention.key_weight.bias = d_attr_bias_K;
  encoder_param.self_attention.value_weight.bias = d_attr_bias_V;
  encoder_param.attr_mask = d_attr_mask;
  encoder_param.self_attention.attention_output_weight.kernel = d_attr_output_kernel;
  encoder_param.self_attention.attention_output_weight.bias = d_attr_output_bias;
  encoder_param.self_layernorm.beta = d_attr_output_layernorm_beta;
  encoder_param.self_layernorm.gamma = d_attr_output_layernorm_gamma;
  encoder_param.ffn.intermediate_weight.kernel = d_inter_kernel;
  encoder_param.ffn.intermediate_weight.bias = d_inter_bias;
  encoder_param.ffn.output_weight.kernel = d_output_kernel;
  encoder_param.ffn.output_weight.bias = d_output_bias;
  encoder_param.ffn_layernorm.beta = d_output_layernorm_beta;
  encoder_param.ffn_layernorm.gamma = d_output_layernorm_gamma;
  encoder_param.transformer_out = d_transformer_out;
  encoder_param.cublas_handle = cublasHandle;
  encoder_param.stream = stream;

  BertEncoderTransformer<EncoderTraits_> *encoder_transformer_ = 
          new BertEncoderTransformer<EncoderTraits_>(allocator, 
                                                    batch_size, 
                                                    from_seq_len,
                                                    to_seq_len,
                                                    head_num, 
                                                    size_per_head);
  encoder_transformer_->initialize(encoder_param);

  int ite = 200;
  //warp up
  for (int i = 0; i < ite; ++i)
    encoder_transformer_->forward();

  struct timeval start, end;
  cudaDeviceSynchronize();
  gettimeofday(&start, NULL);
  for (int i = 0; i < ite; ++i)
  {
    for (int j = 0; j < num_layers; ++j)
      encoder_transformer_->forward();
  }

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  printf("[batch_size %d seq_len %d %d transformer layers] costs %.2f ms\n", batch_size, seq_len, num_layers,
         ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

  delete encoder_transformer_;
  return;
}

