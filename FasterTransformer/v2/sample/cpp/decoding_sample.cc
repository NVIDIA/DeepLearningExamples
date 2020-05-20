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

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/decoding_opennmt.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>

using namespace fastertransformer;

template<typename T>
void device_malloc(T** ptr, int size);

template<typename T>
void decoding_sample(int batch_size,
                    int beam_width,
                    int head_num,
                    int size_per_head,
                    int vocab_size,
                    int seq_len,
                    int decoder_layers,
                    int memory_hidden_units);

int main(int argc, char* argv[])
{
  srand(0);
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);
  
  if(argc != 10)
  {
    printf("[ERROR] decoding_sample batch_size beam_width head_num size_per_head vocab_size seq_len num_layer memory_hidden_units is_fp16\n");
    printf("e.g. ./bin/decoding_sample 32 4 8 64 30000 32 6 768 0\n");
    return 0;
  }

  const int batch_size = atoi(argv[1]);
  const int beam_width = atoi(argv[2]);
  const int head_num = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);
  const int vocab_size = atoi(argv[5]);
  const int seq_len = atoi(argv[6]);
  const int decoder_layers = atoi(argv[7]);
  const int memory_hidden_units = atoi(argv[8]);
  
  if(atoi(argv[9]) == 0)
    decoding_sample<float>(batch_size, beam_width, head_num, size_per_head, vocab_size, seq_len, decoder_layers, memory_hidden_units);
  else if(atoi(argv[9]) == 1)
    decoding_sample<half>(batch_size, beam_width, head_num, size_per_head, vocab_size, seq_len, decoder_layers, memory_hidden_units);
  else
  {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }
  
  return 0;
}

template<typename T>
void device_malloc(T** ptr, int size)
{
  check_cuda_error(cudaMalloc((void**)ptr, sizeof(T) * size));
  T* tmp = new T[size];
  for(int i = 0; i < size; i++) tmp[i] = (T)((float) rand() / (RAND_MAX + 1.0) * 0.02);
  check_cuda_error(cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice));
  delete tmp;
}

template<typename T>
void decoding_sample(int batch_size,
                    int beam_width,
                    int head_num,
                    int size_per_head,
                    int vocab_size,
                    int seq_len,
                    int decoder_layers,
                    int memory_hidden_units)
{
  const int max_seq_len = seq_len;
  const int memory_seq_len = seq_len; 
  const int start_id = 1;
  const int end_id = 2;
  const int hidden_units = head_num * size_per_head;
  const int inner_size = 4 * hidden_units;

  cublasHandle_t cublasHandle;
  check_cuda_error(cublasCreate(&cublasHandle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  DecoderInitParam<T> *param = new DecoderInitParam<T>[decoder_layers];

  for(int i = 0; i < decoder_layers; i++){
    param[i].stream = stream;
    param[i].cublas_handle = cublasHandle;

    T *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel;
    T *d_self_Q_bias, *d_self_K_bias, *d_self_V_bias, *d_self_output_bias;
    T *d_cross_Q_kernel, *d_cross_K_kernel, *d_cross_V_kernel, *d_cross_output_kernel;
    T *d_cross_Q_bias, *d_cross_K_bias, *d_cross_V_bias, *d_cross_output_bias;
    T *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2;
    T *d_self_gamma, *d_self_beta;
    T *d_cross_gamma, *d_cross_beta;
    T *d_ffn_gamma, *d_ffn_beta;
    
    device_malloc(&d_self_Q_kernel, sizeof(T) * hidden_units * hidden_units);
    device_malloc(&d_self_K_kernel, sizeof(T) * hidden_units * hidden_units);
    device_malloc(&d_self_V_kernel, sizeof(T) * hidden_units * hidden_units);
    device_malloc(&d_self_output_kernel, sizeof(T) * hidden_units * hidden_units);
    device_malloc(&d_self_Q_bias, sizeof(T) * hidden_units);
    device_malloc(&d_self_K_bias, sizeof(T) * hidden_units);
    device_malloc(&d_self_V_bias, sizeof(T) * hidden_units);
    device_malloc(&d_self_output_bias, sizeof(T) * hidden_units);

    device_malloc(&d_cross_Q_kernel, sizeof(T) * hidden_units * hidden_units);
    device_malloc(&d_cross_K_kernel, sizeof(T) * memory_hidden_units * hidden_units);
    device_malloc(&d_cross_V_kernel, sizeof(T) * memory_hidden_units * hidden_units);
    device_malloc(&d_cross_output_kernel, sizeof(T) * hidden_units * hidden_units);
    device_malloc(&d_cross_Q_bias, sizeof(T) * hidden_units);
    device_malloc(&d_cross_K_bias, sizeof(T) * hidden_units);
    device_malloc(&d_cross_V_bias, sizeof(T) * hidden_units);
    device_malloc(&d_cross_output_bias, sizeof(T) * hidden_units);

    device_malloc(&d_ffn_bias1, sizeof(T) * inner_size);
    device_malloc(&d_ffn_kernel1, sizeof(T) * inner_size * hidden_units);
    device_malloc(&d_ffn_bias2, sizeof(T) * hidden_units);
    device_malloc(&d_ffn_kernel2, sizeof(T) * inner_size * hidden_units);

    device_malloc(&d_self_gamma, sizeof(T) * hidden_units);
    device_malloc(&d_self_beta, sizeof(T) * hidden_units);
    device_malloc(&d_cross_gamma, sizeof(T) * hidden_units);
    device_malloc(&d_cross_beta, sizeof(T) * hidden_units);
    device_malloc(&d_ffn_gamma, sizeof(T) * hidden_units);
    device_malloc(&d_ffn_beta, sizeof(T) * hidden_units);

    param[i].self_attention.query_weight.kernel = d_self_Q_kernel;
    param[i].self_attention.key_weight.kernel = d_self_K_kernel;
    param[i].self_attention.value_weight.kernel = d_self_V_kernel;
    param[i].self_attention.attention_output_weight.kernel = d_self_output_kernel;
    param[i].self_attention.query_weight.bias = d_self_Q_bias;
    param[i].self_attention.key_weight.bias = d_self_K_bias;
    param[i].self_attention.value_weight.bias = d_self_V_bias;
    param[i].self_attention.attention_output_weight.bias = d_self_output_bias;

    param[i].cross_attention.query_weight.kernel = d_cross_Q_kernel;
    param[i].cross_attention.key_weight.kernel = d_cross_K_kernel;
    param[i].cross_attention.value_weight.kernel = d_cross_V_kernel;
    param[i].cross_attention.attention_output_weight.kernel = d_cross_output_kernel;
    param[i].cross_attention.query_weight.bias = d_cross_Q_bias;
    param[i].cross_attention.key_weight.bias = d_cross_K_bias;
    param[i].cross_attention.value_weight.bias = d_cross_V_bias;
    param[i].cross_attention.attention_output_weight.bias = d_cross_output_bias;

    param[i].self_layernorm.gamma = d_self_gamma;
    param[i].self_layernorm.beta = d_self_beta;
    param[i].cross_layernorm.gamma = d_cross_gamma;
    param[i].cross_layernorm.beta = d_cross_beta;
    param[i].ffn_layernorm.gamma = d_ffn_gamma;
    param[i].ffn_layernorm.beta = d_ffn_beta;
    param[i].ffn.intermediate_weight.bias = d_ffn_bias1;
    param[i].ffn.output_weight.bias = d_ffn_bias2;
    param[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
    param[i].ffn.output_weight.kernel = d_ffn_kernel2;
  }
  
  DecodingInitParam<T> decoding_params;

  T *d_memory_tensor;
  T *d_embedding_table;
  T* d_embedding_kernel;
  float* d_embedding_bias;
  int* d_output_ids;
  int* d_parent_ids;
  int* d_sequence_lengths;
  int* d_memory_sequence_lengths;
  T *d_gamma, *d_beta;    

  device_malloc(&d_memory_tensor, sizeof(T) * hidden_units * seq_len * batch_size * beam_width);
  device_malloc(&d_embedding_table, sizeof(T) * hidden_units * vocab_size);
  device_malloc(&d_embedding_kernel, sizeof(T) * vocab_size * hidden_units);
  check_cuda_error(cudaMalloc((void**)&d_embedding_bias, sizeof(float) * vocab_size));
  check_cuda_error(cudaMalloc((void**)&d_output_ids, sizeof(int) * (max_seq_len) * batch_size * beam_width));
  check_cuda_error(cudaMalloc((void**)&d_parent_ids, sizeof(int) * (max_seq_len) * batch_size * beam_width));
  check_cuda_error(cudaMalloc((void**)&d_sequence_lengths, sizeof(int) * batch_size * beam_width));
  check_cuda_error(cudaMalloc((void**)&d_memory_sequence_lengths, sizeof(int) * batch_size * beam_width));
  device_malloc(&d_gamma, sizeof(T) * hidden_units);
  device_malloc(&d_beta, sizeof(T) * hidden_units);

  int *h_memory_sequence_lengths = new int[batch_size * beam_width];
  for(int i = 0; i < batch_size * beam_width; i++) h_memory_sequence_lengths[i] = seq_len;
  check_cuda_error(cudaMemcpy(d_memory_sequence_lengths, h_memory_sequence_lengths, sizeof(int) * batch_size * beam_width, cudaMemcpyHostToDevice));

  decoding_params.cublas_handle = cublasHandle;
  decoding_params.stream = stream;
  decoding_params.memory_tensor = d_memory_tensor;
  decoding_params.embedding_table = d_embedding_table;
  decoding_params.embedding_kernel = d_embedding_kernel;
  decoding_params.embedding_bias = d_embedding_bias;
  decoding_params.output_ids = d_output_ids;
  decoding_params.parent_ids = d_parent_ids;
  decoding_params.sequence_length = d_sequence_lengths;
  decoding_params.memory_sequence_length = d_memory_sequence_lengths;
  decoding_params.layernorm.gamma = d_gamma;
  decoding_params.layernorm.beta = d_beta;

  const fastertransformer::OperationType type = sizeof(T) == sizeof(float) ? OperationType::FP32 : OperationType::FP16;
  
  DecodingOpenNMT<type> *decoding = new 
    DecodingOpenNMT<type>(allocator, batch_size, beam_width,
                                         max_seq_len, head_num, size_per_head, 
                                         vocab_size, decoder_layers,
                                         memory_hidden_units, memory_seq_len, 
                                         start_id, end_id);
 
  //warm up
  int ite = 100;
  for(int i = 0; i < ite; ++i)
    decoding->forward(param, decoding_params);

  struct timeval start, end;
  cudaDeviceSynchronize();
  gettimeofday(&start, NULL);

  for(int i = 0; i < ite; ++i)
    decoding->forward(param, decoding_params);
 
  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);
  printf("[batch_size %d beam_width %d head_num %d size_per_head %d seq_len %d decoder_layers %d vocab_size %d] costs %.2f ms\n",
    batch_size, beam_width, head_num, size_per_head, seq_len, decoder_layers, vocab_size,
    ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);
  printf("done\n");

  delete decoding;
  return ;
}