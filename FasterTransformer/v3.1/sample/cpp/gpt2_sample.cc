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
#include "fastertransformer/gpt2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>
#include <cuda_fp16.h>

#include <string>
#include <sstream>
#include <fstream>
#include <vector>

// #define WEIGHTS_ROOT "/workspace/ft2-gpt2/gpt2-withlm-weights/"
// #define PREFIX_STRING "transformer."
#define WEIGHTS_ROOT "./tmp/"
#define PREFIX_STRING "model."

static inline std::string path_to_weights(const char *file, int layernum = -1)
{
  if (layernum == -1)
    return std::string() + WEIGHTS_ROOT + PREFIX_STRING + file;
  else
  {
    char layername[256];
    sprintf(layername, "%sh%d.", PREFIX_STRING, layernum);
    return std::string() + WEIGHTS_ROOT + layername + file;
  }
}

using namespace fastertransformer;

template <typename T>
void device_malloc(T **ptr, int size);

template <typename T>
void decoding_sample(int batch_size,
                     int candidate_num,
                     float probability_threshold,
                     int head_num,
                     int size_per_head,
                     int vocab_size,
                     int seq_len,
                     int decoder_layers);

int main(int argc, char *argv[])
{
  srand(0);
  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  if (argc != 10)
  {
    printf("[ERROR] decoding_sample batch_size candidate_num probability_threshold head_num size_per_head vocab_size seq_len num_layer is_fp16\n");
    printf("e.g. ./bin/decoding_sample 1 1 0.0 12 64 50257 32 12 0\n");
    return 0;
  }

  const int batch_size = atoi(argv[1]);
  const int candidate_num = atoi(argv[2]);
  const float probability_threshold = atof(argv[3]);
  const int head_num = atoi(argv[4]);
  const int size_per_head = atoi(argv[5]);
  const int vocab_size = atoi(argv[6]);
  const int seq_len = atoi(argv[7]);
  const int decoder_layers = atoi(argv[8]);

  if (atoi(argv[9]) == 0)
    decoding_sample<float>(batch_size, candidate_num, probability_threshold, head_num, size_per_head, vocab_size, seq_len, decoder_layers);
  else if (atoi(argv[9]) == 1)
    decoding_sample<half>(batch_size, candidate_num, probability_threshold, head_num, size_per_head, vocab_size, seq_len, decoder_layers);
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
  T *tmp = new T[size];
  srand(123);
  //for(int i = 0; i < size; i++) tmp[i] = (T)((float) rand() / (RAND_MAX + 1.0));// * 0.02);
  for (int i = 0; i < size; i++)
    tmp[i] = (T)((float)rand() / (RAND_MAX + 1.0) * 0.02);
  check_cuda_error(cudaMemcpy(*ptr, tmp, sizeof(T) * size, cudaMemcpyHostToDevice));
  delete[] tmp;
}

template <typename T>
void device_malloc_zero(T **ptr, int size)
{
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  check_cuda_error(cudaMemset(*ptr, 0, sizeof(T) * size));
}

template <typename T>
int init_device_from_csv(T **ptr, std::vector<int> shape, std::string filename, int split = 1)
{
  if (shape.size() > 2)
  {
    printf("[ERROR] shape should have less than two dims \n");
    return -1;
  }
  int dim0 = shape[0], dim1 = 1;
  if (shape.size() == 2)
  {
    dim1 = shape[1];
  }
  size_t size = dim0 * dim1;

  int split_boundary = (dim1 + split - 1) / split;
  size_t size_each = size / split;
  size_t dim1_each = dim1 / split;

  bool dim0_reached = false, dim1_reached = false;
  int i0 = 0, i1;
  std::ifstream file(filename);
  std::vector<T> host_array(size);
  if (file.is_open())
  {
    std::string line;
    while (std::getline(file, line))
    {
      if (i0 == dim0)
      {
        dim0_reached = true;
        break;
      }

      std::stringstream lineStream(line);
      std::string vals;
      i1 = 0;
      while (std::getline(lineStream, vals, ','))
      {
        if (i1 == dim1)
        {
          dim1_reached = true;
          break;
        }
        if (split > 1)
        {
          int idx = i1 / split_boundary;
          int i11 = i1 % split_boundary;
          if (sizeof(T) == sizeof(float))
            host_array[i0 * dim1_each + (idx * size_each) + i11] = std::stof(vals);
          else
            host_array[i0 * dim1_each + (idx * size_each) + i11] = __float2half(std::stof(vals));
        }
        else
        {
          if (sizeof(T) == sizeof(float))
            host_array[i0 * dim1 + i1] = std::stof(vals);
          else
            host_array[i0 * dim1 + i1] = __float2half(std::stof(vals));
        }
        i1++;
      }
      i0++;
    }
  }
  else
  {
    printf("[WARNING] file %s cannot be opened, initializing weights with random values! \n", filename.c_str());
    device_malloc(ptr, size);
    return 0;
  }
  check_cuda_error(cudaMalloc((void **)ptr, sizeof(T) * size));
  cudaMemcpy(*ptr, host_array.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
  if (dim0_reached)
    printf("[WARNING] the file dimension does not match with input dim0! %s, dim0=%d, i0=%d\n", filename.c_str(), dim0, i0);
  if (dim1_reached)
    printf("[WARNING] the file dimension does not match with input dim1! %s, dim1=%d, i1=%d\n", filename.c_str(), dim1, i1);
  return 0;
}

template <typename T>
void decoding_sample(int batch_size,
                     int candidate_num,
                     float probability_threshold,
                     int head_num,
                     int size_per_head,
                     int vocab_size,
                     int seq_len,
                     int decoder_layers)
{
  const int max_seq_len = seq_len;
  // const int start_ids[] = {15496, 11, 616, 3290, 468,
  //                         15496, 11, 616, 3290, 469,
  //                         15496, 11, 616, 3290, 470,
  //                         15496, 11, 616, 3290, 471};
  // const int start_ids[] = {15496};
  // const int start_id = start_ids[0];
  // assert((sizeof(start_ids) / sizeof(start_ids[0])) % batch_size == 0);
  // const int start_ids_len = sizeof(start_ids) / sizeof(start_ids[0]) / batch_size;
  int *start_ids = new int[batch_size];
  for(int i = 0; i < batch_size; i++) start_ids[i] = 50256;
  const int start_ids_len = 1;
  const int start_id = 50256;
  const int end_id = 50256;
  const int hidden_units = head_num * size_per_head;
  const int inner_size = 4 * hidden_units;

  cublasHandle_t cublasHandle;
  check_cuda_error(cublasCreate(&cublasHandle));

  cudaStream_t stream;
  check_cuda_error(cudaStreamCreate(&stream));
  check_cuda_error(cublasSetStream(cublasHandle, stream));

  fastertransformer::Allocator<AllocatorType::CUDA> allocator(0);
  DecoderInitParam<T> *param = new DecoderInitParam<T>[decoder_layers];

  for (int i = 0; i < decoder_layers; i++)
  {
    param[i].stream = stream;
    param[i].cublas_handle = cublasHandle;

    T *d_self_Q_kernel, *d_self_K_kernel, *d_self_V_kernel, *d_self_output_kernel;
    T *d_self_Q_bias, *d_self_K_bias, *d_self_V_bias, *d_self_output_bias;
    T *d_ffn_kernel1, *d_ffn_bias1, *d_ffn_kernel2, *d_ffn_bias2;
    T *d_self_gamma, *d_self_beta;
    T *d_ffn_gamma, *d_ffn_beta;

    init_device_from_csv(&d_self_Q_kernel, {hidden_units, hidden_units * 3}, path_to_weights("attn.c_attn.w.csv", i), 3);
    d_self_K_kernel = d_self_Q_kernel + hidden_units * hidden_units;
    d_self_V_kernel = d_self_K_kernel + hidden_units * hidden_units;
    init_device_from_csv(&d_self_output_kernel, {hidden_units, hidden_units}, path_to_weights("attn.c_proj.w.csv", i));
    T *d_self_bias;
    init_device_from_csv(&d_self_bias, {hidden_units * 3}, path_to_weights("attn.c_attn.b.csv", i));
    d_self_Q_bias = d_self_bias;
    d_self_K_bias = d_self_bias + hidden_units;
    d_self_V_bias = d_self_bias + 2 * hidden_units;
    init_device_from_csv(&d_self_output_bias, {hidden_units}, path_to_weights("attn.c_proj.b.csv", i));

    init_device_from_csv(&d_ffn_bias1, {inner_size}, path_to_weights("mlp.c_fc.b.csv", i));
    init_device_from_csv(&d_ffn_bias2, {hidden_units}, path_to_weights("mlp.c_proj.b.csv", i));
    init_device_from_csv(&d_ffn_kernel1, {hidden_units, inner_size}, path_to_weights("mlp.c_fc.w.csv", i));
    init_device_from_csv(&d_ffn_kernel2, {inner_size, hidden_units}, path_to_weights("mlp.c_proj.w.csv", i));

    init_device_from_csv(&d_self_gamma, {hidden_units}, path_to_weights("ln_1.g.csv", i));
    init_device_from_csv(&d_self_beta, {hidden_units}, path_to_weights("ln_1.b.csv", i));
    init_device_from_csv(&d_ffn_gamma, {hidden_units}, path_to_weights("ln_2.g.csv", i));
    init_device_from_csv(&d_ffn_beta, {hidden_units}, path_to_weights("ln_2.b.csv", i));

    param[i].self_layernorm.gamma = d_self_gamma;
    param[i].self_layernorm.beta = d_self_beta;
    param[i].self_attention.query_weight.kernel = d_self_Q_kernel;
    param[i].self_attention.key_weight.kernel = d_self_K_kernel;
    param[i].self_attention.value_weight.kernel = d_self_V_kernel;
    param[i].self_attention.attention_output_weight.kernel = d_self_output_kernel;
    param[i].self_attention.query_weight.bias = d_self_Q_bias;
    param[i].self_attention.key_weight.bias = d_self_K_bias;
    param[i].self_attention.value_weight.bias = d_self_V_bias;
    param[i].self_attention.attention_output_weight.bias = d_self_output_bias;

    param[i].ffn_layernorm.gamma = d_ffn_gamma;
    param[i].ffn_layernorm.beta = d_ffn_beta;
    param[i].ffn.intermediate_weight.bias = d_ffn_bias1;
    param[i].ffn.output_weight.bias = d_ffn_bias2;
    param[i].ffn.intermediate_weight.kernel = d_ffn_kernel1;
    param[i].ffn.output_weight.kernel = d_ffn_kernel2;
  }

  DecodingInitParam<T> decoding_params;

  T *d_embedding_table;
  T *d_position_encoding_table;
  T *d_embedding_kernel;
  int *d_output_ids;
  T *d_gamma, *d_beta;

  init_device_from_csv(&d_embedding_table, {vocab_size, hidden_units}, path_to_weights("wte.csv"));
  init_device_from_csv(&d_position_encoding_table, {seq_len, hidden_units}, path_to_weights("wpe.csv"));
  d_embedding_kernel = d_embedding_table;
  check_cuda_error(cudaMalloc((void **)&d_output_ids, sizeof(int) * max_seq_len * batch_size));
  init_device_from_csv(&d_gamma, {hidden_units}, path_to_weights("ln_f.g.csv"));
  init_device_from_csv(&d_beta, {hidden_units}, path_to_weights("ln_f.b.csv"));

  decoding_params.cublas_handle = cublasHandle;
  decoding_params.stream = stream;
  decoding_params.embedding_table = d_embedding_table;
  decoding_params.position_encoding_table = d_position_encoding_table;
  decoding_params.embedding_kernel = d_embedding_kernel;
  decoding_params.output_ids = d_output_ids;
  decoding_params.layernorm.gamma = d_gamma;
  decoding_params.layernorm.beta = d_beta;

  const fastertransformer::OperationType type = sizeof(T) == sizeof(float) ? OperationType::FP32 : OperationType::FP16;

  DecodingGpt2<type> *decoding = new DecodingGpt2<type>(allocator, batch_size, 
                                                        max_seq_len, head_num, size_per_head,
                                                        vocab_size, decoder_layers,
                                                        start_id, end_id,
                                                        start_ids, start_ids_len,
                                                        candidate_num, probability_threshold);

  // Doing only one run at first for correctness check
  int ite = 1;
  cudaDeviceSynchronize();
  for (int i = 0; i < ite; ++i)
    decoding->forward(param, decoding_params);
  cudaDeviceSynchronize();

  //warm up
  ite = 1;
  for (int i = 0; i < ite; ++i)
    decoding->forward(param, decoding_params);

  struct timeval start, end;
  cudaDeviceSynchronize();
  gettimeofday(&start, NULL);

  for (int i = 0; i < ite; ++i)
    decoding->forward(param, decoding_params);

  cudaDeviceSynchronize();
  gettimeofday(&end, NULL);

  printf("[INFO] batch_size %d head_num %d size_per_head %d seq_len %d"
         " decoder_layers %d vocab_size %d FT-CPP-gpt2-time %.2f ms\n",
         batch_size, head_num, size_per_head, seq_len, decoder_layers, vocab_size,
         ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

  std::string fName = "out";
  auto outFile = std::ofstream(fName, std::ios::out);

  size_t outCount = max_seq_len * batch_size;
  int *hBuf = new int[outCount];
  cudaDeviceSynchronize();
  cudaMemcpy(hBuf, d_output_ids, outCount * sizeof(int), cudaMemcpyDeviceToHost);

  {
    std::cout << "Writing " << outCount << " elements\n";
    int zerroCount = 0;
    //outFile.precision(5);
    //outFile << std::fixed << std::scientific;
    for (size_t i = 0; i < outCount; i++)
    {
        if (hBuf[i] == int(0)) zerroCount++;
        outFile << hBuf[i] << std::endl;
        std::cout << hBuf[i] << " ";
        if((i+1) % (batch_size) == 0) std::cout << std::endl;
    }
    std::cout << std::endl << "zerroCount = " << zerroCount << std::endl;
  }

  // answer under following settings
  // ./bin/gpt2_sample 1 1 12 64 50257 32 12 768 0
  // start_ids[] = {15496, 11, 616, 3290, 468};
  // 50256 198 464 717 640 314 2497 262 649 2196 286 428 983 11 340 373 257 1643 17185 13 383 9382 547 2495 922 290 612 2492 470 881 284 466 
  delete [] hBuf;
  delete[] param;
  delete decoding;
  return;
}
