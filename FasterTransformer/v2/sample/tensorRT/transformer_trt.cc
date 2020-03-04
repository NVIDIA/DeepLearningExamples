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
#include "fastertransformer/trt_plugin/trt_model.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <iostream>
#include <sys/time.h>

using namespace fastertransformer;

double diffTime(timeval start, timeval end)
{
  return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

template <typename T>
void host_malloc(T** ptr, int size)
{
  (*ptr) = (T*)malloc(sizeof(T) * size);
}


template <typename T>
void run_bert_transformer(int batch_size, int seq_len, int layers, int head_num, int size_per_head){

  int hidden_dim = head_num * size_per_head;

  std::vector<std::vector<T *> > params;

  T *h_from_tensor = NULL, *h_transformer_out = NULL;
  T *h_attr_mask = NULL;
  host_malloc(&h_from_tensor, batch_size * seq_len * hidden_dim);
  host_malloc(&h_transformer_out, batch_size * seq_len * hidden_dim);
  host_malloc(&h_attr_mask, batch_size * seq_len * seq_len);

  for(int i = 0; i < batch_size * seq_len * hidden_dim; ++i)
    h_from_tensor[i] = 0.001f;
  for(int i = 0; i < batch_size * seq_len * seq_len; ++i)
    h_attr_mask[i] = 1.0f;

  for(int i = 0; i < layers; ++i)
  {
    T *h_attr_kernel_Q = NULL, *h_attr_kernel_K = NULL, *h_attr_kernel_V = NULL;
    T *h_attr_bias_Q = NULL, *h_attr_bias_K = NULL, *h_attr_bias_V = NULL;
    T *h_attr_output_kernel = NULL, *h_attr_output_bias = NULL;
    T *h_attr_output_layernorm_beta = NULL;
    T *h_attr_output_layernorm_gamma = NULL;
    T *h_inter_kernel = NULL, *h_inter_bias = NULL;
    T *h_output_kernel = NULL, *h_output_bias = NULL, *h_output_layernorm_beta = NULL, *h_output_layernorm_gamma = NULL;

    host_malloc(&h_attr_kernel_Q, hidden_dim * hidden_dim);
    host_malloc(&h_attr_kernel_K, hidden_dim * hidden_dim);
    host_malloc(&h_attr_kernel_V, hidden_dim * hidden_dim);
    host_malloc(&h_attr_bias_Q, hidden_dim);
    host_malloc(&h_attr_bias_K, hidden_dim);
    host_malloc(&h_attr_bias_V, hidden_dim);
    host_malloc(&h_attr_output_kernel, hidden_dim * hidden_dim);
    host_malloc(&h_attr_output_bias, hidden_dim);
    host_malloc(&h_attr_output_layernorm_beta, hidden_dim);
    host_malloc(&h_attr_output_layernorm_gamma, hidden_dim);
    host_malloc(&h_inter_kernel, hidden_dim * hidden_dim * 4);
    host_malloc(&h_inter_bias, hidden_dim * 4);
    host_malloc(&h_output_kernel, hidden_dim * hidden_dim * 4);
    host_malloc(&h_output_bias, hidden_dim);
    host_malloc(&h_output_layernorm_beta, hidden_dim);
    host_malloc(&h_output_layernorm_gamma, hidden_dim);
    
    for(int i = 0; i < hidden_dim * hidden_dim; ++i)
    {
      h_attr_kernel_Q[i] = 0.001f;
      h_attr_kernel_K[i] = 0.001f;
      h_attr_kernel_V[i] = 0.001f;
      h_attr_output_kernel[i] = 0.0001f * i;
      if(i < hidden_dim)
      {
        h_attr_bias_Q[i] = 0.001f;
        h_attr_bias_K[i] = 0.001f;
        h_attr_bias_V[i] = 0.001f;
        h_attr_output_bias[i] = 0.001f;
        h_attr_output_layernorm_beta[i] = 0.0001f * i;
        h_attr_output_layernorm_gamma[i] = 0.001f * i;
        h_output_bias[i] = 0.001f;
        h_output_layernorm_beta[i] = 0.001f;
        h_output_layernorm_gamma[i] = 0.001f;
      }
      if(i < hidden_dim * 4)
        h_inter_bias[i] = 0.001f;
    }
    for(int i = 0; i < 4 * hidden_dim * hidden_dim; ++i)
    {
      h_inter_kernel[i] = 0.001f;
      h_output_kernel[i] = 0.001f;
    }
    std::vector<T* > layer_param;
    layer_param.push_back(h_attr_kernel_Q);
    layer_param.push_back(h_attr_kernel_K);
    layer_param.push_back(h_attr_kernel_V);
    layer_param.push_back(h_attr_bias_Q);
    layer_param.push_back(h_attr_bias_K);
    layer_param.push_back(h_attr_bias_V);
    layer_param.push_back(h_attr_output_kernel);
    layer_param.push_back(h_attr_output_bias);
    layer_param.push_back(h_attr_output_layernorm_beta);
    layer_param.push_back(h_attr_output_layernorm_gamma);
    layer_param.push_back(h_inter_kernel);
    layer_param.push_back(h_inter_bias);
    layer_param.push_back(h_output_kernel);
    layer_param.push_back(h_output_bias);
    layer_param.push_back(h_output_layernorm_beta);
    layer_param.push_back(h_output_layernorm_gamma);
    params.push_back(layer_param);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  TRT_Transformer<T>* trt_transformer = new TRT_Transformer<T>(batch_size, seq_len, head_num, hidden_dim, layers);
  trt_transformer->build_engine(params);

  trt_transformer->do_inference(batch_size, h_from_tensor, h_attr_mask, h_transformer_out, stream);

  delete trt_transformer;
 
  printf("finished!\n");
}

int main(int argc, char* argv[])
{
  if(argc != 7)
  {
    printf("./transformer_trt batch_size num_layers seq_len head_num size_per_head fp32/fp16\n");
    printf("e.g., ./transformer_trt 1 12 32 12 64 fp32\n");
    printf("e.g., ./transformer_trt 1 12 32 12 64 fp16\n");
    return 0;
  }
  int batch_size = atoi(argv[1]);
  int num_layers = atoi(argv[2]);
  int seq_len = atoi(argv[3]);
  int head_num = atoi(argv[4]);
  int size_per_head = atoi(argv[5]);
  if(strcmp(argv[6], "fp16") == 0)
    run_bert_transformer<half>(batch_size, seq_len, num_layers, head_num, size_per_head);
  else if(strcmp(argv[6], "fp32") == 0)
    run_bert_transformer<float>(batch_size, seq_len, num_layers, head_num, size_per_head);
  else
  {
    printf("the last argument is invalid, it should be fp16 or fp32\n");
    return 0;
  }
}
