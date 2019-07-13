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
#pragma once

#include "fastertransformer/common.h"
#include "fastertransformer/trt_plugin/bert_transformer_plugin.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <iostream>
#include <NvInfer.h>
#include <map>
#include <string>
#include <vector>

template<typename T>
class TRT_Transformer
{
  public:
    TRT_Transformer(const int batch_size, const int seq_len, const int head_num, const int hidden_dim, const int num_layers)
      :batch_size_(batch_size), seq_len_(seq_len), head_num_(head_num), hidden_dim_(hidden_dim), num_layers_(num_layers) 
    {
       dtype_ = TransformerTrtTraits<T>::DataType;
    }

    ~TRT_Transformer()
    {
      check_cuda_error(cudaFree(buffers[input_index_]));
      check_cuda_error(cudaFree(buffers[mask_index_]));
      check_cuda_error(cudaFree(buffers[output_index_]));
      engine_->destroy();
      context_->destroy();
    }

    nvinfer1::Weights point2weight(T* ptr, int size)
    {
      return nvinfer1::Weights{dtype_, ptr, (long)size}; 
    }
    void build_engine(std::vector<std::vector<T* > > &weights)
    {
      assert(weights.size() == num_layers_);
      for(int i = 0; i < num_layers_; ++i)
         assert(weights[i].size() == 16);

      nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
      assert(builder);
      nvinfer1::INetworkDefinition* network = builder->createNetwork();

      auto from_tensor = network->addInput(INPUT_BLOB_NAME, dtype_, nvinfer1::Dims2{seq_len_, hidden_dim_});
      auto mask_tensor = network->addInput(MASK_BLOB_NAME, dtype_, nvinfer1::Dims2{seq_len_, seq_len_});

      assert(input_tensor);
      assert(mask_tensor);

      nvinfer1::ITensor* output_tensor = nullptr;

      for(int i = 0; i < num_layers_; ++i)
      {
        auto plugin = new TransformerPlugin<T>(
          hidden_dim_, head_num_, seq_len_, batch_size_, 
          point2weight(weights[i][0], hidden_dim_ * hidden_dim_),
          point2weight(weights[i][1], hidden_dim_ * hidden_dim_),
          point2weight(weights[i][2], hidden_dim_ * hidden_dim_),
          point2weight(weights[i][3], hidden_dim_),
          point2weight(weights[i][4], hidden_dim_),
          point2weight(weights[i][5], hidden_dim_),
          point2weight(weights[i][6], hidden_dim_ * hidden_dim_),
          point2weight(weights[i][7], hidden_dim_),
          point2weight(weights[i][8], hidden_dim_),
          point2weight(weights[i][9], hidden_dim_),
          point2weight(weights[i][10], hidden_dim_ * hidden_dim_ * 4),
          point2weight(weights[i][11], hidden_dim_ * 4),
          point2weight(weights[i][12], hidden_dim_ * hidden_dim_ * 4),
          point2weight(weights[i][13], hidden_dim_),
          point2weight(weights[i][14], hidden_dim_),
          point2weight(weights[i][15], hidden_dim_)
         );
         assert(plugin);
         ITensor *inputs[] = {from_tensor, mask_tensor};
         auto transformerLayer = network->addPluginV2(inputs, 2, *plugin);

         from_tensor = transformerLayer->getOutput(0);
         output_tensor = from_tensor;
      } 

      output_tensor->setName(OUTPUT_BLOB_NAME);
      network->markOutput(*output_tensor);

      builder->setMaxBatchSize(batch_size_);
      builder->setMaxWorkspaceSize(1 << 20);
      builder->setFp16Mode(false);

      engine_ = builder->buildCudaEngine(*network);
      assert(engine_);

      network->destroy();
      builder->destroy();

      input_index_ = engine_->getBindingIndex(INPUT_BLOB_NAME);
      mask_index_ = engine_->getBindingIndex(MASK_BLOB_NAME);
      output_index_ = engine_->getBindingIndex(OUTPUT_BLOB_NAME);

      check_cuda_error(cudaMalloc(&buffers[input_index_], batch_size_ * seq_len_ * hidden_dim_ * sizeof(T)));
      check_cuda_error(cudaMalloc(&buffers[mask_index_], batch_size_ * seq_len_ * seq_len_ * sizeof(T)));
      check_cuda_error(cudaMalloc(&buffers[output_index_], batch_size_ * seq_len_ * hidden_dim_ * sizeof(T)));

      context_ = engine_->createExecutionContext();
   }

   void do_inference(int batch_size, const T* h_from_tensor, const T* h_attr_mask, T* h_output, cudaStream_t stream)
   {
     cudaMemcpyAsync(buffers[input_index_], h_from_tensor, batch_size * seq_len_ * hidden_dim_ * sizeof(T),
         cudaMemcpyHostToDevice, stream);
     cudaMemcpyAsync(buffers[mask_index_], h_attr_mask, batch_size * seq_len_ * seq_len_ * sizeof(T),
         cudaMemcpyHostToDevice, stream);
     context_->enqueue(batch_size_, buffers, stream, nullptr);
     cudaMemcpyAsync(h_output, buffers[output_index_], batch_size * seq_len_ * hidden_dim_ * sizeof(T),
         cudaMemcpyDeviceToHost, stream);
   }

  private:
    const int batch_size_, seq_len_, head_num_, hidden_dim_, num_layers_;
    nvinfer1::DataType dtype_;
    int inputN_, outputN_, input_index_, mask_index_, output_index_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::map<std::string, nvinfer1::Weights> weightMap_;
    void* buffers[3];
    const char* INPUT_BLOB_NAME = "input";
    const char* MASK_BLOB_NAME = "mask";
    const char* OUTPUT_BLOB_NAME = "prob";
};
