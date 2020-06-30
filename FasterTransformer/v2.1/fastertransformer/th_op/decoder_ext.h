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

#include <vector>
#include <iostream>
#include <cuda_fp16.h>

#include "torch/extension.h"
#include "torch/csrc/cuda/Stream.h"

#include "fastertransformer/open_decoder.h"
#include "fastertransformer/th_op/th_traits.h"
#include "fastertransformer/th_op/utils.h"

namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

class IFTDecoder {
public:
  virtual ~IFTDecoder() {}
  virtual void forward(int batch_size, int seq_len, int mem_hidden_dim, int step,
                       Tensor& input, Tensor& memory, Tensor& memory_seq_lens, Tensor& self_cache, Tensor& mem_cache, Tensor& output) = 0;
};

template <typename T>
class FTDecoder : public IFTDecoder {
public:
  FTDecoder(int head_num, int head_size, const std::vector<Tensor>& w) : _head_num(head_num), _head_size(head_size), _weights(w) {
    int hidden_dim = _head_num * _head_size;
    check_cuda_error(cublasCreate(&_cublasHandle));
    decoder_params.self_layernorm.gamma = get_ptr<T>(_weights[0]);
    decoder_params.self_layernorm.beta = get_ptr<T>(_weights[1]);
    decoder_params.self_attention.query_weight.kernel = get_ptr<T>(_weights[2]);
    decoder_params.self_attention.key_weight.kernel = get_ptr<T>(_weights[3]);
    decoder_params.self_attention.value_weight.kernel = get_ptr<T>(_weights[4]);
    decoder_params.self_attention.query_weight.bias = get_ptr<T>(_weights[5]);
    decoder_params.self_attention.key_weight.bias = get_ptr<T>(_weights[6]);
    decoder_params.self_attention.value_weight.bias = get_ptr<T>(_weights[7]);
    decoder_params.self_attention.attention_output_weight.kernel = get_ptr<T>(_weights[8]);
    decoder_params.self_attention.attention_output_weight.bias = get_ptr<T>(_weights[9]);
    decoder_params.cross_layernorm.gamma = get_ptr<T>(_weights[10]);
    decoder_params.cross_layernorm.beta = get_ptr<T>(_weights[11]);
    decoder_params.cross_attention.query_weight.kernel = get_ptr<T>(_weights[12]);
    decoder_params.cross_attention.key_weight.kernel = get_ptr<T>(_weights[13]);
    decoder_params.cross_attention.value_weight.kernel = get_ptr<T>(_weights[14]);
    decoder_params.cross_attention.query_weight.bias = get_ptr<T>(_weights[15]);
    decoder_params.cross_attention.key_weight.bias = get_ptr<T>(_weights[16]);
    decoder_params.cross_attention.value_weight.bias = get_ptr<T>(_weights[17]);
    decoder_params.cross_attention.attention_output_weight.kernel = get_ptr<T>(_weights[18]);
    decoder_params.cross_attention.attention_output_weight.bias = get_ptr<T>(_weights[19]);
    decoder_params.ffn_layernorm.gamma = get_ptr<T>(_weights[20]);
    decoder_params.ffn_layernorm.beta = get_ptr<T>(_weights[21]);
    decoder_params.ffn.intermediate_weight.kernel = get_ptr<T>(_weights[22]);
    decoder_params.ffn.intermediate_weight.bias = get_ptr<T>(_weights[23]);
    decoder_params.ffn.output_weight.kernel = get_ptr<T>(_weights[24]);
    decoder_params.ffn.output_weight.bias = get_ptr<T>(_weights[25]);
    decoder_params.cublas_handle = _cublasHandle;
  }

  ~FTDecoder() override {
    cublasDestroy(_cublasHandle);
  }

  void forward(int batch_size, int seq_len, int mem_hidden_dim, int step,
               Tensor& input, Tensor& memory, Tensor& memory_seq_lens, Tensor& self_cache, Tensor& mem_cache, Tensor& output) override
  {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    check_cuda_error(cublasSetStream(decoder_params.cublas_handle, stream));
    decoder_params.stream = stream;
    fastertransformer::Allocator<AllocatorType::TH> allocator;
    OpenDecoder<THTraits<T>::OpType>* decoder = 
      new OpenDecoder<THTraits<T>::OpType>(batch_size, seq_len, _head_num, _head_size, mem_hidden_dim);
    
    T* output_ptr = get_ptr<T>(output);
    T* self_cache_ptr = get_ptr<T>(self_cache);
    T* mem_cache_ptr = get_ptr<T>(mem_cache);
    const T* input_ptr = get_ptr<T>(input);
    const T* memory_ptr = get_ptr<T>(memory);
    const int* memory_seq_lens_ptr = get_ptr<int>(memory_seq_lens);
    
    T* K_cache = self_cache_ptr;
    T* V_cache = self_cache_ptr + batch_size * step * _head_num * _head_size;
    T* K_mem_cache = mem_cache_ptr;
    T* V_mem_cache = mem_cache_ptr + batch_size * seq_len * _head_num * _head_size;
    const int decoder_buffer_size = decoder->getWorkspaceSize() * sizeof(T);
    T* decoder_buffer = (T*)allocator.malloc(decoder_buffer_size);

    decoder->initialize(decoder_params, decoder_buffer);
    decoder->forward(input_ptr, memory_ptr, K_cache, V_cache, K_mem_cache, V_mem_cache, memory_seq_lens_ptr, output_ptr, step);
    allocator.free(decoder_buffer);
    delete decoder;
  }

private:
  const int _head_num;
  const int _head_size;
  std::vector<Tensor> _weights;
  cublasHandle_t _cublasHandle;
  DecoderInitParam<T> decoder_params;
};

class FasterTransformerDecoder {
public:
  FasterTransformerDecoder(
    int head_num,
    int head_size,
    Tensor self_layernorm_gamma,
    Tensor self_layernorm_beta,
    Tensor self_kernel_q,
    Tensor self_kernel_k,
    Tensor self_kernel_v,
    Tensor self_bias_q,
    Tensor self_bias_k,
    Tensor self_bias_v,
    Tensor self_output_kernel,
    Tensor self_output_bias,
    Tensor cross_layernorm_gamma,
    Tensor cross_layernorm_beta,
    Tensor cross_kernel_q,
    Tensor cross_kernel_k,
    Tensor cross_kernel_v,
    Tensor cross_bias_q,
    Tensor cross_bias_k,
    Tensor cross_bias_v,
    Tensor cross_output_kernel,
    Tensor cross_output_bias,
    Tensor ffn_layernorm_gamma,
    Tensor ffn_layernorm_beta,
    Tensor inter_kernel,
    Tensor inter_bias,
    Tensor output_kernel,
    Tensor output_bias);

  ~FasterTransformerDecoder();
  
  Tensor forward(Tensor input, Tensor memory, Tensor memory_seq_lens, Tensor self_cache, Tensor mem_cache);

private:
  const at::ScalarType _st;
  IFTDecoder* ftdecoder;
};

} // namespace torch_ext