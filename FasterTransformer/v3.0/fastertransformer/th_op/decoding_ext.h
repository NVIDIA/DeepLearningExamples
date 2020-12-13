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
#include "fastertransformer/decoding_beamsearch.h"
#include "fastertransformer/th_op/th_traits.h"
#include "fastertransformer/th_op/utils.h"

namespace torch_ext {
using namespace fastertransformer;
using torch::Tensor;

class IFTDecoding {
public:
  virtual ~IFTDecoding() {}
  virtual void forward(int batch_size, int beam_size, int max_seq_len, int mem_max_seq_len,
                       Tensor memory, Tensor memory_seq_lens, Tensor output_ids, Tensor parent_ids, Tensor out_seq_lens) = 0;
};

template <typename T>
class FTDecoding : public IFTDecoding {
public:
  FTDecoding(int head_num, int head_size, int mem_hidden_dim, int layer_num, int vocab_size,
             int start_id, int end_id, float beam_search_diversity_rate, const std::vector<Tensor>& w)
  : _head_num(head_num), _head_size(head_size), _mem_hidden_dim(mem_hidden_dim), _layer_num(layer_num), _vocab_size(vocab_size),
  _start_id(start_id), _end_id(end_id), _beam_search_diversity_rate(beam_search_diversity_rate), _weights(w)
  {
    check_cuda_error(cublasCreate(&_cublasHandle));
    decoder_params = new DecoderInitParam<T>[_layer_num];
    const int hidden_dim = _head_num * _head_size;
    for (int i = 0; i < _layer_num; ++i) {
      decoder_params[i].self_layernorm.gamma = get_ptr<T>(_weights[0]) + i * hidden_dim;
      decoder_params[i].self_layernorm.beta = get_ptr<T>(_weights[1]) + i * hidden_dim;
      decoder_params[i].self_attention.query_weight.kernel = get_ptr<T>(_weights[2]) + i * hidden_dim * hidden_dim;
      decoder_params[i].self_attention.key_weight.kernel = get_ptr<T>(_weights[3]) + i * hidden_dim * hidden_dim;
      decoder_params[i].self_attention.value_weight.kernel = get_ptr<T>(_weights[4]) + i * hidden_dim * hidden_dim;
      decoder_params[i].self_attention.query_weight.bias = get_ptr<T>(_weights[5]) + i * hidden_dim;
      decoder_params[i].self_attention.key_weight.bias = get_ptr<T>(_weights[6]) + i * hidden_dim;
      decoder_params[i].self_attention.value_weight.bias = get_ptr<T>(_weights[7]) + i * hidden_dim;
      decoder_params[i].self_attention.attention_output_weight.kernel = get_ptr<T>(_weights[8]) + i * hidden_dim * hidden_dim;
      decoder_params[i].self_attention.attention_output_weight.bias = get_ptr<T>(_weights[9]) + i * hidden_dim;
      decoder_params[i].cross_layernorm.gamma = get_ptr<T>(_weights[10]) + i * hidden_dim;
      decoder_params[i].cross_layernorm.beta = get_ptr<T>(_weights[11]) + i * hidden_dim;
      decoder_params[i].cross_attention.query_weight.kernel = get_ptr<T>(_weights[12]) + i * hidden_dim * hidden_dim;
      decoder_params[i].cross_attention.key_weight.kernel = get_ptr<T>(_weights[13]) + i * mem_hidden_dim * hidden_dim;
      decoder_params[i].cross_attention.value_weight.kernel = get_ptr<T>(_weights[14]) + i * mem_hidden_dim * hidden_dim;
      decoder_params[i].cross_attention.query_weight.bias = get_ptr<T>(_weights[15]) + i * hidden_dim;
      decoder_params[i].cross_attention.key_weight.bias = get_ptr<T>(_weights[16]) + i * hidden_dim;
      decoder_params[i].cross_attention.value_weight.bias = get_ptr<T>(_weights[17]) + i * hidden_dim;
      decoder_params[i].cross_attention.attention_output_weight.kernel = get_ptr<T>(_weights[18]) + i * hidden_dim * hidden_dim;
      decoder_params[i].cross_attention.attention_output_weight.bias = get_ptr<T>(_weights[19]) + i * hidden_dim;
      decoder_params[i].ffn_layernorm.gamma = get_ptr<T>(_weights[20]) + i * hidden_dim;
      decoder_params[i].ffn_layernorm.beta = get_ptr<T>(_weights[21]) + i * hidden_dim;
      decoder_params[i].ffn.intermediate_weight.kernel = get_ptr<T>(_weights[22]) + i * hidden_dim * hidden_dim * 4;
      decoder_params[i].ffn.intermediate_weight.bias = get_ptr<T>(_weights[23]) + i * hidden_dim * 4;
      decoder_params[i].ffn.output_weight.kernel = get_ptr<T>(_weights[24]) + i * hidden_dim * hidden_dim * 4;
      decoder_params[i].ffn.output_weight.bias = get_ptr<T>(_weights[25]) + i * hidden_dim;
      decoder_params[i].cublas_handle = _cublasHandle;
    }
    decoding_params.layernorm.gamma = get_ptr<T>(_weights[26]);
    decoding_params.layernorm.beta = get_ptr<T>(_weights[27]);
    decoding_params.embedding_table = get_ptr<T>(_weights[28]);
    decoding_params.position_encoding_table = get_ptr<T>(_weights[29]);
    decoding_params.embedding_kernel = get_ptr<T>(_weights[30]);
    decoding_params.embedding_bias = get_ptr<float>(_weights[31]);
    decoding_params.cublas_handle = _cublasHandle;
  }

  ~FTDecoding() override {
    cublasDestroy(_cublasHandle);
    delete [] decoder_params;
  }

  void forward(int batch_size, int beam_size, int max_seq_len, int mem_max_seq_len,
               Tensor memory, Tensor memory_seq_lens, Tensor output_ids, Tensor parent_ids, Tensor out_seq_lens) override
  {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    check_cuda_error(cublasSetStream(_cublasHandle, stream));
    decoding_params.stream = stream;
    for(int i = 0; i < _layer_num; ++i)
    {
      decoder_params[i].stream = stream;
      check_cuda_error(cublasSetStream(decoder_params[i].cublas_handle, stream));
    }
    check_cuda_error(cublasSetStream(decoding_params.cublas_handle, stream));

    decoding_params.output_ids = get_ptr<int>(output_ids);
    decoding_params.parent_ids = get_ptr<int>(parent_ids);
    decoding_params.sequence_length = get_ptr<int>(out_seq_lens);
    check_cuda_error(cudaMemset(decoding_params.output_ids, 0, sizeof(int) * batch_size * beam_size * max_seq_len));
    check_cuda_error(cudaMemset(decoding_params.parent_ids, 0, sizeof(int) * batch_size * beam_size * max_seq_len));
    check_cuda_error(cudaMemset(decoding_params.sequence_length, 0, sizeof(int) * batch_size * beam_size));
    decoding_params.memory_tensor = get_ptr<T>(memory);
    decoding_params.memory_sequence_length = get_ptr<int>(memory_seq_lens);

    fastertransformer::Allocator<AllocatorType::TH> allocator;
    DecodingBeamsearch<THTraits<T>::OpType>* decoding = 
      new DecodingBeamsearch<THTraits<T>::OpType>(allocator, batch_size, beam_size, max_seq_len, _head_num, _head_size, _vocab_size,
                                                  _layer_num, _mem_hidden_dim, mem_max_seq_len, _start_id, _end_id, _beam_search_diversity_rate);
    decoding->forward(decoder_params, decoding_params);
    delete decoding;
  }

private:
  const int _head_num;
  const int _head_size;
  const int _mem_hidden_dim;
  const int _layer_num;
  const int _vocab_size;
  const int _start_id;
  const int _end_id;
  const float _beam_search_diversity_rate;
  std::vector<Tensor> _weights;
  cublasHandle_t _cublasHandle;
  DecodingInitParam<T> decoding_params;
  DecoderInitParam<T>* decoder_params;
};

class FasterTransformerDecoding {
public:
  FasterTransformerDecoding(
    int head_num,
    int head_size,
    int mem_hidden_dim,
    int layer_num,
    int vocab_size,
    int start_id,
    int end_id,
    float beam_search_diversity_rate,
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
    Tensor output_bias,
    Tensor decoding_gamma,
    Tensor decoding_beta,
    Tensor embedding_table,
    Tensor position_encoding_table,
    Tensor embedding_kernel,
    Tensor embedding_bias);

  ~FasterTransformerDecoding();
  
  std::vector<Tensor> forward(int batch_size, int beam_size, int max_seq_len, Tensor memory, Tensor memory_seq_lens);

private:
  const at::ScalarType _st;
  IFTDecoding* ftdecoding;
};

Tensor gather_tree(Tensor step_ids, Tensor parent_ids, Tensor max_sequence_lengths, int end_token);

} // namespace torch_ext