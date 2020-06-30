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
/**
 * Decoder transformer
 **/

#pragma once

#include "fastertransformer/common.h"
#include "fastertransformer/common_structure.h"
#include <cuda_runtime.h>
#include <stdlib.h>

namespace fastertransformer
{

template <typename T>
class DecodingInitParam
{
public:
  /* weights for masked_multi_head_attention */
  const T *embedding_table;
  const T *embedding_kernel;
  const float *embedding_bias;

  const T *memory_tensor;
  const int *memory_sequence_length;

  const T *position_encoding_table;

  LayerNormWeight<T> layernorm;

  int *output_ids;
  int *parent_ids;
  int *sequence_length;
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
};

struct TransformerArguments
{
  int batch_size_;
  int seq_len_;
  int head_num_;
  int size_per_head_;
  int hidden_units_;
};

struct DecodingArguments : public TransformerArguments
{
  int decoder_layers_;
  int vocab_size_;
  int start_id_;
  int end_id_;
};

struct DecodingSamplingArguments : public DecodingArguments
{
  int candidate_num_;
  float probability_threshold_;
  size_t temp_storage_size_;
};

struct DecodingBeamsearchArguments : public DecodingArguments
{
  int beam_width_;
  int temp_storage_size_;
  float beam_search_diversity_rate_;
};

} // end of namespace transformer