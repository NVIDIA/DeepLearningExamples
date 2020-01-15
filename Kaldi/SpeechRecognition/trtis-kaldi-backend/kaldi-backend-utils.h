// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "lat/lattice-functions.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

namespace nvidia {
namespace inferenceserver {
namespace custom {
namespace kaldi_cbe {

enum ErrorCodes {
  kSuccess,
  kUnknown,
  kInvalidModelConfig,
  kGpuNotSupported,
  kSequenceBatcher,
  kModelControl,
  kInputOutput,
  kInputName,
  kOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kOutputBuffer,
  kBatchTooBig,
  kTimesteps,
  kChunkTooBig
};

int GetInputTensor(CustomGetNextInputFn_t input_fn, void* input_context,
                   const char* name, const size_t expected_byte_size,
                   std::vector<uint8_t>* input, const void** out);

void LatticeToString(fst::SymbolTable& word_syms,
                     const kaldi::CompactLattice& dlat, std::string* out_str);

int ReadParameter(const ModelConfig& model_config_, const std::string& key,
                  std::string* param);

int ReadParameter(const ModelConfig& model_config_, const std::string& key,
                  int* param);
int ReadParameter(const ModelConfig& model_config_, const std::string& key,
                  float* param);

const char* CustomErrorString(int errcode);

}  // kaldi
}  // custom
}  // inferenceserver
}  // nvidia
