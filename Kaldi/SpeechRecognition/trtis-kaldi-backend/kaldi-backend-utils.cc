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

#include "kaldi-backend-utils.h"

namespace nvidia {
namespace inferenceserver {
namespace custom {
namespace kaldi_cbe {

int GetInputTensor(CustomGetNextInputFn_t input_fn, void* input_context,
                   const char* name, const size_t expected_byte_size,
                   std::vector<uint8_t>* input, const void** out) {
  input->clear();  // reset buffer
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we might copy the chunks into 'input' vector.
  // If possible, we use the data in place
  uint64_t total_content_byte_size = 0;
  while (true) {
    const void* content;
    uint64_t content_byte_size = expected_byte_size - total_content_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) break;

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    total_content_byte_size += content_byte_size;
    if (total_content_byte_size > expected_byte_size) 
	    return kInputSize;

    if (content_byte_size == expected_byte_size) {
      *out = content;
      return kSuccess;
    }

    input->insert(input->end(), static_cast<const uint8_t*>(content),
                  static_cast<const uint8_t*>(content) + content_byte_size);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }
  *out = &input[0];

  return kSuccess;
}

void LatticeToString(fst::SymbolTable& word_syms,
                     const kaldi::CompactLattice& dlat, std::string* out_str) {
  kaldi::CompactLattice best_path_clat;
  kaldi::CompactLatticeShortestPath(dlat, &best_path_clat);

  kaldi::Lattice best_path_lat;
  fst::ConvertLattice(best_path_clat, &best_path_lat);

  std::vector<int32> alignment;
  std::vector<int32> words;
  kaldi::LatticeWeight weight;
  fst::GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  std::ostringstream oss;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = word_syms.Find(words[i]);
    if (s == "") std::cerr << "Word-id " << words[i] << " not in symbol table.";
    oss << s << " ";
  }
  *out_str = std::move(oss.str());
}

int ReadParameter(const ModelConfig& model_config_, const std::string& key,
                  std::string* param) {
  auto it = model_config_.parameters().find(key);
  if (it == model_config_.parameters().end()) {
    std::cerr << "Parameter \"" << key
              << "\" missing from config file. Exiting." << std::endl;
    return kInvalidModelConfig;
  }
  *param = it->second.string_value();
  return kSuccess;
}

int ReadParameter(const ModelConfig& model_config_, const std::string& key,
                  int* param) {
  std::string tmp;
  int err = ReadParameter(model_config_, key, &tmp);
  *param = std::stoi(tmp);
  return err;
}

int ReadParameter(const ModelConfig& model_config_, const std::string& key,
                  float* param) {
  std::string tmp;
  int err = ReadParameter(model_config_, key, &tmp);
  *param = std::stof(tmp);
  return err;
}

const char* CustomErrorString(int errcode) {
  switch (errcode) {
    case kSuccess:
      return "success";
    case kInvalidModelConfig:
      return "invalid model configuration";
    case kGpuNotSupported:
      return "execution on GPU not supported";
    case kSequenceBatcher:
      return "model configuration must configure sequence batcher";
    case kModelControl:
      return "'START' and 'READY' must be configured as the control inputs";
    case kInputOutput:
      return "model must have four inputs and one output with shape [-1]";
    case kInputName:
      return "names for input don't exist";
    case kOutputName:
      return "model output must be named 'OUTPUT'";
    case kInputOutputDataType:
      return "model inputs or outputs data_type cannot be specified";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    case kBatchTooBig:
      return "unable to execute batch larger than max-batch-size";
    case kTimesteps:
      return "unable to execute more than 1 timestep at a time";
    case kChunkTooBig:
      return "a chunk cannot contain more samples than the WAV_DATA dimension";
    default:
      break;
  }

  return "unknown error";
}

}  // kaldi
}  // custom
}  // inferenceserver
}  // nvidia
