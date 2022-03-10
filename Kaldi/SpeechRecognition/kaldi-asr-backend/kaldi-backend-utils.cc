// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <triton/core/tritonserver.h>

using triton::common::TritonJson;

namespace triton {
namespace backend {

TRITONSERVER_Error* GetInputTensor(TRITONBACKEND_Request* request,
                                   const std::string& input_name,
                                   const size_t expected_byte_size,
                                   std::vector<uint8_t>* buffer,
                                   const void** out) {
  buffer->clear();  // reset buffer

  TRITONBACKEND_Input* input;
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInput(request, input_name.c_str(), &input));

  uint64_t input_byte_size;
  uint32_t input_buffer_count;
  RETURN_IF_ERROR(
      TRITONBACKEND_InputProperties(input, nullptr, nullptr, nullptr, nullptr,
                                    &input_byte_size, &input_buffer_count));
  RETURN_ERROR_IF_FALSE(
      input_byte_size == expected_byte_size, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(std::string("unexpected byte size ") +
                  std::to_string(expected_byte_size) + " requested for " +
                  input_name.c_str() + ", received " +
                  std::to_string(input_byte_size)));

  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we might copy the chunks into 'input' vector.
  // If possible, we use the data in place
  uint64_t total_content_byte_size = 0;
  for (uint32_t b = 0; b < input_buffer_count; ++b) {
    const void* input_buffer = nullptr;
    uint64_t input_buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, b, &input_buffer, &input_buffer_byte_size, &input_memory_type,
        &input_memory_type_id));
    RETURN_ERROR_IF_FALSE(input_memory_type != TRITONSERVER_MEMORY_GPU,
                          TRITONSERVER_ERROR_INTERNAL,
                          std::string("expected input tensor in CPU memory"));

    // Skip the copy if input already exists as a single contiguous
    // block
    if ((input_buffer_byte_size == expected_byte_size) && (b == 0)) {
      *out = input_buffer;
      return nullptr;
    }

    buffer->insert(
        buffer->end(), static_cast<const uint8_t*>(input_buffer),
        static_cast<const uint8_t*>(input_buffer) + input_buffer_byte_size);
    total_content_byte_size += input_buffer_byte_size;
  }

  // Make sure we end up with exactly the amount of input we expect.
  RETURN_ERROR_IF_FALSE(
      total_content_byte_size == expected_byte_size,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string(std::string("total unexpected byte size ") +
                  std::to_string(expected_byte_size) + " requested for " +
                  input_name.c_str() + ", received " +
                  std::to_string(total_content_byte_size)));
  *out = &buffer[0];

  return nullptr;
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
    if (s == "") {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          ("word-id " + std::to_string(words[i]) + " not in symbol table")
              .c_str());
    }
    oss << s << " ";
  }
  *out_str = std::move(oss.str());
}

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, std::string* param) {
  TritonJson::Value value;
  RETURN_ERROR_IF_FALSE(
      params.Find(key.c_str(), &value), TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration is missing the parameter ") + key);
  RETURN_IF_ERROR(value.MemberAsString("string_value", param));
  return nullptr;  // success
}

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, int* param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = std::stoi(tmp);
  return nullptr;  // success
}

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, float* param) {
  std::string tmp;
  RETURN_IF_ERROR(ReadParameter(params, key, &tmp));
  *param = std::stof(tmp);
  return nullptr;  // success
}

}  // namespace backend
}  // namespace triton
