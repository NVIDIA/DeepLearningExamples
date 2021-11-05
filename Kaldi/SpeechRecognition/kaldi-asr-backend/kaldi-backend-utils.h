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

#pragma once

#include <lat/lattice-functions.h>
#include <triton/backend/backend_common.h>
#include <triton/common/triton_json.h>

namespace triton {
namespace backend {

using triton::common::TritonJson;

#define RETURN_AND_LOG_IF_ERROR(X, MSG)        \
  do {                                         \
    TRITONSERVER_Error* rie_err__ = (X);       \
    if (rie_err__ != nullptr) {                \
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, MSG); \
      return rie_err__;                        \
    }                                          \
  } while (false)

TRITONSERVER_Error* GetInputTensor(TRITONBACKEND_Request* request,
                                   const std::string& input_name,
                                   const size_t expected_byte_size,
                                   std::vector<uint8_t>* input,
                                   const void** out);

TRITONSERVER_Error* LatticeToString(TRITONBACKEND_Request* request,
                                    const std::string& input_name, char* buffer,
                                    size_t* buffer_byte_size);

void LatticeToString(fst::SymbolTable& word_syms,
                     const kaldi::CompactLattice& dlat, std::string* out_str);

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, std::string* param);

TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, int* param);
TRITONSERVER_Error* ReadParameter(TritonJson::Value& params,
                                  const std::string& key, float* param);

}  // namespace backend
}  // namespace triton
