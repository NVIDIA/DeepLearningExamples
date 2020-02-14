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

#include <queue>
#include <string>
#include <vector>

#include "request_grpc.h"

#ifndef TRTIS_KALDI_ASR_CLIENT_H_
#define TRTIS_KALDI_ASR_CLIENT_H_
namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

// time with arbitrary reference
double inline gettime_monotonic() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double time = ts.tv_sec;
  time += (double)(ts.tv_nsec) / 1e9;
  return time;
}

class TRTISASRClient {
  struct ClientContext {
    std::unique_ptr<nic::InferContext> trtis_context;
  };

  std::string url_;
  std::string model_name_;

  std::vector<ClientContext> contextes_;
  int ncontextes_;
  std::vector<uint8_t> chunk_buf_;
  std::vector<int64_t> shape_;
  int max_chunk_byte_size_;
  std::atomic<int> n_in_flight_;
  double started_at_;
  double total_audio_;
  bool print_results_;
  std::mutex stdout_m_;

  struct Result {
    std::string text;
    double latency;
  };

  std::unordered_map<ni::CorrelationID, Result> results_;
  std::mutex results_m_;

 public:
  void CreateClientContext();
  void SendChunk(uint64_t corr_id, bool start_of_sequence, bool end_of_sequence,
                 float* chunk, int chunk_byte_size);
  void WaitForCallbacks();
  void PrintStats();

  TRTISASRClient(const std::string& url, const std::string& model_name,
                 const int ncontextes, bool print_results);
};

#endif  // TRTIS_KALDI_ASR_CLIENT_H_
