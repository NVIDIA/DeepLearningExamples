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

#include <grpc_client.h>

#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef TRITON_KALDI_ASR_CLIENT_H_
#define TRITON_KALDI_ASR_CLIENT_H_

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

class TritonASRClient {
  struct TritonClient {
    std::unique_ptr<nic::InferenceServerGrpcClient> triton_client;
  };

  std::string url_;
  std::string model_name_;

  std::vector<TritonClient> clients_;
  int nclients_;
  std::vector<uint8_t> chunk_buf_;
  std::vector<int64_t> shape_;
  int max_chunk_byte_size_;
  std::atomic<int> n_in_flight_;
  double started_at_;
  double total_audio_;
  bool print_results_;
  bool print_partial_results_;
  bool ctm_;
  std::mutex stdout_m_;
  int samps_per_chunk_;
  float samp_freq_;

  struct Result {
    std::string raw_lattice;
    double latency;
  };

  std::unordered_map<uint64_t, double> start_timestamps_;
  std::mutex start_timestamps_m_;

  std::unordered_map<uint64_t, Result> results_;
  std::mutex results_m_;

 public:
  TritonASRClient(const std::string& url, const std::string& model_name,
                  const int ncontextes, bool print_results,
                  bool print_partial_results, bool ctm, float samp_freq);

  void CreateClientContext();
  void SendChunk(uint64_t corr_id, bool start_of_sequence, bool end_of_sequence,
                 float* chunk, int chunk_byte_size, uint64_t index);
  void WaitForCallbacks();
  void PrintStats(bool print_latency_stats, bool print_throughput);
  void WriteLatticesToFile(
      const std::string& clat_wspecifier,
      const std::unordered_map<uint64_t, std::string>& corr_id_and_keys);
};

#endif  // TRITON_KALDI_ASR_CLIENT_H_
