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

#include "asr_client_imp.h"

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-table.h"

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

void TritonASRClient::CreateClientContext() {
  clients_.emplace_back();
  TritonClient& client = clients_.back();
  FAIL_IF_ERR(nic::InferenceServerGrpcClient::Create(&client.triton_client,
                                                     url_, /*verbose*/ false),
              "unable to create triton client");

  FAIL_IF_ERR(
      client.triton_client->StartStream(
          [&](nic::InferResult* result) {
            double end_timestamp = gettime_monotonic();
            std::unique_ptr<nic::InferResult> result_ptr(result);
            FAIL_IF_ERR(result_ptr->RequestStatus(),
                        "inference request failed");
            std::string request_id;
            FAIL_IF_ERR(result_ptr->Id(&request_id),
                        "unable to get request id for response");
            uint64_t corr_id =
                std::stoi(std::string(request_id, 0, request_id.find("_")));
            bool end_of_stream = (request_id.back() == '1');
            if (!end_of_stream) {
              if (print_partial_results_) {
                std::vector<std::string> text;
                FAIL_IF_ERR(result_ptr->StringData("TEXT", &text),
                            "unable to get TEXT output");
                std::lock_guard<std::mutex> lk(stdout_m_);
                std::cout << "CORR_ID " << corr_id << "\t[partial]\t" << text[0]
                          << '\n';
              }
              return;
            }

            double start_timestamp;
            {
              std::lock_guard<std::mutex> lk(start_timestamps_m_);
              auto it = start_timestamps_.find(corr_id);
              if (it != start_timestamps_.end()) {
                start_timestamp = it->second;
                start_timestamps_.erase(it);
              } else {
                std::cerr << "start_timestamp not found" << std::endl;
                exit(1);
              }
            }

            if (print_results_) {
              std::vector<std::string> text;
              FAIL_IF_ERR(result_ptr->StringData(ctm_ ? "CTM" : "TEXT", &text),
                          "unable to get TEXT or CTM output");
              std::lock_guard<std::mutex> lk(stdout_m_);
              std::cout << "CORR_ID " << corr_id;
              std::cout << (ctm_ ? "\n" : "\t\t");
              std::cout << text[0] << std::endl;
            }

            std::vector<std::string> lattice_bytes;
            FAIL_IF_ERR(result_ptr->StringData("RAW_LATTICE", &lattice_bytes),
                        "unable to get RAW_LATTICE output");

            {
              double elapsed = end_timestamp - start_timestamp;
              std::lock_guard<std::mutex> lk(results_m_);
              results_.insert(
                  {corr_id, {std::move(lattice_bytes[0]), elapsed}});
            }

            n_in_flight_.fetch_sub(1, std::memory_order_relaxed);
          },
          false),
      "unable to establish a streaming connection to server");
}

void TritonASRClient::SendChunk(uint64_t corr_id, bool start_of_sequence,
                                bool end_of_sequence, float* chunk,
                                int chunk_byte_size, const uint64_t index) {
  // Setting options
  nic::InferOptions options(model_name_);
  options.sequence_id_ = corr_id;
  options.sequence_start_ = start_of_sequence;
  options.sequence_end_ = end_of_sequence;
  options.request_id_ = std::to_string(corr_id) + "_" + std::to_string(index) +
                        "_" + (start_of_sequence ? "1" : "0") + "_" +
                        (end_of_sequence ? "1" : "0");

  // Initialize the inputs with the data.
  nic::InferInput* wave_data_ptr;
  std::vector<int64_t> wav_shape{1, samps_per_chunk_};
  FAIL_IF_ERR(
      nic::InferInput::Create(&wave_data_ptr, "WAV_DATA", wav_shape, "FP32"),
      "unable to create 'WAV_DATA'");
  std::shared_ptr<nic::InferInput> wave_data_in(wave_data_ptr);
  FAIL_IF_ERR(wave_data_in->Reset(), "unable to reset 'WAV_DATA'");
  uint8_t* wave_data = reinterpret_cast<uint8_t*>(chunk);
  if (chunk_byte_size < max_chunk_byte_size_) {
    std::memcpy(&chunk_buf_[0], chunk, chunk_byte_size);
    wave_data = &chunk_buf_[0];
  }
  FAIL_IF_ERR(wave_data_in->AppendRaw(wave_data, max_chunk_byte_size_),
              "unable to set data for 'WAV_DATA'");

  // Dim
  nic::InferInput* dim_ptr;
  std::vector<int64_t> shape{1, 1};
  FAIL_IF_ERR(nic::InferInput::Create(&dim_ptr, "WAV_DATA_DIM", shape, "INT32"),
              "unable to create 'WAV_DATA_DIM'");
  std::shared_ptr<nic::InferInput> dim_in(dim_ptr);
  FAIL_IF_ERR(dim_in->Reset(), "unable to reset WAVE_DATA_DIM");
  int nsamples = chunk_byte_size / sizeof(float);
  FAIL_IF_ERR(
      dim_in->AppendRaw(reinterpret_cast<uint8_t*>(&nsamples), sizeof(int32_t)),
      "unable to set data for WAVE_DATA_DIM");

  std::vector<nic::InferInput*> inputs = {wave_data_in.get(), dim_in.get()};

  std::vector<const nic::InferRequestedOutput*> outputs;
  std::shared_ptr<nic::InferRequestedOutput> raw_lattice, text;
  outputs.reserve(2);
  if (end_of_sequence) {
    nic::InferRequestedOutput* raw_lattice_ptr;
    FAIL_IF_ERR(
        nic::InferRequestedOutput::Create(&raw_lattice_ptr, "RAW_LATTICE"),
        "unable to get 'RAW_LATTICE'");
    raw_lattice.reset(raw_lattice_ptr);
    outputs.push_back(raw_lattice.get());

    // Request the TEXT results only when required for printing
    if (print_results_) {
      nic::InferRequestedOutput* text_ptr;
      FAIL_IF_ERR(
          nic::InferRequestedOutput::Create(&text_ptr, ctm_ ? "CTM" : "TEXT"),
          "unable to get 'TEXT' or 'CTM'");
      text.reset(text_ptr);
      outputs.push_back(text.get());
    }
  } else if (print_partial_results_) {
    nic::InferRequestedOutput* text_ptr;
    FAIL_IF_ERR(nic::InferRequestedOutput::Create(&text_ptr, "TEXT"),
                "unable to get 'TEXT'");
    text.reset(text_ptr);
    outputs.push_back(text.get());
  }

  total_audio_ += (static_cast<double>(nsamples) / samp_freq_);

  if (start_of_sequence) {
    n_in_flight_.fetch_add(1, std::memory_order_consume);
  }

  // Record the timestamp when the last chunk was made available.
  if (end_of_sequence) {
    std::lock_guard<std::mutex> lk(start_timestamps_m_);
    start_timestamps_[corr_id] = gettime_monotonic();
  }

  TritonClient* client = &clients_[corr_id % nclients_];
  // nic::InferenceServerGrpcClient& triton_client = *client->triton_client;
  FAIL_IF_ERR(client->triton_client->AsyncStreamInfer(options, inputs, outputs),
              "unable to run model");
}

void TritonASRClient::WaitForCallbacks() {
  while (n_in_flight_.load(std::memory_order_consume)) {
    usleep(1000);
  }
}

void TritonASRClient::PrintStats(bool print_latency_stats,
                                 bool print_throughput) {
  double now = gettime_monotonic();
  double diff = now - started_at_;
  double rtf = total_audio_ / diff;
  if (print_throughput)
    std::cout << "Throughput:\t" << rtf << " RTFX" << std::endl;
  std::vector<double> latencies;
  {
    std::lock_guard<std::mutex> lk(results_m_);
    latencies.reserve(results_.size());
    for (auto& result : results_) latencies.push_back(result.second.latency);
  }
  std::sort(latencies.begin(), latencies.end());
  double nresultsf = static_cast<double>(latencies.size());
  size_t per90i = static_cast<size_t>(std::floor(90. * nresultsf / 100.));
  size_t per95i = static_cast<size_t>(std::floor(95. * nresultsf / 100.));
  size_t per99i = static_cast<size_t>(std::floor(99. * nresultsf / 100.));

  double lat_90 = latencies[per90i];
  double lat_95 = latencies[per95i];
  double lat_99 = latencies[per99i];

  double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
               latencies.size();

  std::cout << std::setprecision(3);
  std::cout << "Latencies:\t90%\t\t95%\t\t99%\t\tAvg\n";
  if (print_latency_stats) {
    std::cout << "\t\t" << lat_90 << "\t\t" << lat_95 << "\t\t" << lat_99
              << "\t\t" << avg << std::endl;
  } else {
    std::cout << "\t\tN/A\t\tN/A\t\tN/A\t\tN/A" << std::endl;
    std::cout << "Latency statistics are printed only when the "
                 "online option is set (-o)."
              << std::endl;
  }
}

TritonASRClient::TritonASRClient(const std::string& url,
                                 const std::string& model_name,
                                 const int nclients, bool print_results,
                                 bool print_partial_results, bool ctm,
                                 float samp_freq)
    : url_(url),
      model_name_(model_name),
      nclients_(nclients),
      print_results_(print_results),
      print_partial_results_(print_partial_results),
      ctm_(ctm),
      samp_freq_(samp_freq) {
  nclients_ = std::max(nclients_, 1);
  for (int i = 0; i < nclients_; ++i) CreateClientContext();

  inference::ModelMetadataResponse model_metadata;
  FAIL_IF_ERR(
      clients_[0].triton_client->ModelMetadata(&model_metadata, model_name),
      "unable to get model metadata");

  for (const auto& in_tensor : model_metadata.inputs()) {
    if (in_tensor.name().compare("WAV_DATA") == 0) {
      samps_per_chunk_ = in_tensor.shape()[1];
    }
  }

  max_chunk_byte_size_ = samps_per_chunk_ * sizeof(float);
  chunk_buf_.resize(max_chunk_byte_size_);
  shape_ = {max_chunk_byte_size_};
  n_in_flight_.store(0);
  started_at_ = gettime_monotonic();
  total_audio_ = 0;
}

void TritonASRClient::WriteLatticesToFile(
    const std::string& clat_wspecifier,
    const std::unordered_map<uint64_t, std::string>& corr_id_and_keys) {
  kaldi::CompactLatticeWriter clat_writer;
  clat_writer.Open(clat_wspecifier);
  std::unordered_map<std::string, size_t> key_count;
  std::lock_guard<std::mutex> lk(results_m_);
  for (auto& p : corr_id_and_keys) {
    uint64_t corr_id = p.first;
    std::string key = p.second;
    const auto iter = key_count[key]++;
    if (iter > 0) {
      key += std::to_string(iter);
    }
    auto it = results_.find(corr_id);
    if (it == results_.end()) {
      std::cerr << "Cannot find lattice for corr_id " << corr_id << std::endl;
      continue;
    }
    const std::string& raw_lattice = it->second.raw_lattice;
    // We could in theory write directly the binary hold in raw_lattice (it is
    // in the kaldi lattice format) However getting back to a CompactLattice
    // object allows us to us CompactLatticeWriter
    std::istringstream iss(raw_lattice);
    kaldi::CompactLattice* clat = NULL;
    kaldi::ReadCompactLattice(iss, true, &clat);
    clat_writer.Write(key, *clat);
  }
  clat_writer.Close();
}
