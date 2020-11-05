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

void TRTISASRClient::CreateClientContext() {
  contextes_.emplace_back();
  ClientContext& client = contextes_.back();
  FAIL_IF_ERR(
      nic::InferGrpcStreamContext::Create(&client.trtis_context,
                                          /*corr_id*/ -1, url_, model_name_,
                                          /*model_version*/ -1,
                                          /*verbose*/ false),
      "unable to create context");
}

void TRTISASRClient::SendChunk(ni::CorrelationID corr_id,
                               bool start_of_sequence, bool end_of_sequence,
                               float* chunk, int chunk_byte_size) {
  ClientContext* client = &contextes_[corr_id % ncontextes_];
  nic::InferContext& context = *client->trtis_context;
  if (start_of_sequence) n_in_flight_.fetch_add(1, std::memory_order_consume);

  // Setting options
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(nic::InferContext::Options::Create(&options),
              "unable to create inference options");
  options->SetBatchSize(1);
  options->SetFlags(0);
  options->SetCorrelationId(corr_id);
  if (start_of_sequence)
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_START,
                     start_of_sequence);
  if (end_of_sequence) {
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_END,
                     end_of_sequence);
    for (const auto& output : context.Outputs()) {
      if (output->Name() == "TEXT" && !print_results_)
        continue;  // no need for text output if not printing
      options->AddRawResult(output);
    }
  }

  FAIL_IF_ERR(context.SetRunOptions(*options), "unable to set context options");
  std::shared_ptr<nic::InferContext::Input> in_wave_data, in_wave_data_dim;
  FAIL_IF_ERR(context.GetInput("WAV_DATA", &in_wave_data),
              "unable to get WAV_DATA");
  FAIL_IF_ERR(context.GetInput("WAV_DATA_DIM", &in_wave_data_dim),
              "unable to get WAV_DATA_DIM");

  // Wave data input
  FAIL_IF_ERR(in_wave_data->Reset(), "unable to reset WAVE_DATA");
  uint8_t* wave_data = reinterpret_cast<uint8_t*>(chunk);
  if (chunk_byte_size < max_chunk_byte_size_) {
    std::memcpy(&chunk_buf_[0], chunk, chunk_byte_size);
    wave_data = &chunk_buf_[0];
  }
  FAIL_IF_ERR(in_wave_data->SetRaw(wave_data, max_chunk_byte_size_),
              "unable to set data for WAVE_DATA");
  // Dim
  FAIL_IF_ERR(in_wave_data_dim->Reset(), "unable to reset WAVE_DATA_DIM");
  int nsamples = chunk_byte_size / sizeof(float);
  FAIL_IF_ERR(in_wave_data_dim->SetRaw(reinterpret_cast<uint8_t*>(&nsamples),
                                       sizeof(int32_t)),
              "unable to set data for WAVE_DATA_DIM");

  total_audio_ += (static_cast<double>(nsamples) / 16000.);  // TODO freq
  double start = gettime_monotonic();
  FAIL_IF_ERR(context.AsyncRun([corr_id, end_of_sequence, start, this](
                                   nic::InferContext* ctx,
                                   const std::shared_ptr<
                                       nic::InferContext::Request>& request) {
    if (end_of_sequence) {
      double elapsed = gettime_monotonic() - start;
      std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
      ctx->GetAsyncRunResults(request, &results);

      if (results.empty()) {
        std::cerr << "Warning: Could not read "
                     "output for corr_id "
                  << corr_id << std::endl;
      } else {
        if (print_results_) {
	  std::string text;
	  FAIL_IF_ERR(results["TEXT"]->GetRawAtCursor(0, &text),
			  "unable to get TEXT output");
          std::lock_guard<std::mutex> lk(stdout_m_);
          std::cout << "CORR_ID " << corr_id << "\t\t" << text << std::endl;
        }

        std::string lattice_bytes;
        FAIL_IF_ERR(results["RAW_LATTICE"]->GetRawAtCursor(0, &lattice_bytes),
                    "unable to get RAW_LATTICE output");
        {
          std::lock_guard<std::mutex> lk(results_m_);
          results_.insert({corr_id, {std::move(lattice_bytes), elapsed}});
        }
      }
      n_in_flight_.fetch_sub(1, std::memory_order_relaxed);
    }
  }),
              "unable to run model");
}

void TRTISASRClient::WaitForCallbacks() {
  int n;
  while ((n = n_in_flight_.load(std::memory_order_consume))) {
    usleep(1000);
  }
}

void TRTISASRClient::PrintStats(bool print_latency_stats) {
  double now = gettime_monotonic();
  double diff = now - started_at_;
  double rtf = total_audio_ / diff;
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

TRTISASRClient::TRTISASRClient(const std::string& url,
                               const std::string& model_name,
                               const int ncontextes, bool print_results)
    : url_(url),
      model_name_(model_name),
      ncontextes_(ncontextes),
      print_results_(print_results) {
  ncontextes_ = std::max(ncontextes_, 1);
  for (int i = 0; i < ncontextes_; ++i) CreateClientContext();

  std::shared_ptr<nic::InferContext::Input> in_wave_data;
  FAIL_IF_ERR(contextes_[0].trtis_context->GetInput("WAV_DATA", &in_wave_data),
              "unable to get WAV_DATA");
  max_chunk_byte_size_ = in_wave_data->ByteSize();
  chunk_buf_.resize(max_chunk_byte_size_);
  shape_ = {max_chunk_byte_size_};
  n_in_flight_.store(0);
  started_at_ = gettime_monotonic();
  total_audio_ = 0;
}

void TRTISASRClient::WriteLatticesToFile(
    const std::string& clat_wspecifier,
    const std::unordered_map<ni::CorrelationID, std::string>&
        corr_id_and_keys) {
  kaldi::CompactLatticeWriter clat_writer;
  clat_writer.Open(clat_wspecifier);
  std::lock_guard<std::mutex> lk(results_m_);
  for (auto& p : corr_id_and_keys) {
    ni::CorrelationID corr_id = p.first;
    const std::string& key = p.second;
    auto it = results_.find(corr_id);
    if(it == results_.end()) {
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
