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

#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include "asr_client_imp.h"
#include "feat/wave-reader.h"  // to read the wav.scp
#include "util/kaldi-table.h"

using kaldi::BaseFloat;

void Usage(char** argv, const std::string& msg = std::string()) {
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: scripts/docker/launch_client.sh [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-i <Number of iterations on the dataset>" << std::endl;
  std::cerr << "\t-c <Number of parallel audio channels>" << std::endl;
  std::cerr << "\t-a <Path to the scp dataset file>" << std::endl;
  std::cerr << "\t-l <Maximum number of samples per chunk. Must correspond to "
               "the server config>"
            << std::endl;
  std::cerr << "\t-u <URL for inference service and its gRPC port>"
            << std::endl;
  std::cerr << "\t-o : Only feed each channel at realtime speed. Simulates "
               "online clients."
            << std::endl;
  std::cerr << "\t-p : Print text outputs" << std::endl;

  std::cerr << std::endl;
  exit(1);
}

int main(int argc, char** argv) {
  std::cout << "\n";
  std::cout << "==================================================\n"
            << "============= TRTIS Kaldi ASR Client =============\n"
            << "==================================================\n"
            << std::endl;

  // kaldi nampespace TODO
  using namespace kaldi;
  typedef kaldi::int32 int32;

  std::string url = "localhost:8001";
  std::string model_name = "kaldi_online";
  std::string wav_rspecifier =
      "scp:/data/datasets/LibriSpeech/test_clean/wav_conv.scp";
  int chunk_length = 8160;
  size_t nchannels = 1000;
  int niterations = 5;
  bool verbose = false;
  float samp_freq = 16000;
  int ncontextes = 10;
  bool online = false;
  bool print_results = false;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "va:u:i:c:ophl:")) != -1) {
    switch (opt) {
      case 'i':
        niterations = std::atoi(optarg);
        break;
      case 'c':
        nchannels = std::atoi(optarg);
        break;
      case 'a':
        wav_rspecifier = optarg;
        break;
      case 'u':
        url = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'o':
        online = true;
        break;
      case 'p':
        print_results = true;
        break;
      case 'l':
        chunk_length = std::atoi(optarg);
        break;
      case 'h':
      case '?':
        Usage(argv);
        break;
    }
  }

  if (niterations <= 0) Usage(argv, "number of iterations must be > 0");
  if (nchannels <= 0) Usage(argv, "number of audio channels must be > 0");
  if (chunk_length <= 0) Usage(argv, "chunk length must be > 0");

  std::cout << "Configuration:" << std::endl;
  std::cout << std::endl;
  std::cout << "Number of iterations\t\t: " << niterations << std::endl;
  std::cout << "Number of parallel channels\t: " << nchannels << std::endl;
  std::cout << "Server URL\t\t\t: " << url << std::endl;
  std::cout << "Print results\t\t\t: " << (print_results ? "Yes" : "No")
            << std::endl;
  std::cout << "Online - Realtime I/O\t\t: " << (online ? "Yes" : "No")
            << std::endl;
  std::cout << std::endl;

  float chunk_seconds = (double)chunk_length / samp_freq;
  // need to read wav files
  SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

  std::atomic<uint64_t> correlation_id;
  correlation_id.store(1);  // 0 = no correlation

  double total_audio = 0;
  // pre-loading data
  // we don't want to measure I/O
  std::vector<std::shared_ptr<WaveData>> all_wav;
  {
    std::cout << "Loading eval dataset..." << std::flush;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string utt = wav_reader.Key();
      std::shared_ptr<WaveData> wave_data = std::make_shared<WaveData>();
      wave_data->Swap(&wav_reader.Value());
      all_wav.push_back(wave_data);
      total_audio += wave_data->Duration();
    }
    std::cout << "done" << std::endl;
  }

  struct Stream {
    std::shared_ptr<WaveData> wav;
    ni::CorrelationID corr_id;
    int offset;
    float send_next_chunk_at;
    std::atomic<bool> received_output;

    Stream(const std::shared_ptr<WaveData>& _wav, ni::CorrelationID _corr_id)
        : wav(_wav), corr_id(_corr_id), offset(0), received_output(true) {
      send_next_chunk_at = gettime_monotonic();
    }
  };
  std::cout << "Opening GRPC contextes..." << std::flush;
  TRTISASRClient asr_client(url, model_name, ncontextes, print_results);
  std::cout << "done" << std::endl;
  std::cout << "Streaming utterances..." << std::flush;
  std::vector<std::unique_ptr<Stream>> curr_tasks, next_tasks;
  curr_tasks.reserve(nchannels);
  next_tasks.reserve(nchannels);
  size_t all_wav_i = 0;
  size_t all_wav_max = all_wav.size() * niterations;
  while (true) {
      while (curr_tasks.size() < nchannels && all_wav_i < all_wav_max) {
        // Creating new tasks
        uint64_t corr_id = correlation_id.fetch_add(1);
        std::unique_ptr<Stream> ptr(new Stream(all_wav[all_wav_i%(all_wav.size())], corr_id));
        curr_tasks.emplace_back(std::move(ptr));
        ++all_wav_i;
      }
      // If still empty, done
      if (curr_tasks.empty()) break;

      for (size_t itask = 0; itask < curr_tasks.size(); ++itask) {
        Stream& task = *(curr_tasks[itask]);

        SubVector<BaseFloat> data(task.wav->Data(), 0);
        int32 samp_offset = task.offset;
        int32 nsamp = data.Dim();
        int32 samp_remaining = nsamp - samp_offset;
        int32 num_samp =
            chunk_length < samp_remaining ? chunk_length : samp_remaining;
        bool is_last_chunk = (chunk_length >= samp_remaining);
        SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
        bool is_first_chunk = (samp_offset == 0);
        if (online) {
          double now = gettime_monotonic();
          double wait_for = task.send_next_chunk_at - now;
          if (wait_for > 0) usleep(wait_for * 1e6);
        }
        asr_client.SendChunk(task.corr_id, is_first_chunk, is_last_chunk,
                             wave_part.Data(), wave_part.SizeInBytes());
        task.send_next_chunk_at += chunk_seconds;
        if (verbose)
          std::cout << "Sending correlation_id=" << task.corr_id
                    << " chunk offset=" << num_samp << std::endl;

        task.offset += num_samp;
        if (!is_last_chunk) next_tasks.push_back(std::move(curr_tasks[itask]));
      }

      curr_tasks.swap(next_tasks);
      next_tasks.clear();
      // Showing activity if necessary
      if (!print_results && !verbose) std::cout << "." << std::flush;
  }
  std::cout << "done" << std::endl;
  std::cout << "Waiting for all results..." << std::flush;
  asr_client.WaitForCallbacks();
  std::cout << "done" << std::endl;
  asr_client.PrintStats();

  return 0;
}
