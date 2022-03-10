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

#include <unistd.h>

#include <iostream>
#include <random>
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
  std::cerr << "\t-b : Print partial (best path) text outputs" << std::endl;
  //std::cerr << "\t-t : Print text with timings (CTM)" << std::endl;

  std::cerr << std::endl;
  exit(1);
}

int main(int argc, char** argv) {
  std::cout << "\n";
  std::cout << "==================================================\n"
            << "============= Triton Kaldi ASR Client ============\n"
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
  int nclients = 10;
  bool online = false;
  bool print_results = false;
  bool print_partial_results = false;
  bool ctm = false;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "va:u:i:c:otpbhl:")) != -1) {
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
      case 'b':
        print_partial_results = true;
        break;
      case 't':
        ctm = true;
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
  std::cout << "Print text outputs\t\t: " << (print_results ? "Yes" : "No")
            << std::endl;
  std::cout << "Print partial text outputs\t: "
            << (print_partial_results ? "Yes" : "No") << std::endl;
  std::cout << "Online - Realtime I/O\t\t: " << (online ? "Yes" : "No")
            << std::endl;
  std::cout << std::endl;

  float samp_freq = 0;
  double total_audio = 0;
  // pre-loading data
  // we don't want to measure I/O
  std::vector<std::shared_ptr<WaveData>> all_wav;
  std::vector<std::string> all_wav_keys;

  // need to read wav files
  SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
  {
    std::cout << "Loading eval dataset..." << std::flush;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string utt = wav_reader.Key();
      std::shared_ptr<WaveData> wave_data = std::make_shared<WaveData>();
      wave_data->Swap(&wav_reader.Value());
      all_wav.push_back(wave_data);
      all_wav_keys.push_back(utt);
      total_audio += wave_data->Duration();
      samp_freq = wave_data->SampFreq();
    }
    std::cout << "done" << std::endl;
  }

  if (all_wav.empty()) {
    std::cerr << "Empty dataset";
    exit(0);
  }

  std::cout << "Loaded dataset with " << all_wav.size()
            << " utterances, frequency " << samp_freq << "hz, total audio "
            << total_audio << " seconds" << std::endl;

  double chunk_seconds = (double)chunk_length / samp_freq;
  double seconds_per_sample = chunk_seconds / chunk_length;

  struct Stream {
    std::shared_ptr<WaveData> wav;
    uint64_t corr_id;
    int offset;
    double send_next_chunk_at;

    Stream(const std::shared_ptr<WaveData>& _wav, uint64_t _corr_id,
           double _send_next_chunk_at)
        : wav(_wav),
          corr_id(_corr_id),
          offset(0),
          send_next_chunk_at(_send_next_chunk_at) {}

    bool operator<(const Stream& other) const {
      return (send_next_chunk_at > other.send_next_chunk_at);
    }
  };

  std::cout << "Opening GRPC contextes..." << std::flush;
  std::unordered_map<uint64_t, std::string> corr_id_and_keys;
  TritonASRClient asr_client(url, model_name, nclients, print_results,
                             print_partial_results, ctm, samp_freq);
  std::cout << "done" << std::endl;
  std::cout << "Streaming utterances..." << std::endl;
  std::priority_queue<Stream> streams;
  size_t all_wav_i = 0;
  size_t all_wav_max = all_wav.size() * niterations;
  uint64_t index = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  bool add_random_offset = true;
  while (true) {
    while (streams.size() < nchannels && all_wav_i < all_wav_max) {
      // Creating new tasks
      uint64_t corr_id = static_cast<uint64_t>(all_wav_i) + 1;
      auto all_wav_i_modulo = all_wav_i % (all_wav.size());
      double stream_will_start_at = gettime_monotonic();
      if (add_random_offset) stream_will_start_at += dis(gen);
      double first_chunk_available_at =
          stream_will_start_at +
          std::min(static_cast<double>(all_wav[all_wav_i_modulo]->Duration()),
                   chunk_seconds);

      corr_id_and_keys.insert({corr_id, all_wav_keys[all_wav_i_modulo]});
      streams.emplace(all_wav[all_wav_i_modulo], corr_id,
                      first_chunk_available_at);
      ++all_wav_i;
    }
    // If still empty, done
    if (streams.empty()) break;

    auto task = streams.top();
    streams.pop();
    if (online) {
      double wait_for = task.send_next_chunk_at - gettime_monotonic();
      if (wait_for > 0) usleep(wait_for * 1e6);
    }
    add_random_offset = false;

    SubVector<BaseFloat> data(task.wav->Data(), 0);
    int32 samp_offset = task.offset;
    int32 nsamp = data.Dim();
    int32 samp_remaining = nsamp - samp_offset;
    int32 num_samp =
        chunk_length < samp_remaining ? chunk_length : samp_remaining;
    bool is_last_chunk = (chunk_length >= samp_remaining);
    SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
    bool is_first_chunk = (samp_offset == 0);
    asr_client.SendChunk(task.corr_id, is_first_chunk, is_last_chunk,
                         wave_part.Data(), wave_part.SizeInBytes(), index++);

    if (verbose)
      std::cout << "Sending correlation_id=" << task.corr_id
                << " chunk offset=" << num_samp << std::endl;

    task.offset += num_samp;
    int32 next_chunk_num_samp = std::min(nsamp - task.offset, chunk_length);
    task.send_next_chunk_at += next_chunk_num_samp * seconds_per_sample;

    if (!is_last_chunk) streams.push(task);

    // Showing activity if necessary
    if (!print_results && !print_partial_results && !verbose &&
        index % nchannels == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "done" << std::endl;
  std::cout << "Waiting for all results..." << std::flush;
  asr_client.WaitForCallbacks();
  std::cout << "done" << std::endl;

  asr_client.PrintStats(
      online,
      !online);  // Print latency if online, do not print throughput if online
  asr_client.WriteLatticesToFile("ark:|gzip -c > /data/results/lat.cuda-asr.gz",
                                 corr_id_and_keys);

  return 0;
}
