/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "TRTISClient.hpp"
#include "WaveFileWriter.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using highres_clock = std::chrono::high_resolution_clock;
using time_type = std::chrono::high_resolution_clock::time_point;

namespace
{

double timeElapsed(const time_type& start, const time_type& end)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
             .count()
         / 1000000.0;
}

std::vector<std::string> loadInputs(const std::string filename)
{
  std::ifstream fin(filename);

  if (!fin.good()) {
    throw std::runtime_error("Failed to open '" + filename + "'.");
  }

  fin.exceptions(std::ifstream::badbit);

  std::vector<std::string> data;
  std::string line;
  while (std::getline(fin, line)) {
    data.emplace_back(line);
  }

  return data;
}

} // namespace

int main(int argc, const char** argv)
{
  std::string url("localhost:8000");

  if (argc < 2 || argc > 3) {
    std::cerr << "Invalid number of arguments: " << (argc - 1) << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "\t" << argv[0] << " <input file> <batch size>" << std::endl;
    return 1;
  }

  const std::string inputFile(argv[1]);

  int batchSize = 1;
  if (argc == 3) {
    batchSize = std::stol(argv[2]);
  }

  TRTISClient client(url);

  try {
    const std::vector<std::string> inputs = loadInputs(inputFile);

    size_t totalChars = 0;
    for (const std::string& seq : inputs) {
      totalChars += seq.size();
    }

    time_type start = highres_clock::now();
    std::vector<std::vector<float>> outputs
        = client.execute(inputs, batchSize, false);
    time_type stop = highres_clock::now();

    size_t totalSamples = 0;
    for (const std::vector<float>& sample : outputs) {
      totalSamples += sample.size();
    }
    const double audioDuration = static_cast<double>(totalSamples) / 22050.0;
    const double duration = timeElapsed(start, stop);

    std::cout << "Total Processing time: " << duration << " sec" << std::endl;
    std::cout << "Processed " << inputs.size() << " sequences for a total of "
              << audioDuration << " seconds of audio:" << std::endl;
    std::cout << "\t" << (totalChars / duration) << " symbols / sec."
              << std::endl;
    std::cout << "\t" << (totalSamples / duration) << " samples / sec."
              << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i) {
      WaveFileWriter::write(
          "./audio/" + std::to_string(i + 1) + ".wav",
          22050,
          outputs[i].data(),
          outputs[i].size());
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
