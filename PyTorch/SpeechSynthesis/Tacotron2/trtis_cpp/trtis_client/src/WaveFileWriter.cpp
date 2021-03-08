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

#include "WaveFileWriter.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/
namespace
{

bool isLittleEndian()
{
  constexpr const uint32_t key = 0x4D3C2B1A;
  const char* const keyBytes = reinterpret_cast<const char*>(&key);
  return keyBytes[0] == 0x1A && keyBytes[1] == 0x2B && keyBytes[2] == 0x3C
         && keyBytes[3] == 0x4D;
}

void writeToStream(
    std::ofstream& stream, const char* const bytes, const size_t len)
{
  stream.write(bytes, len);
  if (!stream.good()) {
    throw std::runtime_error("Failed to write to stream.");
  }
}

void writeToStream(std::ofstream& stream, const std::string& str)
{
  writeToStream(stream, str.c_str(), str.length());
}

void writeToStream(std::ofstream& stream, const uint32_t num)
{
  writeToStream(stream, reinterpret_cast<const char*>(&num), sizeof(num));
}

void writeToStream(std::ofstream& stream, const uint16_t num)
{
  writeToStream(stream, reinterpret_cast<const char*>(&num), sizeof(num));
}

void writeToStream(std::ofstream& stream, const int16_t num)
{
  writeToStream(stream, reinterpret_cast<const char*>(&num), sizeof(num));
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void WaveFileWriter::write(
    const std::string& filename,
    const int frequency,
    const float* const data,
    const size_t numSamples)
{
  if (!isLittleEndian()) {
    throw std::runtime_error("Wave file writing is only implemented for "
                             "little endian architectures.");
  }

  std::ofstream fout(filename, std::ofstream::trunc | std::ofstream::binary);
  if (!fout.good()) {
    throw std::runtime_error("Failed to open '" + filename + "' for writing.");
  }

  const int numChannels = 1;
  const int bytesPerSample = static_cast<int>(sizeof(int16_t));
  const int bitsPerSample = bytesPerSample * 8;

  const int byteRate = frequency * bytesPerSample;

  // write ckID
  writeToStream(fout, "RIFF");

  const size_t ckSizePos = fout.tellp();

  // write cksize -- filled in layer
  writeToStream(fout, static_cast<uint32_t>(0));

  // write WAVID
  writeToStream(fout, "WAVE");

  // write format (PCM 16bit mono)
  writeToStream(fout, "fmt ");
  writeToStream(fout, static_cast<uint32_t>(16));
  writeToStream(fout, static_cast<uint16_t>(0x0001));
  writeToStream(fout, static_cast<uint16_t>(numChannels));
  writeToStream(fout, static_cast<uint32_t>(frequency));
  writeToStream(fout, static_cast<uint32_t>(frequency * byteRate));
  writeToStream(fout, static_cast<uint16_t>(numChannels * bytesPerSample));
  writeToStream(fout, static_cast<uint16_t>(bitsPerSample));

  // write chunk header
  writeToStream(fout, "data");

  const size_t chunkSizePos = fout.tellp();
  writeToStream(fout, static_cast<uint32_t>(0));

  for (size_t i = 0; i < numSamples; ++i) {
    const int16_t sample
        = static_cast<int16_t>(data[i] * std::numeric_limits<int16_t>::max());
    writeToStream(fout, sample);
  }

  const size_t fileLength = fout.tellp();

  // fill in missing numbers
  fout.seekp(ckSizePos);
  writeToStream(fout, static_cast<uint32_t>(fileLength - ckSizePos - 4));

  fout.seekp(chunkSizePos);
  writeToStream(fout, static_cast<uint32_t>(fileLength - chunkSizePos - 4));
}
