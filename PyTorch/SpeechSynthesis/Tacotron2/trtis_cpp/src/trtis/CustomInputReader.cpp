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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CustomInputReader.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace tts;

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

CustomInputReader::CustomInputReader(const CharacterMapping& charMapping) :
    TimedObject("CustomInputReader::read()"),
    m_charMapping(charMapping)
{
  addChild(&m_charMapping);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void CustomInputReader::read(
    void* const inputContext,
    CustomGetNextInputFn_t inputFn,
    const size_t maxLength,
    const int batchSize,
    int32_t* const inputHost,
    int32_t* const inputLengthsHost,
    int32_t* const inputSpacing)
{
  startTiming();

  // read input
  std::vector<char> inputBuffer;
  inputBuffer.reserve(maxLength * batchSize);
  {
    uint64_t sizeBytes;
    const void* nextPtr;
    size_t inputPos = 0;

    while (true) {
      sizeBytes = (maxLength * sizeof(*inputBuffer.data())) - inputPos;
      const bool success = inputFn(inputContext, "INPUT", &nextPtr, &sizeBytes);
      if (!success) {
        throw std::runtime_error("CustomGetNextInputFn_t returned false while "
                                 "reading input tensor.");
      }

      if (nextPtr == nullptr) {
        // input is finished
        break;
      }

      const size_t newSize = inputPos + sizeBytes;
      if (newSize > maxLength * sizeof(*inputBuffer.data())) {
        throw std::runtime_error(
            "Input tensor is larger than expected: "
            "next chunk of size "
            + std::to_string(sizeBytes) + " when already read "
            + std::to_string(inputPos) + " and tensor should be at most "
            + std::to_string(maxLength * sizeof(*inputBuffer.data())));
      }

      inputBuffer.resize((inputPos + sizeBytes) / sizeof(*inputBuffer.data()));
      std::memcpy(
          inputBuffer.data() + (inputPos / sizeof(*inputBuffer.data())),
          nextPtr,
          sizeBytes);
      inputPos = newSize;
    }
  }

  // currently mapping only translates
  // from multi-byte/character to single sequence item, not the reverse, so the
  // length will only decrease

  // first pass to determine maximum length
  int pos = 0;
  int32_t maxInitLen = 0;
  for (int i = 0; i < batchSize; ++i) {
    const int length
        = *reinterpret_cast<const int32_t*>(inputBuffer.data() + pos);
    pos += sizeof(int32_t) + length;
    maxInitLen = std::max(maxInitLen, length);
  }

  pos = 0;
  for (int i = 0; i < batchSize; ++i) {
    const int length
        = *reinterpret_cast<const int32_t*>(inputBuffer.data() + pos);
    pos += sizeof(int32_t);
    size_t outputLen;
    m_charMapping.map(
        inputBuffer.data() + pos,
        length,
        inputHost + maxInitLen * i,
        &outputLen);
    inputLengthsHost[i] = static_cast<int32_t>(outputLen);
    pos += length;
  }

  *inputSpacing = maxInitLen;

  stopTiming();
}

void CustomInputReader::setCharacterMapping(const CharacterMapping& newMapping)
{
  m_charMapping = newMapping;
}
