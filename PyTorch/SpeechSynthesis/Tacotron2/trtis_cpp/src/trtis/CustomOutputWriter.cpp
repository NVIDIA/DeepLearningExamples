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

#include "CustomOutputWriter.hpp"

#include <algorithm>
#include <cstring>

using namespace tts;

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

CustomOutputWriter::CustomOutputWriter() :
    TimedObject("CustomOutputWriter::write()")
{
  // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void CustomOutputWriter::write(
    void* const outputContext,
    CustomGetOutputFn_t outputFn,
    const int batchSize,
    const int samplesSpacing,
    const float* const samplesHost,
    const int32_t* const lengthsHost)
{
  startTiming();

  // determine maximum audio length
  int32_t maxLength
      = batchSize > 0 ? *std::max_element(lengthsHost, lengthsHost + batchSize)
                      : 0;

  // output audio
  {
    std::vector<int64_t> outputDims{batchSize, static_cast<int64_t>(maxLength)};

    float* hostMem;
    if (!outputFn(
            outputContext,
            "OUTPUT",
            outputDims.size(),
            outputDims.data(),
            sizeof(*samplesHost) * maxLength * batchSize,
            (void**)&hostMem)) {
      throw std::runtime_error("CustomGetOutputFn_t returned false.");
    }
    for (int i = 0; i < batchSize; ++i) {
      std::memcpy(
          hostMem + maxLength * i,
          samplesHost + samplesSpacing * i,
          maxLength * sizeof(*samplesHost));
    }
  }

  // output lengths
  {
    std::vector<int64_t> lengthDims{batchSize, 1};

    int32_t* hostMemLen;
    if (!outputFn(
            outputContext,
            "OUTPUT_LENGTH",
            lengthDims.size(),
            lengthDims.data(),
            sizeof(*lengthsHost) * batchSize,
            (void**)&hostMemLen)) {
      throw std::runtime_error("CustomGetOutputFn_t returned false.");
    }
    std::copy(lengthsHost, lengthsHost + batchSize, hostMemLen);
  }

  stopTiming();
}
