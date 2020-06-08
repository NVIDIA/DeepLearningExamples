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

#include "UnitTest.hpp"
#include "cudaMemory.h"
#include "dataShuffler.h"

#include <vector>

using namespace tts;

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST(parseDecoderOutput)
{
  const int chunkSize = 89;
  const int batchSize = 3;
  const int numChannels = 80;
  const int rows = chunkSize;
  const int cols = (numChannels + 1) * batchSize;
  std::vector<float> mat(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      mat[i * cols + j] = static_cast<float>(i * cols + j);
      if ((j % (numChannels + 1)) == numChannels) {
        // gate
        mat[i * cols + j] *= -1.0f;
      }
    }
  }

  CudaMemory<float> matInDev(mat);
  CudaMemory<float> matOutDev(chunkSize * numChannels * batchSize);
  CudaMemory<float> gateOutDev(chunkSize * batchSize);

  DataShuffler::parseDecoderOutput(
      matInDev.data(),
      matOutDev.data(),
      gateOutDev.data(),
      batchSize,
      chunkSize,
      numChannels,
      0);

  const std::vector<float> act = matOutDev.toHost();

  for (int i = 0; i < numChannels * batchSize; ++i) {
    for (int j = 0; j < chunkSize; ++j) {
      EXPECT_EQ(
          act[i * chunkSize + j],
          static_cast<float>(j * cols + (i + (i / numChannels))))
          << "i = " << i << " j = " << j;
    }
  }

  const std::vector<float> actGate = gateOutDev.toHost();
  for (int i = 0; i < batchSize; ++i) {
    for (int j = 0; j < chunkSize; ++j) {
      EXPECT_EQ(
          actGate[i * chunkSize + j],
          -static_cast<float>(
              ((i + 1) * numChannels + i)
              + (j * (numChannels + 1) * batchSize)))
          << "i = " << i << " j = " << j;
    }
  }
}
