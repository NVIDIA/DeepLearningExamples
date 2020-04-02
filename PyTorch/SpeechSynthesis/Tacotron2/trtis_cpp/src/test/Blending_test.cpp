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
#include "blending.h"
#include "cudaMemory.h"

#include <vector>

using namespace tts;

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST(noOverlapNoOffsetBatchSize1)
{
  const int chunkSize = 4000;
  const int batchSize = 1;

  std::vector<float> samplesHost(chunkSize * batchSize);
  for (size_t i = 0; i < samplesHost.size(); ++i) {
    samplesHost[i] = static_cast<float>(i % 1001) / 1000.0f;
  }
  CudaMemory<float> samplesDevice(samplesHost);
  CudaMemory<float> outDevice(samplesHost.size());

  Blending::linear(
      batchSize,
      samplesDevice.data(),
      outDevice.data(),
      chunkSize,
      0,
      chunkSize,
      0,
      0);

  const std::vector<float> outHost = outDevice.toHost();
  for (size_t i = 0; i < samplesHost.size(); ++i) {
    EXPECT_NEAR(samplesHost[i], outHost[i], 1e-6f);
  }
}

TEST(noOverlapNoOffsetBatchSize4)
{
  const int chunkSize = 4000;
  const int batchSize = 4;

  std::vector<float> samplesHost(chunkSize * batchSize);
  for (size_t i = 0; i < samplesHost.size(); ++i) {
    samplesHost[i] = static_cast<float>(i % 1001) / 1000.0f;
  }
  CudaMemory<float> samplesDevice(samplesHost);
  CudaMemory<float> outDevice(samplesHost.size());

  Blending::linear(
      batchSize,
      samplesDevice.data(),
      outDevice.data(),
      chunkSize,
      0,
      chunkSize,
      0,
      0);

  const std::vector<float> outHost = outDevice.toHost();
  for (size_t i = 0; i < samplesHost.size(); ++i) {
    EXPECT_NEAR(samplesHost[i], outHost[i], 1e-6f);
  }
}

TEST(noOverlapOneOffsetBatchSize4)
{
  const int chunkSize = 4000;
  const int batchSize = 4;

  std::vector<float> samplesHost(chunkSize * batchSize);
  for (size_t i = 0; i < samplesHost.size(); ++i) {
    samplesHost[i] = static_cast<float>(i % 1001) / 1000.0f;
  }
  CudaMemory<float> samplesDevice(samplesHost);
  CudaMemory<float> outDevice(samplesHost.size() * 2);
  outDevice.zero();

  Blending::linear(
      batchSize,
      samplesDevice.data(),
      outDevice.data(),
      chunkSize,
      0,
      2 * chunkSize,
      chunkSize,
      0);

  const std::vector<float> outHost = outDevice.toHost();
  for (int b = 0; b < batchSize; ++b) {
    for (int i = 0; i < chunkSize; ++i) {
      const int j = b * (chunkSize * 2) + i;
      EXPECT_EQ(0.0f, outHost[j]) << "i = " << i;
    }
    for (int i = chunkSize; i < chunkSize * 2; ++i) {
      const int j = b * (chunkSize * 2) + i;
      const int k = b * chunkSize + (i - chunkSize);
      EXPECT_NEAR(samplesHost[k], outHost[j], 1e-6f) << "i = " << i;
    }
  }
}
