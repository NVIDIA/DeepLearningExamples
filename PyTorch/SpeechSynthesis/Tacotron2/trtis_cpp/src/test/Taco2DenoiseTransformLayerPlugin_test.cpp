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
#include "taco2DenoiseTransformLayerPlugin.h"
#include "trtUtils.h"

#include "NvInfer.h"

#include <random>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace tts;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename RNG>
std::vector<float> genVec(const size_t size, RNG& rng)
{
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  std::vector<float> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = dist(rng);
  }

  return vec;
}

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST(CPUCompareTestBatch1)
{
  std::mt19937 rng(0);

  const int filterLength = 1026;
  const int inputLength = 935;
  const int inputWidth = filterLength / 2;

  std::vector<float> weightsHost = genVec(inputWidth, rng);
  Taco2DenoiseTransformLayerPlugin layer(
      TRTUtils::toWeights(weightsHost), filterLength, inputLength);

  std::vector<float> inputHost = genVec(filterLength * inputLength, rng);
  CudaMemory<float> inputDevice(inputHost);

  std::vector<Dims> inputDims{Dims4(1, 1, filterLength, inputLength)};

  const std::vector<Dims> outputDims{Dims4(1, 1, filterLength, inputLength)};

  const std::vector<DataType> dataTypes{DataType::kFLOAT};
  const bool broadcast[] = {false};

  layer.configurePlugin(
      inputDims.data(),
      static_cast<int>(inputDims.size()),
      outputDims.data(),
      static_cast<int>(outputDims.size()),
      dataTypes.data(),
      dataTypes.data(),
      broadcast,
      broadcast,
#if NV_TENSORRT_MAJOR < 6
      PluginFormat::kNCHW,
#else
      PluginFormat::kLINEAR,
#endif
      1);

  layer.initialize();

  std::vector<const float*> inputs{inputDevice.data()};

  CudaMemory<float> outputDevice(filterLength * inputLength);
  std::vector<float*> outputs{outputDevice.data()};

  layer.enqueue(
      1,
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      nullptr,
      0);
  CudaUtils::sync(0);

  // perform operations on cpu
  std::vector<float> expOutput(filterLength * inputLength);

  for (int y = 0; y < inputWidth; ++y) {
    for (int x = 0; x < inputLength; ++x) {
      const int idx = y * inputLength + x;
      const int idx2 = (y + inputWidth) * inputLength + x;
      const float real = inputHost[idx];
      const float img = inputHost[idx2];
      float mag = std::sqrt(real * real + img * img);
      const float phase = std::atan2(img, real);
      mag = std::max(0.0f, mag - weightsHost[y]);
      expOutput[idx] = mag * std::cos(phase);
      expOutput[idx2] = mag * std::sin(phase);
    }
  }

  // match outputs
  const std::vector<float> actOutput = outputDevice.toHost();

  ASSERT_EQ(expOutput.size(), actOutput.size());
  for (size_t i = 0; i < expOutput.size(); ++i) {
    EXPECT_NEAR(expOutput[i], actOutput[i], 1e-6) << "i = " << i;
  }
}

TEST(CPUCompareTestBatch4)
{
  std::mt19937 rng(0);

  const int filterLength = 1026;
  const int inputLength = 935;
  const int inputWidth = filterLength / 2;
  const int batchSize = 9;

  std::vector<float> weightsHost = genVec(inputWidth, rng);
  Taco2DenoiseTransformLayerPlugin layer(
      TRTUtils::toWeights(weightsHost), filterLength, inputLength);

  // duplicate vector
  std::vector<float> inputHostBase = genVec(filterLength * inputLength, rng);
  std::vector<float> inputHost;
  for (int i = 0; i < batchSize; ++i) {
    inputHost.insert(
        inputHost.end(), inputHostBase.begin(), inputHostBase.end());
  }

  CudaMemory<float> inputDevice(inputHost);

  std::vector<Dims> inputDims{Dims4(1, 1, filterLength, inputLength)};

  const std::vector<Dims> outputDims{Dims4(1, 1, filterLength, inputLength)};

  const std::vector<DataType> dataTypes{DataType::kFLOAT};
  const bool broadcast[] = {false};

  layer.configurePlugin(
      inputDims.data(),
      static_cast<int>(inputDims.size()),
      outputDims.data(),
      static_cast<int>(outputDims.size()),
      dataTypes.data(),
      dataTypes.data(),
      broadcast,
      broadcast,
      PluginFormat::kLINEAR,
      batchSize);

  layer.initialize();

  std::vector<const float*> inputs{inputDevice.data()};

  CudaMemory<float> outputDevice(filterLength * inputLength * batchSize);
  std::vector<float*> outputs{outputDevice.data()};

  layer.enqueue(
      batchSize,
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      nullptr,
      0);
  CudaUtils::sync(0);

  // perform operations on cpu
  std::vector<float> expOutput(filterLength * inputLength);

  for (int y = 0; y < inputWidth; ++y) {
    for (int x = 0; x < inputLength; ++x) {
      const int idx = y * inputLength + x;
      const int idx2 = (y + inputWidth) * inputLength + x;
      const float real = inputHost[idx];
      const float img = inputHost[idx2];
      float mag = std::sqrt(real * real + img * img);
      const float phase = std::atan2(img, real);
      mag = std::max(0.0f, mag - weightsHost[y]);
      expOutput[idx] = mag * std::cos(phase);
      expOutput[idx2] = mag * std::sin(phase);
    }
  }

  // match outputs
  const std::vector<float> actOutput = outputDevice.toHost();

  ASSERT_EQ(expOutput.size() * batchSize, actOutput.size());
  for (int b = 0; b < batchSize; ++b) {
    for (size_t i = 0; i < expOutput.size(); ++i) {
      EXPECT_NEAR(expOutput[i], actOutput[i + (b * expOutput.size())], 1e-6)
          << "i = " << i << " b " << b;
    }
  }
}
