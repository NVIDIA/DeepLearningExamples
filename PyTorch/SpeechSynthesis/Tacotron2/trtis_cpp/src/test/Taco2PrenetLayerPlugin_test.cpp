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
#include "binding.h"
#include "cudaMemory.h"
#include "cudaUtils.h"
#include "logging.h"
#include "taco2PrenetLayerPlugin.h"
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
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

TEST(CPUCompareTest)
{
  std::mt19937 rng(0);

  const int inputLength = 80;
  const int numDimensions = 256;

  // weights
  std::vector<float> weight1 = genVec(inputLength * numDimensions, rng);
  std::vector<float> weight2 = genVec(numDimensions * numDimensions, rng);

  Taco2PrenetLayerPlugin layer(
      TRTUtils::toWeights(weight1),
      TRTUtils::toWeights(weight2),
      inputLength,
      numDimensions);

  const std::vector<float> inputHost = genVec(numDimensions, rng);
  const std::vector<float> dropoutHost(numDimensions, 1.0f);

  CudaMemory<float> inputDevice(inputHost);
  CudaMemory<float> dropoutDevice(dropoutHost);

  std::vector<Dims> inputDims{Dims3(1, 1, inputLength),
                              Dims2(1, numDimensions)};
  const std::vector<Dims> outputDims{Dims3(1, 1, numDimensions)};
  const std::vector<DataType> dataTypes(2, DataType::kFLOAT);

  const std::vector<DynamicPluginTensorDesc> inDynDesc{
      {{Dims3(-1, 1, inputLength),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims3(1, 1, inputLength),
       Dims3(1, 1, inputLength)},
      {{Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)}};
  const std::vector<DynamicPluginTensorDesc> outDynDesc{
      {{Dims3(-1, 1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims3(1, 1, numDimensions),
       Dims3(1, 1, numDimensions)}};

  layer.configurePlugin(
      inDynDesc.data(), inDynDesc.size(), outDynDesc.data(), outDynDesc.size());

  layer.initialize();

  std::vector<const float*> inputs{inputDevice.data(), dropoutDevice.data()};

  CudaMemory<float> outputDevice(numDimensions);
  std::vector<float*> outputs{outputDevice.data()};

  const std::vector<PluginTensorDesc> inDesc{
      {Dims3(1, 1, inputLength), DataType::kFLOAT, TensorFormat::kLINEAR, 1.0f},
      {Dims2(1, numDimensions), DataType::kFLOAT, TensorFormat::kLINEAR, 1.0f},
  };
  const std::vector<PluginTensorDesc> outDesc{{Dims3(1, 1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f}};

  CudaMemory<uint8_t> workspace(layer.getWorkspaceSize(
      inDesc.data(),
      static_cast<int>(inDesc.size()),
      outDesc.data(),
      static_cast<int>(outDesc.size())));

  layer.enqueue(
      inDesc.data(),
      outDesc.data(),
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      workspace.data(),
      0);
  CudaUtils::sync(0);

  // perform operations on cpu
  std::vector<float> expOutput(numDimensions);

  std::vector<float> intermediate(numDimensions);
  for (int i = 0; i < numDimensions; ++i) {
    float v = 0.0f;
    for (int j = 0; j < inputLength; ++j) {
      v += inputHost[j] * weight1[i * inputLength + j];
    }
    intermediate[i] = v;
  }
  for (int i = 0; i < numDimensions; ++i) {
    intermediate[i] = std::max(0.0f, intermediate[i]) * dropoutHost[i];
  }

  for (int i = 0; i < numDimensions; ++i) {
    float v = 0.0f;
    for (int j = 0; j < numDimensions; ++j) {
      v += intermediate[j] * weight2[i * numDimensions + j];
    }
    expOutput[i] = v;
  }
  for (int i = 0; i < numDimensions; ++i) {
    expOutput[i] = std::max(0.0f, expOutput[i]) * dropoutHost[i];
  }

  // match outputs
  const std::vector<float> actOutput = outputDevice.toHost();

  ASSERT_EQ(expOutput.size(), actOutput.size());
  for (size_t i = 0; i < expOutput.size(); ++i) {
    EXPECT_NEAR(expOutput[i], actOutput[i], 1e-4) << "i = " << i;
  }
}
