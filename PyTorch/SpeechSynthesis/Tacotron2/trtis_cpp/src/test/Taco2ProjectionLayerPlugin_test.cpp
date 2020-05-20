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
#include "taco2ProjectionLayerPlugin.h"
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

  const int hiddenInputLength = 1024;
  const int contextInputLength = 512;
  const int numChannelDimensions = 80;
  const int numGateDimensions = 1;

  const int inputLength = hiddenInputLength + contextInputLength;
  const int numDimensions = numChannelDimensions + numGateDimensions;

  // weights
  std::vector<float> weightChannel
      = genVec(inputLength * numChannelDimensions, rng);
  std::vector<float> weightGate = genVec(inputLength * numGateDimensions, rng);

  std::vector<float> biasChannel = genVec(numChannelDimensions, rng);
  std::vector<float> biasGate = genVec(numGateDimensions, rng);

  Taco2ProjectionLayerPlugin layer(
      TRTUtils::toWeights(weightChannel),
      TRTUtils::toWeights(weightGate),
      TRTUtils::toWeights(biasChannel),
      TRTUtils::toWeights(biasGate),
      hiddenInputLength,
      contextInputLength,
      numChannelDimensions,
      numGateDimensions);

  std::vector<float> inputHidden = genVec(hiddenInputLength, rng);
  std::vector<float> inputContext = genVec(contextInputLength, rng);

  CudaMemory<float> inputHiddenDevice(inputHidden);
  CudaMemory<float> inputContextDevice(inputContext);

  std::vector<Dims> inputDims{Dims3(1, 1, hiddenInputLength),
                              Dims3(1, 1, contextInputLength)};
  const std::vector<Dims> outputDims{Dims3(1, 1, numDimensions)};
  const std::vector<DataType> dataTypes(2, DataType::kFLOAT);

  const std::vector<DynamicPluginTensorDesc> inDynDesc{
      {{Dims3(-1, 1, hiddenInputLength),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims3(1, 1, hiddenInputLength),
       Dims3(1, 1, hiddenInputLength)},
      {{Dims3(-1, 1, contextInputLength),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims3(1, 1, contextInputLength),
       Dims3(1, 1, contextInputLength)}};
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

  std::vector<const float*> inputs{inputHiddenDevice.data(),
                                   inputContextDevice.data()};

  CudaMemory<float> outputDevice(numDimensions);
  std::vector<float*> outputs{outputDevice.data()};

  const std::vector<PluginTensorDesc> inDesc{
      {Dims3(1, 1, hiddenInputLength),
       DataType::kFLOAT,
       TensorFormat::kLINEAR,
       1.0f},
      {Dims3(1, 1, contextInputLength),
       DataType::kFLOAT,
       TensorFormat::kLINEAR,
       1.0f},
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

  for (int i = 0; i < numChannelDimensions; ++i) {
    float v = 0.0f;
    for (int j = 0; j < hiddenInputLength; ++j) {
      v += inputHidden[j] * weightChannel[i * inputLength + j];
    }
    for (int j = 0; j < contextInputLength; ++j) {
      v += inputContext[j]
           * weightChannel[i * inputLength + j + hiddenInputLength];
    }
    expOutput[i] = v + biasChannel[i];
  }
  for (int i = 0; i < numGateDimensions; ++i) {
    float v = 0.0f;
    for (int j = 0; j < hiddenInputLength; ++j) {
      v += inputHidden[j] * weightGate[i * inputLength + j];
    }
    for (int j = 0; j < contextInputLength; ++j) {
      v += inputContext[j]
           * weightGate[i * inputLength + j + hiddenInputLength];
    }
    expOutput[i + numChannelDimensions] = v + biasGate[i];
  }

  // match outputs
  const std::vector<float> actOutput = outputDevice.toHost();

  ASSERT_EQ(expOutput.size(), actOutput.size());
  for (size_t i = 0; i < expOutput.size(); ++i) {
    EXPECT_NEAR(expOutput[i], actOutput[i], 1e-4) << "i = " << i;
  }
}
