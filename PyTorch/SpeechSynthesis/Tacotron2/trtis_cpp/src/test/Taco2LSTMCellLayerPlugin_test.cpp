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
#include "taco2LSTMCellLayerPlugin.h"
#include "trtUtils.h"
#include "utils.h"

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
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
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

TEST(CPUCompareFP32I256Test)
{
  std::mt19937 rng(0);

  const int inputLengthFirst = 256;
  const int inputLengthSecond = 512;
  const int inputLength = inputLengthFirst + inputLengthSecond;
  const int numDimensions = 1024;

  // weights
  std::vector<float> inputWeight = genVec(inputLength * numDimensions * 4, rng);
  const std::vector<float> inputBias = genVec(numDimensions * 4, rng);
  std::vector<float> hiddenWeight
      = genVec(numDimensions * numDimensions * 4, rng);
  const std::vector<float> hiddenBias = genVec(numDimensions * 4, rng);

  Taco2LSTMCellLayerPlugin layer(
      TRTUtils::toWeights(inputWeight),
      TRTUtils::toWeights(hiddenWeight),
      TRTUtils::toWeights(inputBias),
      TRTUtils::toWeights(hiddenBias),
      inputLength,
      numDimensions,
      false);

  const std::vector<float> inputFirst = genVec(inputLengthFirst, rng);
  const std::vector<float> inputSecond = genVec(inputLengthSecond, rng);
  const std::vector<float> hiddenState = genVec(numDimensions, rng);
  const std::vector<float> cellState = genVec(numDimensions, rng);

  CudaMemory<float> inputFirstDevice(inputFirst);
  CudaMemory<float> inputSecondDevice(inputSecond);
  CudaMemory<float> hiddenStateDevice(hiddenState);
  CudaMemory<float> cellStateDevice(cellState);

  const std::vector<Dims> inputDims{Dims2(1, inputLengthFirst),
                                    Dims4(1, inputLengthSecond, 1, 1),
                                    Dims2(1, numDimensions),
                                    Dims2(1, numDimensions)};
  const std::vector<Dims> outputDims{Dims2(1, numDimensions),
                                     Dims2(1, numDimensions)};
  const std::vector<DataType> dataTypes(4, DataType::kFLOAT);

  const std::vector<DynamicPluginTensorDesc> inDesc{
      {// INPUT_FIRST_INDEX
       {Dims2(-1, inputLengthFirst),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthFirst),
       Dims2(1, inputLengthFirst)},
      {// INPUT_SECOND_INDEX
       {Dims4(-1, inputLengthSecond, 1, 1),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthSecond),
       Dims2(1, inputLengthSecond)},
      {// HIDDEN_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)},
      {// CELL_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)}};

  const std::vector<DynamicPluginTensorDesc> outDesc{{// HIDDEN
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)},
                                                     {// CELL
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)}};

  layer.configurePlugin(
      inDesc.data(), inDesc.size(), outDesc.data(), outDesc.size());

  layer.initialize();

  const std::vector<const float*> inputs{inputFirstDevice.data(),
                                         inputSecondDevice.data(),
                                         hiddenStateDevice.data(),
                                         cellStateDevice.data()};

  CudaMemory<float> hiddenStateOutDevice(hiddenState.size());
  CudaMemory<float> cellStateOutDevice(hiddenState.size());
  std::vector<float*> outputs{hiddenStateOutDevice.data(),
                              cellStateOutDevice.data()};

  const std::vector<PluginTensorDesc> inConf{{// INPUT_FIRST_INDEX
                                              Dims2(1, inputLengthFirst),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// INPUT_SECOND_INDEX
                                              Dims4(1, inputLengthSecond, 1, 1),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// HIDDEN_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// CELL_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f}};

  const std::vector<PluginTensorDesc> outConf{{// HIDDEN
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f},
                                              {// CELL
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f}};

  CudaMemory<uint8_t> workspace(layer.getWorkspaceSize(
      inConf.data(),
      static_cast<int>(inConf.size()),
      outConf.data(),
      static_cast<int>(outConf.size())));

  layer.enqueue(
      inConf.data(),
      outConf.data(),
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      workspace.data(),
      0);
  CudaUtils::sync(0);

  // perform operations on cpu

  std::vector<float> prod1(4 * numDimensions, 0);
  std::vector<float> prod2(4 * numDimensions, 0);
  std::vector<float> prod3(4 * numDimensions, 0);
  std::vector<float> prod(4 * numDimensions, 0);

  // perform input MV
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthFirst); ++j) {
      val += inputWeight[i * inputLength + j] * inputFirst[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthSecond); ++j) {
      val += inputWeight[i * inputLength + j + inputLengthFirst]
             * inputSecond[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < hiddenBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < hiddenState.size(); ++j) {
      val += hiddenWeight[i * hiddenState.size() + j] * hiddenState[j];
    }
    prod[i] += val;
  }

  // add biases
  for (size_t i = 0; i < inputBias.size(); ++i) {
    prod[i] += inputBias[i] + hiddenBias[i];
  }

  std::vector<float> expHiddenOut(hiddenState);
  std::vector<float> expCellOut(cellState);

  // perform reduction
  for (int row = 0; row < numDimensions; ++row) {
    const float c = cellState[row];
    const float i = Utils::sigmoid(prod[row]);
    const float f = Utils::sigmoid(prod[row + numDimensions]);
    const float g = tanh(prod[row + numDimensions * 2]);
    const float o = Utils::sigmoid(prod[row + numDimensions * 3]);

    const float cPrime = f * c + i * g;
    const float hPrime = o * tanh(cPrime);

    expHiddenOut[row] = hPrime;
    expCellOut[row] = cPrime;
  }

  // copy back to host
  const std::vector<float> actHiddenOut = hiddenStateOutDevice.toHost();
  const std::vector<float> actCellOut = cellStateOutDevice.toHost();

  ASSERT_EQ(expHiddenOut.size(), actHiddenOut.size());
  for (size_t i = 0; i < expHiddenOut.size(); ++i) {
    EXPECT_NEAR(expHiddenOut[i], actHiddenOut[i], 7.5e-4) << "i = " << i;
  }

  ASSERT_EQ(expCellOut.size(), actCellOut.size());
  for (size_t i = 0; i < expCellOut.size(); ++i) {
    EXPECT_NEAR(expCellOut[i], actCellOut[i], 5e-3) << "i = " << i;
  }
}

TEST(CPUCompareFP32I1024Test)
{
  std::mt19937 rng(0);

  const int inputLengthFirst = 1024;
  const int inputLengthSecond = 512;
  const int inputLength = inputLengthFirst + inputLengthSecond;
  const int numDimensions = 1024;

  // weights
  std::vector<float> inputWeight = genVec(inputLength * numDimensions * 4, rng);
  const std::vector<float> inputBias = genVec(numDimensions * 4, rng);
  std::vector<float> hiddenWeight
      = genVec(numDimensions * numDimensions * 4, rng);
  const std::vector<float> hiddenBias = genVec(numDimensions * 4, rng);

  Taco2LSTMCellLayerPlugin layer(
      TRTUtils::toWeights(inputWeight),
      TRTUtils::toWeights(hiddenWeight),
      TRTUtils::toWeights(inputBias),
      TRTUtils::toWeights(hiddenBias),
      inputLength,
      numDimensions,
      false);

  const std::vector<float> inputFirst = genVec(inputLengthFirst, rng);
  const std::vector<float> inputSecond = genVec(inputLengthSecond, rng);
  const std::vector<float> hiddenState = genVec(numDimensions, rng);
  const std::vector<float> cellState = genVec(numDimensions, rng);

  CudaMemory<float> inputFirstDevice(inputFirst);
  CudaMemory<float> inputSecondDevice(inputSecond);
  CudaMemory<float> hiddenStateDevice(hiddenState);
  CudaMemory<float> cellStateDevice(cellState);

  const std::vector<Dims> inputDims{Dims2(1, inputLengthFirst),
                                    Dims4(1, inputLengthSecond, 1, 1),
                                    Dims2(1, numDimensions),
                                    Dims2(1, numDimensions)};
  const std::vector<Dims> outputDims{Dims2(1, numDimensions),
                                     Dims2(1, numDimensions)};
  const std::vector<DataType> dataTypes(4, DataType::kFLOAT);

  const std::vector<DynamicPluginTensorDesc> inDesc{
      {// INPUT_FIRST_INDEX
       {Dims2(-1, inputLengthFirst),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthFirst),
       Dims2(1, inputLengthFirst)},
      {// INPUT_SECOND_INDEX
       {Dims4(-1, inputLengthSecond, 1, 1),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthSecond),
       Dims2(1, inputLengthSecond)},
      {// HIDDEN_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)},
      {// CELL_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)}};

  const std::vector<DynamicPluginTensorDesc> outDesc{{// HIDDEN
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)},
                                                     {// CELL
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)}};

  layer.configurePlugin(
      inDesc.data(), inDesc.size(), outDesc.data(), outDesc.size());

  layer.initialize();

  const std::vector<const float*> inputs{inputFirstDevice.data(),
                                         inputSecondDevice.data(),
                                         hiddenStateDevice.data(),
                                         cellStateDevice.data()};

  CudaMemory<float> hiddenStateOutDevice(hiddenState.size());
  CudaMemory<float> cellStateOutDevice(hiddenState.size());
  std::vector<float*> outputs{hiddenStateOutDevice.data(),
                              cellStateOutDevice.data()};

  const std::vector<PluginTensorDesc> inConf{{// INPUT_FIRST_INDEX
                                              Dims2(1, inputLengthFirst),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// INPUT_SECOND_INDEX
                                              Dims4(1, inputLengthSecond, 1, 1),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// HIDDEN_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// CELL_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f}};

  const std::vector<PluginTensorDesc> outConf{{// HIDDEN
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f},
                                              {// CELL
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f}};

  CudaMemory<uint8_t> workspace(layer.getWorkspaceSize(
      inConf.data(),
      static_cast<int>(inConf.size()),
      outConf.data(),
      static_cast<int>(outConf.size())));

  layer.enqueue(
      inConf.data(),
      outConf.data(),
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      workspace.data(),
      0);
  CudaUtils::sync(0);

  // perform operations on cpu

  std::vector<float> prod1(4 * numDimensions, 0);
  std::vector<float> prod2(4 * numDimensions, 0);
  std::vector<float> prod3(4 * numDimensions, 0);
  std::vector<float> prod(4 * numDimensions, 0);

  // perform input MV
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthFirst); ++j) {
      val += inputWeight[i * inputLength + j] * inputFirst[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthSecond); ++j) {
      val += inputWeight[i * inputLength + j + inputLengthFirst]
             * inputSecond[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < hiddenBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < hiddenState.size(); ++j) {
      val += hiddenWeight[i * hiddenState.size() + j] * hiddenState[j];
    }
    prod[i] += val;
  }

  // add biases
  for (size_t i = 0; i < inputBias.size(); ++i) {
    prod[i] += inputBias[i] + hiddenBias[i];
  }

  std::vector<float> expHiddenOut(hiddenState);
  std::vector<float> expCellOut(cellState);

  // perform reduction
  for (int row = 0; row < numDimensions; ++row) {
    const float c = cellState[row];
    const float i = Utils::sigmoid(prod[row]);
    const float f = Utils::sigmoid(prod[row + numDimensions]);
    const float g = tanh(prod[row + numDimensions * 2]);
    const float o = Utils::sigmoid(prod[row + numDimensions * 3]);

    const float cPrime = f * c + i * g;
    const float hPrime = o * tanh(cPrime);

    expHiddenOut[row] = hPrime;
    expCellOut[row] = cPrime;
  }

  // copy back to host
  const std::vector<float> actHiddenOut = hiddenStateOutDevice.toHost();
  const std::vector<float> actCellOut = cellStateOutDevice.toHost();

  ASSERT_EQ(expHiddenOut.size(), actHiddenOut.size());
  for (size_t i = 0; i < expHiddenOut.size(); ++i) {
    EXPECT_NEAR(expHiddenOut[i], actHiddenOut[i], 7.5e-4) << "i = " << i;
  }

  ASSERT_EQ(expCellOut.size(), actCellOut.size());
  for (size_t i = 0; i < expCellOut.size(); ++i) {
    EXPECT_NEAR(expCellOut[i], actCellOut[i], 5e-3) << "i = " << i;
  }
}

TEST(CPUCompareFP16I256Test)
{
  std::mt19937 rng(0);

  const int inputLengthFirst = 256;
  const int inputLengthSecond = 512;
  const int inputLength = inputLengthFirst + inputLengthSecond;
  const int numDimensions = 1024;

  // weights
  std::vector<float> inputWeight = genVec(inputLength * numDimensions * 4, rng);
  const std::vector<float> inputBias = genVec(numDimensions * 4, rng);
  std::vector<float> hiddenWeight
      = genVec(numDimensions * numDimensions * 4, rng);
  const std::vector<float> hiddenBias = genVec(numDimensions * 4, rng);

  Taco2LSTMCellLayerPlugin layer(
      TRTUtils::toWeights(inputWeight),
      TRTUtils::toWeights(hiddenWeight),
      TRTUtils::toWeights(inputBias),
      TRTUtils::toWeights(hiddenBias),
      inputLength,
      numDimensions,
      true);

  const std::vector<float> inputFirst = genVec(inputLengthFirst, rng);
  const std::vector<float> inputSecond = genVec(inputLengthSecond, rng);
  const std::vector<float> hiddenState = genVec(numDimensions, rng);
  const std::vector<float> cellState = genVec(numDimensions, rng);

  CudaMemory<float> inputFirstDevice(inputFirst);
  CudaMemory<float> inputSecondDevice(inputSecond);
  CudaMemory<float> hiddenStateDevice(hiddenState);
  CudaMemory<float> cellStateDevice(cellState);

  const std::vector<Dims> inputDims{Dims2(1, inputLengthFirst),
                                    Dims4(1, inputLengthSecond, 1, 1),
                                    Dims2(1, numDimensions),
                                    Dims2(1, numDimensions)};
  const std::vector<Dims> outputDims{Dims2(1, numDimensions),
                                     Dims2(1, numDimensions)};
  const std::vector<DataType> dataTypes(4, DataType::kFLOAT);

  const std::vector<DynamicPluginTensorDesc> inDesc{
      {// INPUT_FIRST_INDEX
       {Dims2(-1, inputLengthFirst),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthFirst),
       Dims2(1, inputLengthFirst)},
      {// INPUT_SECOND_INDEX
       {Dims4(-1, inputLengthSecond, 1, 1),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthSecond),
       Dims2(1, inputLengthSecond)},
      {// HIDDEN_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)},
      {// CELL_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)}};

  const std::vector<DynamicPluginTensorDesc> outDesc{{// HIDDEN
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)},
                                                     {// CELL
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)}};

  layer.configurePlugin(
      inDesc.data(), inDesc.size(), outDesc.data(), outDesc.size());

  layer.initialize();

  const std::vector<const float*> inputs{inputFirstDevice.data(),
                                         inputSecondDevice.data(),
                                         hiddenStateDevice.data(),
                                         cellStateDevice.data()};

  CudaMemory<float> hiddenStateOutDevice(hiddenState.size());
  CudaMemory<float> cellStateOutDevice(hiddenState.size());
  std::vector<float*> outputs{hiddenStateOutDevice.data(),
                              cellStateOutDevice.data()};

  const std::vector<PluginTensorDesc> inConf{{// INPUT_FIRST_INDEX
                                              Dims2(1, inputLengthFirst),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// INPUT_SECOND_INDEX
                                              Dims4(1, inputLengthSecond, 1, 1),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// HIDDEN_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// CELL_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f}};

  const std::vector<PluginTensorDesc> outConf{{// HIDDEN
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f},
                                              {// CELL
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f}};

  CudaMemory<uint8_t> workspace(layer.getWorkspaceSize(
      inConf.data(),
      static_cast<int>(inConf.size()),
      outConf.data(),
      static_cast<int>(outConf.size())));

  layer.enqueue(
      inConf.data(),
      outConf.data(),
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      workspace.data(),
      0);
  CudaUtils::sync(0);

  // perform operations on cpu

  std::vector<float> prod1(4 * numDimensions, 0);
  std::vector<float> prod2(4 * numDimensions, 0);
  std::vector<float> prod3(4 * numDimensions, 0);
  std::vector<float> prod(4 * numDimensions, 0);

  // perform input MV
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthFirst); ++j) {
      val += inputWeight[i * inputLength + j] * inputFirst[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthSecond); ++j) {
      val += inputWeight[i * inputLength + j + inputLengthFirst]
             * inputSecond[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < hiddenBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < hiddenState.size(); ++j) {
      val += hiddenWeight[i * hiddenState.size() + j] * hiddenState[j];
    }
    prod[i] += val;
  }

  // add biases
  for (size_t i = 0; i < inputBias.size(); ++i) {
    prod[i] += inputBias[i] + hiddenBias[i];
  }

  std::vector<float> expHiddenOut(hiddenState);
  std::vector<float> expCellOut(cellState);

  // perform reduction
  for (int row = 0; row < numDimensions; ++row) {
    const float c = cellState[row];
    const float i = Utils::sigmoid(prod[row]);
    const float f = Utils::sigmoid(prod[row + numDimensions]);
    const float g = tanh(prod[row + numDimensions * 2]);
    const float o = Utils::sigmoid(prod[row + numDimensions * 3]);

    const float cPrime = f * c + i * g;
    const float hPrime = o * tanh(cPrime);

    expHiddenOut[row] = hPrime;
    expCellOut[row] = cPrime;
  }

  // copy back to host
  const std::vector<float> actHiddenOut = hiddenStateOutDevice.toHost();
  const std::vector<float> actCellOut = cellStateOutDevice.toHost();

  ASSERT_EQ(expHiddenOut.size(), actHiddenOut.size());
  for (size_t i = 0; i < expHiddenOut.size(); ++i) {
    EXPECT_NEAR(expHiddenOut[i], actHiddenOut[i], 4.5e-1) << "i = " << i;
  }

  ASSERT_EQ(expCellOut.size(), actCellOut.size());
  for (size_t i = 0; i < expCellOut.size(); ++i) {
    EXPECT_NEAR(expCellOut[i], actCellOut[i], 4.5e-1) << "i = " << i;
  }
}

TEST(CPUCompareFP16I1024Test)
{
  std::mt19937 rng(0);

  const int inputLengthFirst = 1024;
  const int inputLengthSecond = 512;
  const int inputLength = inputLengthFirst + inputLengthSecond;
  const int numDimensions = 1024;

  // weights
  std::vector<float> inputWeight = genVec(inputLength * numDimensions * 4, rng);
  const std::vector<float> inputBias = genVec(numDimensions * 4, rng);
  std::vector<float> hiddenWeight
      = genVec(numDimensions * numDimensions * 4, rng);
  const std::vector<float> hiddenBias = genVec(numDimensions * 4, rng);

  Taco2LSTMCellLayerPlugin layer(
      TRTUtils::toWeights(inputWeight),
      TRTUtils::toWeights(hiddenWeight),
      TRTUtils::toWeights(inputBias),
      TRTUtils::toWeights(hiddenBias),
      inputLength,
      numDimensions,
      true);

  const std::vector<float> inputFirst = genVec(inputLengthFirst, rng);
  const std::vector<float> inputSecond = genVec(inputLengthSecond, rng);
  const std::vector<float> hiddenState = genVec(numDimensions, rng);
  const std::vector<float> cellState = genVec(numDimensions, rng);

  CudaMemory<float> inputFirstDevice(inputFirst);
  CudaMemory<float> inputSecondDevice(inputSecond);
  CudaMemory<float> hiddenStateDevice(hiddenState);
  CudaMemory<float> cellStateDevice(cellState);

  const std::vector<Dims> inputDims{Dims2(1, inputLengthFirst),
                                    Dims4(1, inputLengthSecond, 1, 1),
                                    Dims2(1, numDimensions),
                                    Dims2(1, numDimensions)};
  const std::vector<Dims> outputDims{Dims2(1, numDimensions),
                                     Dims2(1, numDimensions)};
  const std::vector<DataType> dataTypes(4, DataType::kFLOAT);

  const std::vector<DynamicPluginTensorDesc> inDesc{
      {// INPUT_FIRST_INDEX
       {Dims2(-1, inputLengthFirst),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthFirst),
       Dims2(1, inputLengthFirst)},
      {// INPUT_SECOND_INDEX
       {Dims4(-1, inputLengthSecond, 1, 1),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, inputLengthSecond),
       Dims2(1, inputLengthSecond)},
      {// HIDDEN_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)},
      {// CELL_INDEX
       {Dims2(-1, numDimensions),
        DataType::kFLOAT,
        TensorFormat::kLINEAR,
        1.0f},
       Dims2(1, numDimensions),
       Dims2(1, numDimensions)}};

  const std::vector<DynamicPluginTensorDesc> outDesc{{// HIDDEN
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)},
                                                     {// CELL
                                                      {Dims2(-1, numDimensions),
                                                       DataType::kFLOAT,
                                                       TensorFormat::kLINEAR,
                                                       1.0f},
                                                      Dims2(1, numDimensions),
                                                      Dims2(1, numDimensions)}};

  layer.configurePlugin(
      inDesc.data(), inDesc.size(), outDesc.data(), outDesc.size());

  layer.initialize();

  const std::vector<const float*> inputs{inputFirstDevice.data(),
                                         inputSecondDevice.data(),
                                         hiddenStateDevice.data(),
                                         cellStateDevice.data()};

  CudaMemory<float> hiddenStateOutDevice(hiddenState.size());
  CudaMemory<float> cellStateOutDevice(hiddenState.size());
  std::vector<float*> outputs{hiddenStateOutDevice.data(),
                              cellStateOutDevice.data()};

  const std::vector<PluginTensorDesc> inConf{{// INPUT_FIRST_INDEX
                                              Dims2(1, inputLengthFirst),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// INPUT_SECOND_INDEX
                                              Dims4(1, inputLengthSecond, 1, 1),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// HIDDEN_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f},
                                             {// CELL_INDEX
                                              Dims2(1, numDimensions),
                                              DataType::kFLOAT,
                                              TensorFormat::kLINEAR,
                                              1.0f}};

  const std::vector<PluginTensorDesc> outConf{{// HIDDEN
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f},
                                              {// CELL
                                               Dims2(1, numDimensions),
                                               DataType::kFLOAT,
                                               TensorFormat::kLINEAR,
                                               1.0f}};

  CudaMemory<uint8_t> workspace(layer.getWorkspaceSize(
      inConf.data(),
      static_cast<int>(inConf.size()),
      outConf.data(),
      static_cast<int>(outConf.size())));

  layer.enqueue(
      inConf.data(),
      outConf.data(),
      reinterpret_cast<const void* const*>(inputs.data()),
      reinterpret_cast<void**>(outputs.data()),
      workspace.data(),
      0);
  CudaUtils::sync(0);

  // perform operations on cpu

  std::vector<float> prod1(4 * numDimensions, 0);
  std::vector<float> prod2(4 * numDimensions, 0);
  std::vector<float> prod3(4 * numDimensions, 0);
  std::vector<float> prod(4 * numDimensions, 0);

  // perform input MV
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthFirst); ++j) {
      val += inputWeight[i * inputLength + j] * inputFirst[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < inputBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < static_cast<size_t>(inputLengthSecond); ++j) {
      val += inputWeight[i * inputLength + j + inputLengthFirst]
             * inputSecond[j];
    }
    prod[i] += val;
  }
  for (size_t i = 0; i < hiddenBias.size(); ++i) {
    double val = 0;
    for (size_t j = 0; j < hiddenState.size(); ++j) {
      val += hiddenWeight[i * hiddenState.size() + j] * hiddenState[j];
    }
    prod[i] += val;
  }

  // add biases
  for (size_t i = 0; i < inputBias.size(); ++i) {
    prod[i] += inputBias[i] + hiddenBias[i];
  }

  std::vector<float> expHiddenOut(hiddenState);
  std::vector<float> expCellOut(cellState);

  // perform reduction
  for (int row = 0; row < numDimensions; ++row) {
    const float c = cellState[row];
    const float i = Utils::sigmoid(prod[row]);
    const float f = Utils::sigmoid(prod[row + numDimensions]);
    const float g = tanh(prod[row + numDimensions * 2]);
    const float o = Utils::sigmoid(prod[row + numDimensions * 3]);

    const float cPrime = f * c + i * g;
    const float hPrime = o * tanh(cPrime);

    expHiddenOut[row] = hPrime;
    expCellOut[row] = cPrime;
  }

  // copy back to host
  const std::vector<float> actHiddenOut = hiddenStateOutDevice.toHost();
  const std::vector<float> actCellOut = cellStateOutDevice.toHost();

  ASSERT_EQ(expHiddenOut.size(), actHiddenOut.size());
  for (size_t i = 0; i < expHiddenOut.size(); ++i) {
    EXPECT_NEAR(expHiddenOut[i], actHiddenOut[i], 4.5e-1) << "i = " << i;
  }

  ASSERT_EQ(expCellOut.size(), actCellOut.size());
  for (size_t i = 0; i < expCellOut.size(); ++i) {
    EXPECT_NEAR(expCellOut[i], actCellOut[i], 4.5e-1) << "i = " << i;
  }
}
