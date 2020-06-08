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

#include "encoderBuilder.h"
#include "convBatchNormCreator.h"
#include "cudaUtils.h"
#include "dims1.h"
#include "encoderInstance.h"
#include "lstm.h"
#include "utils.h"

#include <cassert>

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const char* const INPUT_NAME = EncoderInstance::INPUT_NAME;
constexpr const char* const INPUT_MASK_NAME = EncoderInstance::INPUT_MASK_NAME;
constexpr const char* const INPUT_LENGTH_NAME = EncoderInstance::INPUT_LENGTH_NAME;
constexpr const char* const OUTPUT_NAME = EncoderInstance::OUTPUT_NAME;
constexpr const char* const OUTPUT_PROCESSED_NAME = EncoderInstance::OUTPUT_PROCESSED_NAME;
constexpr const char* const ENGINE_NAME = EncoderInstance::ENGINE_NAME;
} // namespace

using namespace nvinfer1;

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

EncoderBuilder::EncoderBuilder(const int numEmbeddingDimensions, const int numEncodingDimensions,
    const int numAttentionDimensions, const int inputLength)
    : mNumEmbeddingDimensions(numEmbeddingDimensions)
    , mNumEncodingDimensions(numEncodingDimensions)
    , mNumAttentionDimensions(numAttentionDimensions)
    , mInputLength(inputLength)
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

TRTPtr<ICudaEngine> EncoderBuilder::build(
    IBuilder& builder,
    IModelImporter& importer,
    const int maxBatchSize,
    const bool useFP16)
{
  TRTPtr<INetworkDefinition> network(builder.createNetworkV2(0));
  network->setName("Tacotron2_Encoder");

  // EMBEDDING ////////////////////////////////////////////////////////////////

  ITensor* input
      = network->addInput(INPUT_NAME, DataType::kINT32, Dims2(1, mInputLength));

  const LayerData* embeddingData = importer.getWeights({"embedding"});

  const int numSymbols
      = embeddingData->get("weight").count / mNumEmbeddingDimensions;
  assert(
      numSymbols * mNumEmbeddingDimensions
      == embeddingData->get("weight").count);

  ILayer* const lookupLayer = network->addConstant(
      Dims3(1, numSymbols, mNumEmbeddingDimensions),
      embeddingData->get("weight"));
  lookupLayer->setName("embedding.constant");
  ILayer* const gatherLayer
      = network->addGather(*lookupLayer->getOutput(0), *input, 1);
  gatherLayer->setName("embedding.gather");
  IShuffleLayer* const embTransLayer
      = network->addShuffle(*gatherLayer->getOutput(0));
  embTransLayer->setFirstTranspose({0, 1, 3, 2});
  embTransLayer->setReshapeDimensions(Dims3(mNumEmbeddingDimensions, -1, 1));
  embTransLayer->setName("embedding.transpose");

  input = embTransLayer->getOutput(0);

  // ENCODING /////////////////////////////////////////////////////////////////

  ITensor* inputMask = network->addInput(
      INPUT_MASK_NAME, DataType::kFLOAT, Dims3(1, mInputLength, 1));
  ITensor* inputLength
      = network->addInput(INPUT_LENGTH_NAME, DataType::kINT32, Dims1(1));

  ILayer* const inputMaskLayer = network->addElementWise(
      *input, *inputMask, ElementWiseOperation::kPROD);
  input = inputMaskLayer->getOutput(0);

  // we need to ensure layer data is around during network construction
  ConvBatchNormCreator convBatchNormCreator;
  for (int layer = 0; layer < 3; ++layer) {
    const LayerData* const convData
        = importer.getWeights({"encoder",
                               "convolutions",
                               std::to_string(layer),
                               "conv_layer",
                               "conv"});
    const LayerData* const normData = importer.getWeights(
        {"encoder", "convolutions", std::to_string(layer), "batch_norm"});
    ILayer* const convLayer = convBatchNormCreator.add(
        *network,
        input,
        *convData,
        *normData,
        "relu",
        "encoder.convolutions." + std::to_string(layer));

    ILayer* const maskLayer = network->addElementWise(
        *convLayer->getOutput(0), *inputMask, ElementWiseOperation::kPROD);

    input = maskLayer->getOutput(0);
    }

    IShuffleLayer* const transposeLayer = network->addShuffle(*input);
    transposeLayer->setFirstTranspose({2, 1, 0});
    transposeLayer->setName("encoder.convolutions.transpose");

    const LayerData* const lstmData = importer.getWeights({"encoder", "lstm"});
    ILayer* const lstmLayer = LSTM::addPaddedBidirectional(
        network.get(), transposeLayer->getOutput(0), inputLength, mNumEncodingDimensions, *lstmData);
    lstmLayer->setName("encoder.lstm");

    ILayer* const outputMaskLayer
        = network->addElementWise(*lstmLayer->getOutput(0), *inputMask, ElementWiseOperation::kPROD);
    outputMaskLayer->setName("encoder.mask");

    ITensor* const output = outputMaskLayer->getOutput(0);

    output->setName(OUTPUT_NAME);
    network->markOutput(*output);

    // MEMORY ///////////////////////////////////////////////////////////////////

    IShuffleLayer* const memTransLayer = network->addShuffle(*output);
    memTransLayer->setReshapeDimensions(Dims4(-1, mNumEncodingDimensions, 1, 1));

    const LayerData* const linearData
        = importer.getWeights({"decoder", "attention_layer", "memory_layer", "linear_layer"});
    ILayer* const linearLayer = network->addFullyConnected(*memTransLayer->getOutput(0), mNumAttentionDimensions,
        linearData->get("weight"), Weights{DataType::kFLOAT, 0, 0});
    linearLayer->setName("decoder.attention_layer.memory_layer.linear_layer");

    ITensor* const outputProcessed = linearLayer->getOutput(0);
    outputProcessed->setName(OUTPUT_PROCESSED_NAME);
    network->markOutput(*outputProcessed);

    // build engine
    TRTPtr<IBuilderConfig> config(builder.createBuilderConfig());

    config->setMaxWorkspaceSize(1ULL << 29); // 512 MB
    if (useFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    builder.setMaxBatchSize(maxBatchSize);
    TRTPtr<ICudaEngine> engine(
        builder.buildEngineWithConfig(*network, *config));

    if (!engine)
    {
        throw std::runtime_error("Failed to build Tacotron2::Encoder engine.");
    }

    return engine;
}

} // namespace tts
