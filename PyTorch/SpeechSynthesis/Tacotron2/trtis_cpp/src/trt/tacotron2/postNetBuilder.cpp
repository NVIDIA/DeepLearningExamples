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

#include "postNetBuilder.h"
#include "convBatchNormCreator.h"
#include "postNetInstance.h"

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const int NUM_LAYERS = 5;
constexpr const char* const INPUT_NAME = PostNetInstance::INPUT_NAME;
constexpr const char* const OUTPUT_NAME = PostNetInstance::OUTPUT_NAME;
constexpr const char* const ENGINE_NAME = PostNetInstance::ENGINE_NAME;

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

PostNetBuilder::PostNetBuilder(const int numChannels, const int maxChunkSize, const int numDimensions)
    : mNumChannels(numChannels)
    , mMaxChunkSize(maxChunkSize)
    , mNumDimensions(numDimensions)
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

TRTPtr<ICudaEngine> PostNetBuilder::build(
    IBuilder& builder,
    IModelImporter& importer,
    const int maxBatchSize,
    const bool useFP16)
{
  TRTPtr<INetworkDefinition> network(builder.createNetworkV2(0));
  network->setName("Tacotron2_PostNet");

  ITensor* const input = network->addInput(
      INPUT_NAME, DataType::kFLOAT, Dims4{1, mNumChannels, mMaxChunkSize, 1});

  ITensor* convInput = input;
  ConvBatchNormCreator convBatchNormCreator;
  for (int layer = 0; layer < NUM_LAYERS; ++layer) {
    const LayerData* const convData
        = importer.getWeights({"postnet",
                               "convolutions",
                               std::to_string(layer),
                               "conv_layer",
                               "conv"});
    const LayerData* const normData = importer.getWeights(
        {"postnet", "convolutions", std::to_string(layer), "batch_norm"});

    ILayer* convLayer;
    if (layer == 0) {
      // first layer
      convLayer = convBatchNormCreator.add(
          *network,
          convInput,
          *convData,
          *normData,
          "tanh",
          "postnet.convolutions." + std::to_string(layer));
    } else if (layer == NUM_LAYERS - 1) {
      // last layer
      convLayer = convBatchNormCreator.add(
          *network,
          convInput,
          *convData,
          *normData,
          "none",
          "postnet.convolutions." + std::to_string(layer));
    } else {
      // intermediate layer
      convLayer = convBatchNormCreator.add(
          *network,
          convInput,
          *convData,
          *normData,
          "tanh",
          "postnet.convolutions." + std::to_string(layer));
    }
    convInput = convLayer->getOutput(0);
    }

    // perform the addition
    ILayer* const sumLayer = network->addElementWise(*convInput, *input, ElementWiseOperation::kSUM);
    sumLayer->setName("postnet.elementwise_sum");

    // and transpose before output
    IShuffleLayer* const transLayer = network->addShuffle(*sumLayer->getOutput(0));
    transLayer->setFirstTranspose({0, 2, 1, 3});
    transLayer->setName("postnet.transpose");

    ITensor* const output = transLayer->getOutput(0);

    output->setName(OUTPUT_NAME);
    network->markOutput(*output);

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
        throw std::runtime_error("Failed to build Tacotron2::PostNet engine.");
    }

    return engine;
}

} // namespace tts
