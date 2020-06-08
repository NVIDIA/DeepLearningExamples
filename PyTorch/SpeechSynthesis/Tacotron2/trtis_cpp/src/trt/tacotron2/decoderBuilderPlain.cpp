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

#include "decoderBuilderPlain.h"
#include "attentionLayerCreator.h"
#include "decoderInstance.h"
#include "dims5.h"
#include "engineCache.h"
#include "lstm.h"
#include "utils.h"

#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const int NUM_PRENET_LAYERS = 2;

constexpr const char* const INPUT_MASK_NAME = DecoderInstance::INPUT_MASK_NAME;
constexpr const char* const INPUT_LENGTH_NAME = DecoderInstance::INPUT_LENGTH_NAME;
constexpr const char* const INPUT_DROPOUT_NAME = DecoderInstance::INPUT_DROPOUT_NAME;
constexpr const char* const INPUT_LASTFRAME_NAME = DecoderInstance::INPUT_LASTFRAME_NAME;
constexpr const char* const INPUT_MEMORY_NAME = DecoderInstance::INPUT_MEMORY_NAME;
constexpr const char* const INPUT_PROCESSED_NAME = DecoderInstance::INPUT_PROCESSED_NAME;
constexpr const char* const INPUT_WEIGHTS_NAME = DecoderInstance::INPUT_WEIGHTS_NAME;
constexpr const char* const INPUT_CONTEXT_NAME = DecoderInstance::INPUT_CONTEXT_NAME;
constexpr const char* const INPUT_ATTENTIONHIDDEN_NAME = DecoderInstance::INPUT_ATTENTIONHIDDEN_NAME;
constexpr const char* const INPUT_ATTENTIONCELL_NAME = DecoderInstance::INPUT_ATTENTIONCELL_NAME;
constexpr const char* const INPUT_DECODERHIDDEN_NAME = DecoderInstance::INPUT_DECODERHIDDEN_NAME;
constexpr const char* const INPUT_DECODERCELL_NAME = DecoderInstance::INPUT_DECODERCELL_NAME;
constexpr const char* const OUTPUT_ATTENTIONHIDDEN_NAME = DecoderInstance::OUTPUT_ATTENTIONHIDDEN_NAME;
constexpr const char* const OUTPUT_ATTENTIONCELL_NAME = DecoderInstance::OUTPUT_ATTENTIONCELL_NAME;
constexpr const char* const OUTPUT_CONTEXT_NAME = DecoderInstance::OUTPUT_CONTEXT_NAME;
constexpr const char* const OUTPUT_WEIGHTS_NAME = DecoderInstance::OUTPUT_WEIGHTS_NAME;
constexpr const char* const OUTPUT_DECODERHIDDEN_NAME = DecoderInstance::OUTPUT_DECODERHIDDEN_NAME;
constexpr const char* const OUTPUT_DECODERCELL_NAME = DecoderInstance::OUTPUT_DECODERCELL_NAME;
constexpr const char* const OUTPUT_CHANNELS_NAME = DecoderInstance::OUTPUT_CHANNELS_NAME;
constexpr const char* const OUTPUT_GATE_NAME = DecoderInstance::OUTPUT_GATE_NAME;

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DecoderBuilderPlain::DecoderBuilderPlain(const int inputLength, const int numDim, const int numChannels)
    : mInputLength(inputLength)
    , mNumEncodingDim(numDim)
    , mNumPrenetDim(256)
    , mNumAttentionRNNDim(1024)
    , mNumAttentionDim(128)
    , mNumAttentionFilters(32)
    , mAttentionKernelSize(31)
    , mNumLSTMDim(1024)
    , mNumChannels(numChannels)
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

TRTPtr<ICudaEngine> DecoderBuilderPlain::build(
    IBuilder& builder,
    IModelImporter& importer,
    const int maxBatchSize,
    const bool useFP16)
{
  TRTPtr<INetworkDefinition> network(builder.createNetworkV2(0));
  network->setName("Tacotron2_DecoderWithoutPlugins");

  // PRENET ///////////////////////////////////////////////////////////////////
  ITensor* prenetInput = network->addInput(
      INPUT_LASTFRAME_NAME, DataType::kFLOAT, Dims4{1, mNumChannels + 1, 1, 1});
  ITensor* dropoutInput = network->addInput(
      INPUT_DROPOUT_NAME, DataType::kFLOAT, Dims4{1, mNumPrenetDim, 1, 1});

  ISliceLayer* inputSlice = network->addSlice(
      *prenetInput,
      Dims4(0, 0, 0, 0),
      Dims4(1, mNumChannels, 1, 1),
      Dims4(1, 1, 1, 1));
  inputSlice->setName("decoder.frame_slice");
  prenetInput = inputSlice->getOutput(0);

  for (int layer = 0; layer < NUM_PRENET_LAYERS; ++layer) {
    const LayerData* const linearData = importer.getWeights(
        {"decoder", "prenet", "layers", std::to_string(layer), "linear_layer"});

    ILayer* const linearLayer = network->addFullyConnected(
        *prenetInput,
        mNumPrenetDim,
        linearData->get("weight"),
        Weights{DataType::kFLOAT, nullptr, 0});
    linearLayer->setName(
        std::string(
            "decoder.prenet.layers." + std::to_string(layer) + ".linear_layer")
            .c_str());

    ILayer* const reluLayer = network->addActivation(
        *linearLayer->getOutput(0), ActivationType::kRELU);
    reluLayer->setName(
        std::string("decoder.prenet.layers." + std::to_string(layer) + ".relu")
            .c_str());

    IElementWiseLayer* const elemLayer = network->addElementWise(
        *reluLayer->getOutput(0), *dropoutInput, ElementWiseOperation::kPROD);
    elemLayer->setName(
        std::string(
            "decoder.prenet.layers." + std::to_string(layer) + ".dropout")
            .c_str());

    prenetInput = elemLayer->getOutput(0);
    }
    ITensor* const prenetOutput = prenetInput;

    // ATTENTION LSTM ///////////////////////////////////////////////////////////
    ITensor* const attentionContextInput
        = network->addInput(INPUT_CONTEXT_NAME, DataType::kFLOAT, Dims3{1, 1, mNumEncodingDim});
    ITensor* const attentionRNNHidden
        = network->addInput(INPUT_ATTENTIONHIDDEN_NAME, DataType::kFLOAT, Dims3{1, 1, mNumAttentionRNNDim});
    ITensor* const attentionRNNCell
        = network->addInput(INPUT_ATTENTIONCELL_NAME, DataType::kFLOAT, Dims3{1, 1, mNumAttentionRNNDim});

    const LayerData* const lstmData = importer.getWeights({"decoder", "attention_rnn"});

    IShuffleLayer* const prenetShuffle = network->addShuffle(*prenetOutput);
    prenetShuffle->setReshapeDimensions(Dims3{1, 1, -1});

    std::array<ITensor*, 2> lstmInputs{prenetShuffle->getOutput(0), attentionContextInput};
    IConcatenationLayer* lstmConcatLayer
        = network->addConcatenation(lstmInputs.data(), static_cast<int>(lstmInputs.size()));
    lstmConcatLayer->setAxis(2);
    lstmConcatLayer->setName("decoder.attention_rnn.concat");

    ILayer* attentionLSTMLayer = LSTM::addUnidirectionalCell(network.get(), lstmConcatLayer->getOutput(0),
        attentionRNNHidden, attentionRNNCell, mNumAttentionRNNDim, *lstmData);

    ITensor* const attentionHiddenOut = attentionLSTMLayer->getOutput(1);
    ITensor* const attentionCellOut = attentionLSTMLayer->getOutput(2);

    attentionLSTMLayer->setName("decoder.attention_rnn");

    attentionHiddenOut->setName(OUTPUT_ATTENTIONHIDDEN_NAME);
    network->markOutput(*attentionHiddenOut);

    attentionCellOut->setName(OUTPUT_ATTENTIONCELL_NAME);
    network->markOutput(*attentionCellOut);

    // ATTENTION ////////////////////////////////////////////////////////////////

    ITensor* const inputMemory
        = network->addInput(INPUT_MEMORY_NAME, DataType::kFLOAT, Dims3{1, mInputLength, mNumEncodingDim});
    ITensor* const inputProcessedMemory
        = network->addInput(INPUT_PROCESSED_NAME, DataType::kFLOAT, Dims5{1, mInputLength, mNumAttentionDim, 1, 1});
    ITensor* const inputWeights = network->addInput(INPUT_WEIGHTS_NAME, DataType::kFLOAT, Dims4{1, 2, mInputLength, 1});
    ITensor* const inputMask = network->addInput(INPUT_MASK_NAME, DataType::kFLOAT, Dims3{1, 1, mInputLength});

    ITensor* const inputMaskLength = network->addInput(INPUT_LENGTH_NAME, DataType::kINT32, Dims2{1, 1});

    // reshape data to go from {1,1,X} to {1,1,X,1,1}
    IShuffleLayer* const queryShuffleLayer = network->addShuffle(*attentionHiddenOut);
    queryShuffleLayer->setReshapeDimensions(Dims5{1, 1, attentionHiddenOut->getDimensions().d[2], 1, 1});
    queryShuffleLayer->setName("decoder.attention_layer.query_layer.unsqueeze");
    ITensor* const queryInput = queryShuffleLayer->getOutput(0);

    const LayerData* const queryData
        = importer.getWeights({"decoder", "attention_layer", "query_layer", "linear_layer"});
    ILayer* const queryLayer = network->addFullyConnected(
        *queryInput, mNumAttentionDim, queryData->get("weight"), Weights{DataType::kFLOAT, nullptr, 0});
    queryLayer->setName("decoder.attention_layer.query_layer.linear_layer");

    // build location layers
    const LayerData* const locationConvData
        = importer.getWeights({"decoder", "attention_layer", "location_layer", "location_conv", "conv"});
    const LayerData* const locationLinearData
        = importer.getWeights({"decoder", "attention_layer", "location_layer", "location_dense", "linear_layer"});
    ILayer* const locationLayer
        = AttentionLayerCreator::addLocation(*network, inputWeights, mNumAttentionDim, mNumAttentionFilters,
            mAttentionKernelSize, *locationConvData, *locationLinearData, "decoder.attention_layer.location_layer");

    const LayerData* const energyData = importer.getWeights({"decoder", "attention_layer", "v", "linear_layer"});

    IShuffleLayer* const locationShuffleLayer = network->addShuffle(*locationLayer->getOutput(0));
    locationShuffleLayer->setReshapeDimensions(
        Dims5{locationLayer->getOutput(0)->getDimensions().d[0], locationLayer->getOutput(0)->getDimensions().d[1],
            locationLayer->getOutput(0)->getDimensions().d[2], locationLayer->getOutput(0)->getDimensions().d[3], 1});

    ILayer* const energyLayer = AttentionLayerCreator::addEnergy(*network, queryLayer->getOutput(0),
        locationShuffleLayer->getOutput(0), inputProcessedMemory, *energyData, "decoder.attention_layer.v");

    IShuffleLayer* const squeezeEnergyLayer = network->addShuffle(*energyLayer->getOutput(0));
    squeezeEnergyLayer->setReshapeDimensions(
        Dims4(energyLayer->getOutput(0)->getDimensions().d[0], energyLayer->getOutput(0)->getDimensions().d[1],
            energyLayer->getOutput(0)->getDimensions().d[2], energyLayer->getOutput(0)->getDimensions().d[3]));

    ILayer* const softMaxLayer = AttentionLayerCreator::addPaddedSoftMax(*network, squeezeEnergyLayer->getOutput(0),
        inputMask, inputMaskLength, "decoder.attention_layer.softmax_layer");

    IShuffleLayer* const transLayer = network->addShuffle(*softMaxLayer->getOutput(0));
    transLayer->setFirstTranspose({2, 1, 0});
    transLayer->setName("decoder.attention_layer.softmax_transpose");

    ITensor* const attentionWeight = transLayer->getOutput(0);

    ILayer* const sliceWeightsLayer
        = network->addSlice(*inputWeights, Dims4{0, 1, 0, 0}, Dims4{1, 1, mInputLength, 1}, Dims4{1, 1, 1, 1});

    IShuffleLayer* const squeezeWeightsLayer = network->addShuffle(*sliceWeightsLayer->getOutput(0));
    squeezeWeightsLayer->setReshapeDimensions(Dims3(sliceWeightsLayer->getOutput(0)->getDimensions().d[0],
        sliceWeightsLayer->getOutput(0)->getDimensions().d[1], sliceWeightsLayer->getOutput(0)->getDimensions().d[2]));

    ILayer* const sumLayer
        = network->addElementWise(*attentionWeight, *squeezeWeightsLayer->getOutput(0), ElementWiseOperation::kSUM);
    sumLayer->setName("decoder.attention_layer.weight_sum_layer");

    std::vector<ITensor*> weightOutputs{attentionWeight, sumLayer->getOutput(0)};
    IConcatenationLayer* const outputWeightConcat
        = network->addConcatenation(weightOutputs.data(), static_cast<int>(weightOutputs.size()));
    outputWeightConcat->setAxis(2);
    outputWeightConcat->setName("decoder.attention_weights.concat");

    ITensor* const attentionWeightOutput = outputWeightConcat->getOutput(0);

#if NV_TENSORRT_MAJOR < 6
    ILayer* const mmLayer = network->addMatrixMultiply(*attentionWeight, false, *inputMemory, false);
#else
    ILayer* const mmLayer
        = network->addMatrixMultiply(*attentionWeight, MatrixOperation::kNONE, *inputMemory, MatrixOperation::kNONE);
#endif
    mmLayer->setName("decoder.attention_layer.mm");

    ITensor* const attentionContextOutput = mmLayer->getOutput(0);

    attentionWeightOutput->setName(OUTPUT_WEIGHTS_NAME);
    network->markOutput(*attentionWeightOutput);

    attentionContextOutput->setName(OUTPUT_CONTEXT_NAME);
    network->markOutput(*attentionContextOutput);

    // DECODER LSTM /////////////////////////////////////////////////////////////

    ITensor* const inputDecoderHidden
        = network->addInput(INPUT_DECODERHIDDEN_NAME, DataType::kFLOAT, Dims3{1, 1, mNumLSTMDim});
    ITensor* const inputDecoderCell
        = network->addInput(INPUT_DECODERCELL_NAME, DataType::kFLOAT, Dims3{1, 1, mNumLSTMDim});

    const LayerData* const decoderLSTMData = importer.getWeights({"decoder", "decoder_rnn"});

    std::array<ITensor*, 2> decoderLSTMConcatInputs{attentionHiddenOut, attentionContextOutput};

    IConcatenationLayer* concatLayer = network->addConcatenation(decoderLSTMConcatInputs.data(), 2);
    concatLayer->setAxis(2);
    concatLayer->setName("decoder.decoder_rnn.concat");

    ILayer* const decoderLSTMLayer = LSTM::addUnidirectionalCell(
        network.get(), concatLayer->getOutput(0), inputDecoderHidden, inputDecoderCell, mNumLSTMDim, *decoderLSTMData);
    decoderLSTMLayer->setName("decoder.decoder_rnn");

    ITensor* const decoderHiddenOut = decoderLSTMLayer->getOutput(1);
    ITensor* const decoderCellOut = decoderLSTMLayer->getOutput(2);

    decoderHiddenOut->setName(OUTPUT_DECODERHIDDEN_NAME);
    network->markOutput(*decoderHiddenOut);

    decoderCellOut->setName(OUTPUT_DECODERCELL_NAME);
    network->markOutput(*decoderCellOut);

    // PROJECTION ///////////////////////////////////////////////////////////////
    const LayerData* const channelData = importer.getWeights({"decoder", "linear_projection", "linear_layer"});
    const LayerData* const gateData = importer.getWeights({"decoder", "gate_layer", "linear_layer"});

    IShuffleLayer* const projHiddenShuffleLayer = network->addShuffle(*decoderHiddenOut);
    projHiddenShuffleLayer->setReshapeDimensions(Dims4{1, -1, 1, 1});
    projHiddenShuffleLayer->setName("decoder.decoder_rnn.hidden.unsqueeze");

    IShuffleLayer* const projContextShuffleLayer = network->addShuffle(*attentionContextOutput);
    projContextShuffleLayer->setReshapeDimensions(Dims4{1, -1, 1, 1});
    projContextShuffleLayer->setName("decoder.attention_context.unsqueeze");

    std::array<ITensor*, 2> projectionInputs{
        projHiddenShuffleLayer->getOutput(0), projContextShuffleLayer->getOutput(0)};
    IConcatenationLayer* const projConcatLayer
        = network->addConcatenation(projectionInputs.data(), projectionInputs.size());
    projConcatLayer->setAxis(1);
    projConcatLayer->setName("decoder.projection.concat");

    // we'll merge these two tensors layer wise for the weights
    std::vector<float> projectionWeightData(channelData->get("weight").count + gateData->get("weight").count);
    std::copy(static_cast<const float*>(channelData->get("weight").values),
        static_cast<const float*>(channelData->get("weight").values) + channelData->get("weight").count,
        projectionWeightData.data());
    std::copy(static_cast<const float*>(gateData->get("weight").values),
        static_cast<const float*>(gateData->get("weight").values) + gateData->get("weight").count,
        projectionWeightData.data() + channelData->get("weight").count);

    std::vector<float> projectionBiasData(channelData->get("bias").count + gateData->get("bias").count);
    std::copy(static_cast<const float*>(channelData->get("bias").values),
        static_cast<const float*>(channelData->get("bias").values) + channelData->get("bias").count,
        projectionBiasData.data());
    std::copy(static_cast<const float*>(gateData->get("bias").values),
        static_cast<const float*>(gateData->get("bias").values) + gateData->get("bias").count,
        projectionBiasData.data() + channelData->get("bias").count);

    ILayer* const projLayer = network->addFullyConnected(*projConcatLayer->getOutput(0), mNumChannels + 1,
        Weights{DataType::kFLOAT, projectionWeightData.data(), static_cast<int64_t>(projectionWeightData.size())},
        Weights{DataType::kFLOAT, projectionBiasData.data(), static_cast<int64_t>(projectionBiasData.size())});

    projLayer->setName("decoder.linear_projection.linear_layer");
    ITensor* const outputChannels = projLayer->getOutput(0);

    outputChannels->setName(OUTPUT_CHANNELS_NAME);
    network->markOutput(*outputChannels);

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
        throw std::runtime_error("Failed to build Tacotron2::DecoderPlain engine.");
    }

    return engine;
}

} // namespace tts
