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

#include "decoderBuilderPlugins.h"
#include "decoderInstance.h"
#include "dims5.h"
#include "engineCache.h"
#include "pluginBuilder.h"
#include "trtUtils.h"

#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

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
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

void configureDims(const INetworkDefinition* const network, IOptimizationProfile* optProfile,
    const std::string& inputName, const int maxBatchSize, const int minInputLength, const int maxInputLength,
    const int optInputLength)
{
    for (int inputIdx = 0; inputIdx < network->getNbInputs(); ++inputIdx)
    {
        const ITensor* const input = network->getInput(inputIdx);
        if (std::string(input->getName()) == inputName)
        {
            const Dims defDims = input->getDimensions();
            Dims maxDims = defDims;
            Dims minDims = defDims;
            Dims optDims = defDims;

            bool foundBatch = false;
            bool foundLength = false;
            for (int d = 0; d < defDims.nbDims; ++d)
            {
                if (defDims.d[d] == -1)
                {
                    if (!foundBatch)
                    {
                        maxDims.d[d] = maxBatchSize;
                        minDims.d[d] = 1;
                        optDims.d[d] = 1;
                        foundBatch = true;
                    }
                    else if (!foundLength)
                    {
                        maxDims.d[d] = maxInputLength;
                        minDims.d[d] = minInputLength;
                        optDims.d[d] = optInputLength;
                        foundLength = true;
                    }
                    else
                    {
                        throw std::runtime_error("Unknown third dynamic dimension: " + std::to_string(d));
                    }
                }
            }

            if (!foundBatch || !foundLength)
            {
                throw std::runtime_error("Failed to find all dynamic dimensions");
            }

            if (!optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, minDims))
            {
                throw std::runtime_error("Failed to set minimum dimensions of " + TRTUtils::dimsToString(minDims)
                    + " for " + inputName + ".");
            }
            if (!optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, maxDims))
            {
                throw std::runtime_error("Failed to set maximum dimensions of " + TRTUtils::dimsToString(maxDims)
                    + " for " + inputName + ".");
            }
            if (!optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, optDims))
            {
                throw std::runtime_error("Failed to set optimal dimensions of " + TRTUtils::dimsToString(optDims)
                    + " for " + inputName + ".");
            }

            // success
            return;
        }
    }

    throw std::runtime_error("Unable to find input: '" + inputName + "'.");
}

void configureDefaultDims(const INetworkDefinition* const network, IOptimizationProfile* optProfile,
    const std::string& inputName, const int maxBatchSize)
{
    for (int inputIdx = 0; inputIdx < network->getNbInputs(); ++inputIdx)
    {
        const ITensor* const input = network->getInput(inputIdx);
        if (std::string(input->getName()) == inputName)
        {
            const Dims defDims = input->getDimensions();
            Dims maxDims = defDims;
            Dims minDims = defDims;
            Dims optDims = defDims;

            bool foundBatch = false;
            for (int d = 0; d < defDims.nbDims; ++d)
            {
                if (defDims.d[d] == -1)
                {
                    if (!foundBatch)
                    {
                        maxDims.d[d] = maxBatchSize;
                        minDims.d[d] = 1;
                        optDims.d[d] = 1;
                        foundBatch = true;
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Unknown second dynamic dimension for " + inputName + ": " + std::to_string(d));
                    }
                }
            }

            if (!foundBatch)
            {
                throw std::runtime_error("Failed to find all dynamic dimensions");
            }

            if (!optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, minDims))
            {
                throw std::runtime_error("Failed to set minimum dimensions of " + TRTUtils::dimsToString(minDims)
                    + " for " + inputName + ".");
            }
            if (!optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, maxDims))
            {
                throw std::runtime_error("Failed to set maximum dimensions of " + TRTUtils::dimsToString(maxDims)
                    + " for " + inputName + ".");
            }
            if (!optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, optDims))
            {
                throw std::runtime_error("Failed to set optimal dimensions of " + TRTUtils::dimsToString(optDims)
                    + " for " + inputName + ".");
            }

            // success
            return;
        }
    }

    throw std::runtime_error("Unable to find input: '" + inputName + "'.");
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DecoderBuilderPlugins::DecoderBuilderPlugins(const int numDim, const int numChannels)
    : mNumEncodingDim(numDim)
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

TRTPtr<ICudaEngine> DecoderBuilderPlugins::build(
    IBuilder& builder,
    IModelImporter& importer,
    const int maxBatchSize,
    const int minInputLength,
    const int maxInputLength,
    const bool useFP16)
{
    if (maxBatchSize > 1)
    {
        throw std::runtime_error(
            "DecoderBuilderPlugins only supports batch size of 1: " + std::to_string(maxBatchSize));
    }

    TRTPtr<INetworkDefinition> network(builder.createNetworkV2(
        1U << static_cast<int>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    network->setName("Tacotron2_DecoderWithPlugins");

    // PRENET ///////////////////////////////////////////////////////////////////
    ITensor* prenetInput = network->addInput(INPUT_LASTFRAME_NAME, DataType::kFLOAT, Dims4{-1, mNumChannels + 1, 1, 1});
    ITensor* dropoutInput = network->addInput(INPUT_DROPOUT_NAME, DataType::kFLOAT, Dims4{-1, mNumPrenetDim, 1, 1});

    const LayerData* const prenetData1 = importer.getWeights({"decoder", "prenet", "layers", "0", "linear_layer"});
    const LayerData* const prenetData2 = importer.getWeights({"decoder", "prenet", "layers", "1", "linear_layer"});

    PluginBuilder prenetBuilder("Taco2Prenet", "0.1.0");
    prenetBuilder.setField("InputLength", mNumChannels);
    prenetBuilder.setField("Dimension", mNumPrenetDim);
    prenetBuilder.setField("weight1", prenetData1->get("weight"));
    prenetBuilder.setField("weight2", prenetData2->get("weight"));
    TRTPtr<IPluginV2> prenet = prenetBuilder.make("decoder.prenet");

    std::vector<ITensor*> prenetInputs{prenetInput, dropoutInput};
    ILayer* const prenetLayer
        = network->addPluginV2(prenetInputs.data(), static_cast<int>(prenetInputs.size()), *prenet);
    prenetLayer->setName("decoder.prenet");
    ITensor* const prenetOutput = prenetLayer->getOutput(0);

    // ATTENTION LSTM ///////////////////////////////////////////////////////////
    ITensor* const attentionContextInput
        = network->addInput(INPUT_CONTEXT_NAME, DataType::kFLOAT, Dims3{-1, 1, mNumEncodingDim});
    ITensor* const attentionRNNHidden
        = network->addInput(INPUT_ATTENTIONHIDDEN_NAME, DataType::kFLOAT, Dims3{-1, 1, mNumAttentionRNNDim});
    ITensor* const attentionRNNCell
        = network->addInput(INPUT_ATTENTIONCELL_NAME, DataType::kFLOAT, Dims3{-1, 1, mNumAttentionRNNDim});

    const LayerData* const lstmData = importer.getWeights({"decoder", "attention_rnn"});

    std::vector<ITensor*> attentionLSTMInputs{
        prenetOutput, attentionContextInput, attentionRNNHidden, attentionRNNCell};

    PluginBuilder attLSTMCellBuilder("Taco2LSTMCell", "0.1.0");
    attLSTMCellBuilder.setField("Length",
        static_cast<int32_t>(
            TRTUtils::getTensorSize(*attentionLSTMInputs[0]) + TRTUtils::getTensorSize(*attentionLSTMInputs[1])));
    attLSTMCellBuilder.setField("Dimension", mNumAttentionRNNDim);
    attLSTMCellBuilder.setField("FP16", static_cast<int32_t>(useFP16));
    attLSTMCellBuilder.setField("weight_ih", lstmData->get("weight_ih"));
    attLSTMCellBuilder.setField("weight_hh", lstmData->get("weight_hh"));
    attLSTMCellBuilder.setField("bias_ih", lstmData->get("bias_ih"));
    attLSTMCellBuilder.setField("bias_hh", lstmData->get("bias_hh"));
    TRTPtr<IPluginV2> attentionLSTM
        = attLSTMCellBuilder.make("decoder.attention_rnn");

    ILayer* const attentionLSTMLayer = network->addPluginV2(
        attentionLSTMInputs.data(), static_cast<int>(attentionLSTMInputs.size()), *attentionLSTM);

    ITensor* const attentionHiddenOut = attentionLSTMLayer->getOutput(0);
    ITensor* const attentionCellOut = attentionLSTMLayer->getOutput(1);

    attentionLSTMLayer->setName("decoder.attention_rnn");

    attentionHiddenOut->setName(OUTPUT_ATTENTIONHIDDEN_NAME);
    network->markOutput(*attentionHiddenOut);

    attentionCellOut->setName(OUTPUT_ATTENTIONCELL_NAME);
    network->markOutput(*attentionCellOut);

    // ATTENTION ////////////////////////////////////////////////////////////////

    ITensor* const inputMemory = network->addInput(INPUT_MEMORY_NAME, DataType::kFLOAT, Dims3(-1, -1, mNumEncodingDim));
    ITensor* const inputProcessedMemory
        = network->addInput(INPUT_PROCESSED_NAME, DataType::kFLOAT, Dims5(-1, -1, mNumAttentionDim, 1, 1));
    ITensor* const inputWeights = network->addInput(INPUT_WEIGHTS_NAME, DataType::kFLOAT, Dims4(-1, 2, -1, 1));

    const LayerData* const queryData
        = importer.getWeights({"decoder", "attention_layer", "query_layer", "linear_layer"});
    const LayerData* const locationConvData
        = importer.getWeights({"decoder", "attention_layer", "location_layer", "location_conv", "conv"});
    const LayerData* const locationLinearData
        = importer.getWeights({"decoder", "attention_layer", "location_layer", "location_dense", "linear_layer"});
    const LayerData* const energyData = importer.getWeights({"decoder", "attention_layer", "v", "linear_layer"});

    std::vector<ITensor*> attentionInputs{inputMemory, inputProcessedMemory, inputWeights, attentionHiddenOut};

    PluginBuilder attBuilder("Taco2Attention", "0.1.0");
    attBuilder.setField("EncodingDimension", mNumEncodingDim);
    attBuilder.setField("QueryDimension", mNumAttentionRNNDim);
    attBuilder.setField("NumFilters", mNumAttentionFilters);
    attBuilder.setField("ConvKernelSize", mAttentionKernelSize);
    attBuilder.setField("AttentionDimension", mNumAttentionDim);
    attBuilder.setField("QueryWeight", queryData->get("weight"));
    attBuilder.setField("ConvWeight", locationConvData->get("weight"));
    attBuilder.setField("LocationWeight", locationLinearData->get("weight"));
    attBuilder.setField("EnergyWeight", energyData->get("weight"));

    TRTPtr<IPluginV2> attention = attBuilder.make("decoder.attention_layer");

    ILayer* const attentionLayer
        = network->addPluginV2(attentionInputs.data(), static_cast<int>(attentionInputs.size()), *attention);
    attentionLayer->setName("decoder.attention_layer");
    ITensor* const attentionContextOutput = attentionLayer->getOutput(0);
    ITensor* const attentionWeightOutput = attentionLayer->getOutput(1);

    attentionWeightOutput->setName(OUTPUT_WEIGHTS_NAME);
    network->markOutput(*attentionWeightOutput);

    attentionContextOutput->setName(OUTPUT_CONTEXT_NAME);
    network->markOutput(*attentionContextOutput);

    // DECODER LSTM /////////////////////////////////////////////////////////////

    ITensor* const inputDecoderHidden
        = network->addInput(INPUT_DECODERHIDDEN_NAME, DataType::kFLOAT, Dims3{-1, 1, mNumLSTMDim});
    ITensor* const inputDecoderCell
        = network->addInput(INPUT_DECODERCELL_NAME, DataType::kFLOAT, Dims3{-1, 1, mNumLSTMDim});

    const LayerData* const decoderLSTMData = importer.getWeights({"decoder", "decoder_rnn"});

    std::vector<ITensor*> decoderLSTMInputs{
        attentionHiddenOut, attentionContextOutput, inputDecoderHidden, inputDecoderCell};

    PluginBuilder decoderLSTMCellBuilder("Taco2LSTMCell", "0.1.0");
    decoderLSTMCellBuilder.setField("Length",
        static_cast<int32_t>(
            TRTUtils::getTensorSize(*decoderLSTMInputs[0]) + TRTUtils::getTensorSize(*decoderLSTMInputs[1])));
    decoderLSTMCellBuilder.setField("Dimension", mNumLSTMDim);
    decoderLSTMCellBuilder.setField("FP16", static_cast<int32_t>(useFP16));
    decoderLSTMCellBuilder.setField("weight_ih", decoderLSTMData->get("weight_ih"));
    decoderLSTMCellBuilder.setField("weight_hh", decoderLSTMData->get("weight_hh"));
    decoderLSTMCellBuilder.setField("bias_ih", decoderLSTMData->get("bias_ih"));
    decoderLSTMCellBuilder.setField("bias_hh", decoderLSTMData->get("bias_hh"));
    TRTPtr<IPluginV2> decoderLSTM
        = decoderLSTMCellBuilder.make("decoder.decoder_rnn");

    ILayer* const decoderLSTMLayer
        = network->addPluginV2(decoderLSTMInputs.data(), static_cast<int>(decoderLSTMInputs.size()), *decoderLSTM);
    decoderLSTMLayer->setName("decoder.decoder_rnn");

    ITensor* const decoderHiddenOut = decoderLSTMLayer->getOutput(0);
    ITensor* const decoderCellOut = decoderLSTMLayer->getOutput(1);

    decoderHiddenOut->setName(OUTPUT_DECODERHIDDEN_NAME);
    network->markOutput(*decoderHiddenOut);

    decoderCellOut->setName(OUTPUT_DECODERCELL_NAME);
    network->markOutput(*decoderCellOut);

    // PROJECTION ///////////////////////////////////////////////////////////////
    const LayerData* const channelData = importer.getWeights({"decoder", "linear_projection", "linear_layer"});
    const LayerData* const gateData = importer.getWeights({"decoder", "gate_layer", "linear_layer"});

    PluginBuilder projBuilder("Taco2Projection", "0.1.0");
    projBuilder.setField("HiddenInputLength", static_cast<int32_t>(TRTUtils::getTensorSize(*decoderHiddenOut)));
    projBuilder.setField("ContextInputLength", static_cast<int32_t>(TRTUtils::getTensorSize(*attentionContextOutput)));
    projBuilder.setField("ChannelDimension", mNumChannels);
    projBuilder.setField("GateDimension", 1);
    projBuilder.setField("ChannelWeights", channelData->get("weight"));
    projBuilder.setField("GateWeights", gateData->get("weight"));
    projBuilder.setField("ChannelBias", channelData->get("bias"));
    projBuilder.setField("GateBias", gateData->get("bias"));
    TRTPtr<IPluginV2> proj
        = projBuilder.make("decoder.linear_projection.linear_layer");

    std::vector<ITensor*> projInputs{decoderHiddenOut, attentionContextOutput};

    ILayer* const projLayer = network->addPluginV2(projInputs.data(), static_cast<int>(projInputs.size()), *proj);

    projLayer->setName("decoder.linear_projection.linear_layer");
    ITensor* const outputChannels = projLayer->getOutput(0);

    outputChannels->setName(OUTPUT_CHANNELS_NAME);
    network->markOutput(*outputChannels);

    TRTPtr<IBuilderConfig> config(builder.createBuilderConfig());

    config->setMaxWorkspaceSize(1ULL << 29); // 512 MB
    if (useFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    builder.setMaxBatchSize(maxBatchSize);

    IOptimizationProfile* const optProfile = builder.createOptimizationProfile();

    // the optimimum input length should actually matter, so we'll just take
    // the average
    const int optInputLength = (minInputLength + maxInputLength) / 2;

    // memory dimensions
    configureDims(
        network.get(), optProfile, INPUT_MEMORY_NAME, maxBatchSize, minInputLength, maxInputLength, optInputLength);

    // processed memory dimensions
    configureDims(
        network.get(), optProfile, INPUT_PROCESSED_NAME, maxBatchSize, minInputLength, maxInputLength, optInputLength);

    // weights dimensions
    configureDims(
        network.get(), optProfile, INPUT_WEIGHTS_NAME, maxBatchSize, minInputLength, maxInputLength, optInputLength);

    // set the batch dimension on the rest
    configureDefaultDims(network.get(), optProfile, INPUT_DROPOUT_NAME, maxBatchSize);
    configureDefaultDims(network.get(), optProfile, INPUT_LASTFRAME_NAME, maxBatchSize);
    configureDefaultDims(network.get(), optProfile, INPUT_CONTEXT_NAME, maxBatchSize);
    configureDefaultDims(network.get(), optProfile, INPUT_ATTENTIONHIDDEN_NAME, maxBatchSize);
    configureDefaultDims(network.get(), optProfile, INPUT_ATTENTIONCELL_NAME, maxBatchSize);
    configureDefaultDims(network.get(), optProfile, INPUT_DECODERHIDDEN_NAME, maxBatchSize);
    configureDefaultDims(network.get(), optProfile, INPUT_DECODERCELL_NAME, maxBatchSize);

    config->addOptimizationProfile(optProfile);

    TRTPtr<ICudaEngine> engine(
        builder.buildEngineWithConfig(*network, *config));

    if (!engine)
    {
        throw std::runtime_error("Failed to build Tacotron2::DecoderPlugins engine.");
    }

    return engine;
}

} // namespace tts
