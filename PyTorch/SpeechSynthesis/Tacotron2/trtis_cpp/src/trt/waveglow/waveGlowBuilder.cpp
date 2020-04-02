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

#include "waveGlowBuilder.h"
#include "logging.h"
#include "trtUtils.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nvinfer1;
using IParser = nvonnxparser::IParser;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const ENGINE_NAME = "waveglow_chunk160_fp16";
constexpr const char* const MEL_INPUT_NAME = "spect";
constexpr const char* const Z_INPUT_NAME = "z";
constexpr const char* const OUTPUT_NAME = "audio";
} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

WaveGlowBuilder::WaveGlowBuilder(const std::string& modelPath, std::shared_ptr<ILogger> logger)
    : mOnnxModelPath(modelPath)
    , mLogger(logger)
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

TRTPtr<ICudaEngine> WaveGlowBuilder::build(
    IBuilder& builder, const int maxBatchSize, const bool useFP16)
{
    // configure tensor-rt objects
    TRTPtr<INetworkDefinition> network(builder.createNetworkV2(
        1U << static_cast<int>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    network->setName("WaveGlow");

    TRTPtr<IParser> parser{nvonnxparser::createParser(*network, *mLogger)};
    if (!parser->parseFromFile(mOnnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kERROR)))
    {
        throw std::runtime_error("Failed to parse ONNX network. Parser failed.");
    }

    if (network->getOutput(0) == nullptr)
    {
        throw std::runtime_error("Failed to parse ONNX network. Null output.");
    }

    // set all inputs to FP32
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        if (network->getInput(i)->getType() != DataType::kFLOAT)
        {
            network->getInput(i)->setType(DataType::kFLOAT);
            if (network->getInput(i)->getType() != DataType::kFLOAT)
            {
                throw std::runtime_error("WaveGlowBuilder expects non 32-bit input for " + std::to_string(i));
            }
        }
    }

    // set output to FP32 and name
    ITensor* output = network->getOutput(0);
    if (output->getType() == DataType::kHALF)
    {
        // convert from half to full
        network->unmarkOutput(*output);
        IIdentityLayer* const identLayer = network->addIdentity(*output);
        identLayer->setPrecision(DataType::kFLOAT);
        output = identLayer->getOutput(0);
        assert(output->getType() == DataType::kFLOAT);
        network->markOutput(*output);

        std::cout << "Changing output to be 32-bit" << std::endl;
    }
    output->setName(OUTPUT_NAME);

    // rename z
    network->getInput(1)->setName(Z_INPUT_NAME);

    // add transpose to mel spectrogram
    ITensor* const originalInput = network->getInput(0);
    originalInput->setName("toBeRemoved");
    const Dims originalDims = originalInput->getDimensions();
    if (originalDims.nbDims != 4)
    {
        throw std::runtime_error("Invalid WaveGlow input of " + TRTUtils::dimsToString(originalDims));
    }

    ITensor* const spectInput = network->addInput(MEL_INPUT_NAME, DataType::kFLOAT,
        Dims4(originalDims.d[0], originalDims.d[3], originalDims.d[2], originalDims.d[1]));

    ILayer* const firstLayer = network->getLayer(0);

    IShuffleLayer* const transLayer = network->addShuffle(*spectInput);
    transLayer->setFirstTranspose({0, 3, 2, 1});

    firstLayer->setInput(0, *transLayer->getOutput(0));

    network->removeTensor(*originalInput);

    TRTPtr<IBuilderConfig> config(builder.createBuilderConfig());
    config->setMaxWorkspaceSize(1ULL << 29);
    if (useFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    Dims minSpectDims = spectInput->getDimensions();
    minSpectDims.d[0] = 1;
    Dims maxSpectDims = minSpectDims;
    maxSpectDims.d[0] = maxBatchSize;

    Dims minZDims = TRTUtils::getInputByName(*network, Z_INPUT_NAME)->getDimensions();
    minZDims.d[0] = 1;
    Dims maxZDims = minZDims;
    maxZDims.d[0] = maxBatchSize;

    TRTUtils::printDimensions("spect", minSpectDims);
    TRTUtils::printDimensions("z", minZDims);
    TRTUtils::printDimensions("spect", maxSpectDims);
    TRTUtils::printDimensions("z", maxZDims);

    IOptimizationProfile* const optProfile = builder.createOptimizationProfile();
    optProfile->setDimensions(MEL_INPUT_NAME, OptProfileSelector::kMIN, minSpectDims);
    optProfile->setDimensions(MEL_INPUT_NAME, OptProfileSelector::kMAX, maxSpectDims);
    optProfile->setDimensions(MEL_INPUT_NAME, OptProfileSelector::kOPT, minSpectDims);

    optProfile->setDimensions(Z_INPUT_NAME, OptProfileSelector::kMIN, minZDims);
    optProfile->setDimensions(Z_INPUT_NAME, OptProfileSelector::kMAX, maxZDims);
    optProfile->setDimensions(Z_INPUT_NAME, OptProfileSelector::kOPT, minZDims);

    config->addOptimizationProfile(optProfile);

    TRTPtr<ICudaEngine> engine(
        builder.buildEngineWithConfig(*network, *config));
    if (!engine)
    {
        throw std::runtime_error("Failed to build WaveGlow engine.");
    }

    return engine;
}

} // namespace tts
