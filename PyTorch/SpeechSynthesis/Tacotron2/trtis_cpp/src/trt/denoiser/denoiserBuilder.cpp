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

#include "denoiserBuilder.h"
#include "denoiserStreamingInstance.h"
#include "pluginBuilder.h"
#include "trtUtils.h"

#include <iostream>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const char* const INPUT_NAME = DenoiserStreamingInstance::INPUT_NAME;
constexpr const char* const OUTPUT_NAME = DenoiserStreamingInstance::OUTPUT_NAME;
} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DenoiserBuilder::DenoiserBuilder(int sampleLength, int filterLength, int numOverlap, int winLength)
    : mChunkSize(sampleLength)
    , mFilterLength(filterLength)
    , mHopLength(filterLength / numOverlap)
    , mWinLength(winLength)
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

TRTPtr<ICudaEngine> DenoiserBuilder::build(
    IModelImporter& importer,
    IBuilder& builder,
    const int maxBatchSize,
    const bool useFP16)
{
  TRTPtr<INetworkDefinition> network(builder.createNetworkV2(0));
  network->setName("Denoiser");

  const int cutoff = mFilterLength / 2 + 1;

  const LayerData* const stftData = importer.getWeights({"denoiser", "stft"});
  const LayerData* const denoiserData = importer.getWeights({"denoiser"});

  ITensor* const input = network->addInput(
      INPUT_NAME, DataType::kFLOAT, Dims4(1, 1, 1, mChunkSize));

    // forward transform
    #if NV_TENSORRT_MAJOR < 7 
    IConvolutionLayer* const convLayer = network->addConvolution(
        *input, cutoff * 2, DimsHW(1, mFilterLength), stftData->get("forward_basis"), Weights{});
    convLayer->setPadding(DimsHW(0, mFilterLength / 2));
    convLayer->setStride(DimsHW(1, mHopLength));
    #else
    IConvolutionLayer* const convLayer = network->addConvolutionNd(
        *input, cutoff * 2, Dims2(1, mFilterLength), stftData->get("forward_basis"), Weights{});
    convLayer->setPaddingNd(Dims2(0, mFilterLength / 2));
    convLayer->setStrideNd(Dims2(1, mHopLength));
    #endif
    convLayer->setName("forward_transform_layer");

    // use plugin to compute magnitude and phase
    PluginBuilder denoiseTransformBuilder("Taco2DenoiseTransform", "0.1.0");
    denoiseTransformBuilder.setField(
        "InputLength", static_cast<int32_t>(TRTUtils::getTensorSize(*convLayer->getOutput(0)) / (cutoff * 2)));
    denoiseTransformBuilder.setField("FilterLength", cutoff * 2);
    denoiseTransformBuilder.setField("Weights", denoiserData->get("bias_spec"));
    TRTPtr<IPluginV2> denoise = denoiseTransformBuilder.make("denoise_layer");

    std::vector<ITensor*> denoiseInputs{convLayer->getOutput(0)};
    ILayer* const denoiseLayer
        = network->addPluginV2(denoiseInputs.data(), static_cast<int>(denoiseInputs.size()), *denoise);

    // inverse transform
    #if NV_TENSORRT_MAJOR < 7 
    IDeconvolutionLayer* const deconvLayer = network->addDeconvolution(
        *denoiseLayer->getOutput(0), 1, DimsHW(1, mFilterLength), stftData->get("inverse_basis"), {});
    deconvLayer->setStride(DimsHW(1, mHopLength));
    #else
    IDeconvolutionLayer* const deconvLayer = network->addDeconvolutionNd(
        *denoiseLayer->getOutput(0), 1, Dims2(1, mFilterLength), stftData->get("inverse_basis"), {});
    deconvLayer->setStrideNd(Dims2(1, mHopLength));
    #endif
    deconvLayer->setName("inverse_transform_layer");

    // apply windowing
    PluginBuilder modulationRemovalBuilder("Taco2ModulationRemoval", "0.1.0");
    modulationRemovalBuilder.setField(
        "InputLength", static_cast<int32_t>(TRTUtils::getTensorSize(*deconvLayer->getOutput(0))));
    modulationRemovalBuilder.setField("FilterLength", static_cast<int32_t>(mFilterLength));
    modulationRemovalBuilder.setField("HopLength", static_cast<int32_t>(mHopLength));
    modulationRemovalBuilder.setField("Weights", stftData->get("win_sq"));
    TRTPtr<IPluginV2> modRemoval
        = modulationRemovalBuilder.make("modulation_removal_layer");

    std::vector<ITensor*> modRemovalInputs{deconvLayer->getOutput(0)};
    ILayer* const modRemovalLayer
        = network->addPluginV2(modRemovalInputs.data(), static_cast<int>(modRemovalInputs.size()), *modRemoval);

    ITensor* const output = modRemovalLayer->getOutput(0);
    output->setName(OUTPUT_NAME);
    network->markOutput(*output);

    assert(TRTUtils::getTensorSize(*output) == static_cast<size_t>(mChunkSize));

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
        throw std::runtime_error("Failed to build Denoiser engine.");
    }

    return engine;
}

} // namespace tts
