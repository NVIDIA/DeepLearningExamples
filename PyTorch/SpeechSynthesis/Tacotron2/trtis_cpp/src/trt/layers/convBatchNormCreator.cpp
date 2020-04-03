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

#include "convBatchNormCreator.h"
#include "layerData.h"
#include "trtUtils.h"

#include "NvInfer.h"

#include <cmath>
#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const float EPS = 1e-5f;
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

ILayer* ConvBatchNormCreator::add(INetworkDefinition& network, ITensor* const input, const LayerData& convData,
    const LayerData& normData, const std::string& activation, const std::string& name)
{
    // base the number of channels based on the output size of the batch norm
    const int numChannels = static_cast<int>(normData.get("bias").count);

    // CONVOLUTION //////////////////////////////////////////////////////////////

    const std::vector<float>& convWeight = newVector(static_cast<const float*>(convData.get("weight").values),
        static_cast<const float*>(convData.get("weight").values) + convData.get("weight").count);
    const std::vector<float>& convBias = newVector(static_cast<const float*>(convData.get("bias").values),
        static_cast<const float*>(convData.get("bias").values) + convData.get("bias").count);
    #if NV_TENSORRT_MAJOR < 7 
    IConvolutionLayer* const convLayer = network.addConvolution(
        *input, numChannels, DimsHW(5, 1), TRTUtils::toWeights(convWeight), TRTUtils::toWeights(convBias));
    convLayer->setPadding({2, 0});
    #else
    IConvolutionLayer* const convLayer = network.addConvolutionNd(
        *input, numChannels, Dims2(5, 1), TRTUtils::toWeights(convWeight), TRTUtils::toWeights(convBias));
    convLayer->setPaddingNd(Dims2(2, 0));
    #endif
    convLayer->setName((name + ".conv_layer").c_str());

    ITensor* const batchInput = convLayer->getOutput(0);

    // BATCH NORM ///////////////////////////////////////////////////////////////

    // create vectors
    std::vector<float>& negativeMeanWeights = newVector(static_cast<const float*>(normData.get("running_mean").values),
        static_cast<const float*>(normData.get("running_mean").values) + normData.get("running_mean").count);
    std::vector<float>& scaleWeights = newVector(static_cast<const float*>(normData.get("weight").values),
        static_cast<const float*>(normData.get("weight").values) + normData.get("weight").count);
    const std::vector<float>& normBias = newVector(static_cast<const float*>(normData.get("bias").values),
        static_cast<const float*>(normData.get("bias").values) + normData.get("bias").count);

    const Weights emptyWeights{DataType::kFLOAT, nullptr, 0};

    // check input
    if (negativeMeanWeights.size() != scaleWeights.size())
    {
        throw std::runtime_error("Mismatch between 'running_mean' and 'weight' sizes: "
            + std::to_string(negativeMeanWeights.size()) + " " + std::to_string(scaleWeights.size()) + ".");
    }
    if (static_cast<size_t>(normData.get("running_var").count) != scaleWeights.size())
    {
        throw std::runtime_error("Size of 'running_var' does not match 'running_mean':"
            + std::to_string(normData.get("running_var").count) + " vs. " + std::to_string(scaleWeights.size()));
    }

    // create negative mean values
    for (float& val : negativeMeanWeights)
    {
        val = -val;
    }

    // compute scaling matrix
    // weight / sqrt(var(x) + eps)
    const float* varWeights = static_cast<const float*>(normData.get("running_var").values);
    for (size_t i = 0; i < scaleWeights.size(); ++i)
    {
        const float den = std::sqrt(varWeights[i] + EPS);
        scaleWeights[i] /= den;
    }

    // x - mean(x)
    ILayer* const shiftedLayer = network.addScale(
        *batchInput, ScaleMode::kCHANNEL, TRTUtils::toWeights(negativeMeanWeights), emptyWeights, emptyWeights);
    shiftedLayer->setName((name + ".shift").c_str());

    // ((x - mean(x)) / sqrt(var(x) + eps)) * weight + bias
    ILayer* const scaleLayer = network.addScale(*shiftedLayer->getOutput(0), ScaleMode::kCHANNEL,
        TRTUtils::toWeights(normBias), TRTUtils::toWeights(scaleWeights), emptyWeights);
    scaleLayer->setName((name + ".scale").c_str());

    ITensor* const actInput = scaleLayer->getOutput(0);

    // ACTIVATION ///////////////////////////////////////////////////////////////

    ILayer* outputLayer;

    if (activation == "relu")
    {
        outputLayer = network.addActivation(*actInput, ActivationType::kRELU);
        outputLayer->setName((name + ".relu").c_str());
    }
    else if (activation == "tanh")
    {
        outputLayer = network.addActivation(*actInput, ActivationType::kTANH);
        outputLayer->setName((name + ".tanh").c_str());
    }
    else if (activation == "none")
    {
        outputLayer = scaleLayer;
    }
    else
    {
        throw std::runtime_error("Unknown activation '" + activation + "'.");
    }

    return outputLayer;
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

std::vector<float>& ConvBatchNormCreator::newVector(const float* const begin, const float* const end)
{
    mData.emplace_back(new std::vector<float>(begin, end));

    return *mData.back().get();
}

} // namespace tts
