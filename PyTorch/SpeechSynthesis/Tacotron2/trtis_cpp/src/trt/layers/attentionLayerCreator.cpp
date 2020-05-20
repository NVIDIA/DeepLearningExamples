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

#include "attentionLayerCreator.h"
#include "dims5.h"
#include "layerData.h"

#include "NvInfer.h"

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

ILayer* AttentionLayerCreator::addLocation(INetworkDefinition& network, ITensor* const input, const int attentionDim,
    const int numFilters, const int kernelSize, const LayerData& convData, const LayerData& linearData,
    const std::string& name)
{
    // conv layer
    const int padding = (kernelSize - 1) / 2;
    #if NV_TENSORRT_MAJOR < 7
    IConvolutionLayer* const convLayer = network.addConvolution(
        *input, numFilters, DimsHW{kernelSize, 1}, convData.get("weight"), {DataType::kFLOAT, nullptr, 0});
    convLayer->setPadding({padding, 0});
    #else
    IConvolutionLayer* const convLayer = network.addConvolutionNd(
        *input, numFilters, Dims2(kernelSize, 1), convData.get("weight"), {DataType::kFLOAT, nullptr, 0});
    convLayer->setPaddingNd(Dims2(padding, 0));
    #endif
    convLayer->setName((name + ".conv_layer").c_str());

    // need to tranpose
    IShuffleLayer* const transLayer = network.addShuffle(*convLayer->getOutput(0));
    transLayer->setFirstTranspose({0, 2, 1, 3});
    transLayer->setReshapeDimensions(Dims5{1, convLayer->getOutput(0)->getDimensions().d[2],
        convLayer->getOutput(0)->getDimensions().d[1], 1, convLayer->getOutput(0)->getDimensions().d[3]});
    transLayer->setName((name + ".transpose").c_str());

    // fully connected layer
    ILayer* const linearLayer = network.addFullyConnected(
        *transLayer->getOutput(0), attentionDim, linearData.get("weight"), Weights{DataType::kFLOAT, 0, 0});
    linearLayer->setName((name + ".linear_layer").c_str());
    return linearLayer;
}

ILayer* AttentionLayerCreator::addEnergy(INetworkDefinition& network, ITensor* const input1, ITensor* const input2,
    ITensor* const input3, const LayerData& linearData, const std::string& name)
{
    // summation
    ILayer* const add1Layer = network.addElementWise(*input1, *input2, ElementWiseOperation::kSUM);
    add1Layer->setName((name + ".0.elementwise_sum").c_str());
    ILayer* const add2Layer = network.addElementWise(*add1Layer->getOutput(0), *input3, ElementWiseOperation::kSUM);
    add2Layer->setName((name + ".1.elementwise_sum").c_str());

    // activation
    ILayer* const actLayer = network.addActivation(*add2Layer->getOutput(0), ActivationType::kTANH);
    actLayer->setName((name + ".tanh").c_str());

    // fully connected layer
    ILayer* const linearLayer = network.addFullyConnected(
        *actLayer->getOutput(0), 1, linearData.get("weight"), Weights{DataType::kFLOAT, 0, 0});
    linearLayer->setName((name + ".linear_layer").c_str());
    return linearLayer;
}

ILayer* AttentionLayerCreator::addPaddedSoftMax(INetworkDefinition& network, ITensor* const input,
    ITensor* const inputMask, ITensor* const inputSegments, const std::string& name)
{
    // make our inputs 2 dimensional
    IShuffleLayer* const maskShuffleLayer = network.addShuffle(*inputMask);
    maskShuffleLayer->setReshapeDimensions(Dims2{1, -1});
    maskShuffleLayer->setName((name + ".mask_reshape").c_str());

    IShuffleLayer* const inputShuffleLayer = network.addShuffle(*input);
    inputShuffleLayer->setReshapeDimensions(Dims2{1, -1});
    inputShuffleLayer->setName((name + ".input_reshape").c_str());

    // perform softmax over non-padding elements
    ILayer* const softMaxLayer = network.addRaggedSoftMax(*inputShuffleLayer->getOutput(0), *inputSegments);
    softMaxLayer->setName((name + ".ragged_softmax").c_str());

    // zero padding
    ILayer* const maskLayer = network.addElementWise(
        *softMaxLayer->getOutput(0), *maskShuffleLayer->getOutput(0), ElementWiseOperation::kPROD);
    maskLayer->setName((name + ".mask").c_str());

    // return three dimensional output
    IShuffleLayer* const outShuffle = network.addShuffle(*maskLayer->getOutput(0));
    outShuffle->setReshapeDimensions(Dims3{-1, 1, 1});
    outShuffle->setName((name + ".transpose").c_str());

    return outShuffle;
}

} // namespace tts
