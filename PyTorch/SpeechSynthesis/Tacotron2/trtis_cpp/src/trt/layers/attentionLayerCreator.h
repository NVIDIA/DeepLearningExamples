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

#ifndef TT2I_COMPOSITELAYERS_H
#define TT2I_COMPOSITELAYERS_H

#include <string>
#include <vector>

namespace nvinfer1
{
class INetworkDefinition;
class ITensor;
class ILayer;
} // namespace nvinfer1

namespace tts
{

class LayerData;

class AttentionLayerCreator
{
public:
    /**
     * @brief Add a location layer to the given network.
     *
     * @param network The network to add to.
     * @param input The input tensor.
     * @param attentionDim The number of dimensions.
     * @param numFilters The number of filters
     * @param kernelSize The size of each kernel.
     * @param convData The convolution data.
     * @param linearData The linear data for the fully connected layer.
     * @param name The name to prefix the layers with.
     *
     * @return The last of the newly added layers.
     */
    static nvinfer1::ILayer* addLocation(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* input,
        int attentionDim, int numFilters, int kernelSize, const LayerData& convData, const LayerData& linearData,
        const std::string& name);

    /**
     * @brief Add an energy layer to the given network.
     *
     * @param network The network.
     * @param input1 The first input to be summed.
     * @param input2 The second input to be summed.
     * @param input3 The third input to be summed.
     * @param linearData The data for the fully connected layer.
     * @param name The name to prefix layers with.
     *
     * @return The last layer of the newly added layers.
     */
    static nvinfer1::ILayer* addEnergy(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* input1,
        nvinfer1::ITensor* input2, nvinfer1::ITensor* input3, const LayerData& linearData, const std::string& name);

    /**
     * @brief Perform a softmax on padded input.
     *
     * @param network The network being built.
     * @param input The padded input.
     * @param inputMask The mask.
     * @param inputSegments The length of the input.
     * @param name The name to prefix the layers with.
     *
     * @return The last layer.
     */
    static nvinfer1::ILayer* addPaddedSoftMax(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* input,
        nvinfer1::ITensor* inputMask, nvinfer1::ITensor* inputSegments, const std::string& name);
};

} // namespace tts

#endif
