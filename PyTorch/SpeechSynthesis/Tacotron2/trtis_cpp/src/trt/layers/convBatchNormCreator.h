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

#ifndef TT2I_CONVBATCHNORMCREATOR_H
#define TT2I_CONVBATCHNORMCREATOR_H

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
class INetworkDefinition;
class ILayer;
class ITensor;
} // namespace nvinfer1

namespace tts
{

class LayerData;

class ConvBatchNormCreator
{
public:
    /**
     * @brief Add a 1d-convolution plus batch normalization followed by
     * activation to the network,
     * where the convolution has kernel size of 5, and padding 2 (to preserve
     * shape).
     * ```
     * y = conv(x)
     * z = ( (y-Mean[y]) / sqrt(Var[y]+eps) ) * weight + bias
     * ```
     *
     * WARNING: This sets pointers from the network to this object's members,
     * and so this object must not be destroyed or moved while until after the
     * lifetime of the network has ended.
     *
     * @param network The network to add to.
     * @param input The input tensor.
     * @param convData The LayerData object that has `weight` and `bias` for the
     * convolution.
     * @param normData The LayerData object that has `running_mean`,
     * `running_var`, `weight`, and `bias` entries for the batch norm.
     * @param activation May be "relu", "tanh", or "none".
     * @param name The name to prefix the layers with.
     *
     * @return The last of the newly added layers.
     */
    nvinfer1::ILayer* add(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor* input, const LayerData& convData,
        const LayerData& normData, const std::string& activation, const std::string& name);

private:
    std::vector<std::unique_ptr<std::vector<float>>> mData{};

    /**
     * @brief Create a new vector to be stored inside of this object.
     *
     * @param begin The starting iterator to initialize with.
     * @param end The ending iterator to initialize with.
     *
     * @return The vector.
     */
    std::vector<float>& newVector(const float* begin, const float* end);
};

} // namespace tts

#endif
