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

#ifndef TT2I_LSTM_H
#define TT2I_LSTM_H

#include "layerData.h"

namespace nvinfer1
{
class INetworkDefinition;
class ITensor;
class ILayer;
} // namespace nvinfer1

namespace tts
{

class LSTM
{
public:
    /**
     * @brief Add a new bidirection LSTM layer to the network with padding at the
     * end of the sequence, and with a number of
     * hidden layers equal to half the number of output layers.
     *
     * @param network The network to add to.
     * @param input The input tensor.
     * @param inputLength The length of each input sequence.
     * @param numDimensions The number of output dimensions of the LSTM.
     * @param lstmData The LSTM weights (must be in
     * scope until the network is finished building).
     * @param name The name to prefix the layers with.
     *
     * @return The last of the newly added layrs.
     */
    static nvinfer1::ILayer* addPaddedBidirectional(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
        nvinfer1::ITensor* inputLength, int numDimensions, const LayerData& lstmData);

    /**
     * @brief Add a new unidirection LSTM layer to the network, with a number of
     * hidden layers equal to half the number of output layers.
     *
     * @param network The network to add to.
     * @param input The input tensor.
     * @param input The input hidden states.
     * @param input The input cell states.
     * @param numDimensions The number of output dimensions of the LSTM.
     * @param lstmData The LSTM weights (must be in
     * scope until the network is finished building).
     *
     * @return The last of the newly added layrs.
     */
    static nvinfer1::ILayer* addUnidirectionalCell(nvinfer1::INetworkDefinition* network, nvinfer1::ITensor* input,
        nvinfer1::ITensor* hiddenStatesIn, nvinfer1::ITensor* cellStatesIn, int numDimensions,
        const LayerData& lstmData);
};

} // namespace tts

#endif
