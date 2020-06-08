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

#include "lstm.h"

#include "NvInfer.h"

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

ILayer* LSTM::addPaddedBidirectional(INetworkDefinition* const network, ITensor* const input,
    ITensor* const inputLength, const int numDimensions, const LayerData& lstmData)
{
    // build LSTM
    const int hiddenSize = numDimensions / 2;
    IRNNv2Layer* lstm = network->addRNNv2(*input, 1, hiddenSize, input->getDimensions().d[1], RNNOperation::kLSTM);
    lstm->setDirection(RNNDirection::kBIDIRECTION);
    lstm->setSequenceLengths(*inputLength);

    {
        const int64_t inputBlockSize = numDimensions * hiddenSize;

        // pytorch weights are stored in "weight_ih_l0" = {W_ii|W_if|W_ig|W_io}
        const float* inputWeights = (const float*) lstmData.get("weight_ih_l0").values;
        Weights wii{DataType::kFLOAT, (void*) (inputWeights), inputBlockSize};
        Weights wif{DataType::kFLOAT, (void*) (inputWeights + inputBlockSize), inputBlockSize};
        Weights wig{DataType::kFLOAT, (void*) (inputWeights + 2 * inputBlockSize), inputBlockSize};
        Weights wio{DataType::kFLOAT, (void*) (inputWeights + 3 * inputBlockSize), inputBlockSize};

        lstm->setWeightsForGate(0, RNNGateType::kINPUT, true, wii);
        lstm->setWeightsForGate(0, RNNGateType::kCELL, true, wig);
        lstm->setWeightsForGate(0, RNNGateType::kFORGET, true, wif);
        lstm->setWeightsForGate(0, RNNGateType::kOUTPUT, true, wio);

        const float* inputBias = (const float*) lstmData.get("bias_ih_l0").values;
        Weights bii{DataType::kFLOAT, (void*) (inputBias), hiddenSize};
        Weights bif{DataType::kFLOAT, (void*) (inputBias + hiddenSize), hiddenSize};
        Weights big{DataType::kFLOAT, (void*) (inputBias + 2 * hiddenSize), hiddenSize};
        Weights bio{DataType::kFLOAT, (void*) (inputBias + 3 * hiddenSize), hiddenSize};

        lstm->setBiasForGate(0, RNNGateType::kINPUT, true, bii);
        lstm->setBiasForGate(0, RNNGateType::kCELL, true, big);
        lstm->setBiasForGate(0, RNNGateType::kFORGET, true, bif);
        lstm->setBiasForGate(0, RNNGateType::kOUTPUT, true, bio);

        const int64_t hiddenBlockSize = hiddenSize * hiddenSize;

        // pytorch weights are stored in "weight_hh_l0" = {W_hi|W_hf|W_hg|W_ho}
        const float* hiddenWeights = (const float*) lstmData.get("weight_hh_l0").values;
        Weights whi{DataType::kFLOAT, (void*) (hiddenWeights), hiddenBlockSize};
        Weights whf{DataType::kFLOAT, (void*) (hiddenWeights + hiddenBlockSize), hiddenBlockSize};
        Weights whg{DataType::kFLOAT, (void*) (hiddenWeights + 2 * hiddenBlockSize), hiddenBlockSize};
        Weights who{DataType::kFLOAT, (void*) (hiddenWeights + 3 * hiddenBlockSize), hiddenBlockSize};

        lstm->setWeightsForGate(0, RNNGateType::kINPUT, false, whi);
        lstm->setWeightsForGate(0, RNNGateType::kCELL, false, whg);
        lstm->setWeightsForGate(0, RNNGateType::kFORGET, false, whf);
        lstm->setWeightsForGate(0, RNNGateType::kOUTPUT, false, who);

        const float* hiddenBias = (const float*) lstmData.get("bias_hh_l0").values;
        Weights bhi{DataType::kFLOAT, (void*) (hiddenBias), hiddenSize};
        Weights bhf{DataType::kFLOAT, (void*) (hiddenBias + hiddenSize), hiddenSize};
        Weights bhg{DataType::kFLOAT, (void*) (hiddenBias + 2 * hiddenSize), hiddenSize};
        Weights bho{DataType::kFLOAT, (void*) (hiddenBias + 3 * hiddenSize), hiddenSize};

        lstm->setBiasForGate(0, RNNGateType::kINPUT, false, bhi);
        lstm->setBiasForGate(0, RNNGateType::kCELL, false, bhg);
        lstm->setBiasForGate(0, RNNGateType::kFORGET, false, bhf);
        lstm->setBiasForGate(0, RNNGateType::kOUTPUT, false, bho);
    }

    {
        const int64_t inputBlockSize = numDimensions * hiddenSize;

        // pytorch weights are stored in "weight_ih_l0" = {W_ii|W_if|W_ig|W_io}
        const float* inputWeights = (const float*) lstmData.get("weight_ih_l0_reverse").values;
        Weights wii{DataType::kFLOAT, (void*) (inputWeights), inputBlockSize};
        Weights wif{DataType::kFLOAT, (void*) (inputWeights + inputBlockSize), inputBlockSize};
        Weights wig{DataType::kFLOAT, (void*) (inputWeights + 2 * inputBlockSize), inputBlockSize};
        Weights wio{DataType::kFLOAT, (void*) (inputWeights + 3 * inputBlockSize), inputBlockSize};

        lstm->setWeightsForGate(1, RNNGateType::kINPUT, true, wii);
        lstm->setWeightsForGate(1, RNNGateType::kCELL, true, wig);
        lstm->setWeightsForGate(1, RNNGateType::kFORGET, true, wif);
        lstm->setWeightsForGate(1, RNNGateType::kOUTPUT, true, wio);

        const float* inputBias = (const float*) lstmData.get("bias_ih_l0_reverse").values;
        Weights bii{DataType::kFLOAT, (void*) (inputBias), hiddenSize};
        Weights bif{DataType::kFLOAT, (void*) (inputBias + hiddenSize), hiddenSize};
        Weights big{DataType::kFLOAT, (void*) (inputBias + 2 * hiddenSize), hiddenSize};
        Weights bio{DataType::kFLOAT, (void*) (inputBias + 3 * hiddenSize), hiddenSize};

        lstm->setBiasForGate(1, RNNGateType::kINPUT, true, bii);
        lstm->setBiasForGate(1, RNNGateType::kCELL, true, big);
        lstm->setBiasForGate(1, RNNGateType::kFORGET, true, bif);
        lstm->setBiasForGate(1, RNNGateType::kOUTPUT, true, bio);

        const int64_t hiddenBlockSize = hiddenSize * hiddenSize;

        // pytorch weights are stored in "weight_hh_l0" = {W_hi|W_hf|W_hg|W_ho}
        const float* hiddenWeights = (const float*) lstmData.get("weight_hh_l0_reverse").values;
        Weights whi{DataType::kFLOAT, (void*) (hiddenWeights), hiddenBlockSize};
        Weights whf{DataType::kFLOAT, (void*) (hiddenWeights + hiddenBlockSize), hiddenBlockSize};
        Weights whg{DataType::kFLOAT, (void*) (hiddenWeights + 2 * hiddenBlockSize), hiddenBlockSize};
        Weights who{DataType::kFLOAT, (void*) (hiddenWeights + 3 * hiddenBlockSize), hiddenBlockSize};

        lstm->setWeightsForGate(1, RNNGateType::kINPUT, false, whi);
        lstm->setWeightsForGate(1, RNNGateType::kCELL, false, whg);
        lstm->setWeightsForGate(1, RNNGateType::kFORGET, false, whf);
        lstm->setWeightsForGate(1, RNNGateType::kOUTPUT, false, who);

        const float* hiddenBias = (const float*) lstmData.get("bias_hh_l0_reverse").values;
        Weights bhi{DataType::kFLOAT, (void*) (hiddenBias), hiddenSize};
        Weights bhf{DataType::kFLOAT, (void*) (hiddenBias + hiddenSize), hiddenSize};
        Weights bhg{DataType::kFLOAT, (void*) (hiddenBias + 2 * hiddenSize), hiddenSize};
        Weights bho{DataType::kFLOAT, (void*) (hiddenBias + 3 * hiddenSize), hiddenSize};

        lstm->setBiasForGate(1, RNNGateType::kINPUT, false, bhi);
        lstm->setBiasForGate(1, RNNGateType::kCELL, false, bhg);
        lstm->setBiasForGate(1, RNNGateType::kFORGET, false, bhf);
        lstm->setBiasForGate(1, RNNGateType::kOUTPUT, false, bho);
    }

    return lstm;
}

ILayer* LSTM::addUnidirectionalCell(INetworkDefinition* const network, ITensor* const input,
    ITensor* const hiddenStatesIn, ITensor* const cellStatesIn, const int numDimensions, const LayerData& lstmData)
{
    // build LSTM
    const int hiddenSize = numDimensions;
    const int inputLength = input->getDimensions().d[2];
    IRNNv2Layer* lstm = network->addRNNv2(*input, 1, hiddenSize, input->getDimensions().d[1], RNNOperation::kLSTM);
    lstm->setDirection(RNNDirection::kUNIDIRECTION);

    const int64_t inputBlockSize = inputLength * hiddenSize;

    // pytorch weights are stored in "weight_ih" = {W_ii|W_if|W_ig|W_io}
    const float* inputWeights = (const float*) lstmData.get("weight_ih").values;
    Weights wii{DataType::kFLOAT, (void*) (inputWeights), inputBlockSize};
    Weights wif{DataType::kFLOAT, (void*) (inputWeights + inputBlockSize), inputBlockSize};
    Weights wig{DataType::kFLOAT, (void*) (inputWeights + 2 * inputBlockSize), inputBlockSize};
    Weights wio{DataType::kFLOAT, (void*) (inputWeights + 3 * inputBlockSize), inputBlockSize};

    lstm->setWeightsForGate(0, RNNGateType::kINPUT, true, wii);
    lstm->setWeightsForGate(0, RNNGateType::kCELL, true, wig);
    lstm->setWeightsForGate(0, RNNGateType::kFORGET, true, wif);
    lstm->setWeightsForGate(0, RNNGateType::kOUTPUT, true, wio);

    const float* inputBias = (const float*) lstmData.get("bias_ih").values;
    Weights bii{DataType::kFLOAT, (void*) (inputBias), hiddenSize};
    Weights bif{DataType::kFLOAT, (void*) (inputBias + hiddenSize), hiddenSize};
    Weights big{DataType::kFLOAT, (void*) (inputBias + 2 * hiddenSize), hiddenSize};
    Weights bio{DataType::kFLOAT, (void*) (inputBias + 3 * hiddenSize), hiddenSize};

    lstm->setBiasForGate(0, RNNGateType::kINPUT, true, bii);
    lstm->setBiasForGate(0, RNNGateType::kCELL, true, big);
    lstm->setBiasForGate(0, RNNGateType::kFORGET, true, bif);
    lstm->setBiasForGate(0, RNNGateType::kOUTPUT, true, bio);

    const int64_t hiddenBlockSize = hiddenSize * hiddenSize;

    // pytorch weights are stored in "weight_hh" = {W_hi|W_hf|W_hg|W_ho}
    const float* hiddenWeights = (const float*) lstmData.get("weight_hh").values;
    Weights whi{DataType::kFLOAT, (void*) (hiddenWeights), hiddenBlockSize};
    Weights whf{DataType::kFLOAT, (void*) (hiddenWeights + hiddenBlockSize), hiddenBlockSize};
    Weights whg{DataType::kFLOAT, (void*) (hiddenWeights + 2 * hiddenBlockSize), hiddenBlockSize};
    Weights who{DataType::kFLOAT, (void*) (hiddenWeights + 3 * hiddenBlockSize), hiddenBlockSize};

    lstm->setWeightsForGate(0, RNNGateType::kINPUT, false, whi);
    lstm->setWeightsForGate(0, RNNGateType::kCELL, false, whg);
    lstm->setWeightsForGate(0, RNNGateType::kFORGET, false, whf);
    lstm->setWeightsForGate(0, RNNGateType::kOUTPUT, false, who);

    const float* hiddenBias = (const float*) lstmData.get("bias_hh").values;
    Weights bhi{DataType::kFLOAT, (void*) (hiddenBias), hiddenSize};
    Weights bhf{DataType::kFLOAT, (void*) (hiddenBias + hiddenSize), hiddenSize};
    Weights bhg{DataType::kFLOAT, (void*) (hiddenBias + 2 * hiddenSize), hiddenSize};
    Weights bho{DataType::kFLOAT, (void*) (hiddenBias + 3 * hiddenSize), hiddenSize};

    lstm->setBiasForGate(0, RNNGateType::kINPUT, false, bhi);
    lstm->setBiasForGate(0, RNNGateType::kCELL, false, bhg);
    lstm->setBiasForGate(0, RNNGateType::kFORGET, false, bhf);
    lstm->setBiasForGate(0, RNNGateType::kOUTPUT, false, bho);

    lstm->setHiddenState(*hiddenStatesIn);
    lstm->setCellState(*cellStatesIn);

    return lstm;
}

} // namespace tts
