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

#ifndef TT2I_LSTMCELLKERNEL_H
#define TT2I_LSTMCELLKERNEL_H

#include "cudaMemory.h"

#include "cuda_runtime.h"

namespace nvinfer1
{
namespace plugin
{

class Taco2LSTMCellKernel
{
public:
    /**
     * @brief Create a new Taco2LSTMCellKernel.
     *
     * @param inputWeightsHost The weight matrix for the input (Wi).
     * @param hiddenWeightsHost The weight matrix for the hidden states (Wh).
     * @param inputBiasHost The input bias (Bi).
     * @param hiddenBiasHost The hidden bias (Bh).
     * @param inputLength The length of the input.
     * @param numDimension The number of hidden dimensions.
     * @param useFP16 Whether or not to use fp16 format weights.
     */
    Taco2LSTMCellKernel(const float* inputWeightsHost, const float* hiddenWeightsHost, const float* inputBiasHost,
        const float* hiddenBiasHost, const int inputLength, const int numDimension, bool useFP16);

    /**
     * @brief Execute an LSTM cell.
     *
     * @param inputA The first half of the input vector.
     * @param inputB The second half of the input vector.
     * @param hiddenIn The hidden states (input).
     * @param cellIn The cell states (input).
     * @param hiddenOut The hidden states (output).
     * @param cellOut The cell states (output).
     * @param inputLengthA The length of the first input.
     * @param inputLengthB The length of the second input.
     * @param numDimensions The number of dimensions.
     * @param stream The stream to execute on.
     */
    void execute(const float* inputA, const float* inputB, const float* hiddenIn, const float* cellIn, float* hiddenOut,
        float* cellOut, int inputLengthA, int inputLengthB, cudaStream_t stream);

private:
    int mInputLength;
    int mNumDimension;

    bool mFp16;

    tts::CudaMemory<float> mWeightsDevice;
    tts::CudaMemory<float> mBiasDevice;
};

} // namespace plugin
} // namespace nvinfer1

#endif
