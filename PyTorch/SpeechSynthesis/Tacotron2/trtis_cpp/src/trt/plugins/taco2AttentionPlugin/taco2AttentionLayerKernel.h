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

#ifndef TT2I_ATTENTIONLAYERKERNEL_H
#define TT2I_ATTENTIONLAYERKERNEL_H

#include "cudaMemory.h"
#include "cuda_runtime.h"

#include "cublas_v2.h"
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Taco2AttentionLayerKernel
{
public:
    /**
     * @brief Create a new Taco2AttentionLayerKernel.
     *
     * @param queryWeights The query weights.
     * @param convWeights The convolution weights.
     * @param locationWeights The location weights.
     * @param energyWeights The energy weights.
     * @param encLength The encoding length.
     * @param queryDimension The number of query dimensions.
     * @param numFilters The number of convolution filters.
     * @param convKernelSize The convolution kernel size.
     * @param attDimension The number of attention dimensions.
     */
    Taco2AttentionLayerKernel(const std::vector<float>& queryWeights, const std::vector<float>& convWeights,
        const std::vector<float>& locationWeights, const std::vector<float>& energyWeights, int encLength,
        int queryDimension, int numFilters, int convKernelSize, int attDimension);

    // delete copy constructor and operator
    Taco2AttentionLayerKernel(const Taco2AttentionLayerKernel& other) = delete;
    Taco2AttentionLayerKernel& operator=(const Taco2AttentionLayerKernel& other) = delete;

    /**
     * @brief Destructor.
     */
    ~Taco2AttentionLayerKernel();

    /**
     * @brief Execute this kernel.
     *
     * @param memoryDevice The "Memory" tensor on the device.
     * @param processedMemoryDevice The "Processed Memory" tensor on the
     * device.
     * @param weightsDevice The "Weights" tensor for input on the device.
     * @param attentionHiddenDevice The hidden states from the attention LSTM
     * on the device.
     * @param outputContextDevice The attention context on the device to write
     * to.
     * @param outputWeightsDevice The "Weights" tensor to use as output.
     * @param inputLength The length of the input to process (number chars).
     * @param workspace The workspace.
     * @param stream The stream to operate on.
     */
    void execute(const float* memoryDevice, const float* processedMemoryDevice, const float* weightsDevice,
        const float* attentionHiddenDevice, float* const outputContextDevice, float* const outputWeightsDevice,
        const int inputLength, float* const workspace, cudaStream_t stream);

private:
    static const float ONE;
    static const float ZERO;

    int mNumEncodingDimension;
    int mNumQueryDimension;
    int mNumFilters;
    int mConvKernelSize;
    int mNumAttentionDimension;

    tts::CudaMemory<float> mQueryWeightsDevice;
    tts::CudaMemory<float> mConvWeightsDevice;
    tts::CudaMemory<float> mLocationWeightsDevice;
    tts::CudaMemory<float> mEnergyWeightsDevice;

    cublasHandle_t mCublasHandle;
};

} // namespace plugin
} // namespace nvinfer1

#endif
