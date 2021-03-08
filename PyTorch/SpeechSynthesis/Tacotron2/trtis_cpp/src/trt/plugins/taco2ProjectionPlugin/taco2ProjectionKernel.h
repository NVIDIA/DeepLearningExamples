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

#ifndef TT2I_PROJECTIONKERNEL_H
#define TT2I_PROJECTIONKERNEL_H

#include "cudaMemory.h"

#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Taco2ProjectionKernel
{
public:
    /**
     * @brief Create a new Taco2ProjectionKernel.
     *
     * @param fc1WeightsHost The weights of the first fully connected layer.
     * @param fc2WeightsHost The weights of the second fully connected layer.
     * @param inputLength The length of the input.
     * @param numDimension The number of dimensions of the FC layers.
     */
    Taco2ProjectionKernel(const std::vector<float>& fcWeightsHost, const std::vector<float>& fcBiasHost,
        int inputLength1, int inputLength2, int numDimension);

    /**
     * @brief Execute this kernel.
     *
     * @param input1Device The first input on the device.
     * @param input2Device The second input on the device.
     * @param outputDevice THe output on the device.
     * @param scratchDevice The scratch space on the device.
     * @param stream The stream to operate on.
     */
    void execute(const float* input1Device, const float* input2Device, float* outputDevice, cudaStream_t stream);

private:
    int mInput1Length;
    int mInput2Length;
    int mInputLength;
    int mNumDimension;
    tts::CudaMemory<float> mWeightsDevice;
    tts::CudaMemory<float> mBiasDevice;
};

} // namespace plugin
} // namespace nvinfer1

#endif
