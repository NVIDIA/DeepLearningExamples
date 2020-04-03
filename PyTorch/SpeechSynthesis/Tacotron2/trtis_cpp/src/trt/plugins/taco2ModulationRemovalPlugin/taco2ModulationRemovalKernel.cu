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

#include "taco2ModulationRemovalKernel.h"
#include "taco2Utils.h"

#include <algorithm>
#include <cassert>
#include <cfloat>

namespace nvinfer1
{
namespace plugin
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int WINDOW_SIZE = 1024;
} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

__global__ void modulationRemovalKernel(const int batchSize, const float* const weightsDevice,
    const float* const inputDevice, float* const outputDevice, const int inputLength, const int hopLength,
    const float scale)
{
    // load weights into shared memory
    __shared__ float localWeights[WINDOW_SIZE];
    for (int i = threadIdx.x; i < WINDOW_SIZE; i += blockDim.x)
    {
        localWeights[i] = weightsDevice[i];
    }

    __syncthreads();

    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < inputLength - WINDOW_SIZE)
    {
        const int inIdx = idx + (WINDOW_SIZE / 2);
        // start the window over the first overlap, and slide it until the last
        // overlap for this point
        float sum = 0.0f;
        const int windowOffset = inIdx % hopLength;
        for (int j = windowOffset; j < WINDOW_SIZE; j += hopLength)
        {
            if (inIdx - j >= 0)
            {
                sum += localWeights[j];
            }
        }

        // normal all non-zero values
        for (int i = 0; i < batchSize; ++i)
        {
            float val = inputDevice[inIdx + inputLength * i];
            if (sum > FLT_MIN)
            {
                val /= sum;
            }

            val *= scale;

            outputDevice[idx + (inputLength - WINDOW_SIZE) * i] = val;
        }
    }
}

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void Taco2ModulationRemovalKernel::compute(const int batchSize, const float* const weightsDevice,
    const float* const inputDevice, float* const outputDevice, const int inputLength, const int filterLength,
    const int hopLength, cudaStream_t stream)
{
    assert(filterLength == WINDOW_SIZE);

    const dim3 grid(taco2::Taco2Utils::roundUpBlocks(inputLength - filterLength, WINDOW_SIZE));
    const dim3 block(WINDOW_SIZE);

    modulationRemovalKernel<<<grid, block, 0, stream>>>(batchSize, weightsDevice, inputDevice, outputDevice,
        inputLength, hopLength, static_cast<float>(filterLength) / static_cast<float>(hopLength));
}

} // namespace plugin
} // namespace nvinfer1
