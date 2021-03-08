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

#include "taco2ProjectionKernel.h"

#include "cuda_runtime.h"

#include <stdexcept>
#include <string>

using namespace tts;

namespace nvinfer1
{
namespace plugin
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int PROJECTION_COL_SIZE = 512;
constexpr const int WARP_SIZE = 32;
} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

template <typename T, int NUM_THREADS>
__device__ inline T warpSum(T const initVal)
{
    constexpr const uint32_t mask = 0xffffffff >> (WARP_SIZE - NUM_THREADS);
    T val = initVal;
#pragma unroll
    for (int d = NUM_THREADS / 2; d > 0; d /= 2)
    {
        val += __shfl_down_sync(mask, val, d, NUM_THREADS);
    }

    return val;
}

template <typename T, int BLOCK_SIZE>
__device__ T cooperativeSum(T const initVal, T* const buffer)
{
    // first all warps reduce to single value
    assert(BLOCK_SIZE % WARP_SIZE == 0);
    assert(BLOCK_SIZE <= WARP_SIZE * WARP_SIZE);

    T val = warpSum<T, WARP_SIZE>(initVal);
    if (threadIdx.x % WARP_SIZE == 0)
    {
        buffer[threadIdx.x / WARP_SIZE] = val;
    }
    __syncthreads();

    if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE))
    {
        val = warpSum<T, BLOCK_SIZE / WARP_SIZE>(buffer[threadIdx.x]);
    }

    return val;
}

__device__ inline void sumReduce(float* const array, const int len)
{
    for (int d = 1; d < blockDim.x; d *= 2)
    {
        if (threadIdx.x % (d * 2) == 0 && threadIdx.x + d < len)
        {
            array[threadIdx.x] += array[threadIdx.x + d];
        }
        __syncthreads();
    }
}

template <int INPUT_1_LENGTH, int INPUT_2_LENGTH>
__global__ void projectionKernel(const float* const weights, const float* const bias, const float* const input1,
    const float* const input2, float* const output)
{
    __shared__ float shared[PROJECTION_COL_SIZE];

    // perform mat vec
    float v = 0.0f;
    constexpr const int inputLength = INPUT_1_LENGTH + INPUT_2_LENGTH;
    for (int col = threadIdx.x; col < INPUT_1_LENGTH; col += PROJECTION_COL_SIZE)
    {
        // load chunk
        if (col < INPUT_1_LENGTH)
        {
            v += input1[col] * weights[blockIdx.x * inputLength + col];
        }
    }

    for (int col = threadIdx.x; col < INPUT_2_LENGTH; col += PROJECTION_COL_SIZE)
    {
        // load chunk
        if (col < INPUT_2_LENGTH)
        {
            v += input2[col] * weights[blockIdx.x * inputLength + (col + INPUT_1_LENGTH)];
        }
    }

    v = cooperativeSum<float, PROJECTION_COL_SIZE>(v, shared);

    // add bias and write
    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = bias[blockIdx.x] + v;
    }
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2ProjectionKernel::Taco2ProjectionKernel(const std::vector<float>& fcWeightsHost,
    const std::vector<float>& fcBiasHost, const int input1Length, const int input2Length, const int numDimension)
    : mInput1Length(input1Length)
    , mInput2Length(input2Length)
    , mInputLength(input1Length + input2Length)
    , mNumDimension(numDimension)
    , mWeightsDevice()
    , mBiasDevice()
{
    const size_t numExpectedWeights = mInputLength * mNumDimension;
    const size_t numExpectedBias = mNumDimension;

    if (numExpectedWeights != fcWeightsHost.size())
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedWeights) + " weights for FC but got "
            + std::to_string(fcWeightsHost.size()) + " instead.");
    }
    if (numExpectedBias != fcBiasHost.size())
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedBias) + " biases for FC but got "
            + std::to_string(fcBiasHost.size()) + " instead.");
    }

    // copy up weights to GPU in row major and concatenated
    mWeightsDevice = CudaMemory<float>(fcWeightsHost);
    mBiasDevice = CudaMemory<float>(fcBiasHost);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Taco2ProjectionKernel::execute(
    const float* input1Device, const float* input2Device, float* outputDevice, cudaStream_t stream)
{
    const dim3 grid(mNumDimension);
    const dim3 block(PROJECTION_COL_SIZE);

    if (mInput1Length != 1024)
    {
        throw std::runtime_error(
            "Plugin is configured to only handle hidden "
            "input length of 1024, but got "
            + std::to_string(mInput1Length));
    }
    if (mInput2Length != 512)
    {
        throw std::runtime_error(
            "Plugin is configured to only handle context "
            "input length of 512, but got "
            + std::to_string(mInput1Length));
    }

    projectionKernel<1024, 512><<<grid, block, 0, stream>>>(
        mWeightsDevice.data(),
        mBiasDevice.data(),
        input1Device,
        input2Device,
        outputDevice);
}

} // namespace plugin
} // namespace nvinfer1
