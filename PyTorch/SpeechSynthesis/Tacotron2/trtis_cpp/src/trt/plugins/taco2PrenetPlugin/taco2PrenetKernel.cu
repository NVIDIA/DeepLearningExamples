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

#include "taco2PrenetKernel.h"
#include "taco2Utils.h"

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
constexpr const int PRENET_ROW_SIZE = 8;
constexpr const int PRENET_COL_SIZE = 32;
} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

__device__ inline float relu(const float a)
{
    return a * (a > 0.0f);
}

template <typename T, int NUM_THREADS>
__device__ inline T warpSum(T const initVal)
{
    constexpr const uint32_t mask = NUM_THREADS < 32 ? (1u << NUM_THREADS) - 1 : 0xffffffff;
    T val = initVal;
#pragma unroll
    for (int d = NUM_THREADS / 2; d > 0; d /= 2)
    {
        val += __shfl_down_sync(mask, val, d, NUM_THREADS);
    }

    return val;
}

template <int ROW_SIZE, int COL_SIZE, int INPUT_LENGTH>
__global__ void prenetKernel(const float* const weights, const float* const inputFrame, const float* const dropout,
    float* const output, const int numDimension)
{
    __shared__ float shared[ROW_SIZE * COL_SIZE];

    constexpr const int blockSize = ROW_SIZE * COL_SIZE;
    const int tid = threadIdx.x + threadIdx.y * COL_SIZE;

    assert(numDimension % ROW_SIZE == 0);

    float v = 0.0f;
    { // perform mat vec
        const int row = threadIdx.y + blockIdx.y * ROW_SIZE;
        for (int colStart = 0; colStart < INPUT_LENGTH; colStart += blockSize)
        {
            // load chunk
            if (colStart + tid < INPUT_LENGTH)
            {
                shared[tid] = inputFrame[colStart + tid];
            }

            __syncthreads();

            for (int col = threadIdx.x; col < blockSize; col += COL_SIZE)
            {
                if (col + colStart < INPUT_LENGTH)
                {
                    v += shared[col] * weights[row * INPUT_LENGTH + (col + colStart)];
                }
            }
            __syncthreads();
        }

        v = warpSum<float, COL_SIZE>(v);
    }

    if (threadIdx.x % COL_SIZE == 0)
    {
        shared[threadIdx.y] = v;
    }
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < ROW_SIZE)
    {
        const int row = threadIdx.x + blockIdx.y * ROW_SIZE;
        const float a = relu(shared[threadIdx.x]);
        const float b = a * dropout[row];
        output[row] = b;
    }
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2PrenetKernel::Taco2PrenetKernel(const std::vector<float>& fc1WeightsHost, const std::vector<float>& fc2WeightsHost,
    const int inputLength, const int numDimension)
    : mInputLength(inputLength)
    , mNumDimension(numDimension)
    , mWeights1Device()
    , mWeights2Device()
{
    const size_t numExpectedWeights1 = mInputLength * mNumDimension;
    const size_t numExpectedWeights2 = mNumDimension * mNumDimension;

    if (numExpectedWeights1 != fc1WeightsHost.size())
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedWeights1) + " weights for FC1 but get "
            + std::to_string(fc1WeightsHost.size()) + " instead.");
    }
    if (numExpectedWeights2 != fc2WeightsHost.size())
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedWeights2) + " weights for FC2 but get "
            + std::to_string(fc2WeightsHost.size()) + " instead.");
    }

    // copy up weights to GPU in column major and concatenated
    {
        std::vector<float> trans1(mInputLength * mNumDimension);
        for (int i = 0; i < mNumDimension; ++i)
        {
            for (int j = 0; j < mInputLength; ++j)
            {
                trans1[i * mInputLength + j] = fc1WeightsHost[i * mInputLength + j];
            }
        }
        mWeights1Device = CudaMemory<float>(trans1);
    }

    {
        std::vector<float> trans2(mNumDimension * mNumDimension);
        for (int i = 0; i < mNumDimension; ++i)
        {
            for (int j = 0; j < mNumDimension; ++j)
            {
                trans2[i * mNumDimension + j] = fc2WeightsHost[i * mNumDimension + j];
            }
        }
        mWeights2Device = CudaMemory<float>(trans2);
    }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Taco2PrenetKernel::execute(const float* inputDevice, const float* dropoutDevice, float* outputDevice,
    float* workspaceDevice, cudaStream_t stream)
{
    const int numBlocks = taco2::Taco2Utils::roundUpBlocks(mNumDimension, PRENET_ROW_SIZE);

    const dim3 grid(1, numBlocks);
    const dim3 block(PRENET_COL_SIZE, PRENET_ROW_SIZE);

    assert(mInputLength == 80);
    prenetKernel<PRENET_ROW_SIZE, PRENET_COL_SIZE, 80>
        <<<grid, block, 0, stream>>>(
            mWeights1Device.data(),
            inputDevice,
            dropoutDevice,
            workspaceDevice,
            mNumDimension);

    assert(mNumDimension == 256);
    prenetKernel<PRENET_ROW_SIZE, PRENET_COL_SIZE, 256>
        <<<grid, block, 0, stream>>>(
            mWeights2Device.data(),
            workspaceDevice,
            dropoutDevice,
            outputDevice,
            mNumDimension);
}

} // namespace plugin
} // namespace nvinfer1
