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

#include "taco2LSTMCellKernel.h"
#include "taco2Utils.h"

#include "cuda_fp16.h"

#include <cassert>
#include <cmath>
#include <iostream>
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

constexpr const int BLOCK_COL_SIZE = 128;

// must be at least 4 to allow computation of i,f,g,o by a single block
constexpr const int BLOCK_ROWS_PER_THREAD = 4;

} // namespace

/******************************************************************************
 * CUDA KERNELS ***************************************************************
 *****************************************************************************/

__device__ inline float sigmoid(const float x)
{
    return 1.0f / (1.0f + exp(-x));
}

__device__ inline float dot2(const float2 a, const __half2 b)
{
    float2 bf = __half22float2(b);
    return a.x * bf.x + a.y * bf.y;
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

// template <typename T, int BLOCK_SIZE>
//__device__ T cooperativeSum(T const initVal, T* const buffer)
//{
//  // first all warps reduce to single value
//  assert(BLOCK_SIZE % WARP_SIZE == 0);
//  assert(BLOCK_SIZE <= WARP_SIZE * WARP_SIZE);
//
//  int val = warpSum<T, WARP_SIZE>(initVal);
//  if (threadIdx.x % WARP_SIZE == 0) {
//    buffer[threadIdx.x / WARP_SIZE] = val;
//  }
//  __syncthreads();
//
//  if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) {
//    val = warpSum<T, BLOCK_SIZE / WARP_SIZE>(buffer[threadIdx.x]);
//  }
//
//  return val;
//}

__device__ void sumBlock(float* const shared)
{
    constexpr const int chunkSize = BLOCK_COL_SIZE / BLOCK_ROWS_PER_THREAD;
    const int tid = threadIdx.x % chunkSize;
    const int chunkId = threadIdx.x / chunkSize;

    assert(chunkSize <= 32);

    float val = 0.0f;
#pragma unroll
    for (int i = tid; i < BLOCK_COL_SIZE; i += chunkSize)
    {
        val += shared[chunkId * BLOCK_COL_SIZE + i];
    }

    val = warpSum<float, chunkSize>(val);
    if (tid == 0)
    {
        shared[chunkId * BLOCK_COL_SIZE] = val;
    }
    __syncthreads();
}

template <int INPUT_LENGTH_A, int INPUT_LENGTH_B, int NUM_DIMENSIONS>
__global__ void lstmCellRowHalfKernel(const __half2* const weights, const float* const bias, const float2* const inputA,
    const float2* const inputB, const float2* const hiddenIn, const float* const cellIn, float* const hiddenOut,
    float* const cellOut)
{
    __shared__ float shared[BLOCK_COL_SIZE * BLOCK_ROWS_PER_THREAD];

    const int rowOffset = blockIdx.x * BLOCK_ROWS_PER_THREAD;

    {
        constexpr const int numCols = INPUT_LENGTH_A + INPUT_LENGTH_B + NUM_DIMENSIONS;

        float values[BLOCK_ROWS_PER_THREAD];

        for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
        {
            values[row] = 0.0f;
        }

        // input A
        for (int col = threadIdx.x; col < INPUT_LENGTH_A / 2; col += BLOCK_COL_SIZE)
        {
            const float2 v = inputA[col];
            for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
            {
                values[row] += dot2(v, weights[(rowOffset + row) * (numCols / 2) + col]);
            }
        }

        // input B
        for (int col = threadIdx.x; col < INPUT_LENGTH_B / 2; col += BLOCK_COL_SIZE)
        {
            const float2 v = inputB[col];
            for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
            {
                values[row] += dot2(v, weights[(rowOffset + row) * (numCols / 2) + (INPUT_LENGTH_A / 2) + col]);
            }
        }

        // hidden input
        for (int col = threadIdx.x; col < NUM_DIMENSIONS / 2; col += BLOCK_COL_SIZE)
        {
            const float2 v = hiddenIn[col];
            for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
            {
                values[row] += dot2(
                    v, weights[(rowOffset + row) * (numCols / 2) + ((INPUT_LENGTH_A + INPUT_LENGTH_B) / 2) + col]);
            }
        }

        // place outputs into shared memory for reduction
        for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
        {
            shared[row * BLOCK_COL_SIZE + threadIdx.x] = values[row];
        }
    }

    __syncthreads();

    sumBlock(shared);

    {
        const int globalRow = rowOffset + threadIdx.x;

        // add bias and functify (first four threads only)
        if (threadIdx.x < BLOCK_ROWS_PER_THREAD)
        {
            float sum = shared[threadIdx.x * BLOCK_COL_SIZE] + bias[globalRow];
            if (threadIdx.x % 4 == 2)
            {
                // g gets tanh
                sum = tanh(sum);
            }
            else
            {
                // everything else gets sigmoid
                sum = sigmoid(sum);
            }
            shared[threadIdx.x * BLOCK_COL_SIZE] = sum;

            __syncwarp(0x0000000f);

            if ((threadIdx.x % 4) == 0)
            {
                const int stateRow = globalRow / 4;

                const float i = shared[(threadIdx.x + 0) * BLOCK_COL_SIZE];
                const float f = shared[(threadIdx.x + 1) * BLOCK_COL_SIZE];
                const float g = shared[(threadIdx.x + 2) * BLOCK_COL_SIZE];
                const float o = shared[(threadIdx.x + 3) * BLOCK_COL_SIZE];

                const float c = cellIn[stateRow];

                const float cPrime = f * c + i * g;
                const float hPrime = o * tanh(cPrime);

                cellOut[stateRow] = cPrime;
                hiddenOut[stateRow] = hPrime;
            }
        }
    }
}

template <int INPUT_LENGTH_A, int INPUT_LENGTH_B, int NUM_DIMENSIONS>
__global__ void lstmCellRowFloatKernel(const float* const weights, const float* const bias, const float* const inputA,
    const float* const inputB, const float* const hiddenIn, const float* const cellIn, float* const hiddenOut,
    float* const cellOut)
{
    __shared__ float shared[BLOCK_COL_SIZE * BLOCK_ROWS_PER_THREAD];

    const int rowOffset = blockIdx.x * BLOCK_ROWS_PER_THREAD;

    {
        constexpr const int numCols = NUM_DIMENSIONS + INPUT_LENGTH_A + INPUT_LENGTH_B;

        float values[BLOCK_ROWS_PER_THREAD];

        for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
        {
            values[row] = 0.0f;
        }

        // input A
        for (int col = threadIdx.x; col < INPUT_LENGTH_A; col += BLOCK_COL_SIZE)
        {
            const float v = inputA[col];
            for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
            {
                values[row] += v * weights[(rowOffset + row) * numCols + col];
            }
        }

        // input B
        for (int col = threadIdx.x; col < INPUT_LENGTH_B; col += BLOCK_COL_SIZE)
        {
            const float v = inputB[col];
            for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
            {
                values[row] += v * weights[(rowOffset + row) * numCols + INPUT_LENGTH_A + col];
            }
        }

        // hidden input
        for (int col = threadIdx.x; col < NUM_DIMENSIONS; col += BLOCK_COL_SIZE)
        {
            const float v = hiddenIn[col];
            for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
            {
                values[row] += v * weights[(rowOffset + row) * numCols + (INPUT_LENGTH_A + INPUT_LENGTH_B) + col];
            }
        }

        // place outputs into shared memory for reduction
        for (int row = 0; row < BLOCK_ROWS_PER_THREAD; ++row)
        {
            shared[row * BLOCK_COL_SIZE + threadIdx.x] = values[row];
        }
    }

    __syncthreads();

    sumBlock(shared);

    {
        const int globalRow = rowOffset + threadIdx.x;

        // add bias and functify (first four threads only)
        if (threadIdx.x < BLOCK_ROWS_PER_THREAD)
        {
            float sum = shared[threadIdx.x * BLOCK_COL_SIZE] + bias[globalRow];
            if (threadIdx.x % 4 == 2)
            {
                // g gets tanh
                sum = tanh(sum);
            }
            else
            {
                // everything else gets sigmoid
                sum = sigmoid(sum);
            }
            shared[threadIdx.x * BLOCK_COL_SIZE] = sum;
        }

        __syncwarp(0x0000000f);

        if (threadIdx.x < BLOCK_ROWS_PER_THREAD && (threadIdx.x % 4) == 0)
        {
            const int stateRow = globalRow / 4;

            const float i = shared[(threadIdx.x + 0) * BLOCK_COL_SIZE];
            const float f = shared[(threadIdx.x + 1) * BLOCK_COL_SIZE];
            const float g = shared[(threadIdx.x + 2) * BLOCK_COL_SIZE];
            const float o = shared[(threadIdx.x + 3) * BLOCK_COL_SIZE];

            const float c = cellIn[stateRow];

            const float cPrime = f * c + i * g;
            const float hPrime = o * tanh(cPrime);

            cellOut[stateRow] = cPrime;
            hiddenOut[stateRow] = hPrime;
        }
    }
}

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

size_t stride(const size_t i, const size_t n, const size_t s)
{
    return ((i * (n / s)) % n) + (i / s);
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2LSTMCellKernel::Taco2LSTMCellKernel(
    const float* const inputWeightsHost,
    const float* const hiddenWeightsHost,
    const float* const inputBiasHost,
    const float* const hiddenBiasHost,
    const int inputLength,
    const int numDimension,
    const bool useFP16) :
    mInputLength(inputLength),
    mNumDimension(numDimension),
    mFp16(useFP16),
    mWeightsDevice(),
    mBiasDevice()
{
    const size_t numRows = 4 * mNumDimension;
    { // combine weights into single matrix on device [W_i W_h], in column
        // major order, and in i_0, f_0, g_0, o_0, ... i_n, f_n, g_n, o_n order.
        std::vector<float> weightCat((mNumDimension + mInputLength) * numRows);
        // row wise strided
        for (size_t i = 0; i < numRows; ++i)
        {
            for (size_t j = 0; j < static_cast<size_t>(mInputLength); ++j)
            {
                weightCat[i * (mInputLength + mNumDimension) + j]
                    = inputWeightsHost[stride(i, numRows, 4) * mInputLength + j];
            }
        }
        for (size_t i = 0; i < numRows; ++i)
        {
            for (size_t j = 0; j < static_cast<size_t>(mNumDimension); ++j)
            {
                weightCat[i * (mInputLength + mNumDimension) + mInputLength + j]
                    = hiddenWeightsHost[stride(i, numRows, 4) * mNumDimension + j];
            }
        }
        if (mFp16)
        {
            // copy to device as floats
            CudaMemory<float> weightsFloatDevice(weightCat);

            // convert to halfs
            mWeightsDevice = CudaMemory<float>(
                taco2::Taco2Utils::roundUpBlocks(weightsFloatDevice.size(), 2));
            taco2::Taco2Utils::floatsToHalves(
                weightsFloatDevice.data(),
                mWeightsDevice.data(),
                weightsFloatDevice.size());
        }
        else
        {
          mWeightsDevice = CudaMemory<float>(weightCat);
        }
    }

    { // add biases togethor before moving to device [b_i + b_h],
        // and in i_0, f_0, g_0, o_0, ... i_n, f_n, g_n, o_n order.
        std::vector<float> biasSum(numRows);
        for (size_t i = 0; i < biasSum.size(); ++i)
        {
            const size_t j = stride(i, numRows, 4);
            assert(j < numRows);
            biasSum[i] = inputBiasHost[j] + hiddenBiasHost[j];
        }
        mBiasDevice = CudaMemory<float>(biasSum.size());
        taco2::Taco2Utils::copyHostToDevice(static_cast<float*>(mBiasDevice.data()), biasSum.data(), biasSum.size());
    }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Taco2LSTMCellKernel::execute(const float* const inputA, const float* const inputB, const float* const hiddenIn,
    const float* const cellIn, float* const hiddenOut, float* const cellOut, const int inputLengthA,
    const int inputLengthB, cudaStream_t stream)
{
    assert(inputLengthA + inputLengthB == mInputLength);
    const int numBlocks = taco2::Taco2Utils::roundUpBlocks(mNumDimension * 4, BLOCK_ROWS_PER_THREAD);

    const dim3 grid(numBlocks);
    const dim3 block(BLOCK_COL_SIZE);

    assert(mNumDimension == 1024);
    assert(inputLengthB == 512);

    if (mFp16)
    {
        if (inputLengthA == 256)
        {
          lstmCellRowHalfKernel<256, 512, 1024><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half2*>(mWeightsDevice.data()),
              mBiasDevice.data(),
              reinterpret_cast<const float2*>(inputA),
              reinterpret_cast<const float2*>(inputB),
              reinterpret_cast<const float2*>(hiddenIn),
              cellIn,
              hiddenOut,
              cellOut);
        }
        else if (inputLengthA == 1024)
        {
          lstmCellRowHalfKernel<1024, 512, 1024><<<grid, block, 0, stream>>>(
              reinterpret_cast<const __half2*>(mWeightsDevice.data()),
              mBiasDevice.data(),
              reinterpret_cast<const float2*>(inputA),
              reinterpret_cast<const float2*>(inputB),
              reinterpret_cast<const float2*>(hiddenIn),
              cellIn,
              hiddenOut,
              cellOut);
        }
        else
        {
            throw std::runtime_error("Unsupported Input A length of " + std::to_string(inputLengthA));
        }
    }
    else
    {
        if (inputLengthA == 256)
        {
          lstmCellRowFloatKernel<256, 512, 1024><<<grid, block, 0, stream>>>(
              mWeightsDevice.data(),
              mBiasDevice.data(),
              inputA,
              inputB,
              hiddenIn,
              cellIn,
              hiddenOut,
              cellOut);
        }
        else if (inputLengthA == 1024)
        {
          lstmCellRowFloatKernel<1024, 512, 1024><<<grid, block, 0, stream>>>(
              mWeightsDevice.data(),
              mBiasDevice.data(),
              inputA,
              inputB,
              hiddenIn,
              cellIn,
              hiddenOut,
              cellOut);
        }
        else
        {
            throw std::runtime_error("Unsupported Input A length of " + std::to_string(inputLengthA));
        }
    }
}

} // namespace plugin
} // namespace nvinfer1
