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

#include "dropoutGenerator.h"
#include "taco2Utils.h"

#include <stdexcept>

using namespace taco2;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int DROPOUT_BLOCK_SIZE = 256;
} // namespace

/******************************************************************************
 * CUDA KERNELS ***************************************************************
 *****************************************************************************/

__global__ void dropoutKernel(curandState_t* const states, const int numStates, float* const outValues,
    const int numValues, const float dropProbability, const float scale)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numStates)
    {
        // load random state information from global memory
        curandState_t localState = states[tid];

        for (int index = tid; index < numValues; index += numStates)
        {
            outValues[index] = scale * (curand_uniform(&localState) < dropProbability);
        }

        // save random state information back to global memory
        states[tid] = localState;
    }
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DropoutGenerator::DropoutGenerator(
    const int maxBatchSize,
    const int maxChunkSize,
    const int numValues,
    const float prob,
    const unsigned int seed) :
    mProb(prob),
    mMaxChunkSize(maxChunkSize),
    mNumValues(numValues),
    mGeneratedChunks(0),
    mBatchSize(0),
    mDropoutDevice(maxBatchSize * maxChunkSize * numValues),
    mRand(mNumValues, seed)
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void DropoutGenerator::reset(unsigned int seed)
{
    mRand.setSeed(seed);
}

void DropoutGenerator::generate(const int batchSize, const int numChunks, cudaStream_t stream)
{
    if (numChunks > mMaxChunkSize)
    {
        throw std::runtime_error("Cannot generate more chunks than maximum: " + std::to_string(numChunks) + " vs. "
            + std::to_string(mMaxChunkSize));
    }

    const dim3 grid(
        Taco2Utils::roundUpBlocks(mRand.size(), DROPOUT_BLOCK_SIZE));
    const dim3 block(DROPOUT_BLOCK_SIZE);

    const float scale = 1.0f / (1.0f - mProb);

    assert(mRand.size() <= grid.x * block.x);

    mBatchSize = batchSize;
    mGeneratedChunks = numChunks;

    dropoutKernel<<<grid, block, 0, stream>>>(
        mRand.getRandomStates(),
        mRand.size(),
        mDropoutDevice.data(),
        mGeneratedChunks * mNumValues * mBatchSize,
        mProb,
        scale);
}

const float* DropoutGenerator::get(const int chunk) const
{
    if (chunk > mGeneratedChunks)
    {
        throw std::runtime_error("Cannot chunk past number generated: " + std::to_string(chunk) + " vs. "
            + std::to_string(mGeneratedChunks));
    }

    return mDropoutDevice.data() + chunk * mNumValues * mBatchSize;
}

} // namespace tts
