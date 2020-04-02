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

#include "normalDistribution.h"

#include <cassert>
#include <string>

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int NORMAL_DIST_BLOCK_SIZE = 512;
} // namespace

/******************************************************************************
 * CUDA KERNELS ***************************************************************
 *****************************************************************************/

__global__ void normalDistributionKernel(
    curandState_t* const states, const int numStates, float* const outValues, const int numValues)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numStates)
    {
        // load random state information from global memory
        curandState_t localState = states[tid];

        for (int index = tid; index < numValues; index += numStates)
        {
            outValues[index] = curand_normal(&localState);
        }

        // save random state information back to global memory
        states[tid] = localState;
    }
}

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{
int roundUpBlocks(const int num, const int blockSize)
{
    return ((num - 1) / blockSize) + 1;
}
} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

NormalDistribution::NormalDistribution(const int numStates, const uint32_t seed)
    : mRand(numStates)
{
    setSeed(seed, 0);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void NormalDistribution::setSeed(const uint32_t seed, cudaStream_t stream)
{
    mRand.setSeed(seed, stream);
}

void NormalDistribution::generate(float* const outValues, const int numValues, cudaStream_t stream)
{
    const dim3 grid(roundUpBlocks(mRand.size(), NORMAL_DIST_BLOCK_SIZE));
    const dim3 block(NORMAL_DIST_BLOCK_SIZE);

    assert(mRand.size() <= grid.x * block.x);

    normalDistributionKernel<<<grid, block, 0, stream>>>(mRand.getRandomStates(), mRand.size(), outValues, numValues);
}

} // namespace tts
