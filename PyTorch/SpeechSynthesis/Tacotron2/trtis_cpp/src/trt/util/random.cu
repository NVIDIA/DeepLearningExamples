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

#include "cudaUtils.h"
#include "random.h"

#include <cassert>

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int BLOCK_SIZE = 256;
}

/******************************************************************************
 * CUDA KERNELS ***************************************************************
 *****************************************************************************/

__global__ void initRandStateKernel(curandState_t* const states, const int numStates, const uint32_t seed)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numStates)
    {
        curand_init(seed, index, 0, states + index);
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

Random::Random(const int numStates, const unsigned int seed)
    : mNumStates(numStates)
    , mRandStateDevice(nullptr)
{
    CudaUtils::alloc(&mRandStateDevice, mNumStates);

    setSeed(seed);
}

Random::~Random()
{
    CudaUtils::free(&mRandStateDevice);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Random::setSeed(unsigned int seed, cudaStream_t stream)
{
    assert(mNumStates == 0 || mRandStateDevice);

    const dim3 grid(roundUpBlocks(mNumStates, BLOCK_SIZE));
    const dim3 block(BLOCK_SIZE);

    initRandStateKernel<<<grid, block, 0, stream>>>(mRandStateDevice, mNumStates, seed);
}

curandState_t* Random::getRandomStates()
{
    assert(mNumStates == 0 || mRandStateDevice);
    return mRandStateDevice;
}

} // namespace tts
