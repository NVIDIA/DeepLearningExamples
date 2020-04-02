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

#include "blending.h"

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int BLOCK_SIZE = 1024;
}

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

__global__ void linearBlendingKernel(const float* const newChunk, float* const base, const int chunkLength,
    const int overlapSize, const int spacing, const int offset)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < chunkLength)
    {
        const float weight
            = offset > 0 && idx < overlapSize ? static_cast<float>(idx) / static_cast<float>(overlapSize) : 1.0f;

        const int inputIdx = idx + (blockIdx.y * chunkLength);
        const int outputIdx = idx + offset + (blockIdx.y * spacing);

        float newValue;
        if (weight < 1.0f)
        {
            newValue = (1.0f - weight) * base[outputIdx] + newChunk[inputIdx] * weight;
        }
        else
        {
            newValue = newChunk[inputIdx];
        }

        base[outputIdx] = newValue;
    }
}

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{
static int roundUpBlocks(const int num, const int blockSize)
{
    return ((num - 1) / blockSize) + 1;
}
} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void Blending::linear(const int batchSize, const float* const newChunk, float* const base, const int chunkSize,
    const int overlapSize, const int outputSequenceSpacing, const int outputSequenceOffset, cudaStream_t stream)
{
    const int blocksPerChunk = roundUpBlocks(chunkSize, BLOCK_SIZE);

    const dim3 grid(blocksPerChunk, batchSize);
    const dim3 block(BLOCK_SIZE);

    linearBlendingKernel<<<grid, block, 0, stream>>>(
        newChunk, base, chunkSize, overlapSize, outputSequenceSpacing, outputSequenceOffset);
}
