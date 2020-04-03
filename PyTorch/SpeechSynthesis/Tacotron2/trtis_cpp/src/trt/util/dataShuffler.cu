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

#include "dataShuffler.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int TRANSPOSE_BLOCK_SIZE = 32;
constexpr const int FRAME_TRANSFER_BLOCK_SIZE = 1024;
} // namespace

/******************************************************************************
 * CUDA KERNELS ***************************************************************
 *****************************************************************************/

__global__ void parseDecoderOutputKernel(const float* const matIn, float* const matOut, float* const gateOut,
    const int batchSize, const int chunkSize, const int numChannels)
{
    __shared__ float buffer[TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE];

    const int xIn = blockDim.x * blockIdx.x + threadIdx.x;
    const int yIn = blockDim.y * blockIdx.y + threadIdx.y;

    const int nCols = batchSize * (numChannels + 1);
    const int nRows = chunkSize;

    // all threads load a block cooperatively
    if (xIn < nCols && yIn < nRows)
    {
        buffer[blockDim.x * threadIdx.y + threadIdx.x] = matIn[yIn * nCols + xIn];
    }

    __syncthreads();

    const int xOut = blockDim.x * blockIdx.x + threadIdx.y;
    const int yOut = blockDim.y * blockIdx.y + threadIdx.x;

    // all threads write the block tranposed cooperatively
    if (xOut < nCols && yOut < nRows)
    {
        if (xOut % (numChannels + 1) != numChannels)
        {
            matOut[(xOut - (xOut / (numChannels + 1))) * nRows + yOut] = buffer[blockDim.y * threadIdx.x + threadIdx.y];
        }
        else
        {
            gateOut[(xOut / (numChannels + 1)) * nRows + yOut] = buffer[blockDim.y * threadIdx.x + threadIdx.y];
        }
    }
}

__global__ void transposeMatrixKernel(const float* const matIn, float* const matOut, const int nRows, const int nCols)
{
    __shared__ float buffer[TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE];

    const int xIn = blockDim.x * blockIdx.x + threadIdx.x;
    const int yIn = blockDim.y * blockIdx.y + threadIdx.y;

    // all threads load a block cooperatively
    if (xIn < nCols && yIn < nRows)
    {
        buffer[blockDim.x * threadIdx.y + threadIdx.x] = matIn[yIn * nCols + xIn];
    }

    __syncthreads();

    const int xOut = blockDim.x * blockIdx.x + threadIdx.y;
    const int yOut = blockDim.y * blockIdx.y + threadIdx.x;

    // all threads write the block tranposed cooperatively
    if (xOut < nCols && yOut < nRows)
    {
        matOut[xOut * nRows + yOut] = buffer[blockDim.y * threadIdx.x + threadIdx.y];
    }
}

__global__ void shuffleMelsKernel(
    const float* const matIn, float* const matOut, int batchSize, int numChannels, int chunkSize, int compactLength)
{
    // each block is assigned a chunk, with blockIdx.x corresponding to chunk
    // number and blockIdx.y corresponding to batch number
    const int chunkInStart = (blockIdx.x * batchSize + blockIdx.y) * chunkSize * numChannels;
    const int chunkOutStart = (blockIdx.y * compactLength + blockIdx.x * chunkSize) * numChannels;

    for (int row = threadIdx.y; row < chunkSize; row += blockDim.y)
    {
        if (row + blockIdx.x * chunkSize < compactLength)
        {
            for (int col = threadIdx.x; col < numChannels; col += blockDim.x)
            {
                const int offset = row * numChannels + col;
                matOut[chunkOutStart + offset] = matIn[chunkInStart + offset];
            }
        }
    }
}

__global__ void frameTransferKernel(const float* const in, float* const out, const int inputSequenceSpacing,
    const int inputSequenceOffset, const int chunkSize, const int outputSequenceSpacing, const int outputSequenceOffset)
{
    const int chunkIdx = threadIdx.x + blockIdx.x * blockDim.x;
    const int inIdx = chunkIdx + blockIdx.y * inputSequenceSpacing + inputSequenceOffset;
    const int outIdx = chunkIdx + blockIdx.y * outputSequenceSpacing + outputSequenceOffset;

    if (chunkIdx < chunkSize)
    {
        out[outIdx] = in[inIdx];
    }
}

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

static int roundUpBlocks(const int num, const int blockSize)
{
    return ((num - 1) / blockSize) + 1;
}

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void DataShuffler::parseDecoderOutput(const float* const in, float* const out, float* const gateOut,
    const int batchSize, const int chunkSize, const int numChannels, cudaStream_t stream)
{
    const dim3 grid(roundUpBlocks(batchSize * (numChannels + 1), TRANSPOSE_BLOCK_SIZE),
        roundUpBlocks(chunkSize, TRANSPOSE_BLOCK_SIZE));

    const dim3 block(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE);

    parseDecoderOutputKernel<<<grid, block, 0, stream>>>(in, out, gateOut, batchSize, chunkSize, numChannels);
    cudaError_t err = cudaStreamSynchronize(stream);
    assert(err == cudaSuccess);
}

void DataShuffler::transposeMatrix(
    const float* const in, float* const out, const int nRows, const int nCols, cudaStream_t stream)
{
    const dim3 grid(roundUpBlocks(nCols, TRANSPOSE_BLOCK_SIZE), roundUpBlocks(nRows, TRANSPOSE_BLOCK_SIZE));
    const dim3 block(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE);

    transposeMatrixKernel<<<grid, block, 0, stream>>>(in, out, nRows, nCols);
    cudaError_t err = cudaStreamSynchronize(stream);
    assert(err == cudaSuccess);
}

void DataShuffler::shuffleMels(const float* const in, float* const out, const int batchSize, const int numChannels,
    const int chunkSize, const int numChunks, const int compactLength, cudaStream_t stream)
{
    const dim3 grid(numChunks, batchSize);
    const dim3 block(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE);

    shuffleMelsKernel<<<grid, block, 0, stream>>>(in, out, batchSize, numChannels, chunkSize, compactLength);
    cudaError_t err = cudaStreamSynchronize(stream);
    assert(err == cudaSuccess);
}

void DataShuffler::frameTransfer(const float* const in, float* const out, const int inputSequenceSpacing,
    const int inputSequenceOffset, const int chunkSize, const int numChunks, const int outputSequenceSpacing,
    const int outputSequenceOffset, cudaStream_t stream)
{
    const int blocksPerChunk = roundUpBlocks(chunkSize, FRAME_TRANSFER_BLOCK_SIZE);

    const dim3 grid(blocksPerChunk, numChunks);
    const dim3 block(FRAME_TRANSFER_BLOCK_SIZE);

    frameTransferKernel<<<grid, block, 0, stream>>>(
        in, out, inputSequenceSpacing, inputSequenceOffset, chunkSize, outputSequenceSpacing, outputSequenceOffset);
}

} // namespace tts
