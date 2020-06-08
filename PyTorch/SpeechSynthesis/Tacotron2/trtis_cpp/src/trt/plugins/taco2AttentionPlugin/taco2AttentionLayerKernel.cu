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

#include "taco2AttentionLayerKernel.h"
#include "taco2Utils.h"

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
constexpr const int ENERGY_BLOCK_SIZE = 128;
constexpr const int CONV_BLOCK_SIZE = 128;
constexpr const int QUERY_NUM_COLS = 1024;
constexpr const int QUERY_COL_SIZE = 128;
constexpr const int WARP_SIZE = 32;

static_assert(QUERY_NUM_COLS % QUERY_COL_SIZE == 0, "QUERY_NUM_COLS must be a multiple of QUERY_COL_SIZE");

} // namespace

const float Taco2AttentionLayerKernel::ONE = 1.0f;
const float Taco2AttentionLayerKernel::ZERO = 0.0f;

/******************************************************************************
 * CUDA KERNELS ***************************************************************
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

    if (BLOCK_SIZE > WARP_SIZE)
    {
        if (threadIdx.x % WARP_SIZE == 0)
        {
            buffer[threadIdx.x / WARP_SIZE] = val;
        }

        __syncthreads();

        if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE))
        {
            val = warpSum<T, BLOCK_SIZE / WARP_SIZE>(buffer[threadIdx.x]);
        }
    }

    return val;
}

__global__ void attentionQueryGemvKernel(const float* const weights, const float* const input, float* const output,
    const int inputLength, const int outputLength)
{
    __shared__ float shared[QUERY_COL_SIZE];

    assert(gridDim.x == outputLength);
    assert(inputLength == QUERY_NUM_COLS);

    // perform mat vec
    float v = 0.0f;
    for (int col = threadIdx.x; col < QUERY_NUM_COLS; col += QUERY_COL_SIZE)
    {
        // load chunk
        v += input[col] * weights[blockIdx.x * QUERY_NUM_COLS + col];
    }

    v = cooperativeSum<float, QUERY_COL_SIZE>(v, shared);

    // add bias and write
    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = v;
    }
}

__global__ void attentionEnergyKernel(const float* const query, const float* const processedMemory,
    const float* const location, const float* const weights, const int inputLength, float* const blockSums)
{
    // first every thread must load their 'query' cell
    const float q = query[threadIdx.x];

    // should be 32x128 = 4k
    __shared__ float summation[ENERGY_BLOCK_SIZE];

    // iterate over rows to create sums and perform tanh
    const int gIdx = blockIdx.x * ENERGY_BLOCK_SIZE + threadIdx.x;
    const float v = q + processedMemory[gIdx] + location[gIdx];
    float val = tanh(v) * weights[threadIdx.x];

    val = cooperativeSum<float, ENERGY_BLOCK_SIZE>(val, summation);

    // perform simplistic reduction
    if (threadIdx.x == 0)
    {
        // write summation back to shared memory
        blockSums[blockIdx.x] = exp(val);
    }
}

__global__ void attentionNormalizeAndSumKernel(
    const float* const elemAccumsIn, float* const elems, const int numElems, const float* const blockSums)
{
    __shared__ float sums[ENERGY_BLOCK_SIZE];
    __shared__ float invSum;

    // each block sums up the blockSums on its own
    float v = 0;
    for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
    {
        v += blockSums[i];
    }

    v = cooperativeSum<float, ENERGY_BLOCK_SIZE>(v, sums);
    if (threadIdx.x == 0)
    {
        invSum = 1.0f / v;
    }

    __syncthreads();

    // normalize and sum
    float* const elemAccumsOut = elems + numElems;
    for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < numElems; i += gridDim.x * blockDim.x)
    {
        const float val = blockSums[i] * invSum;
        elems[i] = val;
        elemAccumsOut[i] = val + elemAccumsIn[i];
    }
}

__global__ void attentionConvolutionKernel(const float* const convWeights, const float* const attWeights,
    float* const output, const int inputLength, const int kernelSize)
{
    __shared__ float kernels[32 * 2];
    __shared__ float input[(CONV_BLOCK_SIZE + 32) * 2];
    __shared__ float sum[CONV_BLOCK_SIZE * 2];

    const int halfKernel = (kernelSize - 1) / 2;
    const int inputOffset = 32 - halfKernel;

    // all threads work to populate the shared memory kernels
    if (threadIdx.x < kernelSize)
    {
        kernels[threadIdx.x + threadIdx.y * 32]
            = convWeights[blockIdx.x * (kernelSize * 2) + (threadIdx.x + threadIdx.y * kernelSize)];
    }

    // set initial input zero for second half
    if (threadIdx.x < 32)
    {
        if (threadIdx.x < halfKernel || threadIdx.x - halfKernel >= inputLength)
        {
            input[CONV_BLOCK_SIZE + threadIdx.x + threadIdx.y * (CONV_BLOCK_SIZE + 32)] = 0;
        }
        else
        {
            input[CONV_BLOCK_SIZE + threadIdx.x + threadIdx.y * (CONV_BLOCK_SIZE + 32)]
                = attWeights[threadIdx.x - halfKernel + threadIdx.y * inputLength];
        }
    }
    __syncthreads();

    for (int i = 0; i < inputLength; i += CONV_BLOCK_SIZE)
    {
        // shift second half into first half
        if (threadIdx.x < 32)
        {
            input[threadIdx.x + threadIdx.y * (CONV_BLOCK_SIZE + 32)]
                = input[CONV_BLOCK_SIZE + threadIdx.x + threadIdx.y * (CONV_BLOCK_SIZE + 32)];
        }
        __syncthreads();

        // copy in second half
        float v = 0;
        if (i + threadIdx.x + inputOffset < inputLength)
        {
            v = attWeights[i + threadIdx.x + inputOffset + threadIdx.y * inputLength];
        }
        input[32 + threadIdx.x + threadIdx.y * (CONV_BLOCK_SIZE + 32)] = v;

        __syncthreads();

        // multiply with kernel
        float a = 0.0f;
        for (int j = 0; j < kernelSize; ++j)
        {
            const int k = threadIdx.x + j + threadIdx.y * (CONV_BLOCK_SIZE + 32);
            a += input[k] * kernels[j + threadIdx.y * 32];
        }

        sum[threadIdx.x + threadIdx.y * CONV_BLOCK_SIZE] = a;

        __syncthreads();

        // write to global memory
        if (threadIdx.y == 0 && threadIdx.x + i < inputLength)
        {
            output[(blockIdx.x * inputLength) + i + threadIdx.x]
                = sum[threadIdx.x] + sum[threadIdx.x + CONV_BLOCK_SIZE];
        }
    }
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2AttentionLayerKernel::Taco2AttentionLayerKernel(
    const std::vector<float>& queryWeightsHost,
    const std::vector<float>& convWeightsHost,
    const std::vector<float>& locationWeightsHost,
    const std::vector<float>& energyWeightsHost,
    const int encLength,
    const int numQueryDimension,
    const int numFilters,
    const int convKernelSize,
    const int numAttentionDimension) :
    mNumEncodingDimension(encLength),
    mNumQueryDimension(numQueryDimension),
    mNumFilters(numFilters),
    mConvKernelSize(convKernelSize),
    mNumAttentionDimension(numAttentionDimension),
    mQueryWeightsDevice(),
    mConvWeightsDevice(),
    mLocationWeightsDevice(),
    mEnergyWeightsDevice(),
    mCublasHandle{}
{
    const size_t numExpectedQueryWeights = mNumAttentionDimension * mNumQueryDimension;
    const size_t numExpectedConvWeights = mNumFilters * mConvKernelSize * 2;
    const size_t numExpectedLocationWeights = mNumAttentionDimension * mNumFilters;
    const size_t numExpectedEnergyWeights = mNumAttentionDimension;

    if (queryWeightsHost.size() != numExpectedQueryWeights)
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedQueryWeights) + " query weights but got "
            + std::to_string(queryWeightsHost.size()) + " instead.");
    }
    else if (convWeightsHost.size() != numExpectedConvWeights)
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedConvWeights) + " convolution weights but got "
            + std::to_string(convWeightsHost.size()) + " instead.");
    }
    else if (locationWeightsHost.size() != numExpectedLocationWeights)
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedLocationWeights) + " location weights but got "
            + std::to_string(locationWeightsHost.size()) + " instead.");
    }
    else if (energyWeightsHost.size() != numExpectedEnergyWeights)
    {
        throw std::runtime_error("Expected " + std::to_string(numExpectedEnergyWeights) + " energy weights but got "
            + std::to_string(energyWeightsHost.size()) + " instead.");
    }

    // copy up weights to GPU

    // keep in row major [128x1024]
    mQueryWeightsDevice = CudaMemory<float>(queryWeightsHost);

    // convolution has [32x2x31] weights (filters x kernel size).
    mConvWeightsDevice = CudaMemory<float>(convWeightsHost);

    // transpose from column major [32x128] to column major [128x32]
    std::vector<float> transLocationWeights(locationWeightsHost.size());
    for (int j = 0; j < mNumAttentionDimension; ++j)
    {
        for (int i = 0; i < mNumFilters; ++i)
        {
            transLocationWeights[i * mNumAttentionDimension + j] = locationWeightsHost[j * mNumFilters + i];
        }
    }
    mLocationWeightsDevice = CudaMemory<float>(transLocationWeights);

    // energy FC is [1x128]
    mEnergyWeightsDevice = CudaMemory<float>(energyWeightsHost);

    // initialize cublas
    if (cublasCreate(&mCublasHandle) != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to create cublas handle.");
    }
}

Taco2AttentionLayerKernel::~Taco2AttentionLayerKernel()
{
    cublasDestroy(mCublasHandle);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Taco2AttentionLayerKernel::execute(const float* const memoryDevice, const float* const processedMemoryDevice,
    const float* const weightsDevice, const float* const attentionHiddenDevice, float* const outputContextDevice,
    float* const outputWeightsDevice, const int inputLength, float* const workspace, cudaStream_t stream)
{
    float* const queryOutput = workspace;
    float* const convOutput = queryOutput + mNumAttentionDimension;
    float* const elemSum = convOutput + (inputLength * mNumFilters);
    float* const energyScratch = elemSum + (inputLength * mNumAttentionDimension);

    cublasSetStream(mCublasHandle, stream);

    // launch fully connected layer to parse LSTM hidden states -
    // multiplying 128x1024 weights with 1024 inputs, to get 128 outputs
    {
        const dim3 grid(mNumAttentionDimension);
        const dim3 block(QUERY_COL_SIZE);

        attentionQueryGemvKernel<<<grid, block, 0, stream>>>(
            mQueryWeightsDevice.data(),
            attentionHiddenDevice,
            queryOutput,
            mNumQueryDimension,
            mNumAttentionDimension);
    }

    // perform convolution
    {
        const dim3 grid(mNumFilters);
        const dim3 block(CONV_BLOCK_SIZE, 2);

        // only works for 2 channels
        assert(mConvKernelSize <= CONV_BLOCK_SIZE);

        attentionConvolutionKernel<<<grid, block, 0, stream>>>(
            mConvWeightsDevice.data(),
            weightsDevice,
            convOutput,
            inputLength,
            mConvKernelSize);
    }

    // location linear layer - 128x128x32
    cublasStatus_t err = cublasSgemm(
        mCublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        mNumAttentionDimension,
        inputLength,
        mNumFilters,
        &ONE,
        mLocationWeightsDevice.data(),
        mNumAttentionDimension,
        convOutput,
        inputLength,
        &ZERO,
        elemSum,
        mNumAttentionDimension);

    if (err != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Location layer failed in cublas.");
    }

    // perform energy calculation
    {
        const int numBlocks = inputLength;

        if (ENERGY_BLOCK_SIZE != mNumAttentionDimension)
        {
            throw std::runtime_error("mNumAttentionDimension must be " + std::to_string(ENERGY_BLOCK_SIZE));
        }

        const dim3 grid(numBlocks);
        const dim3 block(ENERGY_BLOCK_SIZE);

        attentionEnergyKernel<<<grid, block, 0, stream>>>(
            queryOutput,
            processedMemoryDevice,
            elemSum,
            mEnergyWeightsDevice.data(),
            inputLength,
            energyScratch);

        attentionNormalizeAndSumKernel<<<grid, block, 0, stream>>>(
            weightsDevice + inputLength, outputWeightsDevice, inputLength, energyScratch);
    }

    // finally perform mmLayer
    err = cublasSgemv(mCublasHandle, CUBLAS_OP_N, mNumEncodingDimension, inputLength, &ONE, memoryDevice,
        mNumEncodingDimension, outputWeightsDevice, 1, &ZERO, outputContextDevice, 1);
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error("Matrix multiply layer failed in cublas.");
    }
}

} // namespace plugin
} // namespace nvinfer1
