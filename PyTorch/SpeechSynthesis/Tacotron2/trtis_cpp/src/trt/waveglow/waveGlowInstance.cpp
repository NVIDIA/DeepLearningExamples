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

#include "waveGlowInstance.h"
#include "blending.h"
#include "cudaUtils.h"
#include "dataShuffler.h"
#include "engineCache.h"
#include "normalDistribution.h"

#include "NvOnnxParser.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

using namespace nvinfer1;
using IParser = nvonnxparser::IParser;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const int NUM_OVERLAP = 0;
constexpr const char* const MEL_INPUT_NAME = "spectrograms";
constexpr const char* const Z_INPUT_NAME = "z";
constexpr const char* const OUTPUT_NAME = "waveglow_output";
} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

WaveGlowInstance::WaveGlowInstance(TRTPtr<ICudaEngine> eng) :
    TimedObject("WaveGlowInstance::infer()"),
    mStreamingInstance(std::move(eng)),
    mFrequency(22050),
    mNumOverlap(std::min(NUM_OVERLAP, mStreamingInstance.getMaxOutputLength())),
    mIndependentChunkSize(
        mStreamingInstance.getMelSpacing()
        - (mNumOverlap / getNumberOfSamplesPerFrame())),
    mIndependentChunkSampleSize(
        mIndependentChunkSize * getNumberOfSamplesPerFrame()),
    mNumChunkMels(getMaxBatchSize()),
    mNumChunkSamples(getMaxBatchSize()),
    mInputFrame(
        mStreamingInstance.getMelSpacing()
        * mStreamingInstance.getNumMelChannels() * getMaxBatchSize()),
    mOutputFrame(
        mStreamingInstance.getRequiredOutputBufferSize(getMaxBatchSize()))
{
    if (mIndependentChunkSampleSize + mNumOverlap != mStreamingInstance.getMaxOutputLength())
    {
        throw std::runtime_error("Overlap size must be a multiple of the number of samples per frame: "
            + std::to_string(mNumOverlap) + "/" + std::to_string(getNumberOfSamplesPerFrame()));
    }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void WaveGlowInstance::infer(const int batchSize, const float* const melsDevice, const int melSpacing,
    const int* const numMels, const int numMaxSamples, float* outputDevice, int* numSamplesOut)
{
    startTiming();

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        throw std::runtime_error("Failed to create stream.");
    }
    mStreamingInstance.startInference(batchSize, stream);

    // compute maximum number of chunks we'll need
    int maxNumMels = 0;
    for (int i = 0; i < batchSize; ++i)
    {
        if (numMels[i] > maxNumMels)
        {
            maxNumMels = numMels[i];
        }
    }

    const int numChunks = ((maxNumMels - 1) / mIndependentChunkSize) + 1;
    const int totalChunkSize = mStreamingInstance.getMelSpacing() * mStreamingInstance.getNumMelChannels();

    for (int i = 0; i < batchSize; ++i)
    {
        numSamplesOut[i] = 0;
    }

    for (int chunk = 0; chunk < numChunks; ++chunk)
    {
        const int inputIdx = chunk * mIndependentChunkSize;

        DataShuffler::frameTransfer(
            melsDevice,
            mInputFrame.data(),
            melSpacing * mStreamingInstance.getNumMelChannels(),
            inputIdx * mStreamingInstance.getNumMelChannels(),
            totalChunkSize,
            batchSize,
            totalChunkSize,
            0,
            stream);

        for (int i = 0; i < batchSize; ++i)
        {
            mNumChunkMels[i] = std::min(mStreamingInstance.getMelSpacing(), std::max(0, numMels[i] - inputIdx));
        }

        mStreamingInstance.inferNext(
            stream,
            mInputFrame.data(),
            mNumChunkMels.data(),
            mOutputFrame.data(),
            mNumChunkSamples.data());

        Blending::linear(
            batchSize,
            mOutputFrame.data(),
            outputDevice,
            mStreamingInstance.getOutputSpacing(),
            mNumOverlap,
            numMaxSamples,
            chunk * mIndependentChunkSampleSize,
            stream);

        for (int i = 0; i < batchSize; ++i)
        {
            numSamplesOut[i] += mNumChunkSamples[i];
        }

        CudaUtils::sync(stream);
    }

    cudaStreamDestroy(stream);
    stopTiming();
}

int WaveGlowInstance::getNumberOfSamplesPerFrame() const
{
    return mStreamingInstance.getNumberOfSamplesPerFrame();
}

int WaveGlowInstance::getFrequency() const
{
    return mFrequency;
}

int WaveGlowInstance::getMaxBatchSize() const
{
    return mStreamingInstance.getMaxBatchSize();
}

} // namespace tts
