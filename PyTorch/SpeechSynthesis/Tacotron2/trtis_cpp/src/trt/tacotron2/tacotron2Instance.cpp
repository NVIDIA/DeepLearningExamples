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

#include "tacotron2Instance.h"
#include "dataShuffler.h"

#include "NvInfer.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Tacotron2Instance::Tacotron2Instance(
    TRTPtr<ICudaEngine> encoderEngine,
    TRTPtr<ICudaEngine> decoderPlainEngine,
    TRTPtr<ICudaEngine> decoderPluginsEngine,
    TRTPtr<ICudaEngine> postnetEngine) :
    TimedObject("Tacotron2Instance::infer()"),
    mStreamingInstance(
        std::move(encoderEngine),
        std::move(decoderPlainEngine),
        std::move(decoderPluginsEngine),
        std::move(postnetEngine)),
    mChunkSize(mStreamingInstance.getMaxBatchSize()),
    mNumMelChunks(mStreamingInstance.getChunkSize()),
    mEarlyExit(true),
    mOutputShuffledDevice(getRequiredOutputSize(
        getMaxBatchSize(), getMaximumInputLength() * 10 + 100))
{
    // setup timing
    addChild(&mStreamingInstance);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Tacotron2Instance::infer(const int batchSize, const int* const inputDevice, const int inputSpacing,
    const int* const inputLength, const int maxOutputLength, float* const outputDevice, int* const outputLength)
{
    startTiming();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    mStreamingInstance.startInference(
        batchSize, inputDevice, inputSpacing, inputLength, stream);

    // do decoding
    float* intermediateOutput;

    if (batchSize > 1)
    {
        if (static_cast<size_t>(getRequiredOutputSize(batchSize, maxOutputLength)) > mOutputShuffledDevice.size())
        {
          mOutputShuffledDevice = CudaMemory<float>(
              getRequiredOutputSize(batchSize, maxOutputLength));
        }
        intermediateOutput = mOutputShuffledDevice.data();
    }
    else
    {
        intermediateOutput = outputDevice;
    }

    std::fill(outputLength, outputLength + batchSize, 0);
    const int numBlocks = ((maxOutputLength - 1) / mNumMelChunks) + 1;
    bool moreToDo = false;
    for (int block = 0; block < numBlocks; ++block)
    {
        const int numFramesTotal = block * mNumMelChunks;
        const int offset
            = block * batchSize * mStreamingInstance.getChunkSize() * mStreamingInstance.getNumMelChannels();
        if (numFramesTotal + mNumMelChunks > maxOutputLength)
        {
            mStreamingInstance.setNextChunkSize(maxOutputLength - numFramesTotal);
        }

        moreToDo = mStreamingInstance.inferNext(
            intermediateOutput + offset, mChunkSize.data(), stream);

        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            if (mEarlyExit)
            {
                outputLength[batchIndex] += mChunkSize[batchIndex];
            }
            else
            {
                outputLength[batchIndex] += mStreamingInstance.getChunkSize();
            }
        }

        if (mEarlyExit && !moreToDo)
        {
            break;
        }
    }
    if (mEarlyExit && moreToDo)
    {
        std::cerr << "One or more sequences failed to finish." << std::endl;
    }

    if (batchSize > 1)
    {
        // take the output from TNC to NTC

        // re-shuffle final output
        DataShuffler::shuffleMels(
            mOutputShuffledDevice.data(),
            outputDevice,
            batchSize,
            mStreamingInstance.getNumMelChannels(),
            mStreamingInstance.getChunkSize(),
            numBlocks,
            maxOutputLength,
            stream);
    }

    CudaUtils::sync(stream);
    cudaStreamDestroy(stream);

    stopTiming();
}

void Tacotron2Instance::setEarlyExit(const bool earlyExit)
{
    mEarlyExit = earlyExit;
}

void Tacotron2Instance::setSeed(const unsigned int seed)
{
    mStreamingInstance.setSeed(seed);
}

int Tacotron2Instance::getNumMelChannels() const
{
    return mStreamingInstance.getNumMelChannels();
}

int Tacotron2Instance::getMaximumInputLength() const
{
    return mStreamingInstance.getMaximumInputLength();
}

int Tacotron2Instance::getMaxBatchSize() const
{
    return mStreamingInstance.getMaxBatchSize();
}

int Tacotron2Instance::getRequiredOutputSize(const int batchSize, const int maxFrames) const
{
    if (batchSize > getMaxBatchSize())
    {
        throw std::runtime_error("Maximum batch size is " + std::to_string(getMaxBatchSize()) + " but got "
            + std::to_string(batchSize) + ".");
    }
    const int numMelChunks = mStreamingInstance.getChunkSize();
    const int frameCeil = (((maxFrames - 1) / mNumMelChunks) + 1) * numMelChunks;
    return batchSize * frameCeil * getNumMelChannels();
}

void Tacotron2Instance::usePlugins(const bool usePlugins)
{
    mStreamingInstance.usePlugins(usePlugins);
}

bool Tacotron2Instance::willUsePlugins(const int batchSize) const
{
    return mStreamingInstance.willUsePlugins(batchSize);
}

} // namespace tts
