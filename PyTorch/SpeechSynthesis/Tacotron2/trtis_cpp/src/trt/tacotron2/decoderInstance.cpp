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

#include "decoderInstance.h"
#include "checkedCopy.h"
#include "cudaUtils.h"
#include "dataShuffler.h"
#include "trtUtils.h"
#include "utils.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DecoderInstance::DecoderInstance(
    TRTPtr<ICudaEngine> engine, const int maxChunkSize) :
    TimedObject("DecoderInstance::infer()"),
    EngineDriver(std::move(engine)),
    mContext(getEngine().createExecutionContext()),
    mMaxChunkSize(maxChunkSize),
    mNextChunkSize(mMaxChunkSize),
    mNumChannels(
        TRTUtils::getBindingSize(getEngine(), OUTPUT_CHANNELS_NAME) - 1),
    mStopThreshold(0.5),
    mBatchSize(-1),
    mSeed(0),
    mLastChunkSize(getMaxBatchSize(), 0),
    mDone(getMaxBatchSize(), true),
    mDropout(
        getMaxBatchSize(),
        maxChunkSize,
        TRTUtils::getBindingSize(getEngine(), INPUT_DROPOUT_NAME),
        0.5,
        0),
    mDecoderInputDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_LASTFRAME_NAME)),
    mGateOutputDevice(mMaxChunkSize * getMaxBatchSize()),
    mOutputTransposedDevice(
        getMaxBatchSize() * mMaxChunkSize
        * TRTUtils::getBindingSize(getEngine(), OUTPUT_CHANNELS_NAME)),
    mOutputGateHost(mMaxChunkSize * getMaxBatchSize())
{
    reset();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void DecoderInstance::infer(cudaStream_t stream, const int batchSize, const float* const inputDevice,
    const float* const inputProcessedDevice, const float* const inputMaskDevice, const int32_t* const inputLengthHost,
    const int32_t* const inputLengthDevice, float* const outputDevice)
{
    startTiming();
    if (batchSize > getMaxBatchSize())
    {
        throw std::runtime_error("Maximum batch size is " + std::to_string(getMaxBatchSize()) + " but got "
            + std::to_string(batchSize) + ".");
    }

    mBatchSize = batchSize;

    // generate dropout for entire chunk ahead of time
    mDropout.generate(mBatchSize, mNextChunkSize, stream);

    const float* lastFrame = mDecoderInputDevice.data();
    for (int chunk = 0; chunk < mNextChunkSize; ++chunk)
    {
      float* const thisFrame = mOutputTransposedDevice.data()
                               + mBatchSize * (mNumChannels + 1) * chunk;

      // convert data to gpu for tensor rt
      decode(
          stream,
          *mContext,
          mBatchSize,
          lastFrame,
          inputDevice,
          inputProcessedDevice,
          inputMaskDevice,
          inputLengthHost,
          inputLengthDevice,
          mDropout.get(chunk),
          thisFrame);

      lastFrame = thisFrame;
    }

    // we need to save the last frame in mDecoderInputDevice for the next run
    CheckedCopy::deviceToDeviceAsync(
        mDecoderInputDevice.data(),
        lastFrame,
        mBatchSize * (mNumChannels + 1),
        stream);

    // transpose my output data from TNC to NCT and pull out gate output
    DataShuffler::parseDecoderOutput(
        mOutputTransposedDevice.data(),
        outputDevice,
        mGateOutputDevice.data(),
        mBatchSize,
        mMaxChunkSize,
        mNumChannels,
        stream);

    CheckedCopy::deviceToHostAsync(
        mOutputGateHost.data(),
        mGateOutputDevice.data(),
        mMaxChunkSize * mBatchSize,
        stream);
    CudaUtils::sync(stream);

    // reduce gate output
    std::fill(mLastChunkSize.begin(), mLastChunkSize.end(), 0);
    for (int batchIndex = 0; batchIndex < mBatchSize; ++batchIndex)
    {
        for (int chunk = 0; chunk < mNextChunkSize; ++chunk)
        {
            if (!mDone[batchIndex])
            {
                ++mLastChunkSize[batchIndex];
                if (Utils::sigmoid(
                        mOutputGateHost
                            .data()[batchIndex * mMaxChunkSize + chunk])
                    > mStopThreshold) {
                  mDone[batchIndex] = true;
                }
            }
        }
    }
    stopTiming();
}

const int* DecoderInstance::lastChunkSize() const
{
    return mLastChunkSize.data();
}

bool DecoderInstance::isAllDone() const
{
    // only reduce active batch
    return std::accumulate(
        mDone.cbegin(), mDone.cbegin() + mBatchSize, true, [](const bool a, const bool b) { return a && b; });
}

void DecoderInstance::reset()
{
    mNextChunkSize = mMaxChunkSize;

    std::fill(mDone.begin(), mDone.end(), false);

    mDropout.reset(mSeed);

    // relies on zeros
    CudaUtils::zero(mDecoderInputDevice.data(), mDecoderInputDevice.size());
}

void DecoderInstance::setNextChunkSize(const int chunkSize)
{
    if (chunkSize > mMaxChunkSize)
    {
        throw std::runtime_error(
            "Invalid next chunk size: " + std::to_string(chunkSize) + " > " + std::to_string(mMaxChunkSize));
    }
    mNextChunkSize = chunkSize;
}

void DecoderInstance::setSeed(const unsigned int seed)
{
    mSeed = seed;
}

int DecoderInstance::getNextChunkSize() const
{
    return mNextChunkSize;
}

int DecoderInstance::getMaxChunkSize() const
{
    return mMaxChunkSize;
}

} // namespace tts
