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

#include "waveGlowStreamingInstance.h"
#include "cudaUtils.h"
#include "trtUtils.h"

#include "NvInfer.h"

#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const MEL_INPUT_NAME = "spect";
constexpr const char* const Z_INPUT_NAME = "z";
constexpr const char* const OUTPUT_NAME = "audio";

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

void setBatchDimensions(IExecutionContext* const context, const int batchSize)
{
  const ICudaEngine& engine = context->getEngine();

  Dims melDims = engine.getBindingDimensions(0);
  melDims.d[0] = batchSize;
  context->setBindingDimensions(0, melDims);
}
} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

WaveGlowStreamingInstance::WaveGlowStreamingInstance(
    TRTPtr<ICudaEngine>&& eng) :
    TimedObject("WaveGlowStreamingInstance::infer()"),
    EngineDriver(std::move(eng)),
    mChunkSize(TRTUtils::getBindingDimension(getEngine(), MEL_INPUT_NAME, 2)),
    mSamplesPerFrame(256),
    mChunkSampleSize(
        TRTUtils::getNonBatchBindingSize(getEngine(), OUTPUT_NAME)),
    mTruncatedChunkSampleSize(mSamplesPerFrame * mChunkSize),
    mInputChannels(
        TRTUtils::getBindingDimension(getEngine(), MEL_INPUT_NAME, 3)),
    mZChannels(TRTUtils::getBindingDimension(getEngine(), Z_INPUT_NAME, 1)),
    mBatchSize(1),
    mBinding(),
    mContext(getEngine().createExecutionContext()),
    mRand(mChunkSampleSize, 0),
    mZ(TRTUtils::getMaxBindingSize(getEngine(), Z_INPUT_NAME))
{
    const int zChunkSize = TRTUtils::getBindingDimension(getEngine(), Z_INPUT_NAME, 1);
    if (zChunkSize * mZChannels > mChunkSampleSize)
    {
        throw std::runtime_error("Expected z to be of dimension at most: " + std::to_string(mZChannels) + "x"
            + std::to_string(mChunkSampleSize / mZChannels) + " but engine has " + std::to_string(mZChannels) + "x"
            + std::to_string(zChunkSize));
    }

    // generate z vector
    mRand.setSeed(0, 0);

    // set batch size to 1 by default
    setBatchDimensions(mContext.get(), mBatchSize);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void WaveGlowStreamingInstance::startInference(const int batchSize, cudaStream_t stream)
{
  bool newBatchSize = mBatchSize != batchSize;
  mBatchSize = batchSize;

  mRand.generate(mZ.data(), mZ.size(), stream);

  if (newBatchSize) {
    // only set batch dimensions if they have changed
    setBatchDimensions(mContext.get(), mBatchSize);
  }

  const ICudaEngine& engine = mContext->getEngine();
  mBinding.setBinding(engine, Z_INPUT_NAME, mZ.data());
}

void WaveGlowStreamingInstance::inferNext(cudaStream_t stream, const float* const melsDevice, const int* const numMels,
    float* outputDevice, int* numSamplesOut)
{
    startTiming();

    const ICudaEngine& engine = mContext->getEngine();

    for (int batchIdx = 0; batchIdx < mBatchSize; ++batchIdx)
    {
        if (numMels[batchIdx] > mChunkSize)
        {
            throw std::runtime_error("Cannot work on chunk of " + std::to_string(numMels[batchIdx]) + ", maximum is "
                + std::to_string(mChunkSize));
        }
    }

    // queue up work on the GPU
    mBinding.setBinding(engine, MEL_INPUT_NAME, melsDevice);
    mBinding.setBinding(engine, OUTPUT_NAME, outputDevice);
    if (!mContext->enqueueV2(mBinding.getBindings(), stream, nullptr))
    {
        throw std::runtime_error("Failed to enqueue WaveGlow.");
    }

    // then do CPU work as needed
    for (int batchIdx = 0; batchIdx < mBatchSize; ++batchIdx)
    {
        numSamplesOut[batchIdx] = numMels[batchIdx] * mSamplesPerFrame;
    }

    CudaUtils::sync(stream);
    stopTiming();
}

int WaveGlowStreamingInstance::getNumberOfSamplesPerFrame() const
{
    return mSamplesPerFrame;
}

int WaveGlowStreamingInstance::getMelSpacing() const
{
    return mChunkSize;
}

int WaveGlowStreamingInstance::getNumMelChannels() const
{
    return mInputChannels;
}

int WaveGlowStreamingInstance::getMaxOutputLength() const
{
    return mChunkSize * mSamplesPerFrame;
}

int WaveGlowStreamingInstance::getOutputSpacing() const
{
    return getMaxOutputLength() + 768;
}

int WaveGlowStreamingInstance::getRequiredOutputBufferSize(const int batchSize) const
{
    return getOutputSpacing() * batchSize;
}

} // namespace tts
