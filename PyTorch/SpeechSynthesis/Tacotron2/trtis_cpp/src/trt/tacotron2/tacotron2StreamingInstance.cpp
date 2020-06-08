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

#include "tacotron2StreamingInstance.h"
#include "checkedCopy.h"
#include "cudaUtils.h"
#include "dataShuffler.h"
#include "decoderInstancePlain.h"
#include "decoderInstancePlugins.h"
#include "encoderInstance.h"
#include "maskGenerator.h"
#include "postNetInstance.h"
#include "trtUtils.h"
#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Tacotron2StreamingInstance::Tacotron2StreamingInstance(
    TRTPtr<ICudaEngine> encoderEngine,
    TRTPtr<ICudaEngine> decoderPlainEngine,
    TRTPtr<ICudaEngine> decoderPluginsEngine,
    TRTPtr<ICudaEngine> postnetEngine) :
    TimedObject("Tacotron2StreamingInstance::infer()"),
    mEncoder(std::make_shared<EncoderInstance>(std::move(encoderEngine))),
    mDecoderPlain(
        decoderPlainEngine
            ? std::make_shared<DecoderInstancePlain>(
                  std::move(decoderPlainEngine),
                  TRTUtils::getBindingDimension(
                      *postnetEngine, PostNetInstance::OUTPUT_NAME, 1))
            : nullptr),
    mDecoderPlugins(
        decoderPluginsEngine
            ? std::make_shared<DecoderInstancePlugins>(
                  std::move(decoderPluginsEngine),
                  TRTUtils::getBindingDimension(
                      *postnetEngine, PostNetInstance::OUTPUT_NAME, 1))
            : nullptr),
    mPostnet(std::make_shared<PostNetInstance>(std::move(postnetEngine))),
    mMaxInputLength(mEncoder->getInputLength()),
    mNumMelChannels(mPostnet->getNumMelChannels()),
    mNumMelChunks(mPostnet->getMelChunkSize()),
    mMaxBatchSize(std::min(
        mEncoder->getMaxBatchSize(),
        std::min(
            mDecoderPlain ? mDecoderPlain->getMaxBatchSize()
                          : mDecoderPlugins->getMaxBatchSize(),
            mPostnet->getMaxBatchSize()))),
    mBatchSize(0),
    mUsePlugins(mDecoderPlugins),
    mInUseDecoder(nullptr),
    mPaddedInputDevice(mMaxBatchSize * mMaxInputLength),
    mInputMaskDevice(mMaxBatchSize * mMaxInputLength),
    mInputLengthsDevice(mMaxBatchSize),
    mEncodingDevice(
        mMaxBatchSize * mMaxInputLength * mEncoder->getNumDimensions()),
    mProcessedEncodingDevice(
        mMaxBatchSize * mMaxInputLength
        * mEncoder->getNumProcessedDimensions()),
    mMelChunkDevice(
        mMaxBatchSize * mNumMelChannels * mPostnet->getMelChunkSize()),
    mInputLengthHost(nullptr)
{
    assert(mNumMelChannels == mPostnet->getNumMelChannels());

    // build timing structure
    addChild(mEncoder.get());
    if (mDecoderPlain)
    {
        addChild(mDecoderPlain.get());
    }
    if (mDecoderPlugins)
    {
        addChild(mDecoderPlugins.get());
    }
    addChild(mPostnet.get());
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void Tacotron2StreamingInstance::startInference(
    const int batchSize,
    const int* const inputDevice,
    const int inputSpacing,
    const int* const inputLength,
    cudaStream_t stream)
{
    startTiming();

    mBatchSize = batchSize;
    mInputLengthHost = inputLength;

    if (mBatchSize < 0 || mBatchSize > mMaxBatchSize)
    {
        throw std::runtime_error(
            "Maximum batch size is " + std::to_string(mMaxBatchSize) + " but got " + std::to_string(mBatchSize) + ".");
    }
    for (int i = 0; i < mBatchSize; ++i)
    {
        if (mInputLengthHost[i] > mMaxInputLength)
        {
            throw std::runtime_error("Model not configured for lengths greater than " + std::to_string(mMaxInputLength)
                + ". Given " + std::to_string(mInputLengthHost[i]) + " for sequence " + std::to_string(i) + ".");
        }
    }

    // copy input to padded location and set zeros
    CudaUtils::zeroAsync(
        mPaddedInputDevice.data(), mMaxInputLength * mBatchSize, stream);
    for (int i = 0; i < mBatchSize; ++i)
    {
        const int offset = mMaxInputLength * i;
        const int length = mInputLengthHost[i];
        CheckedCopy::deviceToDeviceAsync(
            mPaddedInputDevice.data() + offset,
            inputDevice + (i * inputSpacing),
            length,
            stream);
    }

    CheckedCopy::hostToDeviceAsync(
        mInputLengthsDevice.data(), mInputLengthHost, mBatchSize, stream);

    MaskGenerator::generate(
        mInputLengthsDevice.data(),
        mMaxInputLength,
        mBatchSize,
        mInputMaskDevice.data(),
        stream);

    mEncoder->infer(
        stream,
        mBatchSize,
        mPaddedInputDevice.data(),
        mInputMaskDevice.data(),
        mInputLengthsDevice.data(),
        mEncodingDevice.data(),
        mProcessedEncodingDevice.data());

    // configure decoder
    if (willUsePlugins(mBatchSize))
    {
        if (!mDecoderPlugins)
        {
            std::cerr << "This tacotron2 decoder engine with plugins is missing. "
                         "Do you need to rebuild the engine?"
                      << std::endl;
            throw std::runtime_error("Missing mDecoderPlugins engine");
        }

        mInUseDecoder = mDecoderPlugins.get();
    }
    else
    {
        if (!mDecoderPlain)
        {
            std::cerr << "This tacotron2 decoder engine without plugins is missing. "
                         "Do you need to rebuild the engine?"
                      << std::endl;
            throw std::runtime_error("Missing mDecoderPlain engine");
        }

        mInUseDecoder = mDecoderPlain.get();
    }
    mInUseDecoder->reset(stream);

    stopTiming();
}

bool Tacotron2StreamingInstance::inferNext(
    float* const outputDevice, int* const outputLength, cudaStream_t stream)
{
    startTiming();
    if (!mInUseDecoder)
    {
        throw std::runtime_error(
            "Tacotron2StreamingInstance::inferNext() cannot "
            "be called until Tacotron2StreamingInstance::startInference() is "
            "called.");
    }
    else if (mBatchSize <= 0 || mBatchSize > mMaxBatchSize)
    {
        throw std::runtime_error(
        "Tacotron2StreamingInstance::inferNext() has an "
        "invalid batch size of "
        + std::to_string(mBatchSize)
        + " set. "
          "This is an internal error.");
    }
    else if (!mInputLengthHost)
    {
        throw std::runtime_error("mInputLengthHost not set.");
    }

    // do decoding
    mInUseDecoder->infer(
        stream,
        mBatchSize,
        mEncodingDevice.data(),
        mProcessedEncodingDevice.data(),
        mInputMaskDevice.data(),
        mInputLengthHost,
        mInputLengthsDevice.data(),
        mMelChunkDevice.data());

    // call postnet
    mPostnet->infer(stream, mBatchSize, mMelChunkDevice.data(), outputDevice);

    cudaStreamSynchronize(stream);

    for (int batchIndex = 0; batchIndex < mBatchSize; ++batchIndex)
    {
        outputLength[batchIndex] = mInUseDecoder->lastChunkSize()[batchIndex];
    }

    stopTiming();

    return !mInUseDecoder->isAllDone();
}

void Tacotron2StreamingInstance::setSeed(const unsigned int seed)
{
    if (mDecoderPlain)
    {
        mDecoderPlain->setSeed(seed);
    }
    if (mDecoderPlugins)
    {
        mDecoderPlugins->setSeed(seed);
    }
    resetInference();
}

int Tacotron2StreamingInstance::getNumMelChannels() const
{
    return mNumMelChannels;
}

int Tacotron2StreamingInstance::getChunkSize() const
{
    return mNumMelChunks;
}

int Tacotron2StreamingInstance::getMaximumInputLength() const
{
    return mMaxInputLength;
}

int Tacotron2StreamingInstance::getMaxBatchSize() const
{
    return mMaxBatchSize;
}

void Tacotron2StreamingInstance::usePlugins(const bool usePlugins)
{
    if (mDecoderPlugins)
    {
        mUsePlugins = usePlugins;
    }
    else
    {
        throw std::runtime_error(
            "Cannot enable plugins. No plugin engine "
            "available for use.");
    }
    resetInference();
}

bool Tacotron2StreamingInstance::willUsePlugins(const int batchSize) const
{
    return mUsePlugins && mDecoderPlugins && batchSize <= mDecoderPlugins->getMaxBatchSize();
}

void Tacotron2StreamingInstance::setNextChunkSize(const int chunkSize)
{
    if (!mInUseDecoder)
    {
        throw std::runtime_error(
            "Cannot set next chunk size until "
            "Tacotron2StreamingInstance::startInference() has been called.");
    }
    else if (chunkSize <= 0 || chunkSize > getChunkSize())
    {
        throw std::runtime_error("Invalid chunk size of " + std::to_string(chunkSize)
            + " passed to Tacotron2StreamingInstance::setNextChunkSize().");
    }

    mInUseDecoder->setNextChunkSize(chunkSize);
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void Tacotron2StreamingInstance::resetInference()
{
    mInUseDecoder = nullptr;
    mBatchSize = 0;
}

} // namespace tts
