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

#include "speechDataBuffer.h"
#include "checkedCopy.h"
#include "cudaUtils.h"

#include "cuda_runtime.h"

#include <cassert>

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

SpeechDataBuffer::SpeechDataBuffer(
    const int inputSpacing,
    const int melSpacing,
    const int samplesSpacing,
    const int maxBatchSize) :
    TimedObject("SpeechDataBuffer::copyToDevice()/copyFromDevice()"),
    mInputDevice(inputSpacing * maxBatchSize),
    mMelsDevice(melSpacing * maxBatchSize),
    mSamplesDevice(samplesSpacing * maxBatchSize)
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void SpeechDataBuffer::copyToDevice(const int32_t* const inputHost, const size_t size)
{
    if (size > mInputDevice.size())
    {
        throw std::runtime_error("Cannot copy input larger than device input: " + std::to_string(size) + "/"
            + std::to_string(mInputDevice.size()));
    }
    startTiming();
    CheckedCopy::hostToDevice(mInputDevice.data(), inputHost, size);
    stopTiming();
}

void SpeechDataBuffer::copyToDevice(const int batchSize, const std::vector<int32_t>* const inputHost, int& spacing)
{
    startTiming();

    spacing = 0;
    for (int i = 0; i < batchSize; ++i)
    {
        const int inputSize = static_cast<int>(inputHost[i].size());
        if (inputSize > spacing)
        {
            spacing = inputSize;
        }
    }
    const size_t size = spacing * static_cast<size_t>(batchSize);
    if (size > mInputDevice.size())
    {
        throw std::runtime_error("Cannot copy input larger than device input: " + std::to_string(size) + "/"
            + std::to_string(mInputDevice.size()));
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int i = 0; i < batchSize; ++i)
    {
      CheckedCopy::hostToDeviceAsync(
          mInputDevice.data() + (spacing * i),
          inputHost[i].data(),
          inputHost[i].size(),
          stream);
    }
    CudaUtils::sync(stream);
    cudaStreamDestroy(stream);

    stopTiming();
}

void SpeechDataBuffer::copyFromDevice(
    float* const melsHost, const size_t melsSize, float* const samplesHost, const size_t samplesSize)
{
    if (melsHost && melsSize > mMelsDevice.size())
    {
        throw std::runtime_error("Cannot copy mels larger than device mels: " + std::to_string(melsSize) + "/"
            + std::to_string(mMelsDevice.size()));
    }
    if (samplesSize > mSamplesDevice.size())
    {
        throw std::runtime_error("Cannot copy samples larger than device samples: " + std::to_string(samplesSize) + "/"
            + std::to_string(mSamplesDevice.size()));
    }

    startTiming();
    CheckedCopy::deviceToHost(samplesHost, mSamplesDevice.data(), samplesSize);
    if (melsHost)
    {
      CheckedCopy::deviceToHost(melsHost, mMelsDevice.data(), melsSize);
    }
    stopTiming();
}

void SpeechDataBuffer::copyFromDevice(const int batchSize, std::vector<float>* const samplesHost,
    const int sampleSpacing, const int* const samplesLengths)
{
    startTiming();
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < batchSize; ++i)
    {
        assert(samplesLengths[i] <= sampleSpacing);
        samplesHost[i].resize(samplesLengths[i]);
        CheckedCopy::deviceToHostAsync(
            samplesHost[i].data(),
            mSamplesDevice.data() + (sampleSpacing * i),
            samplesLengths[i],
            stream);
    }

    CudaUtils::sync(stream);
    cudaStreamDestroy(stream);

    stopTiming();
}

const int32_t* SpeechDataBuffer::getInputOnDevice() const
{
  return mInputDevice.data();
}

float* SpeechDataBuffer::getMelsOnDevice()
{
  return mMelsDevice.data();
}

float* SpeechDataBuffer::getSamplesOnDevice()
{
  return mSamplesDevice.data();
}

} // namespace tts
