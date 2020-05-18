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

#include "speechSynthesizer.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr int MAX_NUM_MELS_PER_CHAR = 10;
}

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

int maxMelsFromChars(const int numChars)
{
  return numChars * MAX_NUM_MELS_PER_CHAR + 100;
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

SpeechSynthesizer::SpeechSynthesizer(
    std::shared_ptr<Tacotron2Instance> tacotron,
    std::shared_ptr<WaveGlowInstance> waveglow,
    std::shared_ptr<DenoiserInstance> denoiser) :
    TimedObject("SpeechSynthsizer::infer()"),
    mMaxBatchSize(
        std::min(tacotron->getMaxBatchSize(), waveglow->getMaxBatchSize())),
    mNumMaxMels(maxMelsFromChars(tacotron->getMaximumInputLength())),
    mNumSymbols(mMaxBatchSize),
    mNumFrames(mMaxBatchSize),
    mNumSamples(mMaxBatchSize),
    mTacotron(tacotron),
    mWaveglow(waveglow),
    mDenoiser(denoiser),
    mBuffer(
        mTacotron->getMaximumInputLength(),
        getMelSpacing() * mTacotron->getNumMelChannels(),
        getMaxOutputSize(),
        mMaxBatchSize)
{
    addChild(mTacotron.get());
    addChild(mWaveglow.get());
    if (mDenoiser)
    {
        addChild(mDenoiser.get());
    }
    addChild(&mBuffer);
}

SpeechSynthesizer::SpeechSynthesizer(
    std::shared_ptr<Tacotron2Instance> tacotron, std::shared_ptr<WaveGlowInstance> waveglow)
    : SpeechSynthesizer(tacotron, waveglow, std::shared_ptr<DenoiserInstance>(nullptr))
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void SpeechSynthesizer::infer(const int batchSize, const int* const inputDevice, const int inputSpacing,
    const int* const inputLength, float* const samplesDevice, int* const numSamples, float* const outputMelsDevice,
    int* const outputNumMels)
{
    startTiming();
    if (batchSize > mMaxBatchSize)
    {
        throw std::runtime_error("Maximum batch size is " + std::to_string(mMaxBatchSize) + ". Cannot run with "
            + std::to_string(batchSize));
    }

    // determine whether to use internal storage or expose the intermediate
    // mel spectrograms to the caller
    float* melFramesDevice;
    if (outputMelsDevice)
    {
        melFramesDevice = outputMelsDevice;
    }
    else
    {
        melFramesDevice = mBuffer.getMelsOnDevice();
    }

    int* melLengths;
    if (outputNumMels)
    {
        melLengths = outputNumMels;
    }
    else
    {
        melLengths = mNumFrames.data();
    }
    mTacotron->infer(
        batchSize,
        inputDevice,
        inputSpacing,
        inputLength,
        mNumMaxMels,
        melFramesDevice,
        melLengths);

    mWaveglow->infer(
        batchSize,
        melFramesDevice,
        mNumMaxMels,
        melLengths,
        getMaxOutputSize(),
        samplesDevice,
        numSamples);

    if (mDenoiser)
    {
        mDenoiser->infer(batchSize, samplesDevice, getMaxOutputSize(), numSamples, samplesDevice);
    }

    stopTiming();
}

void SpeechSynthesizer::inferFromHost(const int batchSize, const int* const inputHost, const int inputSpacing,
    const int* const inputLength, float* const samplesHost, int* const numSamples, float* const outputMelsHost,
    int* const outputNumMels)
{
    if (batchSize > mMaxBatchSize)
    {
        throw std::runtime_error("Maximum batch size is " + std::to_string(mMaxBatchSize) + ". Cannot run with "
            + std::to_string(batchSize));
    }

    startTiming();
    // copy data to GPU and do any lazy allocation
    const size_t inputSize = inputSpacing * batchSize;
    mBuffer.copyToDevice(inputHost, inputSize);
    stopTiming();

    infer(batchSize, mBuffer.getInputOnDevice(), inputSpacing, inputLength, mBuffer.getSamplesOnDevice(), numSamples,
        mBuffer.getMelsOnDevice(), outputNumMels);

    startTiming();
    const size_t melSize = mTacotron->getNumMelChannels() * getMelSpacing() * batchSize;
    const size_t outputSize = getMaxOutputSize() * batchSize;
    mBuffer.copyFromDevice(outputMelsHost, melSize, samplesHost, outputSize);
    stopTiming();
}

void SpeechSynthesizer::inferFromHost(
    const int batchSize, const std::vector<int32_t>* const inputHost, std::vector<float>* const outputHost)
{
    startTiming();
    if (batchSize > mMaxBatchSize)
    {
        throw std::runtime_error("Maximum batch size is " + std::to_string(mMaxBatchSize) + ". Cannot run with "
            + std::to_string(batchSize));
    }

    // copy data to GPU and do any lazy allocation
    int inputSpacing;
    mBuffer.copyToDevice(batchSize, inputHost, inputSpacing);
    stopTiming();

    // setup input lengths
    for (int i = 0; i < batchSize; ++i)
    {
        mNumSymbols[i] = static_cast<int>(inputHost[i].size());
        assert(mNumSymbols[i] <= inputSpacing);
    }

    infer(batchSize, mBuffer.getInputOnDevice(), inputSpacing, mNumSymbols.data(), mBuffer.getSamplesOnDevice(),
        mNumSamples.data());

    startTiming();
    mBuffer.copyFromDevice(batchSize, outputHost, getMaxOutputSize(), mNumSamples.data());
    stopTiming();
}

int SpeechSynthesizer::getMaxBatchSize() const
{
    return mMaxBatchSize;
}

int SpeechSynthesizer::getMaxInputSize() const
{
    return mTacotron->getMaximumInputLength();
}

int SpeechSynthesizer::getMelSpacing() const
{
  return mNumMaxMels;
}

int SpeechSynthesizer::getMaxOutputSize() const
{
  return mNumMaxMels * mWaveglow->getNumberOfSamplesPerFrame();
}

} // namespace tts
