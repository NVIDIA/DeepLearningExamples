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

#ifndef TT2I_SPEECHSYNTHESIZER_H
#define TT2I_SPEECHSYNTHESIZER_H

#include "denoiserInstance.h"
#include "speechDataBuffer.h"
#include "tacotron2Instance.h"
#include "timedObject.h"
#include "waveGlowInstance.h"

#include <memory>

namespace tts
{

class SpeechSynthesizer : public virtual TimedObject
{
public:
    SpeechSynthesizer(std::shared_ptr<Tacotron2Instance> tacotron, std::shared_ptr<WaveGlowInstance> waveglow);

    SpeechSynthesizer(std::shared_ptr<Tacotron2Instance> tacotron, std::shared_ptr<WaveGlowInstance> waveglow,
        std::shared_ptr<DenoiserInstance> denoiser);

    SpeechSynthesizer(const SpeechSynthesizer& other) = delete;
    SpeechSynthesizer& operator=(const SpeechSynthesizer& other) = delete;

    /**
     * @brief Perform inference in order to generate audio.
     *
     * @param batchSize The size of the batch to run.
     * @param inputDevice The input sequences.
     * @param inputSpacing The spacing between the start of each input sequence
     * in the batch.
     * @param inputLength The length of each input sequence.
     * @param samplesDevice The samples of audio to generate. This must be of
     * length equal to the batchSize times the return of `getMaxOutputSize()`
     * (output).
     * @param numSamples The number of samples in each audio segment (output).
     * @param outputMelsDevice The mels to output on the device (optional).
     * @param outputNumMels The number of mels generated (optional).
     */
    void infer(int batchSize, const int* inputDevice, int inputSpacing, const int* inputLength, float* samplesDevice,
        int* numSamples, float* outputMelsDevice = nullptr, int* outputNumMels = nullptr);

    /**
     * @brief Perform inference in order to generate audio, but taking inputs
     * from host memory.
     *
     * @param batchSize The size of the batch to run.
     * @param inputHost The input sequences.
     * @param inputSpacing The spacing between the start of each input sequence
     * in the batch.
     * @param inputLength The length of each input sequence.
     * @param samplesHost The samples of audio to generate. This must be of
     * length equal to the batchSize times the return of `getMaxOutputSize()`
     * (output).
     * @param numSamples The number of samples in each audio segment (output).
     * @param outputMelsHost The mels to output optional).
     * @param outputNumMels The number of mels generated (optional).
     */
    void inferFromHost(int batchSize, const int* inputHost, int inputSpacing, const int* inputLength,
        float* samplesHost, int* numSamples, float* outputMelsHost = nullptr, int* outputNumMels = nullptr);

    /**
     * @brief Perform inference in order to generate audio, but taking inputs
     * from host memory.
     *
     * @param batchSize The size of the batch to run.
     * @param inputHost The input sequences (must be of length batchSize).
     * @param outputHost The samples of audio to generate (must be of length
     * batchSize).
     */
    void inferFromHost(int batchSize, const std::vector<int32_t>* inputHost, std::vector<float>* outputHost);

    /**
     * @brief Get the maximum batch size that can be ran.
     *
     * @return The maximum batch size.
     */
    int getMaxBatchSize() const;

    /**
     * @brief Get the maximum input size for each item in the batch.
     *
     * @return The maximum input size.
     */
    int getMaxInputSize() const;

    /**
     * @brief Get the maximum number of samples that will be returned for each
     * item in the batch.
     *
     * @return The maximum output size.
     */
    int getMaxOutputSize() const;

    /**
     * @brief Get the spacing in frames between the start of mel-spectrogram
     * sequences in batches.
     *
     * @return The spacing in frames.
     */
    int getMelSpacing() const;

private:
    int mMaxBatchSize;
    int mNumMaxMels;
    std::vector<int> mNumSymbols;
    std::vector<int> mNumFrames;
    std::vector<int> mNumSamples;
    std::shared_ptr<Tacotron2Instance> mTacotron;
    std::shared_ptr<WaveGlowInstance> mWaveglow;
    std::shared_ptr<DenoiserInstance> mDenoiser;

    SpeechDataBuffer mBuffer;
};

} // namespace tts

#endif
