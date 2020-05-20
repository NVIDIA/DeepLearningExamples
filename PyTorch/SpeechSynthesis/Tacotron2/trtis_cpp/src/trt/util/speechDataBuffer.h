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

#ifndef TT2I_SPEECHDATABUFFER_H
#define TT2I_SPEECHDATABUFFER_H

#include "cudaMemory.h"
#include "timedObject.h"

namespace tts
{

class SpeechDataBuffer : public TimedObject
{
public:
    /**
     * @brief Create a new SpeechDataBuffer.
     *
     * @param inputSpacing The maximum spacing between the start of inputs
     * (sequences) in a batch.
     * @param melSpacing The spacing between the start of mel-spectrograms in a
     * batch.
     * @param samplesSpacing The spacing between the start of outputs (samples)
     * in a batch.
     * @param maxBatchSize The maximum batch size.
     */
    SpeechDataBuffer(const int inputSpacing, const int melSpacing, const int samplesSpacing, const int maxBatchSize);

    /**
     * @brief Copy input sequence data from the host to the device.
     *
     * @param inputHost The input on the host.
     * @param size The number of elements to copy. Must be a multiple of
     * inputSpacing.
     */
    void copyToDevice(const int32_t* inputHost, size_t size);

    /**
     * @brief Copy input sequence data from the host to the device.
     *
     * @param batchSize The number of items in the batch.
     * @param inputHost The batch items on the host.
     * @param spacing The spacing between the start of batch items on the GPU
     * (output).
     */
    void copyToDevice(int batchSize, const std::vector<int32_t>* inputHost, int& spacing);

    /**
     * @brief Copy output from the device to the host.
     *
     * @param melsHost The location on the host to copy mel spectrograms to.
     * @param melsSize The number of mel-spectrograms copied.
     * @param samplesHost The location on the host to copy waveform samples to.
     * @param samplesSize The number of samples copied.
     */
    void copyFromDevice(float* melsHost, size_t melsSize, float* samplesHost, size_t samplesSize);

    /**
     * @brief Copy output from the device to the host.
     *
     * @param batchSize The number of items in the batch.
     * @param samplesHost The vectors on the host to fill with waveform audio.
     * @param sampleSpacing The spacing start of each item in the batch on the
     * device.
     * @param samplesLengths The length of each item in the batch.
     */
    void copyFromDevice(int batchSize, std::vector<float>* samplesHost, int sampleSpacing, const int* samplesLengths);

    /**
     * @brief Get the input sequences on the device.
     *
     * @return The input sequences.
     */
    const int32_t* getInputOnDevice() const;

    /**
     * @brief Get the mel-spectrograms on the device.
     *
     * @return The mel-spectrograms.
     */
    float* getMelsOnDevice();

    /**
     * @brief The waveform audio samples on the device.
     *
     * @return The audio samples.
     */
    float* getSamplesOnDevice();

private:
  CudaMemory<int32_t> mInputDevice;
  CudaMemory<float> mMelsDevice;
  CudaMemory<float> mSamplesDevice;
};

} // namespace tts

#endif
