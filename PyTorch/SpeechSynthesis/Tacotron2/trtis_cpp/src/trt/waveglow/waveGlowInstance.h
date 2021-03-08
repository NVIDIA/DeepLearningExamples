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

#ifndef TT2I_WAVEGLOWINSTANCE_H
#define TT2I_WAVEGLOWINSTANCE_H

#include "timedObject.h"
#include "trtPtr.h"
#include "waveGlowStreamingInstance.h"

namespace nvinfer1
{
class ICudaEngine;
}

namespace tts
{

class WaveGlowInstance : public TimedObject
{
public:
    static constexpr const char* const ENGINE_NAME = "waveglow_chunk160_fp16";

    /**
     * @brief Create a new WaveGlowInstance from a deserialied engine.
     *
     * @param engine The deserialized engine.
     */
    WaveGlowInstance(TRTPtr<nvinfer1::ICudaEngine> engine);

    // disable copying
    WaveGlowInstance(const WaveGlowInstance& other) = delete;
    WaveGlowInstance& operator=(const WaveGlowInstance& other) = delete;

    /**
     * @brief Perform inference on a set of mel-scale spectrograms.
     *
     * @param batchSize The number of items in the batch.
     * @param mels The mel-scale spectro grams in batch, sequence, channel
     * order.
     * @param melSpacing The offset from the start of subsequent spectrograms.
     * @param numMels The number of spectrograms per batch item (must be less or
     * equal to melSpacing).
     * @param numMaxSamples The maximum number of samples to generate per batch
     * item.
     * @param samples The location to output samples to (each will start at item
     * ID x numMaxSamples).
     * @param numSamples The number of samples for each output.
     */
    void infer(const int batchSize, const float* mels, const int melSpacing, const int* numMels,
        const int numMaxSamples, float* samples, int* numSamples);

    /**
     * @brief Get the number of samples that will be generated per mel-scale
     * spectrogram.
     *
     * @return The number of samples.
     */
    int getNumberOfSamplesPerFrame() const;

    /**
     * @brief Get the frequency of the generated audio.
     *
     * @return The frequency.
     */
    int getFrequency() const;

    /**
     * @brief Get the maximum batch size this object can perform inference with.
     *
     * @return The maximum batch size.
     */
    int getMaxBatchSize() const;

private:
    WaveGlowStreamingInstance mStreamingInstance;

    int mFrequency;
    int mNumOverlap;
    int mIndependentChunkSize;
    int mIndependentChunkSampleSize;

    std::vector<int> mNumChunkMels;
    std::vector<int> mNumChunkSamples;

    CudaMemory<float> mInputFrame;
    CudaMemory<float> mOutputFrame;
};

} // namespace tts

#endif
