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

#ifndef TT2I_WAVEGLOWSTREAMINGINSTANCE_H
#define TT2I_WAVEGLOWSTREAMINGINSTANCE_H

#include "binding.h"
#include "engineDriver.h"
#include "normalDistribution.h"
#include "timedObject.h"
#include "trtPtr.h"

namespace nvinfer1
{
class ICudaEngine;
}

namespace tts
{

class WaveGlowStreamingInstance : public TimedObject, public EngineDriver
{
public:
    /**
     * @brief Create a new WaveGlowInstance from a deserialied engine.
     *
     * @param engine The deserialized engine.
     */
  WaveGlowStreamingInstance(TRTPtr<nvinfer1::ICudaEngine>&& engine);

  // disable copying
  WaveGlowStreamingInstance(const WaveGlowStreamingInstance& other) = delete;
  WaveGlowStreamingInstance& operator=(const WaveGlowStreamingInstance& other)
      = delete;

  /**
   * @brief Initialize for a new round of inference. This method must be called
   * before calls to `inferNext()`, however, the stream does not need to be
   * synchronized on inbetween.
   *
   * @param stream The stream to initialize on.
   */
  void startInference(int batchSize, cudaStream_t stream);

  /**
   * @brief Perform infernece on a chunk of mel-scale spectrograms. The stream
   * must be synchronized on befor reading the output or modifying the input.
   *
   * @param stream The stream to perform inference on.
   * @param batchSize The number of items in the batch.
   * @param melsDevice The mel-scale spectrograms of all batch items.
   * @param numMels The number of mel-scale spectrograms in each item.
   * @param samplesDevice The output waveform for all items. This should be of
   * size equal to the result of `getRequiredOutputBufferSize(batchSize)`.
   * @param numSamples The number of samples per item generated.
   */
  void inferNext(
      cudaStream_t stream,
      const float* melsDevice,
      const int* numMels,
      float* samplesDevice,
      int* numSamples);

  /**
   * @brief Get the spacing between the start of each input item in terms of
   * mel-spectrograms. This also serves as the maximum input length.
   *
   * @return The number of mel-spectrogram frames.
   */
  int getMelSpacing() const;

  /**
   * @brief Get the maximum number of useful samples that will be produced.
   * The space allocated for the output vector may need to be longer than this.
   * `getRequiredOutputBufferSize()` should be used for determing the amount of
   * space to allocate for output.
   *
   * @return The maximum number.
   */
  int getMaxOutputLength() const;

  /**
   * @brief Get the required size of the output buffer that will be given to
   * `inferNext()`.
   *
   * @param batchSize The number of items in the batch.
   *
   * @return The required size in number of samples (floats).
   */
  int getRequiredOutputBufferSize(const int batchSize) const;

  /**
   * @brief Get the number of samples that will be generated per mel-scale
   * spectrogram.
   *
   * @return The number of samples.
   */
  int getNumberOfSamplesPerFrame() const;

  /**
   * @brief Get the number of mel-scale spectrogram channels expected per frame
   * of the input.
   *
   * @return The number of channels.
   */
  int getNumMelChannels() const;

  /**
   * @brief Get the spacing between start of each item in the batch in the
   * output.
   *
   * @return The spacing.
   */
  int getOutputSpacing() const;

private:
    int mChunkSize;
    int mSamplesPerFrame;
    int mChunkSampleSize;
    int mTruncatedChunkSampleSize;
    int mInputChannels;
    int mZChannels;
    int mBatchSize;
    Binding mBinding;
    TRTPtr<nvinfer1::IExecutionContext> mContext;

    NormalDistribution mRand;

    CudaMemory<float> mZ;
};

} // namespace tts

#endif
