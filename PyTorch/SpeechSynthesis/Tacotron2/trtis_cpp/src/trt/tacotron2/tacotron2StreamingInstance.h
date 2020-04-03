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

#ifndef TT2I_TACOTRON2STREAMINGINSTANCE_H
#define TT2I_TACOTRON2STREAMINGINSTANCE_H

#include "cudaMemory.h"
#include "timedObject.h"
#include "trtPtr.h"

#include <memory>

namespace nvinfer1
{
class ICudaEngine;
}

namespace tts
{

class EncoderInstance;
class DecoderInstance;
class DecoderInstancePlain;
class DecoderInstancePlugins;
class PostNetInstance;

class Tacotron2StreamingInstance : public virtual TimedObject
{
public:
    /**
     * @brief Create a new Tacotron2 instance.
     *
     * @param encoder The built encoder network.
     * @param decoder The built decoder network without plugins.
     * @param decoder The built decoder network with plugins.
     * @param postnet The built postnet network.
     */
  Tacotron2StreamingInstance(
      TRTPtr<nvinfer1::ICudaEngine> encoder,
      TRTPtr<nvinfer1::ICudaEngine> decoderPlain,
      TRTPtr<nvinfer1::ICudaEngine> decoderPlugins,
      TRTPtr<nvinfer1::ICudaEngine> postnet);

  // deleted copy constructor and assignment operator
  Tacotron2StreamingInstance(const Tacotron2StreamingInstance& other) = delete;
  Tacotron2StreamingInstance& operator=(const Tacotron2StreamingInstance& other)
      = delete;

  /**
   * @brief Setup inference for a given input tensor.
   *
   * @param batchSize The number of sequences in the batch.
   * @param inputDevice The input for each item in the batch.
   * @param inputSpacing The spacing between the start of each item in the
   * batch.
   * @param inputLength The length of each input.
   */
  void startInference(
      int batchSize,
      const int* inputDevice,
      int inputSpacing,
      const int* inputLength);

  /**
   * @brief Generate the next chunk of output.
   *
   * @param outputDevice The location to write the output tensor in batch,
   * frame, channel order.
   * @param outputLength The length of each output sequence.
   *
   * @return True if not all sequences have finished.
   */
  bool inferNext(float* outputDevice, int* outputLength);

  /**
   * @brief The random seed to use for dropouts. This resets the
   * inference state, and `startInference()` must be called afterwards.
   *
   * @param seed The seed value.
   */
  void setSeed(unsigned int seed);

  /**
   * @brief Get the number of mels produced at once.
   *
   * @return The number of mels.
   */
  int getChunkSize() const;

  /**
   * @brief Get the number of channels each frame will have.
   *
   * @return The number of channels.
   */
  int getNumMelChannels() const;

  /**
   * @brief Get the maximum length of an input sequence.
   *
   * @return The maximum length of the sequence.
   */
  int getMaximumInputLength() const;

  /**
   * @brief Get the maximum batch size supported by this Tacotron2 instance.
   *
   * @return The maximum batch size.
   */
  int getMaxBatchSize() const;

  /**
   * @brief Set whether or not to use plugins when possible. This resets the
   * inference state, and `startInference()` must be called afterwards.
   *
   * @param usePlugins True to use plugins, false to not.
   */
  void usePlugins(bool usePlugins);

  /**
   * @brief Check whether or not plugins will be used for the given batch size.
   *
   * @param batchSize The batch size.
   *
   * @return True if plugins would be used.
   */
  bool willUsePlugins(int batchSize) const;

  /**
   * @brief Set the number of decoder loops to execute for subsequent calls to
   * nextInfer. The number must be less than or equal to the return of
   * `getMaxChunkSize()`.
   *
   * @param chunkSize The number of frames to generate.
   */
  void setNextChunkSize(int chunkSize);

private:
    // TRT network components
    std::shared_ptr<EncoderInstance> mEncoder;
    std::shared_ptr<DecoderInstancePlain> mDecoderPlain;
    std::shared_ptr<DecoderInstancePlugins> mDecoderPlugins;
    std::shared_ptr<PostNetInstance> mPostnet;

    int mMaxInputLength;
    int mNumMelChannels;
    int mNumMelChunks;
    int mMaxBatchSize;
    int mBatchSize;
    bool mUsePlugins;

    DecoderInstance* mInUseDecoder;

    CudaMemory<int32_t> mPaddedInputDevice;
    CudaMemory<float> mInputMaskDevice;
    CudaMemory<int32_t> mInputLengthsDevice;
    CudaMemory<float> mEncodingDevice;
    CudaMemory<float> mProcessedEncodingDevice;
    CudaMemory<float> mMelChunkDevice;

    const int* mInputLengthHost;

    /**
     * @brief Reset internal variables to prevent `inferNext()` from being
     * called until `startInference()` is called.
     */
    void resetInference();
};

} // namespace tts

#endif
