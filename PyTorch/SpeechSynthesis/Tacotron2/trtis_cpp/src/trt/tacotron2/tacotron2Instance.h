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

#ifndef TT2I_TACOTRON2INSTANCE_H
#define TT2I_TACOTRON2INSTANCE_H

#include "tacotron2StreamingInstance.h"
#include "timedObject.h"
#include "trtPtr.h"

#include <memory>

namespace nvinfer1
{
class ICudaEngine;
}

namespace tts
{

class Tacotron2Instance : public virtual TimedObject
{
public:
    static constexpr const char* const ENGINE_NAME = "tacotron2";

    /**
     * @brief Create a new Tacotron2 instance.
     *
     * @param encoder The built encoder network.
     * @param decoder The built decoder network without plugins.
     * @param decoder The built decoder network with plugins.
     * @param postnet The built postnet network.
     */
    Tacotron2Instance(
        TRTPtr<nvinfer1::ICudaEngine> encoder,
        TRTPtr<nvinfer1::ICudaEngine> decoderPlain,
        TRTPtr<nvinfer1::ICudaEngine> decoderPlugins,
        TRTPtr<nvinfer1::ICudaEngine> postnet);

    /**
     * @brief Perform inference on a given batch of input data.
     *
     * @param batchSize The number of sequences in the batch.
     * @param inputDevice The input for each item in the batch.
     * @param inputSpacing The spacing between the start of each item in the
     * batch.
     * @param inputLength The length of each input.
     * @param maxOutputLength The maximum length of output in frames.
     * @param outputDevice The location to write the output tensor in batch,
     * frame, channel order.
     * @param outputLength The length of each output sequence.
     */
    void infer(int batchSize, const int* inputDevice, int inputSpacing, const int* inputLength, int maxOutputLength,
        float* outputDevice, int* outputLength);

    /**
     * @brief Set whether or not the decoder loop should exit when the stop
     * criteria is satisfied, or the maximum number of iterations should be taken.
     *
     * @param earlyExit Set to true exit when the criteria is met, and false to
     * only exit after all iterations are run.
     */
    void setEarlyExit(bool earlyExit);

    /**
     * @brief The random seed to use for dropouts.
     *
     * @param seed The seed value.
     */
    void setSeed(unsigned int seed);

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
     * @brief Get the size of the `outputDevice` vector required for the giving
     * input parameters.
     *
     * @param batchSize The size of the batch.
     * @param maxFrames The maximum number of frames for each item in the batch.
     *
     * @return The required number of elements in the output vector.
     */
    int getRequiredOutputSize(const int batchSize, const int maxFrames) const;

    /**
     * @brief Set whether or not to use plugins when possible.
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

private:
    Tacotron2StreamingInstance mStreamingInstance;
    std::vector<int> mChunkSize;

    int mNumMelChunks;
    bool mEarlyExit;

    CudaMemory<float> mOutputShuffledDevice;
};

} // namespace tts

#endif
