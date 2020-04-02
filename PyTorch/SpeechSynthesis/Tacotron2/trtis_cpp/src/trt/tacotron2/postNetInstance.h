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

#ifndef TT2I_POSTNETINSTANCE_H
#define TT2I_POSTNETINSTANCE_H

#include "binding.h"
#include "engineDriver.h"
#include "timedObject.h"
#include "trtPtr.h"

#include "NvInfer.h"
#include "cuda_runtime.h"

#include <string>

namespace tts
{

class PostNetInstance : public TimedObject, public EngineDriver
{
public:
    /**
     * @brief Tensor of shape {1 x NUM_CHANNELS x NUM_FRAMES x 1 }
     */
    static constexpr const char* const INPUT_NAME = "input_postnet";

    /**
     * @brief Tensor of s hape {1 x NUM_FRAMES x NUM_CHANNELS x 1}
     */
    static constexpr const char* const OUTPUT_NAME = "output_postnet";
    static constexpr const char* const ENGINE_NAME = "tacotron2_postnet";

    /**
     * @brief Create a new PostNetInstance.
     *
     * @param engine The ICudaEngine containing the built network.
     */
    PostNetInstance(TRTPtr<nvinfer1::ICudaEngine> engine);

    // disable copying
    PostNetInstance(const PostNetInstance& other) = delete;
    PostNetInstance& operator=(const PostNetInstance& other) = delete;

    /**
     * @brief Perform inference through this network (apply the postnet).
     *
     * @param stream The cuda stream.
     * @param batchSize The size of the batch to run.
     * @param inputDevice The input tensor on the GPU.
     * @param outputDevice The output tensor on the GPU.
     */
    void infer(cudaStream_t stream, int batchSize, const void* inputDevice, void* outputDevice);

    /**
     * @brief Get the number of mel-scale spectrograms the postnet processes at
     * once.
     *
     * @return The number mels that will be processed at once.
     */
    int getMelChunkSize() const;

    /**
     * @brief Get the number of mel-scale spectrograms channels the postnet is
     * configured for.
     *
     * @return The number of mel channels.
     */
    int getNumMelChannels() const;

    /**
     * @brief Get the total size of the output tensor.
     *
     * @return The total size.
     */
    int getOutputSize() const;

private:
    Binding mBinding;
    TRTPtr<nvinfer1::IExecutionContext> mContext;
};

} // namespace tts

#endif
