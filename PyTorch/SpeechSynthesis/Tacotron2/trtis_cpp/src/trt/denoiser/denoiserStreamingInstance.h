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

#ifndef TT2I_DENOISERSTREAMINGINSTANCE_H
#define TT2I_DENOISERSTREAMINGINSTANCE_H

#include "binding.h"
#include "engineDriver.h"
#include "timedObject.h"

namespace nvinfer1
{
class ICudaEngine;
class IExecutionContext;
} // namespace nvinfer1

namespace tts
{

class DenoiserStreamingInstance : public TimedObject, public EngineDriver
{
public:
    /**
     * @brief Tensor of shape {1 x INPUT_LENGTH}
     */
    static constexpr const char* const INPUT_NAME = "input_denoiser";

    /**
     * @brief Tensor of shape {1 x OUTPUT_LENGTH}
     */
    static constexpr const char* const OUTPUT_NAME = "output_denoiser";

    /**
     * @brief Create a new denoiser.
     *
     * @param sampleNoise The audio sample of what should be "noise" to be
     * removed.
     * @param sampleLength The number of samples in the "noise".
     * @param filterLength The filter length.
     * @param overlapLength The length of overlap between filters.
     * @param winLength The length of the window.
     */
    DenoiserStreamingInstance(TRTPtr<nvinfer1::ICudaEngine>&& engine);

    /**
     * @brief Start a new session for performing streaming inference. This
     * method should be called before the first call to `inferNext()`.
     */
    void startInference();

    /**
     * @brief Perform inference on a chunk of input.
     *
     * @param batchSize The size of the batch to process.
     * @param inputDevice The input tensor on the device.
     * @param outputDevice The output tensor on the device.
     * @param stream The stream to operate on.
     */
    void inferNext(const int batchSize, const float* inputDevice, float* outputDevice, cudaStream_t stream);

    /**
     * @brief Get the size of the chunk the denoiser will process.
     *
     * @return The size of the chunk.
     */
    int getChunkSize() const
    {
        return mChunkSize;
    }

private:
    Binding mBinding;
    TRTPtr<nvinfer1::IExecutionContext> mContext;
    int mChunkSize;
};

} // namespace tts

#endif
