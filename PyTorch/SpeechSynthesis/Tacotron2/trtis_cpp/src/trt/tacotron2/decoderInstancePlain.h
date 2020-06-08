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

#ifndef TT2I_DECODERINSTANCEPLAIN_H
#define TT2I_DECODERINSTANCEPLAIN_H

#include "binding.h"
#include "cudaMemory.h"
#include "decoderInstance.h"

#include "NvInfer.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>

namespace tts
{

class DecoderInstancePlain : public DecoderInstance
{
public:
    static constexpr const char* const ENGINE_NAME = "tacotron2_decoder_plain";

    /**
     * @brief Create a new DecoderInstancePlain.
     *
     * @param engine The ICudaEngine containing the decoder network.
     * @param maxChunkSize The maximum sized chunk the decoder will process.
     */
    DecoderInstancePlain(
        TRTPtr<nvinfer1::ICudaEngine> engine, int maxChunkSize);

    /**
     * @brief Reset the decoder for new input.
     *
     * @param stream The stream to run on.
     */
    void reset(cudaStream_t stream) override;

  protected:
    /**
     * @brief Decode a single frame of output.
     *
     * @param stream The stream to operate on.
     * @param context The execution context.
     * @param batchSize The size of the batch to process.
     * @param inputLastFrameDevice The last frame of output produced (all 0s
     * for first frame).
     * @param inputMemoryDevice The "Memory" tensor on the device.
     * @param inputProcessedMemoryDevice The "Processed Memory" tensor on the
     * device.
     * @param inputMaskDevice The input mask on the device (1 for i < input
     * length, 0 for i >= input length).
     * @param inputLengthHost The length of each input item on the host.
     * @param inputLengthDevice The length of each input on the device.
     * @param inputDropoutsDevice The dropout vector to use on the device.
     * @param outputFrameDevice The output frame on the device.
     */
    void decode(cudaStream_t stream, nvinfer1::IExecutionContext& context, int batchSize,
        const float* inputLastFrameDevice, const float* inputMemoryDevice, const float* inputProcessedMemoryDevice,
        const float* inputMaskDevice, const int32_t* inputLengthHost, const int32_t* inputLengthDevice,
        const float* inputDropoutsDevice, float* outputFrameDevice) override;

private:
    Binding mBinding;

    CudaMemory<float> mInputWeightsDevice;
    CudaMemory<float> mOutputWeightsDevice;
    CudaMemory<float> mInAttentionHiddenStatesDevice;
    CudaMemory<float> mInAttentionCellStatesDevice;
    CudaMemory<float> mOutAttentionHiddenStatesDevice;
    CudaMemory<float> mOutAttentionCellStatesDevice;

    CudaMemory<float> mInputAttentionContextDevice;
    CudaMemory<float> mOutputAttentionContextDevice;
    CudaMemory<float> mInDecoderHiddenStatesDevice;
    CudaMemory<float> mInDecoderCellStatesDevice;
    CudaMemory<float> mOutDecoderHiddenStatesDevice;
    CudaMemory<float> mOutDecoderCellStatesDevice;
};

} // namespace tts

#endif
