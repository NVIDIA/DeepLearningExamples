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

#ifndef TT2I_DECODERINSTANCE_H
#define TT2I_DECODERINSTANCE_H

#include "binding.h"
#include "dropoutGenerator.h"
#include "engineCache.h"
#include "engineDriver.h"
#include "hostMemory.h"
#include "timedObject.h"
#include "trtPtr.h"

#include "NvInfer.h"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <memory>
#include <string>

namespace tts
{

class DecoderInstance : public TimedObject, public EngineDriver
{
public:
    static constexpr const char* const INPUT_MASK_NAME = "input_decoder_mask";
    static constexpr const char* const INPUT_LENGTH_NAME = "input_decoder_length";
    static constexpr const char* const INPUT_DROPOUT_NAME = "input_decoder_dropout";
    static constexpr const char* const INPUT_LASTFRAME_NAME = "input_decoder_lastframe";
    static constexpr const char* const INPUT_MEMORY_NAME = "input_attention_memory";
    static constexpr const char* const INPUT_PROCESSED_NAME = "input_attention_processed";
    static constexpr const char* const INPUT_WEIGHTS_NAME = "input_attention_weights";
    static constexpr const char* const INPUT_CONTEXT_NAME = "input_attentionlstm_contextinput";
    static constexpr const char* const INPUT_ATTENTIONHIDDEN_NAME = "input_attentionlstm_hidden";
    static constexpr const char* const INPUT_ATTENTIONCELL_NAME = "input_attentionlstm_cell";
    static constexpr const char* const INPUT_DECODERHIDDEN_NAME = "input_decoderlstm_hidden";
    static constexpr const char* const INPUT_DECODERCELL_NAME = "input_decoderlstm_cell";

    static constexpr const char* const OUTPUT_ATTENTIONHIDDEN_NAME = "output_attentionlstm_hidden";
    static constexpr const char* const OUTPUT_ATTENTIONCELL_NAME = "output_attentionlstm_cell";
    static constexpr const char* const OUTPUT_CONTEXT_NAME = "output_attention_context";
    static constexpr const char* const OUTPUT_WEIGHTS_NAME = "output_attention_weight";
    static constexpr const char* const OUTPUT_DECODERHIDDEN_NAME = "output_decoderlstm_hidden";
    static constexpr const char* const OUTPUT_DECODERCELL_NAME = "output_decoderlstm_cell";
    static constexpr const char* const OUTPUT_CHANNELS_NAME = "output_projection_channels";
    static constexpr const char* const OUTPUT_GATE_NAME = "output_projection_gates";

    /**
     * @brief Create a new DecoderInstance.
     *
     * @param engine The ICudaEngine containing the decoder network.
     * @param maxChunkSize The maximum sized chunk the decoder will process.
     */
    DecoderInstance(TRTPtr<nvinfer1::ICudaEngine> engine, int maxChunkSize);

    /**
     * @brief Do inference.
     *
     * @param stream The cuda stream.
     * @param batchSize The size of the batch to perform inference on.
     * @param inputDevice The input tensor on the device (memory).
     * @param inputProcessedDevice The processed input tensor (memory_procssed).
     * @param inputMaskDevice The mask of the input.
     * @param inputLengthHost The length of the input on the host for eeach
     * sequence.
     * @param inputLengthDevice The length of the input in a 1x1 tensor
     * (e.g., [[inputLength]]).
     * @param outputDevice The output tensor on the device.
     */
    virtual void infer(cudaStream_t stream, int batchSize, const float* inputDevice, const float* inputProcessedDevice,
        const float* inputMaskDevice, const int32_t* inputLengthHost, const int32_t* inputLengthDevice,
        float* outputDevice);

    /**
     * @brief Get the size of the last chunk processed.
     *
     * @return The size of the last chunk processed for each item in the batch.
     */
    const int* lastChunkSize() const;

    /**
     * @brief Check if the decoder has finished processing the whole batch.
     *
     * @return True if decoding has finished.
     */
    bool isAllDone() const;

    /**
     * @brief Reset the decoder for new input.
     */
    virtual void reset();

    /**
     * @brief Set the number of decoder loops to execute for subsequent calls to
     * infer. The number must be less than or equal to the return of
     * `getMaxChunkSize()`. By default this is equal to `getMaxChunkSize()`,
     * and upon calls to `reset()` it returns to that value.
     *
     * @param chunkSize The number of frames to generate.
     */
    void setNextChunkSize(int chunkSize);

    /**
     * @brief The random seed to use for dropouts.
     *
     * @param seed The seed value.
     */
    void setSeed(unsigned int seed);

    /**
     * @brief Get maximum size of the next chunk of mel spectrograms frames to be
     * generated.
     *
     * @return The number of frames that will be generated.
     */
    int getNextChunkSize() const;

    /**
     * @brief Get the maximum chunk size that can be generated.
     *
     * @return The maximum chunk size in frames.
     */
    int getMaxChunkSize() const;

    /**
     * @brief Get the dropout values used for inference. Must have called
     * `setSaveDropouts()` to true before calling this method.
     *
     * @return The dropout values.
     */
    std::vector<float> getDropoutRecord() const;

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
    virtual void decode(cudaStream_t stream, nvinfer1::IExecutionContext& context, int batchSize,
        const float* inputLastFrameDevice, const float* inputMemoryDevice, const float* inputProcessedMemoryDevice,
        const float* inputMaskDevice, const int32_t* inputLengthHost, const int32_t* inputLengthDevice,
        const float* inputDropoutsDevice, float* outputFrameDevice)
        = 0;

private:
  TRTPtr<nvinfer1::IExecutionContext> mContext;

  int mMaxChunkSize;
  int mNextChunkSize;
  int mNumChannels;
  float mStopThreshold;

  int mBatchSize;
  unsigned int mSeed;

  std::vector<int> mLastChunkSize;
  std::vector<int> mDone;

  DropoutGenerator mDropout;

  CudaMemory<float> mDecoderInputDevice;
  CudaMemory<float> mGateOutputDevice;
  CudaMemory<float> mOutputTransposedDevice;
  HostMemory<float> mOutputGateHost;
};

} // namespace tts

#endif
