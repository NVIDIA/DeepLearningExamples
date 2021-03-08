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

#ifndef TT2I_ENCODERINSTANCE_H
#define TT2I_ENCODERINSTANCE_H

#include "binding.h"
#include "engineDriver.h"
#include "timedObject.h"
#include "trtPtr.h"

#include "NvInfer.h"
#include "cuda_runtime.h"

#include <string>

namespace tts
{

class EncoderInstance : public TimedObject, public EngineDriver
{
public:
    /**
     * @brief Tensor of shape {1 x INPUT_LENGTH}
     */
    static constexpr const char* const INPUT_NAME = "input_encoder";

    /**
     * @brief Tensor of shape {1 x INPUT_LENGTH x 1}
     */
    static constexpr const char* const INPUT_MASK_NAME = "input_encoder_mask";

    /**
     * @brief Tensor of shape {INPUT_LENGTH}
     */
    static constexpr const char* const INPUT_LENGTH_NAME = "input_encoder_length";

    /**
     * @brief Tensor of shape {1 x INPUT_LENGTH x NUM_DIMENSIONS}
     */
    static constexpr const char* const OUTPUT_NAME = "output_encoder";

    /**
     * @brief Tensor of shape {INPUT_LENGTH x NUM_PROCESSED_DIMENSIONS x 1 x 1}
     */
    static constexpr const char* const OUTPUT_PROCESSED_NAME = "output_processed_encoder";
    static constexpr const char* const ENGINE_NAME = "tacotron2_encoder";

    /**
     * @brief Create a new encoder instance.
     *
     * @param engine The TRT Engine implementing Tacotron2's encoder.
     */
    EncoderInstance(TRTPtr<nvinfer1::ICudaEngine> engine);

    // disable copying
    EncoderInstance(const EncoderInstance& other) = delete;
    EncoderInstance& operator=(const EncoderInstance& other) = delete;

    /**
     * @brief Perform inference.
     *
     * @param stream The CUDA stream.
     * @param batchSize The size of the batch.
     * @param inputDevice The input on the GPU.
     * @param inputMaskDevice The input mask on the GPU (all 1's for the length of
     * the actual input and all 0's for the length of the padding).
     * @param inputLengthDevice The length of the input sequences on the GPU.
     * @param outputDevice The output on the GPU (must be of input length x number
     * of encoding dimensions).
     * @param outputProcessedDevice The output on the GPU processed through the
     * memory layer (must be of input length x number of processed dimensions).
     */
    void infer(cudaStream_t stream, int batchSize, const int32_t* inputDevice, const float* inputMaskDevice,
        const int32_t* inputLengthDevice, float* outputDevice, float* outputProcessedDevice);

    /**
     * @brief Get the length of input (padded size).
     *
     * @return The input length.
     */
    int getInputLength() const;

    /**
     * @brief Get the number of encoding dimensions.
     *
     * @return The number of encoding dimensions.
     */
    int getNumDimensions() const;

    /**
     * @brief Get the number of processed dimensions (attention).
     *
     * @return The number of processed dimensions.
     */
    int getNumProcessedDimensions() const;

private:
    Binding mBinding;
    TRTPtr<nvinfer1::IExecutionContext> mContext;
    int mInputLength;
};

} // namespace tts

#endif
