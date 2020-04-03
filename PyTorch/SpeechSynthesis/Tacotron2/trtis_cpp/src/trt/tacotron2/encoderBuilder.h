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

#ifndef TT2I_ENCODERBUILDER_H
#define TT2I_ENCODERBUILDER_H

#include "IModelImporter.h"
#include "trtPtr.h"

#include <string>

namespace nvinfer1
{
class ICudaEngine;
class IBuilder;
} // namespace nvinfer1

namespace tts
{

class EncoderBuilder
{
public:
    /**
     * @brief Create a new EncoderBuilder.
     *
     * @param numEmbeddingDimensions The number of dimensions in the embedding.
     * @param numEncodingDimensions The number of dimensions in 'memory' output.
     * @param numAttentionDimensions The number of dimensions of the 'processed
     * memory' output.
     * @param inputLength The maximum length of input to support.
     */
    EncoderBuilder(const int numEmbeddingDimensions, const int numEncodingDimensions, const int numAttentionDimensions,
        const int inputLength);

    /**
     * @brief Build a Tacotron2 Encoder engine.
     *
     * @param builder The TRT builder.
     * @param importer The weight importer.
     * @param maxBatchSize The maximum batch size to support.
     * @param useFP16 Whether or not to allow FP16 usage in the build.
     *
     * @return The built engine.
     */
    TRTPtr<nvinfer1::ICudaEngine> build(
        nvinfer1::IBuilder& builder,
        IModelImporter& importer,
        const int maxBatchSize,
        const bool useFP16);

  private:
    int mNumEmbeddingDimensions;
    int mNumEncodingDimensions;
    int mNumAttentionDimensions;
    int mInputLength;
};

} // namespace tts

#endif
