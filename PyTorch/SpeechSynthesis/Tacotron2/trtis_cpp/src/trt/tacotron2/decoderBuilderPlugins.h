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

#ifndef TT2I_DECODERBUILDERPLUGINS_H
#define TT2I_DECODERBUILDERPLUGINS_H

#include "IModelImporter.h"
#include "trtPtr.h"

#include "cuda_runtime.h"

#include <memory>
#include <string>

namespace nvinfer1
{
class INetworkDefinition;
class IBuilder;
} // namespace nvinfer1

namespace tts
{

class DecoderBuilderPlugins
{
public:
    /**
     * @brief Create a new DecoderBuilderPlugins.
     *
     * @param numDim The number of dimensions of the input tensor.
     * @param numChannels The number of channels to be output.
     */
    DecoderBuilderPlugins(int numDim, int numChannels);

    /**
     * @brief Build the ICudaEngine for the decoder.
     *
     * @param builder The engine builder.
     * @param importer The model weight importer.
     * @param maxBatchSize The maximum batch size to support. This must be 1.
     * @param minInputLength The minimum input length to support.
     * @param maxInputLength The maximum input length to support.
     * @param useFP16 Whether or not to allow FP16 usage in the build.
     *
     * @return The built engine.
     */
    TRTPtr<nvinfer1::ICudaEngine> build(
        nvinfer1::IBuilder& builder,
        IModelImporter& importer,
        const int maxBatchSize,
        const int minInputLength,
        const int maxInputLength,
        const bool useFP16);

  private:
    int mNumEncodingDim;
    int mNumPrenetDim;
    int mNumAttentionRNNDim;
    int mNumAttentionDim;
    int mNumAttentionFilters;
    int mAttentionKernelSize;
    int mNumLSTMDim;
    int mNumChannels;
};

} // namespace tts

#endif
