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

#ifndef TT2I_TACOTRON2BUILDER_H
#define TT2I_TACOTRON2BUILDER_H

#include "tacotron2Instance.h"

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
class ICudaEngine;
class IBuilder;
} // namespace nvinfer1

namespace tts
{

class Tacotron2Builder
{
public:
    /**
     * @brief Create a new tacotron2 builder.
     *
     * @param modelFilePath The path to the tacotron2 jit model to load weights
     * from.
     */
    Tacotron2Builder(const std::string& modelFilepath);

    /**
     * @brief Build the set of engines for Tacotron2.
     *
     * @param maxInputLength The maximum input length.
     * @param builder The builder to use.
     * @param maxBatchSize The maximum batch size to build the engines for.
     * @param useFP16 whether or not to allow FP16.
     *
     * @return The build engines.
     */
    std::vector<TRTPtr<nvinfer1::ICudaEngine>> build(
        int maxInputLength,
        nvinfer1::IBuilder& builder,
        const int maxBatchSize,
        const bool useFP16);

  private:
    std::string mModelFilePath;
    int mMelChannels;
};

} // namespace tts

#endif
