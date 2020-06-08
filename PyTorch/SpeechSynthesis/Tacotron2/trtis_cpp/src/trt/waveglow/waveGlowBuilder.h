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

#ifndef TT2I_WAVEGLOWBUILDER_H
#define TT2I_WAVEGLOWBUILDER_H

#include "trtPtr.h"

#include <memory>
#include <string>

namespace nvinfer1
{
class ICudaEngine;
class IBuilder;
class INetworkDefinition;
class ILogger;
} // namespace nvinfer1

namespace tts
{

class WaveGlowBuilder
{
public:
    /**
     * @brief Create a new WaveGlowBuilder.
     *
     * @param modelPath The path of the ONNX file to load.
     * @param logger The logger to use while parsing.
     */
    WaveGlowBuilder(const std::string& modelPath, std::shared_ptr<nvinfer1::ILogger> logger);

    /**
     * @brief Create a new WaveGlow engine.
     *
     * @param builder The builder.
     * @param maxBatchSize The maximum batch size the engine should be able to
     * handle.
     * @param useFP16 Whether or not to allow FP16 in the engine.
     *
     * @return The built engine.
     */
    TRTPtr<nvinfer1::ICudaEngine> build(
        nvinfer1::IBuilder& builder,
        const int maxBatchSize,
        const bool useFP16);

  private:
    std::string mOnnxModelPath;
    std::shared_ptr<nvinfer1::ILogger> mLogger;
};

} // namespace tts

#endif
