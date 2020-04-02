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

#ifndef TT2I_WAVEGLOWLOADER_H
#define TT2I_WAVEGLOWLOADER_H

#include "waveGlowInstance.h"

#include <memory>
#include <string>

// forward declaration
namespace nvinfer1
{
class ILogger;
class IBuilder;
} // namespace nvinfer1

namespace tts
{

class EngineCache;

class WaveGlowLoader
{
public:
    /**
     * @brief Load a new WaveGlowInstance from an engine file or a ONNX file. If
     * an ONNX file is loaded, build and save the engine to same path with a
     * `.eng` suffix.
     *
     * @param cache The engine cache for loading/saving the engine file.
     * @param builder The TRT builder to use.
     * @param logger The logger to use.
     * @param filename The engine file or ONNX file to load.
     * @param fp16 If building an engine from an ONNX file, whether or not to
     * allow operations to be performed using fp16. If loading an engine file,
     * this input is ignored.
     * @param batchSize If building an engine from an ONNX file, the maximum
     * batch size to support. If loading an engine file,
     * this input is ignored.
     *
     * @return The instantiated WaveGlowInstance.
     */
    static std::shared_ptr<WaveGlowInstance> load(EngineCache& cache, nvinfer1::IBuilder& builder,
        std::shared_ptr<nvinfer1::ILogger> logger, const std::string& filename, bool fp16 = true, int batchSize = 8);
};

} // namespace tts

#endif
