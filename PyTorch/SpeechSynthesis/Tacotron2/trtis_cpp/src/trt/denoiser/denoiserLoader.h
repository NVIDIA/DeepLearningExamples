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

#ifndef TT2I_DENOISERLOADER_H
#define TT2I_DENOISERLOADER_H

#include "denoiserInstance.h"

#include <memory>
#include <string>

namespace nvinfer1
{
class IBuilder;
}

namespace tts
{

class EngineCache;

class DenoiserLoader
{
public:
    /**
     * @brief Load a new DenoiserInstance from an engine file or a json file.
     *
     * @param cache The engine cache.
     * @param builder The TensorRT Engine Builder.
     * @param filename The name of the engine/json file.
     * @param fp16 If building an engine from a json file, whether or not to
     * allow fp16 operations. If loading an engine file, this input is ignored.
     * @param batchSize If building an engine from a json file, the maximum batch
     * size to support. If loading an engine file, this input is ignored.
     *
     * @return The newly created DenoiserInstance.
     */
    static std::shared_ptr<DenoiserInstance> load(EngineCache& cache, nvinfer1::IBuilder& builder,
        const std::string& filename, bool fp16 = true, int batchSize = 8);
};

} // namespace tts

#endif
