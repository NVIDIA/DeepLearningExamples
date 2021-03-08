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

#include "waveGlowLoader.h"
#include "engineCache.h"
#include "trtUtils.h"
#include "utils.h"
#include "waveGlowBuilder.h"

#include "NvInfer.h"

#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

std::shared_ptr<WaveGlowInstance> WaveGlowLoader::load(EngineCache& cache, IBuilder& builder,
    std::shared_ptr<ILogger> logger, const std::string& filename, const bool fp16, const int batchSize)
{
  TRTPtr<ICudaEngine> engine;
  if (Utils::hasExtension(filename, ".onnx")) {
    WaveGlowBuilder waveGlowBuilder(filename, logger);
    engine = waveGlowBuilder.build(builder, batchSize, fp16);

    // save generated engine
    const std::string engFilename(filename + ".eng");
    cache.save(*engine, engFilename);
    }
    else if (Utils::hasExtension(filename, ".eng"))
    {
        engine = cache.load(filename);

        if (TRTUtils::getMaxBatchSize(*engine) < batchSize)
        {
            throw std::runtime_error(
          "Engine " + filename
          + " does not support "
            " the requested batch size: "
          + std::to_string(engine->getMaxBatchSize()) + " / "
          + std::to_string(batchSize)
          + ". "
            "Rebuild the engine with the larger batch size.");
        }
    }
    else
    {
        throw std::runtime_error("Unknown model file type: " + filename);
    }

    return std::make_shared<WaveGlowInstance>(std::move(engine));
}

} // namespace tts
