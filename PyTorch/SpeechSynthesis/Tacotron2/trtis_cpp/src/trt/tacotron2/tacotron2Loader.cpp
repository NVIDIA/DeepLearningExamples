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

#include "tacotron2Loader.h"
#include "encoderInstance.h"
#include "engineCache.h"
#include "tacotron2Builder.h"
#include "trtUtils.h"
#include "utils.h"

#include "NvInfer.h"

#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

std::shared_ptr<Tacotron2Instance> Tacotron2Loader::load(EngineCache& cache, IBuilder& builder,
    const std::string& filename, const int inputLength, const bool fp16, const int batchSize)
{
  std::vector<TRTPtr<ICudaEngine>> engines;
  if (Utils::hasExtension(filename, ".pt")
      || Utils::hasExtension(filename, ".json")) {

    Tacotron2Builder tacotron2Builder(filename);
    engines = tacotron2Builder.build(inputLength, builder, batchSize, fp16);

    // save generated engine
    const std::string engFilename(
        filename + "_" + std::to_string(inputLength) + ".eng");
    cache.save(engines, engFilename);
    }
    else if (Utils::hasExtension(filename, ".eng"))
    {
        engines = cache.loadComposite(filename);

        for (size_t i = 0; i < engines.size(); ++i)
        {
          const TRTPtr<ICudaEngine>& engine = engines[i];
          // make sure all engines except the plugin engine can support the
          // batch size, or if we don't have both a plain and plugin engine,
          // make sure the batch size is supported
          if (!(engines.size() == 4 && i == 2)
              && engine->getMaxBatchSize() < batchSize) {
            throw std::runtime_error(
                "Engine " + filename + ":" + std::to_string(i)
                + " does not support "
                  " the requested batch size: "
                + std::to_string(engine->getMaxBatchSize()) + " / "
                + std::to_string(batchSize)
                + ". "
                  "Rebuild the engine with the larger batch size.");
            }
            const int maxLen = TRTUtils::getBindingSize(*engines[0], EncoderInstance::INPUT_NAME);
            if (inputLength > maxLen)
            {
                throw std::runtime_error(
            "Engine " + filename
            + " is built for a "
              "maximum input length of "
            + std::to_string(maxLen) + " but " + std::to_string(inputLength)
            + " is requested. Rebuild the engine "
              "with the larger input size.");
            }
        }
    }
    else
    {
        throw std::runtime_error("Unknown model file type: " + filename);
    }

    if (engines.size() != 4)
    {
        throw std::runtime_error(
            "Invalid engine file, contains " + std::to_string(engines.size()) + " engines, but expected 4.");
    }

    return std::make_shared<Tacotron2Instance>(
        std::move(engines[0]), std::move(engines[1]), std::move(engines[2]), std::move(engines[3]));
}

} // namespace tts
