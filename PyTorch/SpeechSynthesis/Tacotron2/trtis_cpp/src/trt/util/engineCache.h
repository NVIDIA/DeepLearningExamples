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

#ifndef TT2I_ENGINECACHE_H
#define TT2I_ENGINECACHE_H

#include "trtPtr.h"

#include "NvInfer.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tts
{

class EngineCache
{
public:
    /**
     * @brief Create a new EngineCache object.
     *
     * @param logger The logger to use.
     */
    EngineCache(std::shared_ptr<nvinfer1::ILogger> logger);

    /**
     * @brief Load a single TRT engine from a file.
     *
     * @param name The name of the file.
     *
     * @return The instantiated engine.
     */
    TRTPtr<nvinfer1::ICudaEngine> load(const std::string& name);

    /**
     * @brief Load multiple TRT engines from a single file.
     *
     * @param name The name of the file.
     *
     * @return The instantiated engines.
     */
    std::vector<TRTPtr<nvinfer1::ICudaEngine>>
    loadComposite(const std::string& name);

    /**
     * @brief Check if an engine is available for loading.
     *
     * @param name The filename.
     *
     * @return True if the file exists and is accessible.
     */
    bool has(const std::string& name) const;

    /**
     * @brief Save the given engine to a file.
     *
     * @param engine The engine to save.
     * @param name The filename.
     */
    void save(const nvinfer1::ICudaEngine& engine, const std::string& name);

    /**
     * @brief Save multiple engines to a single file.
     *
     * @param engines The set of engines to save.
     * @param name The name of the file to save the engines to.
     */
    void save(
        const std::vector<TRTPtr<nvinfer1::ICudaEngine>>& engines,
        const std::string& name);

  private:
    std::shared_ptr<nvinfer1::ILogger> mLogger;

    static TRTPtr<nvinfer1::IRuntime> mRuntime;
    static std::mutex mMutex;
};

} // namespace tts

#endif
