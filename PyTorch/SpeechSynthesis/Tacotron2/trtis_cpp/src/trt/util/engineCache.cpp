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

#include "engineCache.h"
#include "logging.h"

#include "NvInferPlugin.h"

#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const SEP = "/";
constexpr size_t BUFFER_SIZE = 4 * 1024 * 1024; // 4MB

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
void readNum(std::istream& in, T* num)
{
    const size_t size = sizeof(*num);
    in.read(reinterpret_cast<char*>(num), size);
    const size_t numRead = static_cast<size_t>(in.gcount());
    if (numRead != size)
    {
        throw std::runtime_error(
            "Failed to parse number: " + std::to_string(numRead) + " of " + std::to_string(size) + " bytes read.");
    }
}

std::ifstream openFileForRead(const std::string& filename)
{
    std::ifstream fin(filename);
    if (!fin.good())
    {
        throw std::runtime_error("Failed to open '" + filename + "'.");
    }
    fin.exceptions(std::ofstream::badbit);
    return fin;
}

TRTPtr<ICudaEngine>
loadRawEngine(IRuntime& runtime, std::istream& in, const size_t size)
{
    std::vector<char> data(size);
    in.read(data.data(), size);
    if (static_cast<size_t>(in.gcount()) != size)
    {
        throw std::runtime_error("Failed read entire engine from file.");
    }

    TRTPtr<ICudaEngine> engine(
        runtime.deserializeCudaEngine(data.data(), data.size(), nullptr));

    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize engine.");
    }

    return engine;
}

TRTPtr<ICudaEngine> loadEngine(IRuntime& runtime, std::istream& in)
{
    uint64_t size;
    readNum(in, &size);

    return loadRawEngine(runtime, in, size);
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

EngineCache::EngineCache(std::shared_ptr<ILogger> logger)
    : mLogger(logger)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (!mRuntime)
    {
      mRuntime.reset(createInferRuntime(*mLogger));
    }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

TRTPtr<ICudaEngine> EngineCache::load(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(mMutex);

    std::ifstream fin(filename, std::ifstream::binary);
    fin.exceptions(std::ofstream::badbit);

    fin.seekg(0, std::ifstream::end);
    const size_t size = fin.tellg();
    fin.seekg(0, std::ifstream::beg);

    TRTPtr<ICudaEngine> engine = loadRawEngine(*mRuntime, fin, size);
    if (!engine)
    {
      throw std::runtime_error(
          "Failed to load engine from '" + filename + "'.");
    }

    return engine;
}

std::vector<TRTPtr<ICudaEngine>>
EngineCache::loadComposite(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(mMutex);

    try
    {
        std::ifstream fin = openFileForRead(filename);
        fin.exceptions(std::ofstream::badbit);

        // load the number of engines
        uint32_t numEngines;
        readNum(fin, &numEngines);

        std::vector<TRTPtr<ICudaEngine>> engines;
        for (uint32_t i = 0; i < numEngines; ++i)
        {
            engines.emplace_back(loadEngine(*mRuntime, fin));
        }
        assert(engines.size() == numEngines);

        return engines;
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Failed to load multi-engine from '" + filename + "': " + e.what());
    }
}

bool EngineCache::has(const std::string& filename) const
{
    std::ifstream fin(filename);
    return fin.good();
}

void EngineCache::save(const ICudaEngine& engine, const std::string& filename)
{
  std::ofstream fout(filename, std::ofstream::trunc | std::ofstream::binary);
  if (!fout) {
    throw std::runtime_error(
        "Failed to open file '" + filename + "' for writing.");
  }

  try {
    fout.exceptions(std::ofstream::failbit);
    TRTPtr<IHostMemory> serialData(engine.serialize());
    fout.write(
        static_cast<const char*>(serialData->data()), serialData->size());
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to save engine to '" + filename + "' due to: " + e.what());
    }
}

void EngineCache::save(
    const std::vector<TRTPtr<ICudaEngine>>& engines,
    const std::string& filename)
{
    try
    {
      std::ofstream fout(
          filename, std::ofstream::trunc | std::ofstream::binary);
      if (!fout) {
        throw std::runtime_error(
            "Failed to open file '" + filename + "' for writing.");
      }
      fout.exceptions(std::ofstream::failbit);

      const uint32_t num = static_cast<uint32_t>(engines.size());
      fout.write(reinterpret_cast<const char*>(&num), sizeof(num));

      for (const TRTPtr<ICudaEngine>& engine : engines) {
        if (!engine) {
          throw std::runtime_error("Cannot save null engine.");
        }
        TRTPtr<IHostMemory> serialData(engine->serialize());
        const uint64_t size = serialData->size();
        fout.write(reinterpret_cast<const char*>(&size), sizeof(size));
        fout.write(
            static_cast<const char*>(serialData->data()), serialData->size());
        }
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Failed to save multi-engine to '" + filename + "': " + e.what());
    }
}

TRTPtr<nvinfer1::IRuntime> EngineCache::mRuntime;
std::mutex EngineCache::mMutex;

} // namespace tts
