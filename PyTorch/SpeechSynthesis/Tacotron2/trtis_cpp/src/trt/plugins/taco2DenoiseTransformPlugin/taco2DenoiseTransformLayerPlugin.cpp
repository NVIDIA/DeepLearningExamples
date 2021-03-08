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

#include "taco2DenoiseTransformLayerPlugin.h"
#include "taco2DenoiseTransformKernel.h"
#include "taco2Utils.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h> // cudaError_t
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace tts;

namespace nvinfer1
{
namespace plugin
{

using value_type = Taco2DenoiseTransformLayerPlugin::value_type;

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const PLUGIN_NAME = "Taco2DenoiseTransform";
constexpr const char* const PLUGIN_VERSION = "0.1.0";
constexpr const int NUM_INPUTS = 1;
constexpr const int NUM_OUTPUTS = 1;

} // namespace

/******************************************************************************
 * STATIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2DenoiseTransformLayerPlugin::getName()
{
    return PLUGIN_NAME;
}

const char* Taco2DenoiseTransformLayerPlugin::getVersion()
{
    return PLUGIN_VERSION;
}

Taco2DenoiseTransformLayerPlugin Taco2DenoiseTransformLayerPlugin::deserialize(
    const void* const data, const size_t length)
{
    if (length < sizeof(int32_t) * 2)
    {
        throw std::runtime_error("Invalid serialized size: " + std::to_string(length));
    }

    const int filterLength = static_cast<const int32_t*>(data)[0];
    const int inputLength = static_cast<const int32_t*>(data)[1];
    const int numWeights = filterLength / 2;

    const size_t reqSize = 2 * sizeof(int32_t) + sizeof(value_type) * numWeights;
    if (reqSize != length)
    {
        throw std::runtime_error(
            "Invalid serialized size: " + std::to_string(length) + " / " + std::to_string(reqSize));
    }

    const Weights weights{DataType::kFLOAT, static_cast<const int32_t*>(data) + 2, numWeights};

    Taco2DenoiseTransformLayerPlugin layer(weights, filterLength, inputLength);

    return layer;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2DenoiseTransformLayerPlugin::Taco2DenoiseTransformLayerPlugin(
    const Weights& weights, const int filterLength, const int inputLength) :
    mFilterLength(filterLength),
    mInputLength(inputLength),
    mWeightsHost(taco2::Taco2Utils::toFloatVector(weights)),
    mWeightsDevice(),
    mNamespace()
{
    if (mFilterLength <= 0)
    {
        throw std::runtime_error("Invalid Taco2DenoiseTransform filterLength " + std::to_string(mFilterLength));
    }
    if (mInputLength <= 0)
    {
        throw std::runtime_error("Invalid Taco2DenoiseTransform inputLength: " + std::to_string(mInputLength));
    }

    const int expNumWeights = mFilterLength / 2;
    if (mWeightsHost.size() != static_cast<size_t>(expNumWeights))
    {
        throw std::runtime_error("Incorrect Taco2DenoiseTransform number of weights: "
            + std::to_string(mWeightsHost.size()) + " / " + std::to_string(expNumWeights));
    }
}

Taco2DenoiseTransformLayerPlugin::Taco2DenoiseTransformLayerPlugin(Taco2DenoiseTransformLayerPlugin&& other)
    : mFilterLength(other.mFilterLength)
    , mInputLength(other.mInputLength)
    , mWeightsHost(std::move(other.mWeightsHost))
    , mWeightsDevice(std::move(other.mWeightsDevice))
    , mNamespace(std::move(other.mNamespace))
{
    other.mFilterLength = 0;
    other.mInputLength = 0;
}

Taco2DenoiseTransformLayerPlugin::~Taco2DenoiseTransformLayerPlugin()
{
    destroy();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

Taco2DenoiseTransformLayerPlugin& Taco2DenoiseTransformLayerPlugin::operator=(Taco2DenoiseTransformLayerPlugin&& other)
{
    // defere to constructor
    *this = Taco2DenoiseTransformLayerPlugin(std::move(other));

    return *this;
}

DataType Taco2DenoiseTransformLayerPlugin::getOutputDataType(
    const int /* index */, const DataType* const /* inputTypes */, const int /* nbInputs */) const
{
    return DataType::kFLOAT;
}

bool Taco2DenoiseTransformLayerPlugin::isOutputBroadcastAcrossBatch(
    const int /* outputIndex */, const bool* const /* inputIsBroadCasted */, const int /* nbInputs */) const
{
    return false;
}

bool Taco2DenoiseTransformLayerPlugin::canBroadcastInputAcrossBatch(const int /* inputIndex */) const
{
    return false;
}

const char* Taco2DenoiseTransformLayerPlugin::getPluginType() const
{
    return getName();
}

const char* Taco2DenoiseTransformLayerPlugin::getPluginVersion() const
{
    return getVersion();
}

int Taco2DenoiseTransformLayerPlugin::getNbOutputs() const
{
    return NUM_OUTPUTS;
}

Dims Taco2DenoiseTransformLayerPlugin::getOutputDimensions(
    const int index, const Dims* const /*inputs*/, const int nbInputDims)

{
    if (index >= getNbOutputs())
    {
        throw std::runtime_error("Only has one output.");
    }

    if (nbInputDims != NUM_INPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_INPUTS) + " input tensors: " + std::to_string(nbInputDims));
    }

    // magnitude and phase are of the same size
    return Dims4(1, mFilterLength, 1, mInputLength);
}

bool Taco2DenoiseTransformLayerPlugin::supportsFormat(
    const nvinfer1::DataType type, const nvinfer1::PluginFormat /* format */) const
{
    return type == DataType::kFLOAT;
}

void Taco2DenoiseTransformLayerPlugin::configurePlugin(const nvinfer1::Dims* const inputDims, const int nbInputs,
    const nvinfer1::Dims* const /* outputDims */, const int nbOutputs, const nvinfer1::DataType* const inputTypes,
    const nvinfer1::DataType* const /*outputTypes*/, const bool* const inputIsBroadcast,
    const bool* const /*outputIsBroadcast*/, const nvinfer1::PluginFormat /* format */, const int /* maxBatchSize */)
{
    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_INPUTS) + " input tensors: " + std::to_string(nbInputs));
    }

    if (nbOutputs != NUM_OUTPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_OUTPUTS) + " output tensors: " + std::to_string(nbOutputs));
    }

    for (int i = 0; i < nbInputs; ++i)
    {
        if (inputTypes[i] != DataType::kFLOAT)
        {
            throw std::runtime_error("Only FLOAT supported as input " + std::to_string(i) + " : "
                + std::to_string(static_cast<int>(inputTypes[i])));
        }
        if (inputIsBroadcast[i])
        {
            throw std::runtime_error("Broadcasting input is not supported.");
        }
    }

    // assert dimensions
    {
        const Dims dims = taco2::Taco2Utils::getCompactedDims(inputDims[0], 2);
        if (dims.nbDims != 2 || dims.d[0] != mFilterLength || dims.d[1] != mInputLength)
        {
            throw std::runtime_error("Taco2DenoiseTransform input must be filterLength x inputLength ("
                + std::to_string(mFilterLength) + " x " + std::to_string(mInputLength)
                + ") : " + taco2::Taco2Utils::dimsToString(dims));
        }
    }
}

int Taco2DenoiseTransformLayerPlugin::initialize()
{
    try
    {
      mWeightsDevice = CudaMemory<float>(mWeightsHost);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Taco2DenoiseTransform initialization failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

void Taco2DenoiseTransformLayerPlugin::terminate()
{
  mWeightsDevice.clear();
}

size_t Taco2DenoiseTransformLayerPlugin::getWorkspaceSize(const int /*maxBatchSize*/) const
{
    return 0;
}

int Taco2DenoiseTransformLayerPlugin::enqueue(const int batchSize, const void* const* const inputs,
    void** const outputs, void* const /*workspace*/, cudaStream_t stream)
{
    // name inputs and outputs
    const value_type* const inputDevice = static_cast<const value_type*>(inputs[0]);

    value_type* const outputDevice = static_cast<value_type*>(outputs[0]);

    try
    {
      Taco2DenoiseTransformKernel::compute(
          batchSize,
          inputDevice,
          mWeightsDevice.data(),
          outputDevice,
          mFilterLength / 2,
          mInputLength,
          stream);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to launch Taco2DenoiseTransform kernel due to: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

size_t Taco2DenoiseTransformLayerPlugin::getSerializationSize() const
{
    return sizeof(int32_t) * 2 + sizeof(value_type) * mWeightsHost.size();
}

void Taco2DenoiseTransformLayerPlugin::serialize(void* const buffer) const
{
    static_cast<int32_t*>(buffer)[0] = mFilterLength;
    static_cast<int32_t*>(buffer)[1] = mInputLength;

    value_type* const weights = reinterpret_cast<value_type*>(static_cast<int32_t*>(buffer) + 2);

    memcpy(weights, mWeightsHost.data(), sizeof(value_type) * mWeightsHost.size());
}

void Taco2DenoiseTransformLayerPlugin::destroy()
{
    terminate();
}

IPluginV2Ext* Taco2DenoiseTransformLayerPlugin::clone() const
{
    // call constructor which copy's data
    Taco2DenoiseTransformLayerPlugin clone(
        Weights{DataType::kFLOAT, mWeightsHost.data(), static_cast<int64_t>(mWeightsHost.size())}, mFilterLength,
        mInputLength);

    if (mWeightsDevice.size() > 0)
    {
        // initialize the clone too
        clone.initialize();
    }

    // move it to the heap last to avoid exceptions causing memory leaks
    return new Taco2DenoiseTransformLayerPlugin(std::move(clone));
}

void Taco2DenoiseTransformLayerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2DenoiseTransformLayerPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
