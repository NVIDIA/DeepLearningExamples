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

#include "taco2PrenetLayerPlugin.h"
#include "taco2Utils.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h> // cudaError_t
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{

using value_type = Taco2PrenetLayerPlugin::value_type;

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const PLUGIN_NAME = "Taco2Prenet";
constexpr const char* const PLUGIN_VERSION = "0.1.0";
constexpr const int NUM_INPUTS = 2;

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

const void* offset(const void* ptr, const size_t offset)
{
    return reinterpret_cast<const void*>(static_cast<const uint8_t*>(ptr) + offset);
}

} // namespace

/******************************************************************************
 * STATIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2PrenetLayerPlugin::getName()
{
    return PLUGIN_NAME;
}

const char* Taco2PrenetLayerPlugin::getVersion()
{
    return PLUGIN_VERSION;
}

Taco2PrenetLayerPlugin Taco2PrenetLayerPlugin::deserialize(const void* const data, const size_t length)
{
    if (length < sizeof(int32_t) * 2)
    {
        throw std::runtime_error("Invalid serialized size: " + std::to_string(length));
    }

    const int inputLength = static_cast<const int32_t*>(data)[0];
    const int numDimension = static_cast<const int32_t*>(data)[1];
    const size_t reqSize = 2 * sizeof(int32_t) + sizeof(value_type) * ((inputLength + numDimension) * numDimension);
    if (reqSize != length)
    {
        throw std::runtime_error(
            "Invalid serialized size: " + std::to_string(length) + " / " + std::to_string(reqSize));
    }

    const Weights weights1{DataType::kFLOAT, offset(data, 2 * sizeof(int32_t)), numDimension * inputLength};
    const Weights weights2{DataType::kFLOAT, offset(weights1.values, sizeof(value_type) * numDimension * inputLength),
        numDimension * numDimension};
    Taco2PrenetLayerPlugin layer(weights1, weights2, inputLength, numDimension);

    return layer;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2PrenetLayerPlugin::Taco2PrenetLayerPlugin(
    const Weights& weights1, const Weights& weights2, const int inputLength, const int numDimension)
    : mInputLength(inputLength)
    , mNumDimension(numDimension)
    , mWeights1Host(taco2::Taco2Utils::toFloatVector(weights1))
    , mWeights2Host(taco2::Taco2Utils::toFloatVector(weights2))
    , mKernel()
    , mNamespace()
{

    if (mNumDimension <= 0)
    {
        throw std::runtime_error("Invalid Taco2Prenet dimension: " + std::to_string(mNumDimension));
    }
}

Taco2PrenetLayerPlugin::Taco2PrenetLayerPlugin(Taco2PrenetLayerPlugin&& other)
    : mInputLength(other.mInputLength)
    , mNumDimension(other.mNumDimension)
    , mWeights1Host(std::move(other.mWeights1Host))
    , mWeights2Host(std::move(other.mWeights2Host))
    , mKernel(std::move(other.mKernel))
    , mNamespace(std::move(other.mNamespace))
{
    other.mInputLength = 0;
    other.mNumDimension = 0;
}

Taco2PrenetLayerPlugin::~Taco2PrenetLayerPlugin()
{
    destroy();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

DataType Taco2PrenetLayerPlugin::getOutputDataType(
    const int /* index */, const DataType* const /* inputTypes */, const int /* nbInputs */) const
{
    return DataType::kFLOAT;
}

const char* Taco2PrenetLayerPlugin::getPluginType() const
{
    return getName();
}

const char* Taco2PrenetLayerPlugin::getPluginVersion() const
{
    return getVersion();
}

int Taco2PrenetLayerPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs Taco2PrenetLayerPlugin::getOutputDimensions(
    const int index, const DimsExprs* const inputs, const int nbInputs, IExprBuilder& expBuilder)
{
    if (index >= getNbOutputs())
    {
        throw std::runtime_error("Only has one output.");
    }

    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_INPUTS) + " input tensors: " + std::to_string(nbInputs));
    }
    return DimsExprs{3, {inputs[0].d[0], expBuilder.constant(1), expBuilder.constant(mNumDimension)}};
}

bool Taco2PrenetLayerPlugin::supportsFormatCombination(
    const int pos, const PluginTensorDesc* inOut, const int /* nbInputs */,
    const int /* nbOutputs */)
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

void Taco2PrenetLayerPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, const int nbInputs, const DynamicPluginTensorDesc* out, const int nbOutputs)
{
    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_INPUTS) + " input tensors: " + std::to_string(nbInputs));
    }

    for (int i = 0; i < nbInputs; ++i)
    {
        if (in[i].desc.type != DataType::kFLOAT)
        {
            throw std::runtime_error("Only FLOAT supported as input " + std::to_string(i) + " : "
                + std::to_string(static_cast<int>(in[i].desc.type)));
        }
    }

    // assert dimensions
    {
        bool foundDim = false;
        const Dims dims = in[0].desc.dims;
        for (int d = 1; d < dims.nbDims; ++d)
        {
            if (dims.d[d] != 1)
            {
                if (foundDim || dims.d[d] < mInputLength)
                {
                    throw std::runtime_error("Taco2Prenet input must be 1* x inputLength ("
                        + std::to_string(mInputLength) + ") : " + taco2::Taco2Utils::dimsToString(dims));
                }
                foundDim = true;
            }
        }
        if (!foundDim)
        {
            throw std::runtime_error("Taco2Prenet input must be 1* x inputLength (" + std::to_string(mInputLength)
                + ") x 1* : " + taco2::Taco2Utils::dimsToString(dims));
        }
    }

    {
        bool foundDim = false;
        const Dims dims = in[1].desc.dims;
        for (int d = 1; d < dims.nbDims; ++d)
        {
            if (dims.d[d] != 1)
            {
                if (foundDim || dims.d[d] != mNumDimension)
                {
                    throw std::runtime_error("Taco2Prenet input must be 1* x numDimension ("
                        + std::to_string(mNumDimension) + ") : " + taco2::Taco2Utils::dimsToString(dims));
                }
                foundDim = true;
            }
        }
        if (!foundDim)
        {
            throw std::runtime_error("Query input must be 1* x numDimension (" + std::to_string(mNumDimension)
                + ") x 1* : " + taco2::Taco2Utils::dimsToString(dims));
        }
    }

    if (nbOutputs != 1)
    {
        throw std::runtime_error("Only one output is implemented: " + std::to_string(nbOutputs));
    }
    for (int i = 0; i < nbOutputs; ++i)
    {
        if (out[i].desc.type != DataType::kFLOAT)
        {
            throw std::runtime_error("Only FLOAT supported as output: " + std::to_string(i) + " : "
                + std::to_string(static_cast<int>(out[i].desc.type)));
        }
    }
}

int Taco2PrenetLayerPlugin::initialize()
{
    try
    {
        mKernel.reset(new Taco2PrenetKernel(mWeights1Host, mWeights2Host, mInputLength, mNumDimension));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Taco2PrenetLayerPlugin initialization failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

void Taco2PrenetLayerPlugin::terminate()
{
    mKernel.reset();
}

size_t Taco2PrenetLayerPlugin::getWorkspaceSize(const PluginTensorDesc* in, const int /* nbInputs */,
    const PluginTensorDesc* /* out */, const int /* nbOutputs */) const
{
    return in[0].dims.d[0] * mNumDimension * sizeof(value_type);
}

int Taco2PrenetLayerPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* /* outputDesc */,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const int batchSize = inputDesc[0].dims.d[0];

    if (batchSize != 1)
    {
        // we only support batch size of 1 right now
        std::cerr << "Taco2PrenetLayerPlugin plugin does not support batch size other than 1: got " << batchSize
                  << std::endl;
        std::cerr << "Recompile without plugins to use a larger batch size." << std::endl;
        return 1;
    }
    else if (!mKernel)
    {
        std::cerr << "Taco2PrenetLayerPlugin is not initialized properly." << std::endl;
        return 1;
    }

    // name inputs and outputs
    const value_type* const inputDevice = static_cast<const value_type*>(inputs[0]);
    const value_type* const dropoutDevice = static_cast<const value_type*>(inputs[1]);

    value_type* const outputDevice = static_cast<value_type*>(outputs[0]);

    mKernel->execute(inputDevice, dropoutDevice, outputDevice, static_cast<float*>(workspace), stream);

    return 0;
}

size_t Taco2PrenetLayerPlugin::getSerializationSize() const
{
    return NUM_INPUTS * sizeof(int32_t) + sizeof(value_type) * (mNumDimension + mInputLength) * mNumDimension;
}

void Taco2PrenetLayerPlugin::serialize(void* const buffer) const
{
    static_cast<int32_t*>(buffer)[0] = mInputLength;
    static_cast<int32_t*>(buffer)[1] = mNumDimension;
    float* const weights1 = reinterpret_cast<float*>(static_cast<int32_t*>(buffer) + 2);
    float* const weights2 = weights1 + (mInputLength * mNumDimension);

    memcpy(weights1, mWeights1Host.data(), sizeof(value_type) * mWeights1Host.size());
    memcpy(weights2, mWeights2Host.data(), sizeof(value_type) * mWeights2Host.size());
}

void Taco2PrenetLayerPlugin::destroy()
{
    terminate();
}

IPluginV2DynamicExt* Taco2PrenetLayerPlugin::clone() const
{
    // call constructor which copy's data
    Taco2PrenetLayerPlugin clone(
        Weights{DataType::kFLOAT, mWeights1Host.data(), static_cast<int64_t>(mWeights1Host.size())},
        Weights{DataType::kFLOAT, mWeights2Host.data(), static_cast<int64_t>(mWeights2Host.size())}, mInputLength,
        mNumDimension);

    if (mKernel)
    {
        // initialize the clone too
        clone.initialize();
    }

    // move it to the heap last to avoid exceptions causing memory leaks
    return new Taco2PrenetLayerPlugin(std::move(clone));
}

void Taco2PrenetLayerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2PrenetLayerPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
