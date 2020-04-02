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

#include "taco2ProjectionLayerPlugin.h"
#include "taco2ProjectionKernel.h"
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

using value_type = Taco2ProjectionLayerPlugin::value_type;

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const PLUGIN_NAME = "Taco2Projection";
constexpr const char* const PLUGIN_VERSION = "0.1.0";
constexpr const int NUM_INPUTS = 2;

} // namespace

const float Taco2ProjectionLayerPlugin::ONE = 1.0f;
const float Taco2ProjectionLayerPlugin::ZERO = 0.0f;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

std::vector<value_type> toVector(const Weights& weights)
{
    if (weights.type != DataType::kFLOAT)
    {
        throw std::runtime_error(
            "Invalid data type for Taco2Projection weights: " + std::to_string(static_cast<int>(weights.type)));
    }
    const value_type* const valuesBegin = static_cast<const value_type*>(weights.values);
    const value_type* const valuesEnd = valuesBegin + weights.count;
    return std::vector<value_type>(valuesBegin, valuesEnd);
}

const void* offset(const void* ptr, const size_t offset)
{
    return reinterpret_cast<const void*>(static_cast<const uint8_t*>(ptr) + offset);
}

} // namespace

/******************************************************************************
 * STATIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2ProjectionLayerPlugin::getName()
{
    return PLUGIN_NAME;
}

const char* Taco2ProjectionLayerPlugin::getVersion()
{
    return PLUGIN_VERSION;
}

Taco2ProjectionLayerPlugin Taco2ProjectionLayerPlugin::deserialize(const void* const data, const size_t length)
{
    if (length < 4 * sizeof(int32_t))
    {
        throw std::runtime_error("Invalid serialized size: " + std::to_string(length));
    }

    const int hiddenInputLength = static_cast<const int32_t*>(data)[0];
    const int contextInputLength = static_cast<const int32_t*>(data)[1];
    const int numChannelDimension = static_cast<const int32_t*>(data)[2];
    const int numGateDimension = static_cast<const int32_t*>(data)[3];

    const int inputLength = hiddenInputLength + contextInputLength;
    const int numDimensions = numChannelDimension + numGateDimension;

    const size_t reqSize = 4 * sizeof(int32_t) + sizeof(value_type) * ((inputLength * numDimensions) + numDimensions);
    if (reqSize != length)
    {
        throw std::runtime_error(
            "Invalid serialized size: " + std::to_string(length) + " / " + std::to_string(reqSize));
    }

    const Weights weightsChannel{
        DataType::kFLOAT, offset(data, 4 * sizeof(int32_t)), numChannelDimension * inputLength};
    const Weights weightsGate{DataType::kFLOAT,
        offset(weightsChannel.values, sizeof(value_type) * weightsChannel.count), numGateDimension * inputLength};
    const Weights biasChannel{
        DataType::kFLOAT, offset(weightsGate.values, sizeof(value_type) * weightsGate.count), numChannelDimension};
    const Weights biasGate{
        DataType::kFLOAT, offset(biasChannel.values, sizeof(value_type) * biasChannel.count), numGateDimension};
    Taco2ProjectionLayerPlugin layer(weightsChannel, weightsGate, biasChannel, biasGate, hiddenInputLength,
        contextInputLength, numChannelDimension, numGateDimension);

    return layer;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2ProjectionLayerPlugin::Taco2ProjectionLayerPlugin(const nvinfer1::Weights& weightsChannel,
    const nvinfer1::Weights& weightsGate, const nvinfer1::Weights& biasChannel, const nvinfer1::Weights& biasGate,
    const int hiddenInputLength, const int contextInputLength, const int numChannelDimension,
    const int numGateDimension)
    : mHiddenInputLength(hiddenInputLength)
    , mContextInputLength(contextInputLength)
    , mNumChannelDimension(numChannelDimension)
    , mNumGateDimension(numGateDimension)
    , mWeightsChannel(toVector(weightsChannel))
    , mWeightsGate(toVector(weightsGate))
    , mBiasChannel(toVector(biasChannel))
    , mBiasGate(toVector(biasGate))
    , mKernel()
    , mNamespace()
{
    const size_t expectedWeightsChannel = getTotalInputLength() * mNumChannelDimension;
    if (mWeightsChannel.size() != expectedWeightsChannel)
    {
        throw std::runtime_error("Taco2Projection expected " + std::to_string(expectedWeightsChannel)
            + " channel weights but given " + std::to_string(mWeightsChannel.size()));
    }

    const size_t expectedWeightsGate = getTotalInputLength() * mNumGateDimension;
    if (mWeightsGate.size() != expectedWeightsGate)
    {
        throw std::runtime_error("Taco2Projection expected " + std::to_string(expectedWeightsGate)
            + " gate weights but given " + std::to_string(mWeightsGate.size()));
    }

    const size_t expectedBiasChannel = mNumChannelDimension;
    if (mBiasChannel.size() != expectedBiasChannel)
    {
        throw std::runtime_error("Taco2Projection expected " + std::to_string(expectedBiasChannel)
            + " channel bias but given " + std::to_string(mBiasChannel.size()));
    }

    const size_t expectedBiasGate = mNumGateDimension;
    if (mBiasGate.size() != expectedBiasGate)
    {
        throw std::runtime_error("Taco2Projection expected " + std::to_string(expectedBiasGate)
            + " gate bias but given " + std::to_string(mBiasGate.size()));
    }
}

Taco2ProjectionLayerPlugin::Taco2ProjectionLayerPlugin(Taco2ProjectionLayerPlugin&& other)
    : mHiddenInputLength(other.mHiddenInputLength)
    , mContextInputLength(other.mContextInputLength)
    , mNumChannelDimension(other.mNumChannelDimension)
    , mNumGateDimension(other.mNumGateDimension)
    , mWeightsChannel(std::move(other.mWeightsChannel))
    , mWeightsGate(std::move(other.mWeightsGate))
    , mBiasChannel(std::move(other.mBiasChannel))
    , mBiasGate(std::move(other.mBiasGate))
    , mKernel(std::move(other.mKernel))
    , mNamespace(std::move(other.mNamespace))
{
    other.mHiddenInputLength = 0;
    other.mContextInputLength = 0;
    other.mNumChannelDimension = 0;
    other.mNumGateDimension = 0;
}

Taco2ProjectionLayerPlugin::~Taco2ProjectionLayerPlugin()
{
    destroy();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

Taco2ProjectionLayerPlugin& Taco2ProjectionLayerPlugin::operator=(Taco2ProjectionLayerPlugin&& other)
{
    // defere to constructor
    *this = Taco2ProjectionLayerPlugin(std::move(other));

    return *this;
}

DataType Taco2ProjectionLayerPlugin::getOutputDataType(
    const int /* index */, const DataType* const /* inputTypes */, const int /* nbInputs */) const
{
    return DataType::kFLOAT;
}

const char* Taco2ProjectionLayerPlugin::getPluginType() const
{
    return getName();
}

const char* Taco2ProjectionLayerPlugin::getPluginVersion() const
{
    return getVersion();
}

int Taco2ProjectionLayerPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs Taco2ProjectionLayerPlugin::getOutputDimensions(
    const int outputIndex, const DimsExprs* inputs, const int nbInputs, IExprBuilder& exprBuilder)
{
    if (outputIndex >= getNbOutputs())
    {
        throw std::runtime_error("Only has one output.");
    }

    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_INPUTS) + " input tensors: " + std::to_string(nbInputs));
    }
    return DimsExprs{3, {inputs[0].d[0], exprBuilder.constant(1), exprBuilder.constant(getTotalDimensions())}};
}

bool Taco2ProjectionLayerPlugin::supportsFormatCombination(
    const int pos, const PluginTensorDesc* const inOut, const int /* nbInputs */, const int /* nbOutputs */)
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

void Taco2ProjectionLayerPlugin::configurePlugin(const DynamicPluginTensorDesc* const in, const int nbInputs,
    const DynamicPluginTensorDesc* const out, const int nbOutputs)
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
                if (foundDim || dims.d[d] != mHiddenInputLength)
                {
                    throw std::runtime_error(
                        "First projection input must be 1 x hiddenInputLength"
                        " : "
                        + taco2::Taco2Utils::dimsToString(dims));
                }
                foundDim = true;
            }
        }
        if (!foundDim)
        {
            throw std::runtime_error(
                "First projection input must be 1 x hiddenInputLength"
                " : "
                + taco2::Taco2Utils::dimsToString(dims));
        }
    }

    {
        bool foundDim = false;
        const Dims dims = in[1].desc.dims;
        for (int d = 1; d < dims.nbDims; ++d)
        {
            if (dims.d[d] != 1)
            {
                if (foundDim || dims.d[d] != mContextInputLength)
                {
                    throw std::runtime_error(
                        "Second projection input must be 1 x contextInputLength"
                        " : "
                        + taco2::Taco2Utils::dimsToString(dims));
                }
                foundDim = true;
            }
        }
        if (!foundDim)
        {
            throw std::runtime_error(
                "Second projection input must be 1 x contextInputLength"
                " : "
                + taco2::Taco2Utils::dimsToString(dims));
        }
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

int Taco2ProjectionLayerPlugin::initialize()
{
    try
    {
        // concat projection and gate FC layers
        std::vector<float> hostWeightCat;
        hostWeightCat.insert(hostWeightCat.end(), mWeightsChannel.begin(), mWeightsChannel.end());
        hostWeightCat.insert(hostWeightCat.end(), mWeightsGate.begin(), mWeightsGate.end());

        std::vector<float> hostBiasCat;
        hostBiasCat.insert(hostBiasCat.end(), mBiasChannel.begin(), mBiasChannel.end());
        hostBiasCat.insert(hostBiasCat.end(), mBiasGate.begin(), mBiasGate.end());

        mKernel.reset(new Taco2ProjectionKernel(hostWeightCat, hostBiasCat, mHiddenInputLength, mContextInputLength,
            mNumChannelDimension + mNumGateDimension));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Taco2ProjectionLayerPlugin initialization failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

void Taco2ProjectionLayerPlugin::terminate()
{
    mKernel.reset();
}

size_t Taco2ProjectionLayerPlugin::getWorkspaceSize(
    const PluginTensorDesc* const /* in */, const int /* nbInputs */, const PluginTensorDesc* const /* out */, const int /* nbOutputs */) const
{
    return 0;
}

int Taco2ProjectionLayerPlugin::enqueue(const PluginTensorDesc* const inputDesc, const PluginTensorDesc* /* outputDesc */,
    const void* const* const inputs, void* const* const outputs, void* const /* workspace */, cudaStream_t stream)
{
    const int batchSize = inputDesc[0].dims.d[0];

    if (batchSize != 1)
    {
        // we only support batch size of 1 right now
        std::cerr << "Taco2ProjectionLayerPlugin plugin does not support batch size other "
                     "than 1: got "
                  << batchSize << std::endl;
        std::cerr << "Recompile without plugins to use a larger batch size." << std::endl;
        return 1;
    }
    else if (!mKernel)
    {
        std::cerr << "Taco2ProjectionLayerPlugin is not initialized properly." << std::endl;
        return 1;
    }

    // name inputs and outputs
    const value_type* const hiddenDevice = static_cast<const value_type*>(inputs[0]);
    const value_type* const contextDevice = static_cast<const value_type*>(inputs[1]);

    value_type* const outputDevice = static_cast<value_type*>(outputs[0]);

    mKernel->execute(hiddenDevice, contextDevice, outputDevice, stream);

    return 0;
}

size_t Taco2ProjectionLayerPlugin::getSerializationSize() const
{
    return 4 * sizeof(int32_t)
        + sizeof(value_type) * (getTotalInputLength() * getTotalDimensions() + getTotalDimensions());
}

void Taco2ProjectionLayerPlugin::serialize(void* const buffer) const
{
    static_cast<int32_t*>(buffer)[0] = mHiddenInputLength;
    static_cast<int32_t*>(buffer)[1] = mContextInputLength;
    static_cast<int32_t*>(buffer)[2] = mNumChannelDimension;
    static_cast<int32_t*>(buffer)[3] = mNumGateDimension;

    float* const weightsChannel = reinterpret_cast<float*>(static_cast<int32_t*>(buffer) + 4);
    float* const weightsGate = weightsChannel + (getTotalInputLength() * mNumChannelDimension);
    float* const biasChannel = weightsGate + (getTotalInputLength() * mNumGateDimension);
    float* const biasGate = biasChannel + mNumChannelDimension;

    memcpy(weightsChannel, mWeightsChannel.data(), sizeof(value_type) * mWeightsChannel.size());
    memcpy(weightsGate, mWeightsGate.data(), sizeof(value_type) * mWeightsGate.size());
    memcpy(biasChannel, mBiasChannel.data(), sizeof(value_type) * mBiasChannel.size());
    memcpy(biasGate, mBiasGate.data(), sizeof(value_type) * mBiasGate.size());
}

void Taco2ProjectionLayerPlugin::destroy()
{
    terminate();
}

IPluginV2DynamicExt* Taco2ProjectionLayerPlugin::clone() const
{
    // call constructor which copy's data
    Taco2ProjectionLayerPlugin clone(
        Weights{DataType::kFLOAT, mWeightsChannel.data(), static_cast<int64_t>(mWeightsChannel.size())},
        Weights{DataType::kFLOAT, mWeightsGate.data(), static_cast<int64_t>(mWeightsGate.size())},
        Weights{DataType::kFLOAT, mBiasChannel.data(), static_cast<int64_t>(mBiasChannel.size())},
        Weights{DataType::kFLOAT, mBiasGate.data(), static_cast<int64_t>(mBiasGate.size())}, mHiddenInputLength,
        mContextInputLength, mNumChannelDimension, mNumGateDimension);

    if (mKernel)
    {
        // initialize the clone too
        clone.initialize();
    }

    // move it to the heap last to avoid exceptions causing memory leaks
    return new Taco2ProjectionLayerPlugin(std::move(clone));
}

void Taco2ProjectionLayerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2ProjectionLayerPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

int Taco2ProjectionLayerPlugin::getTotalDimensions() const
{
    return mNumChannelDimension + mNumGateDimension;
}

int Taco2ProjectionLayerPlugin::getTotalInputLength() const
{
    return mHiddenInputLength + mContextInputLength;
}

} // namespace plugin
} // namespace nvinfer1
