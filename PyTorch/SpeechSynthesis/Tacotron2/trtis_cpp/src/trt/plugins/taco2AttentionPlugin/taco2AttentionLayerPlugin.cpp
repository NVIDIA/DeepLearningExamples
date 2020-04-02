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

#include "taco2AttentionLayerPlugin.h"
#include "taco2AttentionLayerKernel.h"
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

using value_type = Taco2AttentionLayerPlugin::value_type;

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const PLUGIN_NAME = "Taco2Attention";
constexpr const char* const PLUGIN_VERSION = "0.1.0";

} // namespace

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
            "Invalid data type for Attention weights: " + std::to_string(static_cast<int>(weights.type)));
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

const char* Taco2AttentionLayerPlugin::getName()
{
    return PLUGIN_NAME;
}

const char* Taco2AttentionLayerPlugin::getVersion()
{
    return PLUGIN_VERSION;
}

Taco2AttentionLayerPlugin Taco2AttentionLayerPlugin::deserialize(const void* const data, const size_t length)
{
    static constexpr const size_t numDims = 5;
    if (length < numDims * sizeof(int32_t))
    {
        throw std::runtime_error("Invalid serialized size: " + std::to_string(length));
    }

    const int numEncodingDimension = static_cast<const int32_t*>(data)[0];
    const int numQueryDimension = static_cast<const int32_t*>(data)[1];
    const int numFilters = static_cast<const int32_t*>(data)[2];
    const int convKernelSize = static_cast<const int32_t*>(data)[3];
    const int numAttentionDimension = static_cast<const int32_t*>(data)[4];

    const int numQueryWeights = numQueryDimension * numAttentionDimension;
    const int numConvWeights = numFilters * 2 * convKernelSize;
    const int numLocationWeights = numFilters * numAttentionDimension;
    const int numEnergyWeights = numAttentionDimension;

    const size_t reqSize = numDims * sizeof(int32_t)
        + sizeof(value_type) * (numQueryWeights + numConvWeights + numLocationWeights + numEnergyWeights);
    if (reqSize != length)
    {
        throw std::runtime_error(
            "Invalid serialized size: " + std::to_string(length) + " / " + std::to_string(reqSize));
    }

    const Weights queryWeights{DataType::kFLOAT, offset(data, numDims * sizeof(int32_t)), numQueryWeights};
    const Weights convWeights{
        DataType::kFLOAT, offset(queryWeights.values, sizeof(value_type) * numQueryWeights), numConvWeights};
    const Weights locationWeights{
        DataType::kFLOAT, offset(convWeights.values, sizeof(value_type) * numConvWeights), numLocationWeights};
    const Weights energyWeights{
        DataType::kFLOAT, offset(locationWeights.values, sizeof(value_type) * numLocationWeights), numEnergyWeights};

    return Taco2AttentionLayerPlugin(numEncodingDimension, numQueryDimension, numFilters, convKernelSize,
        numAttentionDimension, queryWeights, convWeights, locationWeights, energyWeights);
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2AttentionLayerPlugin::Taco2AttentionLayerPlugin(int encDimension, int queryDimension, int numFilters,
    int convKernelSize, int attDimension, const nvinfer1::Weights& queryWeights, const nvinfer1::Weights& convWeights,
    const nvinfer1::Weights& locationWeights, const nvinfer1::Weights& energyWeights)
    : mNumEncodingDimension(encDimension)
    , mNumQueryDimension(queryDimension)
    , mNumFilters(numFilters)
    , mConvKernelSize(convKernelSize)
    , mNumAttentionDimension(attDimension)
    , mQueryWeightsHost(toVector(queryWeights))
    , mConvWeightsHost(toVector(convWeights))
    , mLocationWeightsHost(toVector(locationWeights))
    , mEnergyWeightsHost(toVector(energyWeights))
    , mKernel(nullptr)
    , mNamespace()
{
    const size_t expectedQueryWeights = mNumQueryDimension * mNumAttentionDimension;
    const size_t expectedConvWeights = mNumFilters * mConvKernelSize * 2;
    const size_t expectedLocationWeights = mNumFilters * mNumAttentionDimension;
    const size_t expectedEnergyWeights = mNumAttentionDimension;

    if (mQueryWeightsHost.size() != expectedQueryWeights)
    {
        throw std::runtime_error("Attention expected " + std::to_string(expectedQueryWeights)
            + " query weights but given " + std::to_string(mQueryWeightsHost.size()));
    }
    if (mConvWeightsHost.size() != expectedConvWeights)
    {
        throw std::runtime_error("Attention expected " + std::to_string(expectedConvWeights)
            + " conv weights but given " + std::to_string(mConvWeightsHost.size()));
    }
    if (mLocationWeightsHost.size() != expectedLocationWeights)
    {
        throw std::runtime_error("Attention expected " + std::to_string(expectedLocationWeights)
            + " location weights but given " + std::to_string(mLocationWeightsHost.size()));
    }
    if (mEnergyWeightsHost.size() != expectedEnergyWeights)
    {
        throw std::runtime_error("Attention expected " + std::to_string(expectedEnergyWeights)
            + " energy weights but given " + std::to_string(mEnergyWeightsHost.size()));
    }
}

Taco2AttentionLayerPlugin::Taco2AttentionLayerPlugin(Taco2AttentionLayerPlugin&& other)
    : mNumEncodingDimension(other.mNumEncodingDimension)
    , mNumQueryDimension(other.mNumQueryDimension)
    , mNumFilters(other.mNumFilters)
    , mConvKernelSize(other.mConvKernelSize)
    , mNumAttentionDimension(other.mNumAttentionDimension)
    , mQueryWeightsHost(std::move(other.mQueryWeightsHost))
    , mConvWeightsHost(std::move(other.mConvWeightsHost))
    , mLocationWeightsHost(std::move(other.mLocationWeightsHost))
    , mEnergyWeightsHost(std::move(other.mEnergyWeightsHost))
    , mKernel(std::move(other.mKernel))
    , mNamespace(std::move(other.mNamespace))
{
    other.mNumEncodingDimension = 0;
    other.mNumQueryDimension = 0;
    other.mNumFilters = 0;
    other.mConvKernelSize = 0;
    other.mNumAttentionDimension = 0;
}

Taco2AttentionLayerPlugin::~Taco2AttentionLayerPlugin()
{
    destroy();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

Taco2AttentionLayerPlugin& Taco2AttentionLayerPlugin::operator=(Taco2AttentionLayerPlugin&& other)
{
    // defere to constructor
    *this = Taco2AttentionLayerPlugin(std::move(other));

    return *this;
}

DataType Taco2AttentionLayerPlugin::getOutputDataType(
    const int /* index */, const DataType* const /* inputTypes */, const int /* nbInputs */) const
{
    return DataType::kFLOAT;
}

const char* Taco2AttentionLayerPlugin::getPluginType() const
{
    return getName();
}

const char* Taco2AttentionLayerPlugin::getPluginVersion() const
{
    return getVersion();
}

int Taco2AttentionLayerPlugin::getNbOutputs() const
{
    return 2;
}

DimsExprs Taco2AttentionLayerPlugin::getOutputDimensions(
    const int outputIndex, const DimsExprs* inputs, const int nbInputs, IExprBuilder& exprBuilder)
{
    if (outputIndex >= getNbOutputs())
    {
        throw std::runtime_error(
            "Invalid output index: " + std::to_string(outputIndex) + " / " + std::to_string(getNbOutputs()) + ".");
    }

    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error(
            "Can only handle " + std::to_string(NUM_INPUTS) + " input tensors: " + std::to_string(nbInputs));
    }

    if (outputIndex == CONTEXT_OUTPUT)
    {
        return DimsExprs{
            3, {inputs[MEMORY_INDEX].d[0], exprBuilder.constant(1), exprBuilder.constant(mNumEncodingDimension)}};
    }
    else if (outputIndex == WEIGHT_OUTPUT)
    {
        return DimsExprs{3, {inputs[MEMORY_INDEX].d[0], exprBuilder.constant(2), inputs[MEMORY_INDEX].d[1]}};
    }
    else
    {
        throw std::runtime_error("Unknown output index: " + std::to_string(outputIndex));
    }
}

bool Taco2AttentionLayerPlugin::supportsFormatCombination(
    const int pos, const PluginTensorDesc* const inOut, const int /* nbInputs */, const int /* nbOutputs */)
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

void Taco2AttentionLayerPlugin::configurePlugin(const DynamicPluginTensorDesc* const in, const int nbInputs,
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
    if (in[MEMORY_INDEX].desc.dims.d[2] != mNumEncodingDimension)
    {
        throw std::runtime_error("Memory input must be L x " + std::to_string(mNumEncodingDimension) + " but got "
            + taco2::Taco2Utils::dimsToString(in[MEMORY_INDEX].desc.dims));
    }

    if (in[PROCESSED_MEMORY_INDEX].desc.dims.d[2] != mNumAttentionDimension)
    {
        throw std::runtime_error("Processed Memory input must be L x " + std::to_string(mNumAttentionDimension)
            + " but got " + taco2::Taco2Utils::dimsToString(in[PROCESSED_MEMORY_INDEX].desc.dims));
    }
    if (in[WEIGHT_INDEX].desc.dims.d[1] != 2)
    {
        throw std::runtime_error(
            "Weights input must be 2 x L but got " + taco2::Taco2Utils::dimsToString(in[WEIGHT_INDEX].desc.dims));
    }

    if (taco2::Taco2Utils::getDimensionsSize(in[ATTENTION_HIDDEN_INDEX].desc.dims)
        != static_cast<size_t>(mNumQueryDimension))
    {
        throw std::runtime_error("Attention hidden input must be " + std::to_string(mNumQueryDimension) + " but got "
            + taco2::Taco2Utils::dimsToString(in[ATTENTION_HIDDEN_INDEX].desc.dims) + " ("
            + std::to_string(taco2::Taco2Utils::getDimensionsSize(in[ATTENTION_HIDDEN_INDEX].desc.dims)) + ").");
    }

    if (nbOutputs != NUM_OUTPUTS)
    {
        throw std::runtime_error("Only two outputs is implemented: " + std::to_string(nbOutputs));
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

int Taco2AttentionLayerPlugin::initialize()
{
    try
    {
        mKernel.reset(
            new Taco2AttentionLayerKernel(mQueryWeightsHost, mConvWeightsHost, mLocationWeightsHost, mEnergyWeightsHost,
                mNumEncodingDimension, mNumQueryDimension, mNumFilters, mConvKernelSize, mNumAttentionDimension));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Taco2AttentionLayerPlugin initialization failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

void Taco2AttentionLayerPlugin::terminate()
{
    mKernel.reset();
}

size_t Taco2AttentionLayerPlugin::getWorkspaceSize(
    const PluginTensorDesc* const in, const int nbInputs, const PluginTensorDesc* const /* out */, const int /* nbOutputs */) const
{
    if (nbInputs != NUM_INPUTS) {
      throw std::runtime_error("Invalid number of inputs: " +
          std::to_string(nbInputs) + ", but expected " + std::to_string(NUM_INPUTS));
    }

    const int inputLength = in[MEMORY_INDEX].dims.d[1];
    const int batchSize = in[MEMORY_INDEX].dims.d[0];

    // space for queryOutput (num attention dimensions),
    // convOutput (input length*num filters), elemSum (input length), and
    // energyScratch (inputLength).
    return sizeof(value_type) * batchSize * (mNumAttentionDimension + (inputLength * mNumFilters) + 2 * inputLength);
}

int Taco2AttentionLayerPlugin::enqueue(const PluginTensorDesc* const inputDesc,
const PluginTensorDesc* /* outputDesc */,
    const void* const* const inputs, void* const* const outputs, void* const workspace, cudaStream_t stream)
{
    const int inputLength = inputDesc[MEMORY_INDEX].dims.d[1];
    const int batchSize = inputDesc[MEMORY_INDEX].dims.d[0];

    if (batchSize != 1)
    {
        // we only support batch size of 1 right now
        std::cerr << "Taco2AttentionLayerPlugin plugin does not support batch size other than "
                     "1: got "
                  << batchSize << std::endl;
        std::cerr << "Recompile without plugins to use a larger batch size." << std::endl;
        return 1;
    }

    // name inputs and outputs
    const value_type* const memoryDevice = static_cast<const value_type*>(inputs[MEMORY_INDEX]);
    const value_type* const processedMemoryDevice = static_cast<const value_type*>(inputs[PROCESSED_MEMORY_INDEX]);
    const value_type* const weightsDevice = static_cast<const value_type*>(inputs[WEIGHT_INDEX]);
    const value_type* const attentionHiddenDevice = static_cast<const value_type*>(inputs[ATTENTION_HIDDEN_INDEX]);

    value_type* const outputContextDevice = static_cast<value_type*>(outputs[CONTEXT_OUTPUT]);
    value_type* const outputWeightsDevice = static_cast<value_type*>(outputs[WEIGHT_OUTPUT]);

    try
    {
        mKernel->execute(memoryDevice, processedMemoryDevice, weightsDevice, attentionHiddenDevice, outputContextDevice,
            outputWeightsDevice, inputLength, static_cast<value_type*>(workspace), stream);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Taco2AttentionLayerPlugin failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

size_t Taco2AttentionLayerPlugin::getSerializationSize() const
{
    const int numQueryWeights = mNumQueryDimension * mNumAttentionDimension;
    const int numConvWeights = mNumFilters * 2 * mConvKernelSize;
    const int numLocationWeights = mNumFilters * mNumAttentionDimension;
    const int numEnergyWeights = mNumAttentionDimension;

    return 5 * sizeof(int32_t)
        + sizeof(value_type) * (numQueryWeights + numConvWeights + numLocationWeights + numEnergyWeights);
}

void Taco2AttentionLayerPlugin::serialize(void* const buffer) const
{
    static_cast<int32_t*>(buffer)[0] = mNumEncodingDimension;
    static_cast<int32_t*>(buffer)[1] = mNumQueryDimension;
    static_cast<int32_t*>(buffer)[2] = mNumFilters;
    static_cast<int32_t*>(buffer)[3] = mConvKernelSize;
    static_cast<int32_t*>(buffer)[4] = mNumAttentionDimension;

    float* const queryWeights = reinterpret_cast<float*>(static_cast<int32_t*>(buffer) + 5);
    float* const convWeights = queryWeights + mQueryWeightsHost.size();
    float* const locationWeights = convWeights + mConvWeightsHost.size();
    float* const energyWeights = locationWeights + mLocationWeightsHost.size();

    memcpy(queryWeights, mQueryWeightsHost.data(), sizeof(value_type) * mQueryWeightsHost.size());
    memcpy(convWeights, mConvWeightsHost.data(), sizeof(value_type) * mConvWeightsHost.size());
    memcpy(locationWeights, mLocationWeightsHost.data(), sizeof(value_type) * mLocationWeightsHost.size());
    memcpy(energyWeights, mEnergyWeightsHost.data(), sizeof(value_type) * mEnergyWeightsHost.size());
}

void Taco2AttentionLayerPlugin::destroy()
{
    terminate();
}

IPluginV2DynamicExt* Taco2AttentionLayerPlugin::clone() const
{
    // call constructor which copy's data
    Taco2AttentionLayerPlugin clone(mNumEncodingDimension, mNumQueryDimension, mNumFilters, mConvKernelSize,
        mNumAttentionDimension,
        Weights{DataType::kFLOAT, mQueryWeightsHost.data(), static_cast<int64_t>(mQueryWeightsHost.size())},
        Weights{DataType::kFLOAT, mConvWeightsHost.data(), static_cast<int64_t>(mConvWeightsHost.size())},
        Weights{DataType::kFLOAT, mLocationWeightsHost.data(), static_cast<int64_t>(mLocationWeightsHost.size())},
        Weights{DataType::kFLOAT, mEnergyWeightsHost.data(), static_cast<int64_t>(mEnergyWeightsHost.size())});

    if (mKernel)
    {
        // initialize the clone too
        clone.initialize();
    }

    // move it to the heap last to avoid exceptions causing memory leaks
    return new Taco2AttentionLayerPlugin(std::move(clone));
}

void Taco2AttentionLayerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2AttentionLayerPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
