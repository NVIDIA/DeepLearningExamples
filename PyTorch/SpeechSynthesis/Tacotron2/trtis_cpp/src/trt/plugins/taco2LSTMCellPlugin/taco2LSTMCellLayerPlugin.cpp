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

#include "taco2LSTMCellLayerPlugin.h"
#include "taco2LSTMCellKernel.h"
#include "taco2Utils.h"

#include <cuda_runtime.h> // cudaError_t

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{

using value_type = Taco2LSTMCellLayerPlugin::value_type;

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const PLUGIN_NAME = "Taco2LSTMCell";
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
            "Invalid data type for Taco2LSTMCell weights: " + std::to_string(static_cast<int>(weights.type)));
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

const char* Taco2LSTMCellLayerPlugin::getName()
{
    return PLUGIN_NAME;
}

const char* Taco2LSTMCellLayerPlugin::getVersion()
{
    return PLUGIN_VERSION;
}

Taco2LSTMCellLayerPlugin Taco2LSTMCellLayerPlugin::deserialize(const void* const data, const size_t length)
{
    if (length < 5 * sizeof(int32_t))
    {
        throw std::runtime_error("Invalid serialized size: " + std::to_string(length));
    }

    const int inputLength = static_cast<const int32_t*>(data)[0];
    const int inputLengthFirst = static_cast<const int32_t*>(data)[1];
    const int inputLengthSecond = static_cast<const int32_t*>(data)[2];
    const int numDimension = static_cast<const int32_t*>(data)[3];
    const bool useFP16 = static_cast<const int32_t*>(data)[4];
    const size_t reqSize = 5 * sizeof(int32_t)
        + sizeof(value_type)
            * (4 * inputLength * numDimension + 4 * numDimension * numDimension + 2 * 4 * numDimension);
    if (reqSize != length)
    {
        throw std::runtime_error(
            "Invalid serialized size: " + std::to_string(length) + " / " + std::to_string(reqSize));
    }

    const Weights inputWeights{DataType::kFLOAT, offset(data, sizeof(int32_t) * 5), inputLength * numDimension * 4};
    const Weights hiddenWeights{DataType::kFLOAT, offset(inputWeights.values, sizeof(value_type) * inputWeights.count),
        numDimension * numDimension * 4};
    const Weights inputBias{
        DataType::kFLOAT, offset(hiddenWeights.values, sizeof(value_type) * hiddenWeights.count), numDimension * 4};
    const Weights hiddenBias{
        DataType::kFLOAT, offset(inputBias.values, sizeof(value_type) * inputBias.count), numDimension * 4};

    Taco2LSTMCellLayerPlugin layer(
        inputWeights, hiddenWeights, inputBias, hiddenBias, inputLength, numDimension, useFP16);

    layer.mInputLengthFirst = inputLengthFirst;
    layer.mInputLengthSecond = inputLengthSecond;

    return layer;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2LSTMCellLayerPlugin::Taco2LSTMCellLayerPlugin(const Weights& inputWeights, const Weights& hiddenWeights,
    const Weights& inputBias, const Weights& hiddenBias, const int inputLength, const int numDimension,
    const bool useFP16)
    : mInputLength(inputLength)
    , mInputLengthFirst(0)
    , mInputLengthSecond(0)
    , mNumDimension(numDimension)
    , mInputWeightsHost(toVector(inputWeights))
    , mHiddenWeightsHost(toVector(hiddenWeights))
    , mInputBiasHost(toVector(inputBias))
    , mHiddenBiasHost(toVector(hiddenBias))
    , mNamespace()
    , mCell(nullptr)
    , mUseFP16(useFP16)
{
    // do nothing
    if (mInputLength <= 0)
    {
        throw std::runtime_error("Invalid Taco2LSTMCell length: " + std::to_string(mInputLength));
    }
    if (mNumDimension <= 0)
    {
        throw std::runtime_error("Invalid Taco2LSTMCell dimension: " + std::to_string(mNumDimension));
    }

    const size_t expectedInputWeights = mInputLength * mNumDimension * 4U;
    const size_t expectedHiddenWeights = mNumDimension * mNumDimension * 4U;
    const size_t expectedBias = mNumDimension * 4U;
    if (mInputWeightsHost.size() != expectedInputWeights)
    {
        throw std::runtime_error("Taco2LSTMCell expected " + std::to_string(expectedInputWeights)
            + " input weights but given " + std::to_string(mInputWeightsHost.size()));
    }
    if (mHiddenWeightsHost.size() != expectedHiddenWeights)
    {
        throw std::runtime_error("Taco2LSTMCell expected " + std::to_string(expectedHiddenWeights)
            + " hidden weights but given " + std::to_string(mHiddenWeightsHost.size()));
    }
    if (mInputBiasHost.size() != expectedBias)
    {
        throw std::runtime_error("Taco2LSTMCell expected " + std::to_string(expectedBias) + " input bias but given "
            + std::to_string(mInputBiasHost.size()));
    }
    if (mHiddenBiasHost.size() != expectedBias)
    {
        throw std::runtime_error("Taco2LSTMCell expected " + std::to_string(expectedBias) + " hidden bias but given "
            + std::to_string(mHiddenBiasHost.size()));
    }
}

Taco2LSTMCellLayerPlugin::Taco2LSTMCellLayerPlugin(Taco2LSTMCellLayerPlugin&& other)
    : mInputLength(other.mInputLength)
    , mInputLengthFirst(other.mInputLengthFirst)
    , mInputLengthSecond(other.mInputLengthSecond)
    , mNumDimension(other.mNumDimension)
    , mInputWeightsHost(std::move(other.mInputWeightsHost))
    , mHiddenWeightsHost(std::move(other.mHiddenWeightsHost))
    , mInputBiasHost(std::move(other.mInputBiasHost))
    , mHiddenBiasHost(std::move(other.mHiddenBiasHost))
    , mNamespace(std::move(other.mNamespace))
    , mCell(std::move(other.mCell))
    , mUseFP16(other.mUseFP16)
{
    other.mInputLength = 0;
    other.mInputLengthFirst = 0;
    other.mInputLengthSecond = 0;
    other.mNumDimension = 0;
    other.mUseFP16 = false;
}

Taco2LSTMCellLayerPlugin::~Taco2LSTMCellLayerPlugin()
{
    destroy();
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

Taco2LSTMCellLayerPlugin& Taco2LSTMCellLayerPlugin::operator=(Taco2LSTMCellLayerPlugin&& other)
{
    // defere to constructor
    *this = Taco2LSTMCellLayerPlugin(std::move(other));

    return *this;
}

DataType Taco2LSTMCellLayerPlugin::getOutputDataType(
    const int /* index */, const DataType* const /* inputTypes */, const int /* nbInputs */) const
{
    return DataType::kFLOAT;
}

const char* Taco2LSTMCellLayerPlugin::getPluginType() const
{
    return getName();
}

const char* Taco2LSTMCellLayerPlugin::getPluginVersion() const
{
    return getVersion();
}

int Taco2LSTMCellLayerPlugin::getNbOutputs() const
{
    return 2;
}

DimsExprs Taco2LSTMCellLayerPlugin::getOutputDimensions(
    const int outputIndex, const DimsExprs* inputs, const int nbInputs, IExprBuilder& exprBuilder)
{
    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error("Can only handle three input tensors: " + std::to_string(nbInputs));
    }


    if (outputIndex == 0) {
      // hidden
      return DimsExprs{3, {inputs[INPUT_FIRST_INDEX].d[0], exprBuilder.constant(1), exprBuilder.constant(mNumDimension)}};
    } else if (outputIndex == 1) {
      // cell
      return DimsExprs{3, {inputs[INPUT_FIRST_INDEX].d[0], exprBuilder.constant(1), exprBuilder.constant(mNumDimension)}};
    } else {

      throw std::runtime_error("Invalid output index: " + std::to_string(outputIndex));
    }
}

bool Taco2LSTMCellLayerPlugin::supportsFormatCombination(
    const int pos, const PluginTensorDesc* const inOut, const int /* nbInputs */, const int /* nbOutputs */)
{
    return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
}

void Taco2LSTMCellLayerPlugin::configurePlugin(const DynamicPluginTensorDesc* const in, const int nbInputs,
    const DynamicPluginTensorDesc* const out, const int nbOutputs)
{
    if (nbInputs != NUM_INPUTS)
    {
        throw std::runtime_error("Only three inputs is implemented: " + std::to_string(nbInputs));
    }
    for (int i = 0; i < nbInputs; ++i)
    {
        if (in[i].desc.type != DataType::kFLOAT)
        {
            throw std::runtime_error("Only FLOAT supported as input " + std::to_string(i) + " : "
                + std::to_string(static_cast<int>(in[i].desc.type)));
        }
    }

    if (nbOutputs != 2)
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

    {
        const Dims dims = in[INPUT_FIRST_INDEX].desc.dims;
        bool dimsFound = false;
        for (int d = 1; d < dims.nbDims; ++d)
        {
            if (dims.d[d] != 1)
            {
                if (dimsFound)
                {
                    throw std::runtime_error("Invalid first input dimension: " + taco2::Taco2Utils::dimsToString(dims));
                }
                mInputLengthFirst = dims.d[d];
                dimsFound = true;
            }
        }
        if (!dimsFound)
        {
            throw std::runtime_error("Invalid first input dimension: " + taco2::Taco2Utils::dimsToString(dims));
        }
    }
    {
        const Dims dims = in[INPUT_SECOND_INDEX].desc.dims;
        bool dimsFound = false;
        for (int d = 1; d < dims.nbDims; ++d)
        {
            if (dims.d[d] != 1)
            {
                if (dimsFound)
                {
                    throw std::runtime_error(
                        "Invalid second input dimension: " + taco2::Taco2Utils::dimsToString(dims));
                }
                mInputLengthSecond = dims.d[d];
                dimsFound = true;
            }
        }
        if (!dimsFound)
        {
            throw std::runtime_error("Invalid second input dimension: " + taco2::Taco2Utils::dimsToString(dims));
        }
    }

    if (mInputLengthFirst + mInputLengthSecond != mInputLength)
    {
        throw std::runtime_error("Invalid input lenghts: " + std::to_string(mInputLengthFirst) + " "
            + std::to_string(mInputLengthSecond) + " != " + std::to_string(mInputLength));
    }
}

int Taco2LSTMCellLayerPlugin::initialize()
{
    try
    {
        mCell.reset(new Taco2LSTMCellKernel(mInputWeightsHost.data(), mHiddenWeightsHost.data(), mInputBiasHost.data(),
            mHiddenBiasHost.data(), mInputLength, mNumDimension, mUseFP16));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Taco2LSTMCellLayerPlugin initialization failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

void Taco2LSTMCellLayerPlugin::terminate()
{
    mCell.reset();
}

size_t Taco2LSTMCellLayerPlugin::getWorkspaceSize(const PluginTensorDesc* const /* in */, const int /* nbInputs */,
    const PluginTensorDesc* const /* out */, const int /* nbOutputs */) const
{
    return 0;
}

int Taco2LSTMCellLayerPlugin::enqueue(const PluginTensorDesc* const inputDesc, const PluginTensorDesc* const /* outputDesc */,
    const void* const* const inputs, void* const* const outputs, void* const /*workspace*/, cudaStream_t stream)
{
    const int batchSize = inputDesc[INPUT_FIRST_INDEX].dims.d[0];

    if (batchSize != 1)
    {
        // we only support batch size of 1 right now
        std::cerr << "Taco2LSTMCellLayerPlugin plugin does not support batch size other than 1: got " << batchSize
                  << std::endl;
        std::cerr << "Recompile without plugins to use a larger batch size." << std::endl;
        return 1;
    }
    else if (!mCell)
    {
        std::cerr << "Taco2LSTMCellLayerPlugin is not initialized properly." << std::endl;
        return 1;
    }

    // name inputs and outputs
    const value_type* const inputFirstDevice = static_cast<const value_type*>(inputs[INPUT_FIRST_INDEX]);
    const value_type* const inputSecondDevice = static_cast<const value_type*>(inputs[INPUT_SECOND_INDEX]);
    const value_type* const inputHiddenDevice = static_cast<const value_type*>(inputs[HIDDEN_INDEX]);
    const value_type* const inputCellDevice = static_cast<const value_type*>(inputs[CELL_INDEX]);

    value_type* const outputHiddenDevice = static_cast<value_type*>(outputs[0]);
    value_type* const outputCellDevice = static_cast<value_type*>(outputs[1]);

    // launch kernel to perform lstm on `(Wi+Wh)+(bi+bh)`
    mCell->execute(inputFirstDevice, inputSecondDevice, inputHiddenDevice, inputCellDevice, outputHiddenDevice,
        outputCellDevice, mInputLengthFirst, mInputLengthSecond, stream);

    return 0;
}

size_t Taco2LSTMCellLayerPlugin::getSerializationSize() const
{
    return 5 * sizeof(int32_t) + numInputWeightBytes() + numHiddenWeightBytes() + 2 * numBiasBytes();
}

void Taco2LSTMCellLayerPlugin::serialize(void* const buffer) const
{
    static_cast<int32_t*>(buffer)[0] = mInputLength;
    static_cast<int32_t*>(buffer)[1] = mInputLengthFirst;
    static_cast<int32_t*>(buffer)[2] = mInputLengthSecond;
    static_cast<int32_t*>(buffer)[3] = mNumDimension;
    static_cast<int32_t*>(buffer)[4] = mUseFP16;
    float* const inputWeights = reinterpret_cast<float*>(static_cast<int32_t*>(buffer) + 5);
    float* const hiddenWeights = inputWeights + numInputWeights();
    float* const inputBias = hiddenWeights + numHiddenWeights();
    float* const hiddenBias = inputBias + numBiases();

    memcpy(inputWeights, mInputWeightsHost.data(), numInputWeightBytes());
    memcpy(hiddenWeights, mHiddenWeightsHost.data(), numHiddenWeightBytes());
    memcpy(inputBias, mInputBiasHost.data(), numBiasBytes());
    memcpy(hiddenBias, mHiddenBiasHost.data(), numBiasBytes());
}

void Taco2LSTMCellLayerPlugin::destroy()
{
    terminate();
}

IPluginV2DynamicExt* Taco2LSTMCellLayerPlugin::clone() const
{
    // call constructor which copy's data
    Taco2LSTMCellLayerPlugin clone(
        Weights{DataType::kFLOAT, mInputWeightsHost.data(), static_cast<int64_t>(mInputWeightsHost.size())},
        Weights{DataType::kFLOAT, mHiddenWeightsHost.data(), static_cast<int64_t>(mHiddenWeightsHost.size())},
        Weights{DataType::kFLOAT, mInputBiasHost.data(), static_cast<int64_t>(mInputBiasHost.size())},
        Weights{DataType::kFLOAT, mHiddenBiasHost.data(), static_cast<int64_t>(mHiddenBiasHost.size())}, mInputLength,
        mNumDimension, mUseFP16);
    clone.mInputLengthFirst = mInputLengthFirst;
    clone.mInputLengthSecond = mInputLengthSecond;

    if (mCell)
    {
        // initialize the clone too
        clone.initialize();
    }

    // move it to the heap last to avoid exceptions causing memory leaks
    return new Taco2LSTMCellLayerPlugin(std::move(clone));
}

void Taco2LSTMCellLayerPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2LSTMCellLayerPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

int Taco2LSTMCellLayerPlugin::numInputWeights() const
{
    return mNumDimension * mInputLength * 4;
}

int Taco2LSTMCellLayerPlugin::numHiddenWeights() const
{
    return mNumDimension * mNumDimension * 4;
}

int Taco2LSTMCellLayerPlugin::numBiases() const
{
    return mNumDimension * 4;
}

size_t Taco2LSTMCellLayerPlugin::numInputWeightBytes() const
{
    return numInputWeights() * sizeof(value_type);
}

size_t Taco2LSTMCellLayerPlugin::numHiddenWeightBytes() const
{
    return numHiddenWeights() * sizeof(value_type);
}

size_t Taco2LSTMCellLayerPlugin::numBiasBytes() const
{
    return numBiases() * sizeof(value_type);
}

} // namespace plugin
} // namespace nvinfer1
