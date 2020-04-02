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

#include "trtUtils.h"

#include "NvInfer.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

std::vector<float> TRTUtils::toFloatVector(const Weights& weights)
{
    if (weights.type != DataType::kFLOAT)
    {
        throw std::runtime_error(
            "Invalid data type for LSTMCell weights: " + std::to_string(static_cast<int>(weights.type)));
    }
    const float* const valuesBegin = static_cast<const float*>(weights.values);
    const float* const valuesEnd = valuesBegin + weights.count;

    return std::vector<float>(valuesBegin, valuesEnd);
}

Weights TRTUtils::toWeights(const std::vector<float>& vec)
{
    return Weights{DataType::kFLOAT, vec.data(), static_cast<int64_t>(vec.size())};
}

std::string TRTUtils::dimsToString(const Dims& dim)
{
    std::ostringstream oss;
    oss << "{";
    for (int i = 0; i < dim.nbDims; ++i)
    {
        oss << dim.d[i] << " ";
    }
    oss << "}";
    return oss.str();
}

void TRTUtils::printDimensions(const std::string& name, const Dims& dim)
{
    std::cout << "Tensor '" << name << "' =" << dimsToString(dim) << std::endl;
}

void TRTUtils::printTensor(const std::string& name, const ITensor& tensor)
{
    printDimensions(name, tensor.getDimensions());
}

void TRTUtils::printBindingDimensions(const nvinfer1::ICudaEngine& engine)
{
    std::cout << "Engine '" << engine.getName() << "' bindings:" << std::endl;
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        printDimensions(engine.getBindingName(b), engine.getBindingDimensions(b));
    }
}

size_t TRTUtils::getDimensionsSize(const Dims& dims)
{
    size_t i = 1;
    for (int d = 0; d < dims.nbDims; ++d)
    {
        if (dims.d[d] == -1)
        {
            if (d == 0)
            {
                // ignore batch dimension
            }
            else
            {
                throw std::runtime_error(
                    "Cannot get size of tensor with dynamic "
                    "dimension.");
            }
        }
        else
        {
            assert(dims.d[d] > 0);
            i *= dims.d[d];
        }
    }

    return i;
}

size_t TRTUtils::getTensorSize(const ITensor& tensor)
{
    return getDimensionsSize(tensor.getDimensions());
}

size_t TRTUtils::getMaxBindingSize(const ICudaEngine& engine, const char* const bindingName)
{
    const int binding = engine.getBindingIndex(bindingName);
    if (binding < 0)
    {
        throw std::runtime_error("Failed to find binding named '" + std::string(bindingName) + "'.");
    }
    size_t size = 0;
    for (int o = 0; o < engine.getNbOptimizationProfiles(); ++o)
    {
        size_t optSize = 1;
        const Dims dim = engine.getProfileDimensions(binding, 0, OptProfileSelector::kMAX);
        for (int d = 0; d < dim.nbDims; ++d)
        {
            if (dim.d[d] < 0)
            {
                throw std::runtime_error("Invalid rank in dimensions: " + dimsToString(dim));
            }
            optSize *= dim.d[d];
        }
        if (optSize > size)
        {
            size = optSize;
        }
    }

    return size;
}

size_t TRTUtils::getBindingSize(const ICudaEngine& engine, const char* const bindingName)
{
    const int binding = engine.getBindingIndex(bindingName);
    if (binding < 0)
    {
        throw std::runtime_error("Failed to find binding named '" + std::string(bindingName) + "'.");
    }
    size_t size = 1;
    Dims dim = engine.getBindingDimensions(binding);
    for (int d = 0; d < dim.nbDims; ++d)
    {
        if (dim.d[d] == -1)
        {
            if (d == 0)
            {
                // ignore batch dimension
            }
            else
            {
                throw std::runtime_error(
                    "Dynamic dimension detected in " + std::string(bindingName) + " (" + dimsToString(dim) + ").");
            }
        }
        else
        {
            assert(dim.d[d] > 0);
            size *= dim.d[d];
        }
    }

    return size;
}

size_t TRTUtils::getNonBatchBindingSize(const ICudaEngine& engine, const char* const bindingName)
{
    const int binding = engine.getBindingIndex(bindingName);
    if (binding < 0)
    {
        throw std::runtime_error("Failed to find binding named '" + std::string(bindingName) + "'.");
    }
    size_t size = 1;
    Dims dim = engine.getBindingDimensions(binding);
    for (int d = 1; d < dim.nbDims; ++d)
    {
        size *= dim.d[d];
    }

    return size;
}

int TRTUtils::getMaxBatchSize(const ICudaEngine& engine)
{
    const Dims dims = engine.getBindingDimensions(0);
    int maxDim = 0;
    if (dims.nbDims > 0 && dims.d[0] < 0)
    {
        for (int i = 0; i < engine.getNbOptimizationProfiles(); ++i)
        {
            const int currentDim = engine.getProfileDimensions(0, i, OptProfileSelector::kMAX).d[0];
            if (maxDim < currentDim)
            {
                maxDim = currentDim;
            }
        }
    }
    else
    {
        maxDim = engine.getMaxBatchSize();
    }
    return maxDim;
}

size_t TRTUtils::getBindingDimension(const ICudaEngine& engine, const char* const bindingName, const int dimension)
{
    const int binding = engine.getBindingIndex(bindingName);
    if (binding < 0)
    {
        throw std::runtime_error("Failed to find binding named '" + std::string(bindingName) + "'.");
    }
    Dims dim = engine.getBindingDimensions(binding);
    if (dimension >= dim.nbDims)
    {
        throw std::runtime_error("Invalid dimension " + std::to_string(dimension) + " of " + std::to_string(dim.nbDims)
            + " for " + bindingName);
    }

    return dim.d[dimension];
}

ITensor* TRTUtils::getInputByName(INetworkDefinition& network, const std::string& inputName)
{
    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        ITensor* input = network.getInput(i);
        if (inputName == input->getName())
        {
            return input;
        }
    }

    throw std::runtime_error(
        "Unable to find input '" + inputName + "' in " + std::to_string(network.getNbInputs()) + " inputs.");
}

int TRTUtils::getFirstNonUnitDim(const Dims& dims)
{
    if (dims.nbDims)
    {
        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (dims.d[d] > 1)
            {
                return dims.d[d];
            }
        }
        return 1;
    }
    else
    {
        return 0;
    }
}

Dims TRTUtils::getCompactedDims(const Dims& dims, const int minLength)
{
    Dims cDims{0, {}, {}};
    if (dims.nbDims)
    {
        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (dims.d[d] > 1)
            {
                cDims.d[cDims.nbDims++] = dims.d[d];
            }
        }
        if (cDims.nbDims == 0)
        {
            cDims.nbDims = 1;
            cDims.d[0] = 1;
        }
    }

    if (cDims.nbDims < minLength)
    {
        const int offset = minLength - cDims.nbDims;
        for (int i = cDims.nbDims; i > 0;)
        {
            --i;
            cDims.d[i + offset] = cDims.d[i];
        }
        for (int i = 0; i < offset; ++i)
        {
            cDims.d[i] = 1;
        }
        cDims.nbDims = minLength;
    }

    return cDims;
}

} // namespace tts
