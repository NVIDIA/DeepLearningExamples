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

#include "taco2LSTMCellLayerPluginCreator.h"
#include "taco2LSTMCellLayerPlugin.h"

#include <stdexcept>
#include <vector>

using namespace nvinfer1;

namespace nvinfer1
{
namespace plugin
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const char* const LENGTH_STR = "Length";
constexpr const char* const DIMENSION_STR = "Dimension";
constexpr const char* const FP16_STR = "FP16";
constexpr const char* const INPUT_WEIGHTS_STR = "weight_ih";
constexpr const char* const HIDDEN_WEIGHTS_STR = "weight_hh";
constexpr const char* const INPUT_BIAS_STR = "bias_ih";
constexpr const char* const HIDDEN_BIAS_STR = "bias_hh";

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

PluginFieldCollection* Taco2LSTMCellLayerPluginCreator::getFields()
{
    static PluginFieldCollection* pluginPtr = nullptr;
    static const std::vector<PluginField> fields{{LENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0}, {FP16_STR, nullptr, PluginFieldType::kINT32, 0},
        {INPUT_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {HIDDEN_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {INPUT_BIAS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {HIDDEN_BIAS_STR, nullptr, PluginFieldType::kFLOAT32, 0}};

    if (!pluginPtr)
    {
        pluginPtr
            = static_cast<PluginFieldCollection*>(malloc(sizeof(*pluginPtr) + fields.size() * sizeof(PluginField)));
        pluginPtr->nbFields = static_cast<int>(fields.size());
        pluginPtr->fields = fields.data();
    }

    return pluginPtr;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Taco2LSTMCellLayerPluginCreator::Taco2LSTMCellLayerPluginCreator()
    : mNamespace()
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2LSTMCellLayerPluginCreator::getPluginName() const
{
    return Taco2LSTMCellLayerPlugin::getName();
}

const char* Taco2LSTMCellLayerPluginCreator::getPluginVersion() const
{
    return Taco2LSTMCellLayerPlugin::getVersion();
}

const PluginFieldCollection* Taco2LSTMCellLayerPluginCreator::getFieldNames()
{
    return getFields();
}

IPluginV2* Taco2LSTMCellLayerPluginCreator::createPlugin(const char* const /*name*/, const PluginFieldCollection* fc)
{
    int length = 0;
    int dimension = 0;
    bool fp16 = false;

    Weights inputWeights{DataType::kFLOAT, nullptr, 0};
    Weights hiddenWeights{DataType::kFLOAT, nullptr, 0};
    Weights inputBias{DataType::kFLOAT, nullptr, 0};
    Weights hiddenBias{DataType::kFLOAT, nullptr, 0};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const std::string name(fc->fields[i].name);
        if (name == LENGTH_STR)
        {
            length = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == DIMENSION_STR)
        {
            dimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == FP16_STR)
        {
            fp16 = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == INPUT_WEIGHTS_STR)
        {
            inputWeights.values = fc->fields[i].data;
            inputWeights.count = fc->fields[i].length;
        }
        else if (name == HIDDEN_WEIGHTS_STR)
        {
            hiddenWeights.values = fc->fields[i].data;
            hiddenWeights.count = fc->fields[i].length;
        }
        else if (name == INPUT_BIAS_STR)
        {
            inputBias.values = fc->fields[i].data;
            inputBias.count = fc->fields[i].length;
        }
        else if (name == HIDDEN_BIAS_STR)
        {
            hiddenBias.values = fc->fields[i].data;
            hiddenBias.count = fc->fields[i].length;
        }
        else
        {
            throw std::runtime_error("Unknown plugin field: '" + name + "'");
        }
    }

    return new Taco2LSTMCellLayerPlugin(inputWeights, hiddenWeights, inputBias, hiddenBias, length, dimension, fp16);
}

IPluginV2* Taco2LSTMCellLayerPluginCreator::deserializePlugin(
    const char* const /* layerName */, const void* const serialData, size_t const serialLength)
{
    return new Taco2LSTMCellLayerPlugin(Taco2LSTMCellLayerPlugin::deserialize(serialData, serialLength));
}

void Taco2LSTMCellLayerPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2LSTMCellLayerPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
