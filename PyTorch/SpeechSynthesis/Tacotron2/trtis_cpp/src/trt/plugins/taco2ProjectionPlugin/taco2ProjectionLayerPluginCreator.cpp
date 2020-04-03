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

#include "taco2ProjectionLayerPluginCreator.h"
#include "taco2ProjectionLayerPlugin.h"

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
constexpr const char* const HIDDEN_INPUT_LENGTH_STR = "HiddenInputLength";
constexpr const char* const CONTEXT_INPUT_LENGTH_STR = "ContextInputLength";
constexpr const char* const CHANNEL_DIMENSION_STR = "ChannelDimension";
constexpr const char* const GATE_DIMENSION_STR = "GateDimension";
constexpr const char* const CHANNEL_WEIGHTS_STR = "ChannelWeights";
constexpr const char* const GATE_WEIGHTS_STR = "GateWeights";
constexpr const char* const CHANNEL_BIAS_STR = "ChannelBias";
constexpr const char* const GATE_BIAS_STR = "GateBias";

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

PluginFieldCollection* Taco2ProjectionLayerPluginCreator::getFields()
{
    static PluginFieldCollection* pluginPtr = nullptr;
    static const std::vector<PluginField> fields{{HIDDEN_INPUT_LENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {CONTEXT_INPUT_LENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {CHANNEL_DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0},
        {GATE_DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0},
        {CHANNEL_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {GATE_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {CHANNEL_BIAS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {GATE_BIAS_STR, nullptr, PluginFieldType::kFLOAT32, 0}};

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

Taco2ProjectionLayerPluginCreator::Taco2ProjectionLayerPluginCreator()
    : mNamespace()
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2ProjectionLayerPluginCreator::getPluginName() const
{
    return Taco2ProjectionLayerPlugin::getName();
}

const char* Taco2ProjectionLayerPluginCreator::getPluginVersion() const
{
    return Taco2ProjectionLayerPlugin::getVersion();
}

const PluginFieldCollection* Taco2ProjectionLayerPluginCreator::getFieldNames()
{
    return getFields();
}

IPluginV2* Taco2ProjectionLayerPluginCreator::createPlugin(const char* const /*name*/, const PluginFieldCollection* fc)
{
    int hiddenInputLength = 0;
    int contextInputLength = 0;
    int numChannelDimension = 0;
    int numGateDimension = 0;

    Weights channelWeights{DataType::kFLOAT, nullptr, 0};
    Weights gateWeights{DataType::kFLOAT, nullptr, 0};
    Weights channelBias{DataType::kFLOAT, nullptr, 0};
    Weights gateBias{DataType::kFLOAT, nullptr, 0};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const std::string name(fc->fields[i].name);
        if (name == HIDDEN_INPUT_LENGTH_STR)
        {
            hiddenInputLength = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == CONTEXT_INPUT_LENGTH_STR)
        {
            contextInputLength = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == CHANNEL_DIMENSION_STR)
        {
            numChannelDimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == GATE_DIMENSION_STR)
        {
            numGateDimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == CHANNEL_WEIGHTS_STR)
        {
            channelWeights.values = fc->fields[i].data;
            channelWeights.count = fc->fields[i].length;
        }
        else if (name == GATE_WEIGHTS_STR)
        {
            gateWeights.values = fc->fields[i].data;
            gateWeights.count = fc->fields[i].length;
        }
        else if (name == CHANNEL_BIAS_STR)
        {
            channelBias.values = fc->fields[i].data;
            channelBias.count = fc->fields[i].length;
        }
        else if (name == GATE_BIAS_STR)
        {
            gateBias.values = fc->fields[i].data;
            gateBias.count = fc->fields[i].length;
        }
        else
        {
            throw std::runtime_error("Unknown plugin field: '" + name + "'");
        }
    }

    return new Taco2ProjectionLayerPlugin(channelWeights, gateWeights, channelBias, gateBias, hiddenInputLength,
        contextInputLength, numChannelDimension, numGateDimension);
}

IPluginV2* Taco2ProjectionLayerPluginCreator::deserializePlugin(
    const char* const /* layerName */, const void* const serialData, size_t const serialLength)
{
    return new Taco2ProjectionLayerPlugin(Taco2ProjectionLayerPlugin::deserialize(serialData, serialLength));
}

void Taco2ProjectionLayerPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2ProjectionLayerPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
