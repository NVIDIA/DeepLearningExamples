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

#include "taco2AttentionLayerPluginCreator.h"
#include "taco2AttentionLayerPlugin.h"

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

constexpr const char* const ENCODING_DIMENSION_STR = "EncodingDimension";
constexpr const char* const QUERY_DIMENSION_STR = "QueryDimension";
constexpr const char* const NUM_FILTERS_STR = "NumFilters";
constexpr const char* const CONV_KERNEL_SIZE_STR = "ConvKernelSize";
constexpr const char* const ATTENTION_DIMENSION_STR = "AttentionDimension";
constexpr const char* const QUERY_WEIGHTS_STR = "QueryWeight";
constexpr const char* const CONV_WEIGHTS_STR = "ConvWeight";
constexpr const char* const LOCATION_WEIGHTS_STR = "LocationWeight";
constexpr const char* const ENERGY_WEIGHTS_STR = "EnergyWeight";

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

PluginFieldCollection* Taco2AttentionLayerPluginCreator::getFields()
{
    static PluginFieldCollection* pluginPtr = nullptr;
    static const std::vector<PluginField> fields{
        {ENCODING_DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0},
        {QUERY_DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0},
        {NUM_FILTERS_STR, nullptr, PluginFieldType::kINT32, 0},
        {CONV_KERNEL_SIZE_STR, nullptr, PluginFieldType::kINT32, 0},
        {ATTENTION_DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0},
        {QUERY_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {CONV_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {LOCATION_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {ENERGY_WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0}};

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

Taco2AttentionLayerPluginCreator::Taco2AttentionLayerPluginCreator()
    : mNamespace()
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2AttentionLayerPluginCreator::getPluginName() const
{
    return Taco2AttentionLayerPlugin::getName();
}

const char* Taco2AttentionLayerPluginCreator::getPluginVersion() const
{
    return Taco2AttentionLayerPlugin::getVersion();
}

const PluginFieldCollection* Taco2AttentionLayerPluginCreator::getFieldNames()
{
    return getFields();
}

IPluginV2* Taco2AttentionLayerPluginCreator::createPlugin(const char* const /*name*/, const PluginFieldCollection* fc)
{
    int encDimension = 0;
    int queryDimension = 0;
    int numFilters = 0;
    int convKernelSize = 0;
    int attDimension = 0;

    Weights queryWeights{DataType::kFLOAT, nullptr, 0};
    Weights locationWeights{DataType::kFLOAT, nullptr, 0};
    Weights convWeights{DataType::kFLOAT, nullptr, 0};
    Weights energyWeights{DataType::kFLOAT, nullptr, 0};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const std::string name(fc->fields[i].name);
        if (name == ENCODING_DIMENSION_STR)
        {
            encDimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == QUERY_DIMENSION_STR)
        {
            queryDimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == NUM_FILTERS_STR)
        {
            numFilters = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == CONV_KERNEL_SIZE_STR)
        {
            convKernelSize = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == ATTENTION_DIMENSION_STR)
        {
            attDimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == QUERY_WEIGHTS_STR)
        {
            queryWeights.values = fc->fields[i].data;
            queryWeights.count = fc->fields[i].length;
        }
        else if (name == CONV_WEIGHTS_STR)
        {
            convWeights.values = fc->fields[i].data;
            convWeights.count = fc->fields[i].length;
        }
        else if (name == LOCATION_WEIGHTS_STR)
        {
            locationWeights.values = fc->fields[i].data;
            locationWeights.count = fc->fields[i].length;
        }
        else if (name == ENERGY_WEIGHTS_STR)
        {
            energyWeights.values = fc->fields[i].data;
            energyWeights.count = fc->fields[i].length;
        }
        else
        {
            throw std::runtime_error("Unknown plugin field: '" + name + "'");
        }
    }

    return new Taco2AttentionLayerPlugin(encDimension, queryDimension, numFilters, convKernelSize, attDimension,
        queryWeights, convWeights, locationWeights, energyWeights);
}

IPluginV2* Taco2AttentionLayerPluginCreator::deserializePlugin(
    const char* const /* layerName */, const void* const serialData, size_t const serialLength)
{
    return new Taco2AttentionLayerPlugin(Taco2AttentionLayerPlugin::deserialize(serialData, serialLength));
}

void Taco2AttentionLayerPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2AttentionLayerPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
