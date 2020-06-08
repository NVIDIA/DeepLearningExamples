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

#include "taco2ModulationRemovalLayerPluginCreator.h"
#include "taco2ModulationRemovalLayerPlugin.h"

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
constexpr const char* const INPUTLENGTH_STR = "InputLength";
constexpr const char* const FILTERLENGTH_STR = "FilterLength";
constexpr const char* const HOPLENGTH_STR = "HopLength";
constexpr const char* const WEIGHTS_STR = "Weights";

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

PluginFieldCollection* Taco2ModulationRemovalLayerPluginCreator::getFields()
{
    static PluginFieldCollection* pluginPtr = nullptr;
    static const std::vector<PluginField> fields{
        {INPUTLENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {FILTERLENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {HOPLENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {WEIGHTS_STR, nullptr, PluginFieldType::kFLOAT32, 0},
    };

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

Taco2ModulationRemovalLayerPluginCreator::Taco2ModulationRemovalLayerPluginCreator()
    : mNamespace()
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2ModulationRemovalLayerPluginCreator::getPluginName() const
{
    return Taco2ModulationRemovalLayerPlugin::getName();
}

const char* Taco2ModulationRemovalLayerPluginCreator::getPluginVersion() const
{
    return Taco2ModulationRemovalLayerPlugin::getVersion();
}

const PluginFieldCollection* Taco2ModulationRemovalLayerPluginCreator::getFieldNames()
{
    return getFields();
}

IPluginV2* Taco2ModulationRemovalLayerPluginCreator::createPlugin(
    const char* const /*name*/, const PluginFieldCollection* fc)
{
    int inputLength = 0;
    int filterLength = 0;
    int hopLength = 0;
    Weights weights{DataType::kFLOAT, nullptr, 0};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const std::string name(fc->fields[i].name);
        if (name == INPUTLENGTH_STR)
        {
            inputLength = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == FILTERLENGTH_STR)
        {
            filterLength = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == HOPLENGTH_STR)
        {
            hopLength = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == WEIGHTS_STR)
        {
            weights.values = fc->fields[i].data;
            weights.count = fc->fields[i].length;
        }
        else
        {
            throw std::runtime_error("Unknown plugin field: '" + name + "'");
        }
    }

    return new Taco2ModulationRemovalLayerPlugin(weights, inputLength, filterLength, hopLength);
}

IPluginV2* Taco2ModulationRemovalLayerPluginCreator::deserializePlugin(
    const char* const /* layerName */, const void* const serialData, size_t const serialLength)
{
    return new Taco2ModulationRemovalLayerPlugin(
        Taco2ModulationRemovalLayerPlugin::deserialize(serialData, serialLength));
}

void Taco2ModulationRemovalLayerPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2ModulationRemovalLayerPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
