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

#include "taco2PrenetLayerPluginCreator.h"
#include "taco2PrenetLayerPlugin.h"

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
constexpr const char* const DIMENSION_STR = "Dimension";
constexpr const char* const INPUTLENGTH_STR = "InputLength";
constexpr const char* const WEIGHTS1_STR = "weight1";
constexpr const char* const WEIGHTS2_STR = "weight2";

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

PluginFieldCollection* Taco2PrenetLayerPluginCreator::getFields()
{
    static PluginFieldCollection* pluginPtr = nullptr;
    static const std::vector<PluginField> fields{{INPUTLENGTH_STR, nullptr, PluginFieldType::kINT32, 0},
        {DIMENSION_STR, nullptr, PluginFieldType::kINT32, 0}, {WEIGHTS1_STR, nullptr, PluginFieldType::kFLOAT32, 0},
        {WEIGHTS2_STR, nullptr, PluginFieldType::kFLOAT32, 0}};

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

Taco2PrenetLayerPluginCreator::Taco2PrenetLayerPluginCreator()
    : mNamespace()
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

const char* Taco2PrenetLayerPluginCreator::getPluginName() const
{
    return Taco2PrenetLayerPlugin::getName();
}

const char* Taco2PrenetLayerPluginCreator::getPluginVersion() const
{
    return Taco2PrenetLayerPlugin::getVersion();
}

const PluginFieldCollection* Taco2PrenetLayerPluginCreator::getFieldNames()
{
    return getFields();
}

IPluginV2* Taco2PrenetLayerPluginCreator::createPlugin(const char* const /*name*/, const PluginFieldCollection* fc)
{
    int inputLength = 0;
    int dimension = 0;

    Weights weights1{DataType::kFLOAT, nullptr, 0};
    Weights weights2{DataType::kFLOAT, nullptr, 0};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const std::string name(fc->fields[i].name);
        if (name == INPUTLENGTH_STR)
        {
            inputLength = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == DIMENSION_STR)
        {
            dimension = static_cast<const int32_t*>(fc->fields[i].data)[0];
        }
        else if (name == WEIGHTS1_STR)
        {
            weights1.values = fc->fields[i].data;
            weights1.count = fc->fields[i].length;
        }
        else if (name == WEIGHTS2_STR)
        {
            weights2.values = fc->fields[i].data;
            weights2.count = fc->fields[i].length;
        }
        else
        {
            throw std::runtime_error("Unknown plugin field: '" + name + "'");
        }
    }

    return new Taco2PrenetLayerPlugin(weights1, weights2, inputLength, dimension);
}

IPluginV2* Taco2PrenetLayerPluginCreator::deserializePlugin(
    const char* const /* layerName */, const void* const serialData, size_t const serialLength)
{
    return new Taco2PrenetLayerPlugin(Taco2PrenetLayerPlugin::deserialize(serialData, serialLength));
}

void Taco2PrenetLayerPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* Taco2PrenetLayerPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

} // namespace plugin
} // namespace nvinfer1
