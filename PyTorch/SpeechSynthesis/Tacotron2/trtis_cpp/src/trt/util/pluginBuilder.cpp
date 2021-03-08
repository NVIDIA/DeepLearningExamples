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

#include "pluginBuilder.h"
#include "taco2AttentionLayerPluginCreator.h"
#include "taco2DenoiseTransformLayerPluginCreator.h"
#include "taco2LSTMCellLayerPluginCreator.h"
#include "taco2ModulationRemovalLayerPluginCreator.h"
#include "taco2PrenetLayerPluginCreator.h"
#include "taco2ProjectionLayerPluginCreator.h"

#include <stdexcept>

using namespace nvinfer1;

// register plugins
namespace nvinfer1
{
namespace plugin
{
REGISTER_TENSORRT_PLUGIN(Taco2AttentionLayerPluginCreator);
REGISTER_TENSORRT_PLUGIN(Taco2PrenetLayerPluginCreator);
REGISTER_TENSORRT_PLUGIN(Taco2LSTMCellLayerPluginCreator);
REGISTER_TENSORRT_PLUGIN(Taco2ProjectionLayerPluginCreator);
REGISTER_TENSORRT_PLUGIN(Taco2ModulationRemovalLayerPluginCreator);
REGISTER_TENSORRT_PLUGIN(Taco2DenoiseTransformLayerPluginCreator);
} // namespace plugin
} // namespace nvinfer1

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

PluginBuilder::PluginBuilder(const std::string& pluginName, const std::string& pluginVersion)
    : mCreator(nullptr)
    , mFields()
    , mNames()
    , mScalars()
{
    mCreator = getPluginRegistry()->getPluginCreator(pluginName.c_str(), pluginVersion.c_str());
    if (!mCreator)
    {
        throw std::runtime_error("Failed to create plugin '" + pluginName + "'.");
    }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void PluginBuilder::setField(const std::string& name, const nvinfer1::Weights& weights)
{
    PluginFieldType type;
    if (weights.type == DataType::kFLOAT)
    {
        type = PluginFieldType::kFLOAT32;
    }
    else if (weights.type == DataType::kINT32)
    {
        type = PluginFieldType::kINT32;
    }
    else
    {
        throw std::runtime_error(
            "PluginBuilder: Unsupported data type field type: " + std::to_string(static_cast<int32_t>(weights.type)));
    }

    mNames.emplace_back(new std::string(name));

    setField(PluginField{mNames.back()->c_str(), weights.values, type, static_cast<int32_t>(weights.count)});
}

void PluginBuilder::setField(const std::string& name, const int32_t value)
{
    mScalars.emplace_back(new scalar_t{value});
    mNames.emplace_back(new std::string(name));

    setField(PluginField{
        mNames.back()->c_str(), reinterpret_cast<const void*>(&mScalars.back()->i), PluginFieldType::kINT32, 1});
}

void PluginBuilder::setField(const std::string& name, const float value)
{
    mScalars.emplace_back(new scalar_t{value});
    mNames.emplace_back(new std::string(name));

    setField(PluginField{
        mNames.back()->c_str(), reinterpret_cast<const void*>(&mScalars.back()->f), PluginFieldType::kFLOAT32, 1});
}

TRTPtr<IPluginV2> PluginBuilder::make(const std::string& name)
{
    PluginFieldCollection collection{static_cast<int>(mFields.size()), mFields.data()};

    TRTPtr<IPluginV2> plugin(mCreator->createPlugin(name.c_str(), &collection));
    if (!plugin)
    {
        throw std::runtime_error(
            "Failed to instantiate plugin '" + name + "' with " + std::to_string(mFields.size()) + ".");
    }

    return plugin;
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void PluginBuilder::setField(const nvinfer1::PluginField& field)
{
    mFields.emplace_back(field);
}

} // namespace tts
