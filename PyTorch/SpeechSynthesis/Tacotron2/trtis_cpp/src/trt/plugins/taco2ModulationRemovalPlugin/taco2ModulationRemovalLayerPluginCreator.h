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

#ifndef TT2I_MODULATIONREMOVALLAYERPLUGINCREATOR_H
#define TT2I_MODULATIONREMOVALLAYERPLUGINCREATOR_H

#include "NvInfer.h"

#include <string>

#ifdef DEVEL
// The destructor of nvinfer1::IPluginCreator is non-virtual and public, so
// we need to supress the warning.
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

namespace nvinfer1
{
namespace plugin
{

class Taco2ModulationRemovalLayerPluginCreator : public nvinfer1::IPluginCreator
{
public:
    /**
     * @brief Get the collection of fields for this plugin, with their names only.
     *
     * @return The collection of fields.
     */
    static nvinfer1::PluginFieldCollection* getFields();

    /**
     * @brief Create a new Taco2ModulationRemovalLayerPluginCreator.
     */
    Taco2ModulationRemovalLayerPluginCreator();

    /**
     * @brief Get the name of the plugin.
     *
     * @return The name of the plugin.
     */
    const char* getPluginName() const override;

    /**
     * @brief Get the plugin version.
     *
     * @return The plugin version.
     */
    const char* getPluginVersion() const override;

    /**
     * @brief Get the collection of fields for this plugin.
     *
     * @return The collection of fields.
     */
    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    /**
     * @brief Create a new Taco2ModulationRemovalLayerPlugin.
     *
     * @param name The name (unused currently).
     * @param fc The collection of fields to initialize with.
     *
     * @return The created plugin.
     */
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    /**
     * @brief Create a custom layer by name from a data stream.
     *
     * @param layerName The name of the layer.
     * @param serialData The serialized data for the layer.
     * @param serialLength The length of the serialized data.
     *
     * @return The plugin. Clients must destroy the plugin once all consumers of
     * it have been destroyed.
     */
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    /**
     * @brief Set the namespace for created plugins.
     *
     * @param pluginNamespace The namespace.
     */
    void setPluginNamespace(const char* pluginNamespace) override;

    /**
     * @brief Get the namespace for created plugins.
     *
     * @return The namespace.
     */
    const char* getPluginNamespace() const override;

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#ifdef DEVEL
#pragma GCC diagnostic pop
#endif

#endif
