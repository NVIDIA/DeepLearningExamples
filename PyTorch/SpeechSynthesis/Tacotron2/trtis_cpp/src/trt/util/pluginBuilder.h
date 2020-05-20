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

#ifndef TT2I_PLUGINBUILDER_H
#define TT2I_PLUGINBUILDER_H

#include "NvInfer.h"

#include "trtPtr.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts
{

class PluginBuilder
{
public:
    /**
     * @brief Create a new PluginBuilder class.
     *
     * @param pluginName The name of the plugin.
     * @param pluginVersion The version of the plugin.
     */
    PluginBuilder(const std::string& pluginName, const std::string& pluginVersion);

    // delete copy constructor and assignment operator
    PluginBuilder(const PluginBuilder& other) = delete;
    PluginBuilder& operator=(const PluginBuilder& other) = delete;

    /**
     * @brief Add a field to the plugin.
     *
     * @param name The name of the field.
     * @param data The data for the field.
     * @param length The length of the field.
     */
    void setField(const std::string& name, const nvinfer1::Weights& weights);

    /**
     * @brief Add a scalar field to the plugin.
     *
     * @param name The name of the field.
     * @param value The value of the field.
     */
    void setField(const std::string& name, int32_t value);

    /**
     * @brief Add a scalar field to the plugin.
     *
     * @param name The name of the field.
     * @param value The value of the field.
     */
    void setField(const std::string& name, float value);

    /**
     * @brief Build the plugin instance.
     *
     * @param name The name of the instance.
     *
     * @return The instantiated plugin.
     */
    TRTPtr<nvinfer1::IPluginV2> make(const std::string& name);

  private:
    union scalar_t
    {
        int32_t i;
        float f;

        scalar_t(const int32_t value)
            : i(value)
        {
        }
        scalar_t(const float value)
            : f(value)
        {
        }
    };

    nvinfer1::IPluginCreator* mCreator;
    std::vector<nvinfer1::PluginField> mFields;

    // use a set of unique_ptr's, so that as the array is expanded and
    // reallocated, the memory addresses of the scalars do not change.
    std::vector<std::unique_ptr<std::string>> mNames;
    std::vector<std::unique_ptr<scalar_t>> mScalars;

    /**
     * @brief Set a field to the plugin.
     *
     * @param field The field to set.
     */
    void setField(const nvinfer1::PluginField& field);
};

} // namespace tts

#endif
