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

#ifndef TT2I_PRENETLAYERPLUGIN_H
#define TT2I_PRENETLAYERPLUGIN_H

#include "taco2PrenetKernel.h"

#include "NvInfer.h"

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Taco2PrenetLayerPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    using value_type = float;

    /**
     * @brief Get the name of this plugin.
     *
     * @return The name.
     */
    static const char* getName();

    /**
     * @brief Get the version of this plugin.
     *
     * @return The version.
     */
    static const char* getVersion();

    /**
     * @brief Create a new Taco2PrenetLayerPlugin plugin from serialized data.
     *
     * @param data The data.
     * @param length The length of the data in bytes.
     *
     * @return The instantiated plugin.
     */
    static Taco2PrenetLayerPlugin deserialize(const void* data, size_t length);

    /**
     * @brief Create a new Taco2PrenetLayerPlugin plugin.
     *
     * @param fcWeights1 The weights of the first fully connected layer.
     * @param fcWeights2 The weights of the second fully connected layer.
     * @param intputLength The input length.
     * @param numDimension The number of dimensions.
     */
    Taco2PrenetLayerPlugin(
        const nvinfer1::Weights& fcWeights1, const nvinfer1::Weights& fcWeights2, int intputLength, int numDimension);

    /**
     * @brief The move constructor.
     *
     * @param other The Taco2PrenetLayerPlugin to move.
     */
    Taco2PrenetLayerPlugin(Taco2PrenetLayerPlugin&& other);

    ~Taco2PrenetLayerPlugin();

    // disable copying
    Taco2PrenetLayerPlugin(const Taco2PrenetLayerPlugin& other) = delete;
    Taco2PrenetLayerPlugin& operator=(const Taco2PrenetLayerPlugin& other) = delete;

    /**
     * @brief Return the data type of the plugin output at the requested index.
     *
     * @param index The output index.
     * @param inputTypes The input data types.
     * @param nbInputs The number of inputs.
     *
     * @return The type of output.
     */
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    /**
     * @brief Get the plugin type.
     *
     * @return The plugin type.
     */
    const char* getPluginType() const override;

    /**
     * @brief Get the plugin version.
     *
     * @return The plugin version.
     */
    const char* getPluginVersion() const override;

    /**
     * @brief Get the number of outputs.
     *
     * @return The number of outputs.
     */
    int getNbOutputs() const override;

    /**
     * @brief Get the dimensions of an output tensor.
     *
     * @param outputIndex The index of the output tensor.
     * @param inputs Expressions for dimensions of the input tensors.
     * @param nbInputs The number of input tensors.
     * @param expBuilder Object for generating new expressions.
     *
     * @return The resulting dimensions.
     */
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, IExprBuilder& expBuilder) override;

    /**
     * @brief Check if the given plugin format is supported.
     *
     * @param pos The format position/index in inOut.format[].
     * @param inOut The input and output formats.
     * @param nbInputs The number of inputs.
     * @param nbOutputs The number of outputs.
     *
     * @return True if it is supported.
     */
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    /**
     * @brief Configure this plugin with the given inputs, outputs, and datat
     * types.
     *
     * @param in The input tensor descriptions.
     * @param nbInputs The number of inputs.
     * @param out The output tensor descriptions.
     * @param nbOutputs The number of outputs.
     */
    void configurePlugin(
        const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;

    /**
     * @brief Initialize the plugin.
     *
     * @return 0 if initialization was successful. Non-zero otherwise.
     */
    int initialize() override;

    /**
     * @brief Terminate the plugin (deinitialize).
     */
    void terminate() override;

    /**
     * @brief Get workspace size required by this plugin for up to the given
     * configuration.
     *
     * @param in The input tensor descriptions.
     * @param nbInputs The number of inputs.
     * @param out The output tensor descriptions.
     * @param nbOutputs The number of outputs.
     *
     * @return The workspace size in bytes.
     */
    size_t getWorkspaceSize(
        const PluginTensorDesc* in, int nbInputs, const PluginTensorDesc* out, int nbOutputs) const override;

    /**
     * @brief Set this plugin for execution on the stream.
     *
     * @param inputDesc The input tensor descriptors.
     * @param outputDesc The output tensor descriptors.
     * @param inputs The input tensors.
     * @param outputs The output tensors.
     * @param workspace The allocated workspace.
     * @param stream The stream to operate on.
     *
     * @return 0 if successfully queued, non-zero otherwise.
     */
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) override;

    /**
     * @brief Get the number of bytes occupied by this plugin if serialized.
     *
     * @return The size in bytes.
     */
    size_t getSerializationSize() const override;

    /**
     * @brief Serialize this plugin.
     *
     * @param buffer The buffer to write to.
     */
    void serialize(void* buffer) const override;

    /**
     * @brief Destroy this plugin instance.
     */
    void destroy() override;

    /**
     * @brief Clone this pulgin instance.
     *
     * @return The cloned plugin.
     */
    IPluginV2DynamicExt* clone() const override;

    /**
     * @brief Set the namespace of this plugin.
     *
     * @param pluginNamespace The namespace.
     */
    void setPluginNamespace(const char* pluginNamespace) override;

    /**
     * @brief Get the namespace of this plugin.
     *
     * @return The namespace.
     */
    const char* getPluginNamespace() const override;

private:
    int mInputLength;
    int mNumDimension;
    std::vector<value_type> mWeights1Host;
    std::vector<value_type> mWeights2Host;

    std::unique_ptr<Taco2PrenetKernel> mKernel;

    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
