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

#ifndef TT2I_PROJECTIONLAYERPLUGIN_H
#define TT2I_PROJECTIONLAYERPLUGIN_H

#include "NvInfer.h"

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Taco2ProjectionKernel;

class Taco2ProjectionLayerPlugin : public nvinfer1::IPluginV2DynamicExt
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
     * @brief Create a new Taco2ProjectionLayerPlugin from serialized data.
     *
     * @param data The data.
     * @param length The length of the data in bytes.
     *
     * @return The instantiated plugin.
     */
    static Taco2ProjectionLayerPlugin deserialize(const void* data, size_t length);

    /**
     * @brief Create a new Taco2ProjectionLayerPlugin.
     *
     * @param weightsChannel The weights for channel projection.
     * @param weightsGate The weights for gate projection.
     * @param biasChannel The bias for channels.
     * @param biasGate The bias for the gate.
     * @param hiddenIntputLength The hidden input lentgh.
     * @param contextIntputLength The context input length.
     * @param channelDimension The number of channels.
     * @param gateDimension The number of gates.
     */
    Taco2ProjectionLayerPlugin(const nvinfer1::Weights& weightsChannel, const nvinfer1::Weights& weightsGate,
        const nvinfer1::Weights& biasChannel, const nvinfer1::Weights& biasGate, int hiddenIntputLength,
        int contextIntputLength, int channelDimension, int gateDimension);

    /**
     * @brief The move constructor.
     *
     * @param other The Taco2ProjectionLayerPlugin to move.
     */
    Taco2ProjectionLayerPlugin(Taco2ProjectionLayerPlugin&& other);

    /**
     * @brief The move assignment operator.
     *
     * @param other The Taco2ProjectionLayerPlugin to move.
     *
     * @return This object.
     */
    Taco2ProjectionLayerPlugin& operator=(Taco2ProjectionLayerPlugin&& other);

    /**
     * @brief Destructor.
     */
    ~Taco2ProjectionLayerPlugin();

    // disable copying
    Taco2ProjectionLayerPlugin(const Taco2ProjectionLayerPlugin& other) = delete;
    Taco2ProjectionLayerPlugin& operator=(const Taco2ProjectionLayerPlugin& other) = delete;

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
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& expBuilder) override;

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
     * @param in The input tensor attributes that used for configuration.
     * @param nbInputs The number of inputs.
     * @param out The output tensor attributes that are used for configuration.
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
     * batch size.
     *
     * @param in The input tensor descriptors.
     * @param nbInputs The number of inputs.
     * @param out The output tensor descriptors.
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
     * @param workspace The workspace.
     * @param stream The stream to operate on.
     *
     * @return 0 if successfully queued, non-zero otherwise.
     */
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream);

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
    static const float ONE;
    static const float ZERO;

    int mHiddenInputLength;
    int mContextInputLength;
    int mNumChannelDimension;
    int mNumGateDimension;
    std::vector<value_type> mWeightsChannel;
    std::vector<value_type> mWeightsGate;
    std::vector<value_type> mBiasChannel;
    std::vector<value_type> mBiasGate;
    std::unique_ptr<Taco2ProjectionKernel> mKernel;

    std::string mNamespace;

    int getTotalInputLength() const;

    int getTotalDimensions() const;
};

} // namespace plugin
} // namespace nvinfer1

#endif
