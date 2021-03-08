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

#ifndef TT2I_LSTMCELLLAYERPLUGIN_H
#define TT2I_LSTMCELLLAYERPLUGIN_H

#include "NvInfer.h"

#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Taco2LSTMCellKernel;

class Taco2LSTMCellLayerPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    using value_type = float;

    enum Inputs
    {
        INPUT_FIRST_INDEX = 0,
        INPUT_SECOND_INDEX = 1,
        HIDDEN_INDEX = 2,
        CELL_INDEX = 3,
        NUM_INPUTS = 4
    };

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
     * @brief Create a new Taco2LSTMCellLayerPlugin plugin from serialized data.
     *
     * @param data The data.
     * @param length The length of the data in bytes.
     *
     * @return The instantiated plugin.
     */
    static Taco2LSTMCellLayerPlugin deserialize(const void* data, size_t length);

    /**
     * @brief Create a new Taco2LSTMCellLayerPlugin. The weights and bias are in
     * [i,f,g,o] order.
     *
     * @param inputWeights The input weights (Wi).
     * @param hiddenWeights the hidden weights (Wh).
     * @param inputBias The input bias (Bi).
     * @param hiddenBias The hidden bias (Bh).
     * @param inputLength The input length.
     * @param numDimension The number of hidden dimensions.
     * @param useFP16 Whether or not to store weights in fp16 format.
     */
    Taco2LSTMCellLayerPlugin(const nvinfer1::Weights& inputWeights, const nvinfer1::Weights& hiddenWeights,
        const nvinfer1::Weights& inputBias, const nvinfer1::Weights& hiddenBias, int inputLength, int numDimension,
        bool useFP16);

    /**
     * @brief Move constructor.
     *
     * @param other The Taco2LSTMCellLayerPlugin to move.
     */
    Taco2LSTMCellLayerPlugin(Taco2LSTMCellLayerPlugin&& other);

    /**
     * @brief Move assignment operator.
     *
     * @param other The Taco2LSTMCellLayerPlugin to move.
     *
     * @return This Taco2LSTMCellLayerPlugin.
     */
    Taco2LSTMCellLayerPlugin& operator=(Taco2LSTMCellLayerPlugin&& other);

    /**
     * @brief Destructor.
     */
    ~Taco2LSTMCellLayerPlugin();

    // disable copying
    Taco2LSTMCellLayerPlugin(const Taco2LSTMCellLayerPlugin& other) = delete;
    Taco2LSTMCellLayerPlugin& operator=(const Taco2LSTMCellLayerPlugin& other) = delete;

    /**
     * @brief Return the data tyep of the plugin output at the requested index.
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
     * @param index The index of the output.
     * @param inputs The given inputs.
     * @param nbInputs The number of inputs.
     * @param exprBuilder
     *
     * @return The resulting dimensions.
     */
    nvinfer1::DimsExprs getOutputDimensions(
        int index, const nvinfer1::DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;

    /**
     * @brief Check if the given plugin format is supported.
     *
     * @param pos The format position in inOut.format[].
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
     * @param The input tensor attributes that are used for configuration.
     * @param The number of inputs.
     * @param The output tensor attributes that are used for configuration.
     * @param The number of outputs.
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
     * @param inputDesc The input tensors descriptors.
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
    int mInputLengthFirst;
    int mInputLengthSecond;
    int mNumDimension;
    std::vector<value_type> mInputWeightsHost;
    std::vector<value_type> mHiddenWeightsHost;
    std::vector<value_type> mInputBiasHost;
    std::vector<value_type> mHiddenBiasHost;

    std::string mNamespace;

    std::unique_ptr<Taco2LSTMCellKernel> mCell;

    int numInputWeights() const;
    int numHiddenWeights() const;
    int numBiases() const;
    size_t numInputWeightBytes() const;
    size_t numHiddenWeightBytes() const;
    size_t numBiasBytes() const;

    bool mUseFP16;
};

} // namespace plugin
} // namespace nvinfer1

#endif
