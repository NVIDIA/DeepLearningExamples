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

#ifndef TT2I_DENOISETRANSFORMLAYERPLUGIN_H
#define TT2I_DENOISETRANSFORMLAYERPLUGIN_H

#include "cudaMemory.h"

#include "NvInfer.h"

#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Taco2DenoiseTransformLayerPlugin : public nvinfer1::IPluginV2Ext
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
     * @brief Create a new Taco2DenoiseTransformLayer from serialized data.
     *
     * @param data The data.
     * @param length The length of the data in bytes.
     *
     * @return The instantiated plugin.
     */
    static Taco2DenoiseTransformLayerPlugin deserialize(const void* data, size_t length);

    /**
     * @brief Create a new Taco2DenoiseTransformLayerPlugin.
     *
     * @param weight The weights to use.
     * @param filterLength The length of the filter.
     * @param inputLength The input length.
     */
    Taco2DenoiseTransformLayerPlugin(const nvinfer1::Weights& weight, int filterLength, int inputLength);

    /**
     * @brief Move constructor.
     *
     * @param other The Taco2DenoiseTransformLayerPlugin to move.
     */
    Taco2DenoiseTransformLayerPlugin(Taco2DenoiseTransformLayerPlugin&& other);

    /**
     * @brief The move operator.
     *
     * @param other The Taco2DenoiseTransformLayerPlugin to move.
     *
     * @return This object.
     */
    Taco2DenoiseTransformLayerPlugin& operator=(Taco2DenoiseTransformLayerPlugin&& other);

    /**
     * @brief Destructor.
     */
    ~Taco2DenoiseTransformLayerPlugin();

    // disable copying
    Taco2DenoiseTransformLayerPlugin(const Taco2DenoiseTransformLayerPlugin& other) = delete;
    Taco2DenoiseTransformLayerPlugin& operator=(const Taco2DenoiseTransformLayerPlugin& other) = delete;

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
     * @brief Check if the output will be broadcast across the batch.
     *
     * @param outputIndex The output index.
     * @param inputIsBroadCasted Whether or not the input is broadcasted.
     * @param nbInputs The number of inputs.
     *
     * @return True if the output will be broadcasted.
     */
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadCasted, int nbInputs) const override;

    /**
     * @brief Check if the input can be broadcasted across the batch.
     *
     * @param inputIndex The input index.
     *
     * @return True if the input can be broadcasted.
     */
    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

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
     * @param nbInputDims The number of inputs.
     *
     * @return The resulting dimensions.
     */
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    /**
     * @brief Check if the given plugin format is supported.
     *
     * @param type The data type.
     * @param format The plugin format.
     *
     * @return True if it is supported.
     */
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

    /**
     * @brief Configure this plugin with the given inputs, outputs, and datat
     * types.
     *
     * @param inputDims The input tensors dimensions.
     * @param nbInputs The number of inputs.
     * @param outputDims The output tensor dimensions.
     * @param nbOutputs The number of outputs.
     * @param inputTypes The input data types.
     * @param outputTypes The output data types.
     * @param inputIsBroadcast Whether or not the input is broadcast.
     * @param outputIsBroadcast Whether or not the output is broadcast.
     * @param format The format for the plugin.
     * @param maxBatchSize The maximum batch size that will be used.
     */
    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
        const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) override;

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
     * @param maxBatchSize The maximum number of items in the batch.
     *
     * @return The workspace size in bytes.
     */
    size_t getWorkspaceSize(int maxBatchSize) const override;

    /**
     * @brief Set this plugin for execution on the stream.
     *
     * @param batchSize The number of items in the batch.
     * @param inputs The input tensors.
     * @param outputs The output tensors.
     * @param workspace The workspace.
     * @param stream The stream to operate on.
     *
     * @return 0 if successfully queued, non-zero otherwise.
     */
    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

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
    IPluginV2Ext* clone() const override;

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
    int mFilterLength;
    int mInputLength;
    std::vector<value_type> mWeightsHost;
    tts::CudaMemory<float> mWeightsDevice;

    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
