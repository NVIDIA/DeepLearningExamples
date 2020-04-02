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

#ifndef TT2I_TRTUTILS_H
#define TT2I_TRTUTILS_H

#include <string>
#include <vector>

// forward declare objects passed by reference
namespace nvinfer1
{
class Weights;
class Dims;
class ITensor;
class ICudaEngine;
class INetworkDefinition;
} // namespace nvinfer1

namespace tts
{

class TRTUtils
{
public:
    /**
     * @brief Convert a set of weights to a vector.
     *
     * @param weights The weights.
     *
     * @return The vector.
     */
    static std::vector<float> toFloatVector(const nvinfer1::Weights& weights);

    /**
     * @brief Convert a vector to a Weights object. The vector must not have its
     * underlying data free'd/move'd for the lifetime of the Weights object.
     *
     * @param vec The vector.
     *
     * @return  The Weights object pointing to the vector.
     */
    static nvinfer1::Weights toWeights(const std::vector<float>& vec);

    /**
     * @brief Create a string representation of a Dims object.
     *
     * @param dim The object to create a string representation of.
     *
     * @return The string represenetation.
     */
    static std::string dimsToString(const nvinfer1::Dims& dim);

    /**
     * @brief Print the string representation of the Dims object to stdout.
     *
     * @param name The name to prefix the dimensions with.
     * @param dim The dimensions.
     */
    static void printDimensions(const std::string& name, const nvinfer1::Dims& dim);

    /**
     * @brief Print the input and output sizes of the engine to stdout.
     *
     * @param engine The engine to print the input/output of.
     */
    static void printBindingDimensions(const nvinfer1::ICudaEngine& engine);

    /**
     * @brief Print the string representation of the tensor's Dims object to
     * stdout.
     *
     * @param name The name of the tensor.
     * @param tensor The tensor to print the dimensions of.
     */
    static void printTensor(const std::string& name, const nvinfer1::ITensor& tensor);

    /**
     * @brief Get the total volume of a set of Dimensions (number of elements).
     *
     * @param dims The dimensions.
     *
     * @return The volume/total number of elements.
     */
    static size_t getDimensionsSize(const nvinfer1::Dims& dims);

    /**
     * @brief Get the total number of elements in a tensor.
     *
     * @param tensor The tensor.
     *
     * @return The total number of elements.
     */
    static size_t getTensorSize(const nvinfer1::ITensor& tensor);

    /**
     * @brief Get the maixmum size of an input/output tensor for a given
     * binding in an engine.
     *
     * @param engine The engine.
     * @param binding The binding name.
     *
     * @return The maximum size.
     */
    static size_t getMaxBindingSize(const nvinfer1::ICudaEngine& engine, const char* const binding);

    /**
     * @brief Get the size of an input/output tensor for the given binding in an
     * engine.
     *
     * @param engine The engine.
     * @param bindingName The binding name.
     *
     * @return The number of elements in the binding.
     */
    static size_t getBindingSize(const nvinfer1::ICudaEngine& engine, const char* const bindingName);

    /**
     * @brief Get the size of an input/output tensor for the given binding in an
     * engine excluding the first dimension (assumes it to be explicit batch
     * size).
     *
     * @param engine The engine.
     * @param bindingName The binding name.
     *
     * @return The number of elements in the binding.
     */
    static size_t getNonBatchBindingSize(const nvinfer1::ICudaEngine& engine, const char* const bindingName);

    /**
     * @brief Get the maximum batch size of the engine's first binding,
     * using optimization
     * profiles to determine the size if explicit batching is enabled, or
     * directly querying the engine if implicit batching is used.
     *
     * @param engine The engine.
     *
     * @return The maximum batch size supported.
     */
    static int getMaxBatchSize(const nvinfer1::ICudaEngine& engine);

    /**
     * @brief Get the size of specific dimension of an input/output tensor for the
     * given binding in an engine.
     *
     * @param engine The engine.
     * @param bindingName The binding name.
     * @param dimension The dimension in the tensor to get the size of.
     *
     * @return The size of the dimension.
     */
    static size_t getBindingDimension(
        const nvinfer1::ICudaEngine& engine, const char* const bindingName, const int dimension);

    /**
     * @brief Get the input tensor by its name.
     *
     * @param network The network.
     * @param inputName The tensor name.
     *
     * @return The input tensor.
     */
    static nvinfer1::ITensor* getInputByName(nvinfer1::INetworkDefinition& network, const std::string& inputName);

    /**
     * @brief Get the first dimension with a non-unit size (greater than one). If
     * no dimensions are greater than 1, but the number of dimensions is greater
     * than 0, 1 is returned. If the number of dimensions is 0, then 0 is
     * returned.
     *
     * @param dims The dimensions to search.
     *
     * @return The first non-unit dimension.
     */
    static int getFirstNonUnitDim(const nvinfer1::Dims& dims);

    /**
     * @brief Get a Dims object with all 1's removed down to minLength.
     * If the length of dims is less than minLength, than just dims will be
     * returned. Leading 1's will be insert to reach minLength.
     *
     * @param dims The dimensions to compact.
     *
     * @return The compacted dimensions.
     */
    static nvinfer1::Dims getCompactedDims(const nvinfer1::Dims& dims, const int minLength = 1);
};

} // namespace tts

#endif
