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

#ifndef TT2I_LAYERDATA_H
#define TT2I_LAYERDATA_H

#include "NvInfer.h"

#include <map>
#include <vector>

namespace tts
{

class LayerData
{
public:
    /**
     * @brief Allocate an empty LayerData object.
     */
    LayerData();

    /**
     * @brief Move constructor.
     *
     * @param other The object to move.
     */
    LayerData(LayerData&& other) = default;

    /**
     * @brief Move assignment operator.
     *
     * @param other The object to move.
     *
     * @return This object.
     */
    LayerData& operator=(LayerData&& other) = default;

    // prevent copying
    LayerData(const LayerData& other) = delete;
    LayerData& operator=(const LayerData& other) = delete;

    /**
     * @brief Get the weights with the given name (e.g., "weight", "bias").
     *
     * @param name The name of the weights.
     *
     * @return The weights.
     */
    nvinfer1::Weights get(const std::string& name) const;

    /**
     * @brief Check if weights with the given name exist in this object.
     *
     * @param name The name of the weights.
     *
     * @return The weights.
     */
    bool has(const std::string& name) const;

    /**
     * @brief Insert new weights into this object.
     *
     * @tparam ITER The type of iterator.
     * @param name The name of the weights.
     * @param start The iterator to the start of the weights.
     * @param end The iterator to the end of the weights (exclusive).
     */
    template <typename ITER>
    void add(const std::string& name, const ITER start, const ITER end)
    {
        mKeys.emplace(name, mKeys.size());
        mData.insert(mData.end(), start, end);
        mPrefix.emplace_back(mData.size());
    }

    /**
     * @brief Insert the new weights into this object.
     *
     * @param name The name of the weights.
     * @param vec The vector of weights.
     */
    void add(const std::string& name, const std::vector<float>& vec)
    {
        add(name, vec.begin(), vec.end());
    }

    friend std::ostream& operator<<(std::ostream& stream, const LayerData& data);

private:
    std::map<std::string, size_t> mKeys;
    std::vector<size_t> mPrefix;
    std::vector<float> mData;
};

/**
 * @brief Print out a LayerData object to human readable form to the output
 * stream.
 *
 * @param stream The stream.
 * @param data The LayerData to write.
 *
 * @return The stream.
 */
std::ostream& operator<<(std::ostream& stream, const LayerData& data);

} // namespace tts

#endif
