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

#include "layerData.h"

#include <sstream>
#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

LayerData::LayerData()
    : mKeys()
    , mPrefix{0}
    , mData{}
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

Weights LayerData::get(const std::string& name) const
{
    auto pos = mKeys.find(name);
    if (pos == mKeys.end())
    {
        std::ostringstream ss;
        ss << "Unable to find '" << name << "' in : {";
        for (auto pair : mKeys)
        {
            ss << "'" << pair.first << "', ";
        }
        ss << "}";

        throw std::runtime_error(ss.str());
    }

    const size_t idx = pos->second;

    return Weights{DataType::kFLOAT, (const void*) (mData.data() + mPrefix[idx]),
        static_cast<int64_t>(mPrefix[idx + 1] - mPrefix[idx])};
}

bool LayerData::has(const std::string& name) const
{
    return mKeys.count(name) > 0;
}

/******************************************************************************
 * OUTPUT FUNCTIONS ***********************************************************
 *****************************************************************************/

std::ostream& operator<<(std::ostream& stream, const LayerData& data)
{
    stream << "LayerData: {";
    for (auto pair : data.mKeys)
    {
        stream << pair.first << ":" << (data.mPrefix[pair.second + 1] - data.mPrefix[pair.second]) << ", ";
    }
    stream << "}";
    return stream;
}

} // namespace tts
