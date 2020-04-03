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

#ifndef TT2I_JSONMODELIMPORTER_H
#define TT2I_JSONMODELIMPORTER_H

#include "IModelImporter.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tts
{

class JSONModelImporter : public IModelImporter
{
public:
    /**
     * @brief Create a new JSON model importer. This reads all of the weights
     * from the file and stores them in this object.
     *
     * @param filename The json filename.
     */
    JSONModelImporter(const std::string& filename);

    /**
     * @brief Get the weights associate with the given path.
     *
     * @param path The json key (e.g., "{"conv","0","batch_norm"} for *
     * "conv.0.batch_norm").
     *
     * @return The data for the given key.
     */
    const LayerData* getWeights(const std::vector<std::string>& path) override;

private:
    std::map<std::string, std::unique_ptr<LayerData>> mWeights;
};

} // namespace tts

#endif
