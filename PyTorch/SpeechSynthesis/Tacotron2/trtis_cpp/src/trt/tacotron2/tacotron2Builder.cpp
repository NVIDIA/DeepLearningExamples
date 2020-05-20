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

#include "tacotron2Builder.h"
#include "decoderBuilderPlain.h"
#include "decoderBuilderPlugins.h"
#include "encoderBuilder.h"
#include "jsonModelImporter.h"
#include "postNetBuilder.h"
#include "utils.h"

#include <iostream>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{
constexpr const int NUM_EMBEDDING_DIMENSIONS = 512;
constexpr const int NUM_ENCODING_DIMENSIONS = 512;
constexpr const int NUM_ATTENTION_DIMENSIONS = 128;
constexpr const int MEL_CHANNELS = 80;
constexpr const int MAX_MEL_CHUNK = 80;
constexpr const int TOTAL_CHUNK_SIZE = MEL_CHANNELS * MAX_MEL_CHUNK;
} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Tacotron2Builder::Tacotron2Builder(const std::string& modelFilePath)
    : mModelFilePath(modelFilePath)
    , mMelChannels(MEL_CHANNELS)
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

std::vector<TRTPtr<ICudaEngine>> Tacotron2Builder::build(
    const int maxInputLength,
    IBuilder& builder,
    const int maxBatchSize,
    const bool useFP16)
{
    // configure tensor-rt objects
    std::unique_ptr<IModelImporter> importer;
    if (Utils::hasExtension(mModelFilePath, ".json"))
    {
        importer.reset(new JSONModelImporter(mModelFilePath));
    }
    else
    {
        throw std::runtime_error("Unrecognized model filename type: '" + mModelFilePath + "'");
    }

    std::vector<TRTPtr<ICudaEngine>> engines;
    EncoderBuilder encoderBuilder(
        NUM_EMBEDDING_DIMENSIONS, NUM_ENCODING_DIMENSIONS, NUM_ATTENTION_DIMENSIONS, maxInputLength);
    engines.emplace_back(encoderBuilder.build(builder, *importer, maxBatchSize, useFP16));

    DecoderBuilderPlain decoderBuilderPlain(maxInputLength, NUM_EMBEDDING_DIMENSIONS, mMelChannels);
    engines.emplace_back(decoderBuilderPlain.build(builder, *importer, maxBatchSize, useFP16));

    DecoderBuilderPlugins decoderBuilderPlugins(NUM_EMBEDDING_DIMENSIONS, mMelChannels);
    engines.emplace_back(decoderBuilderPlugins.build(builder, *importer, 1, 1, maxInputLength, useFP16));

    PostNetBuilder postnetBuilder(mMelChannels, MAX_MEL_CHUNK, NUM_EMBEDDING_DIMENSIONS);
    engines.emplace_back(postnetBuilder.build(builder, *importer, maxBatchSize, useFP16));

    return engines;
}

} // namespace tts
