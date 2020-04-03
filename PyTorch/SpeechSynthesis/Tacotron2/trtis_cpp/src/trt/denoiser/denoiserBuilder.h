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

#ifndef TT2I_DENOISER_H
#define TT2I_DENOISER_H

#include "IModelImporter.h"
#include "trtPtr.h"

#include <memory>

namespace nvinfer1
{
class ICudaEngine;
class IBuilder;
} // namespace nvinfer1

namespace tts
{

class DenoiserBuilder
{
public:
    /**
     * @brief Create a new denoiser.
     *
     * @param sampleLength The number of samples.
     * @param filterLength The filter length.
     * @param numOverlap The number of overlapping filters.
     * @param winLength The length of the window.
     */
    DenoiserBuilder(int sampleLength, int filterLength = 1024, int numOverlap = 4, int winLength = 1024);

    /**
     * @brief Create a new Denoiser engine.
     *
     * @param importer The weight importer.
     * @param builder The builder.
     * @param maxBatchSize The maximum batch size to support.
     * @param useFP16 Whether or not to allow FP16 calculations.
     *
     * @return The built engine.
     */
    TRTPtr<nvinfer1::ICudaEngine> build(
        IModelImporter& importer,
        nvinfer1::IBuilder& builder,
        const int maxBatchSize,
        const bool useFP16);

  private:
    int mChunkSize;
    int mFilterLength;
    int mHopLength;
    int mWinLength;
};

} // namespace tts

#endif
