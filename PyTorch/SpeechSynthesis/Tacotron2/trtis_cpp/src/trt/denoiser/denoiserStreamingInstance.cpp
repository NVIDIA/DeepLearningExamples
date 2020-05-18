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

#include "denoiserStreamingInstance.h"
#include "trtUtils.h"

#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DenoiserStreamingInstance::DenoiserStreamingInstance(
    TRTPtr<ICudaEngine>&& engine) :
    TimedObject("DenoiserStreamingInstance::infer()"),
    EngineDriver(std::move(engine)),
    mBinding(),
    mContext(getEngine().createExecutionContext()),
    mChunkSize(TRTUtils::getBindingSize(getEngine(), INPUT_NAME))
{
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void DenoiserStreamingInstance::startInference()
{
    // do nothing
}

void DenoiserStreamingInstance::inferNext(
    const int batchSize, const float* const inputDevice, float* outputDevice, cudaStream_t stream)
{
    startTiming();

    const ICudaEngine& engine = mContext->getEngine();

    mBinding.setBinding(engine, OUTPUT_NAME, outputDevice);
    mBinding.setBinding(engine, INPUT_NAME, inputDevice);

    if (!mContext->enqueue(batchSize, mBinding.getBindings(), stream, nullptr))
    {
        throw std::runtime_error("Failed to run denoiser.");
    }

    stopTiming();
}

} // namespace tts
