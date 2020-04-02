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

#include "encoderInstance.h"
#include "cudaUtils.h"
#include "engineCache.h"
#include "trtUtils.h"

#include <cassert>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

EncoderInstance::EncoderInstance(TRTPtr<ICudaEngine> engine) :
    TimedObject("EncoderInstance::infer()"),
    EngineDriver(std::move(engine)),
    mBinding(),
    mContext(getEngine().createExecutionContext()),
    mInputLength(TRTUtils::getBindingDimension(getEngine(), INPUT_NAME, 1))
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void EncoderInstance::infer(cudaStream_t stream, const int batchSize, const int32_t* const inputDevice,
    const float* const inputMaskDevice, const int32_t* const inputLengthDevice, float* const outputDevice,
    float* const outputProcessedDevice)
{
    startTiming();
    const ICudaEngine& engine = mContext->getEngine();

    mBinding.setBinding(engine, INPUT_NAME, inputDevice);
    mBinding.setBinding(engine, INPUT_MASK_NAME, inputMaskDevice);
    mBinding.setBinding(engine, INPUT_LENGTH_NAME, inputLengthDevice);
    mBinding.setBinding(engine, OUTPUT_NAME, outputDevice);
    mBinding.setBinding(engine, OUTPUT_PROCESSED_NAME, outputProcessedDevice);

    if (!mContext->enqueue(batchSize, mBinding.getBindings(), stream, nullptr))
    {
        throw std::runtime_error("Failed to run encoding.");
    }

    CudaUtils::sync(stream);

    stopTiming();
}

int EncoderInstance::getInputLength() const
{
    return mInputLength;
}

int EncoderInstance::getNumDimensions() const
{
    return TRTUtils::getBindingDimension(getEngine(), OUTPUT_NAME, 2);
}

int EncoderInstance::getNumProcessedDimensions() const
{
    return TRTUtils::getBindingDimension(getEngine(), OUTPUT_PROCESSED_NAME, 1);
}

} // namespace tts
