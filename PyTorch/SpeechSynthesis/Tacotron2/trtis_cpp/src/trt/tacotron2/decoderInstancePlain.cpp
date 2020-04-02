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

#include "decoderInstancePlain.h"
#include "cudaUtils.h"
#include "trtUtils.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DecoderInstancePlain::DecoderInstancePlain(
    TRTPtr<ICudaEngine> engine, const int maxChunkSize) :
    DecoderInstance(std::move(engine), maxChunkSize),
    mBinding(),
    mInputWeightsDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_WEIGHTS_NAME)),
    mOutputWeightsDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_WEIGHTS_NAME)),
    mInAttentionHiddenStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_ATTENTIONHIDDEN_NAME)),
    mInAttentionCellStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_ATTENTIONCELL_NAME)),
    mOutAttentionHiddenStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), OUTPUT_ATTENTIONHIDDEN_NAME)),
    mOutAttentionCellStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), OUTPUT_ATTENTIONCELL_NAME)),
    mInputAttentionContextDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_CONTEXT_NAME)),
    mOutputAttentionContextDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), OUTPUT_CONTEXT_NAME)),
    mInDecoderHiddenStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_DECODERHIDDEN_NAME)),
    mInDecoderCellStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), INPUT_DECODERCELL_NAME)),
    mOutDecoderHiddenStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), OUTPUT_DECODERHIDDEN_NAME)),
    mOutDecoderCellStatesDevice(
        getMaxBatchSize()
        * TRTUtils::getBindingSize(getEngine(), OUTPUT_DECODERCELL_NAME))
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void DecoderInstancePlain::reset()
{
    DecoderInstance::reset();

    mInputWeightsDevice.zero();
    mInAttentionHiddenStatesDevice.zero();
    mInAttentionCellStatesDevice.zero();
    mInputAttentionContextDevice.zero();
    mOutputAttentionContextDevice.zero();
    mInDecoderHiddenStatesDevice.zero();
    mInDecoderCellStatesDevice.zero();
}

/******************************************************************************
 * PROTECTED METHODS **********************************************************
 *****************************************************************************/

void DecoderInstancePlain::decode(cudaStream_t stream, IExecutionContext& context, const int batchSize,
    const float* const inputLastFrameDevice, const float* const inputMemoryDevice,
    const float* const inputProcessedMemoryDevice, const float* const inputMaskDevice,

    const int32_t* const /* inputLengthHost */, const int32_t* const inputLengthDevice,
    const float* const inputDropoutDevice, float* const outputChannelsDevice)
{
    const ICudaEngine& engine = context.getEngine();

    mBinding.setBinding(engine, INPUT_MASK_NAME, inputMaskDevice);
    mBinding.setBinding(engine, INPUT_LENGTH_NAME, inputLengthDevice);
    mBinding.setBinding(engine, INPUT_DROPOUT_NAME, inputDropoutDevice);
    mBinding.setBinding(engine, INPUT_MEMORY_NAME, inputMemoryDevice);
    mBinding.setBinding(engine, INPUT_PROCESSED_NAME, inputProcessedMemoryDevice);
    mBinding.setBinding(engine, INPUT_WEIGHTS_NAME, mInputWeightsDevice.data());
    mBinding.setBinding(engine, INPUT_LASTFRAME_NAME, inputLastFrameDevice);
    mBinding.setBinding(engine, INPUT_CONTEXT_NAME, mInputAttentionContextDevice.data());
    mBinding.setBinding(engine, INPUT_ATTENTIONHIDDEN_NAME, mInAttentionHiddenStatesDevice.data());
    mBinding.setBinding(engine, INPUT_ATTENTIONCELL_NAME, mInAttentionCellStatesDevice.data());
    mBinding.setBinding(engine, INPUT_DECODERHIDDEN_NAME, mInDecoderHiddenStatesDevice.data());
    mBinding.setBinding(engine, INPUT_DECODERCELL_NAME, mInDecoderCellStatesDevice.data());
    mBinding.setBinding(engine, OUTPUT_CONTEXT_NAME, mOutputAttentionContextDevice.data());
    mBinding.setBinding(engine, OUTPUT_WEIGHTS_NAME, mOutputWeightsDevice.data());
    mBinding.setBinding(engine, OUTPUT_ATTENTIONHIDDEN_NAME, mOutAttentionHiddenStatesDevice.data());
    mBinding.setBinding(engine, OUTPUT_ATTENTIONCELL_NAME, mOutAttentionCellStatesDevice.data());
    mBinding.setBinding(engine, OUTPUT_DECODERHIDDEN_NAME, mOutDecoderHiddenStatesDevice.data());
    mBinding.setBinding(engine, OUTPUT_DECODERCELL_NAME, mOutDecoderCellStatesDevice.data());
    mBinding.setBinding(engine, OUTPUT_CHANNELS_NAME, outputChannelsDevice);

    if (!context.enqueue(batchSize, mBinding.getBindings(), stream, nullptr))
    {
        throw std::runtime_error("Failed to run decoder.");
    }

    // swap pointers
    std::swap(mInputWeightsDevice, mOutputWeightsDevice);
    std::swap(mInputAttentionContextDevice, mOutputAttentionContextDevice);

    std::swap(mInAttentionHiddenStatesDevice, mOutAttentionHiddenStatesDevice);
    std::swap(mInAttentionCellStatesDevice, mOutAttentionCellStatesDevice);

    std::swap(mInDecoderHiddenStatesDevice, mOutDecoderHiddenStatesDevice);
    std::swap(mInDecoderCellStatesDevice, mOutDecoderCellStatesDevice);

    // required because of LSTM cells
    CudaUtils::sync(stream);
}

} // namespace tts
