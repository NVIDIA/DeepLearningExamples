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

#include "denoiserInstance.h"
#include "cudaUtils.h"
#include "dataShuffler.h"

#include <stdexcept>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

DenoiserInstance::DenoiserInstance(TRTPtr<ICudaEngine>&& engine) :
    TimedObject("DenoiserInstance::infer()"),
    mStreamingInstance(std::move(engine)),
    mInBufferDevice(
        mStreamingInstance.getChunkSize()
        * mStreamingInstance.getMaxBatchSize()),
    mOutBufferDevice(
        mStreamingInstance.getChunkSize()
        * mStreamingInstance.getMaxBatchSize())
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void DenoiserInstance::infer(const int batchSize, const float* const inputDevice, const int inputSpacing,
    const int* const inputLength, float* outputDevice)
{
    startTiming();

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        throw std::runtime_error("Failed to create stream.");
    }

    const int chunkSize = mStreamingInstance.getChunkSize();

    int maxNumSamples = 0;
    for (int i = 0; i < batchSize; ++i)
    {
        if (inputLength[i] > maxNumSamples)
        {
            maxNumSamples = inputLength[i];
        }
    }

    mStreamingInstance.startInference();

    for (int pos = 0; pos < maxNumSamples; pos += chunkSize)
    {
      DataShuffler::frameTransfer(
          inputDevice,
          mInBufferDevice.data(),
          inputSpacing,
          pos,
          chunkSize,
          batchSize,
          chunkSize,
          0,
          stream);

      mStreamingInstance.inferNext(
          batchSize, mInBufferDevice.data(), mOutBufferDevice.data(), stream);

      DataShuffler::frameTransfer(
          mOutBufferDevice.data(),
          outputDevice,
          chunkSize,
          0,
          chunkSize,
          batchSize,
          inputSpacing,
          pos,
          stream);
    }

    CudaUtils::sync(stream);

    cudaStreamDestroy(stream);

    stopTiming();
}

} // namespace tts
