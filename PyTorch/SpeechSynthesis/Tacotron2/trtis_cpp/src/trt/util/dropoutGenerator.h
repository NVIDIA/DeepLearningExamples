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

#ifndef TT2I_DROPOUTGENERATOR_H
#define TT2I_DROPOUTGENERATOR_H

#include "random.h"

#include "cudaMemory.h"
#include "cuda_runtime.h"

namespace tts
{

class DropoutGenerator
{
public:
    /**
     * @brief Create a new dropout generator.
     *
     * @param maxBatchSize The maximum batch size.
     * @param maxChunkSize The maximum number of chunks to generate at once.
     * @param numValues The number of values to generate dropouts for.
     * @param prob The probability with which to drop values.
     * @param seed The seed to use for the random number generator.
     */
    DropoutGenerator(int maxBatchSize, int maxChunkSize, int numValues, float prob, unsigned int seed = 0);

    /**
     * @brief Reset the random number generator.
     *
     * @param seed The seed to use.
     * @param stream The stream to use.
     */
    void reset(unsigned int seed, cudaStream_t stream);

    /**
     * @brief Generate a new set of dropout values.
     *
     * @param batchSize The size of the batch.
     * @param numChunks The number of chunks to generate.
     * @param stream The stream to generate in.
     */
    void generate(int batchSize, int numChunks, cudaStream_t stream);

    /**
     * @brief Get a pointer to the device memory containing the dropout values.
     * This memory is changed when `generate()` is called.
     *
     * @param chunk The chunk of dropouts to get.
     *
     * @return The memory location.
     */
    const float* get(int chunk) const;

    /**
     * @brief Get the number of values generated with each call to `generate()`.
     *
     * @return
     */
    int size() const
    {
        return mNumValues;
    }

private:
    float mProb;
    int mNumValues;
    int mMaxChunkSize;
    int mGeneratedChunks;
    int mBatchSize;
    CudaMemory<float> mDropoutDevice;
    Random mRand;
};

} // namespace tts

#endif
