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

#ifndef TT2I_BLENDING_H
#define TT2I_BLENDING_H

#include "cuda_runtime.h"

class Blending
{
public:
    /**
     * @brief Linearly blend two overlapping sequences together.
     *
     * @param batchSize The number of items in the batch.
     * @param newChunk The chunk being added/blended with the
     * existing sequence(s).
     * @param base The existin sequence(s).
     * @param chunkSize The size of the new chunk to be added/blended.
     * @param overlapSize The size of the overlap (e.g., the amount to blend).
     * @param outputSequenceSpacing The spacing between sequences in the batch.
     * @param outputSequenceOffset The offset from which to add the new chunk to
     * the existing sequences.
     * @param stream The cuda stream to perform the operation on.
     */
    static void linear(const int batchSize, const float* const newChunk, float* const base, const int chunkSize,
        const int overlapSize, const int outputSequenceSpacing, const int outputSequenceOffset, cudaStream_t stream);
};

#endif
