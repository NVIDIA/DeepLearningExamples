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

#ifndef TT2I_DATASHUFFLE_H
#define TT2I_DATASHUFFLE_H

#include "cuda_runtime.h"

#include <stdint.h>

namespace tts
{

class DataShuffler
{
public:
    /**
     * @brief Take decoder output in TN(C+G) output, where C contains the 80 mel
     * channels and G is the 1 gate output, and output a NCT tensor and gate
     * tensor NGT.
     *
     * @param in The input matrix.
     * @param out The tranposed matrix (output).
     * @param gateOut The gate tensor (output).
     * @param batchSize The batch size.
     * @param chunkSize The number of frames.
     * @param numChannels The number of channels (excluding the output gate).
     * @param stream The stream to perform the operation in.
     */
    static void parseDecoderOutput(const float* in, float* out, float* gateOut, int batchSize, int chunkSize,
        int numChannels, cudaStream_t stream);

    /**
     * @brief Tranpose a matrix on the GPU.
     *
     * @param in The input matrix.
     * @param out The tranposed matrix (output).
     * @param nRows The number of rows.
     * @param nCols The number of columns.
     * @param stream The stream to perform the operation in.
     */
    static void transposeMatrix(const float* in, float* out, int nRows, int nCols, cudaStream_t stream);

    /**
     * @brief Shuffle the mel-spectrograms in output, so that it goes from Chunk,
     * Batch, Channel ordering, to Batch, Chunk, Channel, and each sequence to the
     * specified compactLength.
     *
     * @param in The input.
     * @param out The shuffled output.
     * @param batchSize The size of the batch.
     * @param numChannels The number of channels per frame.
     * @param chunkSize The number of frames per chunk.
     * @param numChunks The number of chunks.
     * @param compactLength The length to compact to.
     * @param stream The stream to operate on.
     */
    static void shuffleMels(const float* in, float* out, int batchSize, int numChannels, int chunkSize, int numChunks,
        int compactLength, cudaStream_t stream);

    /**
     * @brief Scatter a frame of data across several different sequences.
     *
     * @param in The input frames.
     * @param out The output frames.
     * @param inputSequenceSpacing The spacing between the start of each
     * input sequence.
     * @param inputSequenceOffset The offset within each input sequence where
     * the chunks will be taken from.
     * @param chunkSize The size of each chunk.
     * @param numChunks The number of chunks.
     * @param outputSequenceSpacing The spacing between the start of each
     * output sequence.
     * @param outputSequenceOffset The offset within each output sequence where
     * the chunks will be placed.
     * @param stream The stream to operate on.
     */
    static void frameTransfer(const float* in, float* out, int inputSequenceSpacing, int inputSequenceOffset,
        int chunkSize, int numChunks, int outputSequenceSpacing, int outputSequenceOffset, cudaStream_t stream);
};

} // namespace tts

#endif
