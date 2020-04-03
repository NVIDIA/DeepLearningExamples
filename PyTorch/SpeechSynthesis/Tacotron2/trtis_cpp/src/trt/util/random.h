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

#ifndef TT2I_RANDOM_H
#define TT2I_RANDOM_H

#include "cuda_runtime.h"
#include "curand_kernel.h"

namespace tts
{

class Random
{
public:
    /**
     * @brief Create a new Random object.
     *
     * @param numStates The number of internal states to use.
     * @param seed The seed to set.
     */
    Random(int numStates, unsigned int seed = 0);

    // disable copying
    Random(const Random& rand) = delete;
    Random& operator=(const Random& rand) = delete;

    /**
     * @brief Destructor (cleanup random states and memory).
     */
    ~Random();

    /**
     * @brief Set the seed of the number generator.
     *
     * @param seed The seed to use.
     */
    void setSeed(unsigned int seed, cudaStream_t stream = 0);

    /**
     * @brief Get the random states on the device.
     *
     * @return The random states.
     */
    curandState_t* getRandomStates();

    /**
     * @brief Get the number of random states.
     *
     * @return The number.
     */
    int size() const
    {
        return mNumStates;
    }

private:
    int mNumStates;
    curandState_t* mRandStateDevice;
};

} // namespace tts

#endif
