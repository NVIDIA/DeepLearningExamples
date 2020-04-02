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

#ifndef TT2I_CUDAUTILS_H
#define TT2I_CUDAUTILS_H

#include "cuda_runtime.h"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts
{

class CudaUtils
{
public:
    /**
     * @brief Synchronize on the given stream, and throw an exception if an error
     * occurs.
     *
     * @param stream The stream to synchronize on.
     */
    static void sync(cudaStream_t stream);

    /**
     * @brief Print information about the avialable devices and which one is in
     * use to stdout.
     */
    static void printDeviceInformation();

    /**
     * @brief Get the number of SMs on the current device.
     *
     * @return The number of SMs.
     */
    static int getNumSM();

    /**
     * @brief Free the given device pointer.
     *
     * @tparam T The type of pointer.
     * @param ptr The pointer.
     */
    template <typename T>
    static void free(T** const ptr)
    {
        check(cudaFree(*ptr), "CudaUtils::free(ptr)");
        *ptr = nullptr;
    }

    /**
     * @brief Allocte data on the GPU.
     *
     * @tparam T The data type.
     * @param ptr The pointer to set pointing at the allocated memory.
     * @param count The number of elements to allocate.
     */
    template <typename T>
    static void alloc(T** const ptr, const size_t count)
    {
        check(cudaMalloc((void**) ptr, sizeof(T) * count), "CudaUtils::alloc(ptr, count" + std::to_string(count) + ")");
        assert(count == 0 || *ptr);
    }

    /**
     * @brief Allocate pinned memory on the host.
     *
     * @tparam T The data type.
     * @param ptr The pointer to set pointing at the allocated memory.
     * @param count The number of elements to allocate.
     */
    template <typename T>
    static void allocHost(T** const ptr, const size_t count)
    {
      check(
          cudaMallocHost((void**)ptr, sizeof(T) * count),
          "CudaUtils::allocHost(ptr, count)");
      assert(count == 0 || *ptr);
    }

    /**
     * @brief Zero out region of device memory.
     *
     * @tparam T The data type.
     * @param ptr The pointer to the memory.
     * @param count The number of elements to zero.
     */
    template <typename T>
    static void zero(T* const ptr, const size_t count)
    {
        check(cudaMemset(ptr, 0, sizeof(T) * count), "CudaUtils::zero(ptr, count)");
    }

    /**
     * @brief Zero out region of device memory asynchronously.
     *
     * @tparam T The data type.
     * @param ptr The pointer to the memory.
     * @param count The number of elements to zero.
     * @param stream The stream to operate on.
     */
    template <typename T>
    static void zeroAsync(T* const ptr, const size_t count, cudaStream_t stream)
    {
        check(cudaMemsetAsync(ptr, 0, sizeof(T) * count, stream), "CudaUtils::zeroAsync(ptr, count, stream)");
    }

    /**
     * @brief Allocate memory zeroed memory.
     *
     * @tparam T The dat atype.
     * @param ptr The pointer to set point at the allocated memory.
     * @param count The number of elements to allocate and zero.
     */
    template <typename T>
    static void allocZeroed(T** const ptr, const size_t count)
    {
        alloc(ptr, count);
        zero(*ptr, count);
    }

private:
    /**
     * @brief Convert cuda errors into exceptions. Will throw an exception
     * unless `err == cudaSuccess`.
     *
     * @param err The error.
     * @param msg The message to attach to the exception.
     */
    static void check(const cudaError_t err, const std::string& msg = "")
    {
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Encountered error: " + std::to_string(static_cast<int>(err)) + ": " + msg);
        }
    }
};

} // namespace tts

#endif
