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

#ifndef TT2I_CHECKEDCOPY_H
#define TT2I_CHECKEDCOPY_H

#include "cuda_runtime.h"

#include <stdexcept>
#include <string>

namespace tts
{

class CheckedCopy
{
public:
    /**
     * @brief Copy data from the host to the device.
     *
     * @tparam T The data type.
     * @param dst The destination address on the device.
     * @param src The source address on the host.
     * @param count The number of elements to copy.
     */
    template <typename T>
    static void hostToDevice(T* const dst, const T* const src, const size_t count)
    {
        copy(dst, src, count, cudaMemcpyHostToDevice);
    }

    /**
     * @brief Copy data from the host to the device asynchronously.
     *
     * @tparam T The data type.
     * @param dst The destination address on the device.
     * @param src The source address on the host.
     * @param count The number of elements to copy.
     * @param stream The stream to operate on.
     */
    template <typename T>
    static void hostToDeviceAsync(T* const dst, const T* const src, const size_t count, cudaStream_t stream)
    {
        copyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
    }

    /**
     * @brief Copy data from the device to the host.
     *
     * @tparam T The data type.
     * @param dst The destination address on the host.
     * @param src The source address on the device.
     * @param count The number of elements to copy.
     */
    template <typename T>
    static void deviceToHost(T* const dst, const T* const src, const size_t count)
    {
        copy(dst, src, count, cudaMemcpyDeviceToHost);
    }

    /**
     * @brief Copy data from the device to the host asynchronously.
     *
     * @tparam T The data type.
     * @param dst The destination address on the host.
     * @param src The source address on the device.
     * @param count The number of elements to copy.
     * @param stream The stream to operate on.
     */
    template <typename T>
    static void deviceToHostAsync(T* const dst, const T* const src, const size_t count, cudaStream_t stream)
    {
        copyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
    }

    /**
     * @brief Copy data from the device to the device asynchronously.
     *
     * @tparam T The data type.
     * @param dst The destination address on the device.
     * @param src The source address on the device.
     * @param count The number of elements to copy.
     * @param stream The stream to operate on.
     */
    template <typename T>
    static void deviceToDeviceAsync(T* const dst, const T* const src, const size_t count, cudaStream_t stream)
    {
        copyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }

    /**
     * @brief Copy data from the device to the device.
     *
     * @tparam T The data type.
     * @param dst The destination address on the device.
     * @param src The source address on the device.
     * @param count The number of elements to copy.
     */
    template <typename T>
    static void deviceToDevice(T* const dst, const T* const src, const size_t count)
    {
        copy(dst, src, count, cudaMemcpyDeviceToDevice);
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

    /**
     * @brief Perform checked asynchronous memcpy.
     *
     * @tparam T The data type.
     * @param dst The destination address.
     * @param src The source address.
     * @param count The number of elements to copy.
     * @param kind The direction of the copy.
     * @param stream THe stream to operate on.
     */
    template <typename T>
    static void copyAsync(
        T* const dst, const T* const src, const size_t count, const enum cudaMemcpyKind kind, cudaStream_t stream)
    {
        check(cudaMemcpyAsync(dst, src, sizeof(T) * count, kind, stream),
            "CheckedCopy::copyAsync(dst, src, count, kind, stream)");
    }

    /**
     * @brief Perform a synchronous memcpy.
     *
     * @tparam T The data type.
     * @param dst The destination address.
     * @param src The source address.
     * @param count The number of elements to copy.
     * @param kind The direction of the copy.
     */
    template <typename T>
    static void copy(T* const dst, const T* const src, const size_t count, const enum cudaMemcpyKind kind)
    {
        check(cudaMemcpy(dst, src, sizeof(T) * count, kind), "CheckedCopy::copy(dst, src, count, kind)");
    }
};

} // namespace tts

#endif
