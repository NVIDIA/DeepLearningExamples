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

#ifndef TT2I_BINDING_H
#define TT2I_BINDING_H

#include "cudaMemory.h"

#include "NvInfer.h"

#include <unordered_map>
#include <vector>

namespace tts
{

class Binding
{
public:
    Binding();

    /**
     * @brief Set a given binding.
     *
     * @tparam T The type of binding.
     * @param engine The engine to bind to.
     * @param name The name of the binding.
     * @param ptr The pointer to bind.
     */
    template <typename T>
    void setBinding(const nvinfer1::ICudaEngine& engine, const char* const name, T* const ptr)
    {
        setVoidBinding(engine, name, (void*) ptr);
    }

    /**
     * @brief Set a given binding.
     *
     * @tparam T The type of binding.
     * @param engine The engine to bind to.
     * @param name The name of the binding.
     * @param buffer The buffer to bind.
     */
    template <typename T>
    void setBinding(
        const nvinfer1::ICudaEngine& engine,
        const char* const name,
        CudaMemory<T>& buffer)
    {
      setVoidBinding(engine, name, buffer.data());
    }

    /**
     * @brief Get the bindings to pass into TRT. This is non-const because TRT
     * requires `void **` rather than `void * const *`.
     *
     * @return The bindings.
     */
    void** getBindings();

private:
    std::vector<void*> mBindings;

    /**
     * @brief Set a given binding via `void*` pointer.
     *
     * @param engine The engine to bind to.
     * @param name The name of the binding.
     * @param ptr The pointer to bind.
     */
    void setVoidBinding(const nvinfer1::ICudaEngine& engine, const char* const name, void* ptr);
};

} // namespace tts

#endif
