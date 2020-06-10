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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TT2I_CUDAMEMORY_H
#define TT2I_CUDAMEMORY_H

#include "checkedCopy.h"
#include "cudaUtils.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

namespace tts
{

template <typename T>
class CudaMemory
{
public:
  CudaMemory() : m_ptr(nullptr), m_size(0)
  {
    // do nothing
  }

  CudaMemory(const size_t size) : CudaMemory()
  {
    m_size = size;
    CudaUtils::alloc(&m_ptr, m_size);
  }

  CudaMemory(const std::vector<T>& data) : CudaMemory(data.size())
  {
    CheckedCopy::hostToDevice(m_ptr, data.data(), size());
  }

  template <typename U>
  CudaMemory(U start, U end) : CudaMemory(end - start)
  {
    CheckedCopy::hostToDevice(m_ptr, start, size());
  }

  CudaMemory(CudaMemory&& other) : m_ptr(other.m_ptr), m_size(other.m_size)
  {
    other.m_ptr = nullptr;
    other.m_size = 0;
  }

  CudaMemory& operator=(CudaMemory&& other)
  {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_size, other.m_size);

    other.clear();

    return *this;
  }

  // deleted constructors
  CudaMemory(const CudaMemory& other) = delete;
  CudaMemory& operator=(const CudaMemory& other) = delete;

  ~CudaMemory()
  {
    clear();
  }

  operator bool() const
  {
    return m_ptr != nullptr;
  }

  T* data()
  {
    return m_ptr;
  }

  const T* data() const
  {
    return m_ptr;
  }

  size_t size() const
  {
    return m_size;
  }

  std::vector<T> toHost() const
  {
    std::vector<T> host(size());
    CheckedCopy::deviceToHost(host.data(), data(), size());
    return host;
  }

  void zero()
  {
    CudaUtils::zero(data(), size());
  }

  void zeroAsync(cudaStream_t stream)
  {
    CudaUtils::zeroAsync(data(), size(), stream);
  }

  void clear()
  {
    if (m_ptr) {
      CudaUtils::free(&m_ptr);
    }
    m_size = 0;
  }

private:
  T* m_ptr;
  size_t m_size;
};

} // namespace tts

#endif
