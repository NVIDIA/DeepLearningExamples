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

#ifndef TT2I_HOSTMEMORY_H
#define TT2I_HOSTMEMORY_H

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
class HostMemory
{
public:
  HostMemory() : m_ptr(nullptr), m_size(0)
  {
    // do nothing
  }

  HostMemory(const size_t size) : HostMemory()
  {
    m_size = size;
    CudaUtils::allocHost(&m_ptr, m_size);
  }

  HostMemory(const std::vector<T>& data) : HostMemory(data.size())
  {
    memcpy(m_ptr, data.data(), data.size() * sizeof(*m_ptr));
  }

  HostMemory(HostMemory&& other) : m_ptr(other.m_ptr), m_size(other.m_size)
  {
    other.m_ptr = nullptr;
    other.m_size = 0;
  }

  HostMemory& operator=(HostMemory&& other)
  {
    std::swap(m_ptr, other.m_ptr);
    std::swap(m_size, other.m_size);

    other.clear();

    return *this;
  }

  // deleted constructors
  HostMemory(const HostMemory& other) = delete;
  HostMemory& operator=(const HostMemory& other) = delete;

  ~HostMemory()
  {
    clear();
  }

  const T* operator+(const size_t offset) const
  {
    return m_ptr + offset;
  }

  T* operator+(const size_t offset)
  {
    return m_ptr + offset;
  }

  const T* operator+(const int offset) const
  {
    assert(offset >= 0);
    return m_ptr + offset;
  }

  T* operator+(const int offset)
  {
    assert(offset >= 0);
    return m_ptr + offset;
  }

  operator T*()
  {
    return m_ptr;
  }

  operator const T*() const
  {
    return m_ptr;
  }

  operator bool() const
  {
    return m_ptr != nullptr;
  }

  T& operator[](const size_t index)
  {
    return m_ptr[index];
  }

  const T& operator[](const size_t index) const
  {
    return m_ptr[index];
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

  void zero()
  {
    memset(m_ptr, 0, sizeof(*m_ptr) * m_size);
  }

  void clear()
  {
    if (m_ptr) {
      cudaFreeHost(m_ptr);
      m_ptr = nullptr;
    }
    m_size = 0;
  }

private:
  T* m_ptr;
  size_t m_size;
};

} // namespace tts

#endif
