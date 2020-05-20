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

#include "taco2Utils.h"

#include "cuda_fp16.h"

#include <cassert>
#include <sstream>

using namespace nvinfer1;

namespace taco2
{

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

__global__ void floatsToHalvesKernel(
    const float2* const floats, __half2* const halves, const int num)
{
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < num) {
    halves[idx] = __float22half2_rn(floats[idx]);
  }
}

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

int Taco2Utils::roundUpBlocks(const int num, const int blockSize)
{
  if (num < 0) {
    throw std::runtime_error(
        "Taco2Utils::roundUpBlocks(): Number of items must be non-negative: "
        + std::to_string(num));
  } else if (blockSize <= 0) {
    throw std::runtime_error(
        "Taco2Utils::roundUpBlocks(): Invalid block size: "
        + std::to_string(blockSize));
  } else if (num == 0) {
    // avoid underflow
    return 0;
  } else {
    return ((num - 1) / blockSize) + 1;
  }
}

std::vector<float> Taco2Utils::toFloatVector(const Weights& weights)
{
  if (weights.type != DataType::kFLOAT) {
    throw std::runtime_error(
        "Invalid data type for LSTMCell weights: "
        + std::to_string(static_cast<int>(weights.type)));
  }
  const float* const valuesBegin = static_cast<const float*>(weights.values);
  const float* const valuesEnd = valuesBegin + weights.count;

  return std::vector<float>(valuesBegin, valuesEnd);
}

std::string Taco2Utils::dimsToString(const Dims& dim)
{
  std::ostringstream oss;
  oss << "{";
  for (int i = 0; i < dim.nbDims; ++i) {
    oss << dim.d[i] << " ";
  }
  oss << "}";
  return oss.str();
}

size_t Taco2Utils::getDimensionsSize(const Dims& dims)
{
  size_t i = 1;
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] == -1) {
      if (d == 0) {
        // ignore batch dimension
      } else {
        throw std::runtime_error("Cannot get size of tensor with dynamic "
                                 "dimension.");
      }
    } else {
      assert(dims.d[d] > 0);
      i *= dims.d[d];
    }
  }

  return i;
}

Dims Taco2Utils::getCompactedDims(const Dims& dims, const int minLength)
{
  Dims cDims{0, {}, {}};
  if (dims.nbDims) {
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] > 1) {
        cDims.d[cDims.nbDims++] = dims.d[d];
      }
    }
    if (cDims.nbDims == 0) {
      cDims.nbDims = 1;
      cDims.d[0] = 1;
    }
  }

  if (cDims.nbDims < minLength) {
    const int offset = minLength - cDims.nbDims;
    for (int i = cDims.nbDims; i > 0;) {
      --i;
      cDims.d[i + offset] = cDims.d[i];
    }
    for (int i = 0; i < offset; ++i) {
      cDims.d[i] = 1;
    }
    cDims.nbDims = minLength;
  }

  return cDims;
}

void Taco2Utils::floatsToHalves(
    const float* floats, float* halves, const size_t num)
{
  if (num % 2 != 0) {
    throw std::runtime_error("Cannot convert odd number of floats to havles.");
  }
  const size_t halfNum = num / 2;
  const dim3 block(1024);
  const dim3 grid(roundUpBlocks(halfNum, block.x));

  floatsToHalvesKernel<<<grid, block>>>(
      reinterpret_cast<const float2*>(floats),
      reinterpret_cast<__half2*>(halves),
      halfNum);
}

} // namespace taco2
