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

#include "cudaUtils.h"

#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace tts
{

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void CudaUtils::sync(cudaStream_t stream)
{
    check(cudaStreamSynchronize(stream), "CudaUtils::sync(stream)");
}

void CudaUtils::printDeviceInformation()
{
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to get active device: " + std::to_string(err));
    }

    std::cout << "Available devices:" << std::endl;
    int nDevices;
    err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to get device count: " + std::to_string(err));
    }
    for (int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                "Failed to get device properties for device " + std::to_string(i) + " : " + std::to_string(err));
        }
        std::cout << "Device: " << i << " : '" << prop.name << "'";
        std::cout << ", ";
        std::cout << prop.multiProcessorCount << " SMs";

        if (prop.cooperativeLaunch)
        {
            std::cout << ", ";
            std::cout << "support Co-op Launch";
        }

        if (i == device)
        {
            std::cout << " <- [ ACTIVE ]";
        }
        std::cout << std::endl;
    }
}

int CudaUtils::getNumSM()
{
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to get active device: " + std::to_string(err));
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to device properties: " + std::to_string(err));
    }

    return prop.multiProcessorCount;
}

} // namespace tts
