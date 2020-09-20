// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the NVIDIA CORPORATION nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "AddPosEncPlugin.h"
#include <thread>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "cuda_fp16.h"
#include <assert.h>


template<typename T>
__global__ void AddPosEnc(T *pOut, T *pIn, int dTime) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int pos = (blockIdx.x % dTime) + 1;
    int i = threadIdx.x;
    int dModel = blockDim.x;

    float period = pow(10000, 2.0 * (i / 2) / dModel);
    float angle = pos / period;
    float posEnc = (i % 2 == 0) ? sinf(angle) : cosf(angle);

    float input = static_cast<float>(pIn[x]);
    pOut[x] = static_cast<T>(input + posEnc);
}

cudaDeviceProp getCudaDeviceProp() {
    cudaError_t error;
    cudaDeviceProp dev;
    int device;
    cudaGetDevice(&device);
    error = cudaGetDeviceProperties(&dev, device);

    if(error != cudaSuccess)
    {
       printf("Error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    return dev;
}

int AddPosEncPlugin::enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) {
    int dTime = m.inputDim.d[0];
    int dHid = m.inputDim.d[1];

#ifndef NDEBUG
    cudaDeviceProp dev = getCudaDeviceProp();
    assert (dHid <= dev.maxThreadsPerBlock);
#endif 

    if (m.dataType == DataType::kFLOAT) {
        // std::cout << "[AddPosEncPlugin] Running kernel in fp32." << std::endl;
        AddPosEnc<<<nBatch * dTime, dHid>>>((float *)outputs[0], (float *)inputs[0], dTime);
    } else if (m.dataType == DataType::kHALF) {
        // std::cout << "[AddPosEncPlugin] Running kernel in fp16." << std::endl;
        AddPosEnc<<<nBatch * dTime, dHid>>>((__half *)outputs[0], (__half *)inputs[0], dTime);
    }

    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddPosEncPluginCreator);
