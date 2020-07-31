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

#include "RepeatPlugin.h"
#include "cuda_fp16.h"
#include <thread>
#include <cub/cub.cuh>

#define ck(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
        cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << endl;
        return false;
    }
    return true;
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

__global__ void ComputeOffset(float *pRepeatCnt, int *pOffset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    const int maxInputLength = 1024;
    cub::BlockScan<int, maxInputLength>().ExclusiveSum(static_cast<int>(pRepeatCnt[x]), pOffset[x]);
}

template<typename T>
__global__ void RepeatTensor(T *pOut, T *pIn, float *pRepeatCnt, int *pOffset, int maxOutputLength) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int h = threadIdx.x;
    int dTime = gridDim.y;
    int dHid = blockDim.x;

    int offset_time = pOffset[b * dTime + t];
    int duration = static_cast<int>(pRepeatCnt[b * dTime + t]);

    T in = pIn[(b * dTime + t) * dHid + h];
    for (int i=offset_time; i < min(offset_time + duration, maxOutputLength); i++) {
        int offset_batch = b * maxOutputLength;
        pOut[(offset_batch + i) * dHid + h] = in;
    }
}

int RepeatPlugin::enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) {
    int dTime = m.inputDim.d[0];
    int dHid = m.inputDim.d[1];
    int maxOutputLength = m.maxOutputLength;

#ifndef NDEBUG
    cudaDeviceProp dev = getCudaDeviceProp();
    assert (dHid <= dev.maxThreadsPerBlock);
#endif 

    float * pRepeatCnt = (float *)inputs[1];

    // get output time dim offset
    int * pOffset;
    ck(cudaMalloc(&pOffset, nBatch * dTime * sizeof(int)));
    ComputeOffset<<<nBatch, dTime>>>(pRepeatCnt, pOffset);

    if (m.dataType == DataType::kFLOAT || m.dataType == DataType::kINT32) {
        // std::cout << "[RepeatPlugin] Running kernel in fp32" << std::endl;

        float * pIn = (float *)inputs[0];
        float * pOut = (float *)outputs[0];

        dim3 dimGrid(nBatch, dTime);
        dim3 dimBlock(dHid);
        RepeatTensor<<<dimGrid, dimBlock>>>(pOut, pIn, pRepeatCnt, pOffset, maxOutputLength);

    } else if (m.dataType == DataType::kHALF) {
        // std::cout << "[RepeatPlugin] Running kernel in fp16" << std::endl;

        __half * pIn = (__half *)inputs[0];
        __half * pOut = (__half *)outputs[0];

        dim3 dimGrid(nBatch, dTime);
        dim3 dimBlock(dHid);
        RepeatTensor<<<dimGrid, dimBlock>>>(pOut, pIn, pRepeatCnt, pOffset, maxOutputLength);

    }

    return 0;
}

REGISTER_TENSORRT_PLUGIN(RepeatPluginCreator);