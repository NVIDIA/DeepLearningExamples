/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fastertransformer/common.h"
#include "fused_multihead_attention_v2.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#pragma once

namespace fastertransformer
{

class MHARunner
{
public:
    MHARunner(const int numHeads, const int headSize, const int wordSize)
        : mS(0)
        , mB(0)
        , mOmatSize(0)
        , mNumMats(0)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mWordSize(wordSize)
        , mLdQKV(0)
        , mStrideQKV(0)
        , mLdOut(0)
        , mStrideOut(0)
        , mRsqrtHeadSize(1.f / sqrtf(headSize))
    {
    }

    virtual ~MHARunner() = default;

    virtual void setup(const int S, const int B)
    {
        assert(S);
        assert(B);
        mB = B;
        mS = S;

        mLdQKV = 3 * B * mNumHeads * mHeadSize;
        mStrideQKV = 3 * mHeadSize;

        mLdOut = B * mNumHeads * mHeadSize;
        mStrideOut = mHeadSize;
        mOmatSize = S * S;
        mNumMats = B * mNumHeads;
    }

    virtual void run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream) = 0; 
    virtual void run(const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream) = 0; 

    virtual size_t getWorkspaceSize() const = 0;

    virtual bool isValid(int s) const = 0;

protected:

    int mS;
    int mB;
    int mOmatSize;
    int mNumMats;
    int mNumHeads;
    int mHeadSize;
    int mWordSize;
    int mLdQKV;
    int mStrideQKV;
    int mLdOut;
    int mStrideOut;

    float mRsqrtHeadSize;
};

class FusedMHARunnerFP16v2 : public MHARunner
{
public:
    FusedMHARunnerFP16v2(const int numHeads, const int headSize, const int sm);
    ~FusedMHARunnerFP16v2() = default; // for pimpl

    virtual void setup(const int S, const int B) override;

    void run(const void* input, const void* mask, void* workspace, void* output, cudaStream_t stream);
    void run(const void* input, const void* mask, const void* seqlen, void* workspace, void* output, cudaStream_t stream) override; 

    size_t getWorkspaceSize() const override;

    bool isValid(int s) const override;

private:
    int mSm;
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

} // namespace fastertransformer
