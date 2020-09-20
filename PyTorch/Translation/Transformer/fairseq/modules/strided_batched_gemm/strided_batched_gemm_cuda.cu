// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "THC/THC.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/wmma_gemm_traits.h"

// symbol to be automatically resolved by PyTorch libs
extern THCState *state;

cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

void CublasGemm(THCState *state, char transa, char transb, long m, long n, long k,
                    float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                    float beta, half *c, long ldc, long strideC, long batchCount) {
    cublasOperation_t opa = convertTransToCublasOperation(transa);
    cublasOperation_t opb = convertTransToCublasOperation(transb);
 
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    //cublasSetStream(handle, THCState_getCurrentStream(state));
    float fAlpha = alpha;
    float fBeta = beta;
    THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    THCublasCheck(cublasGemmStridedBatchedEx(handle,
                                     opa, opb, (int)m, (int)n, (int)k,
                                     (void*)&fAlpha, a, CUDA_R_16F, (int)lda, strideA,
                                     b, CUDA_R_16F, (int)ldb, strideB,
                                     (void*)&fBeta, c, CUDA_R_16F, (int)ldc, strideC,
                                     (int)batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

template<cutlass::MatrixLayout::Kind A_LAYOUT, cutlass::MatrixLayout::Kind B_LAYOUT, int SRC_A, int SRC_B, int DST_C>
void CutlassGemm_FP32Accum(cudaStream_t stream, long m, long n, long k,
                          float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                          float beta, half *c, long ldc, long strideC, long batchCount) {
  //printf("CUTLASS-> %c%c M: %ld N: %ld K: %ld %d%d%d LDA: %ld LDB: %ld LDC: %ld strideA: %ld strideB: %ld strideC: %ld Alpha: %f Beta: %f\n", ((int)A_LAYOUT == 0 ? 'T' : 'N'), ((int)B_LAYOUT ==0 ? 'T' : 'N'), m, n, k, SRC_A,SRC_B,DST_C, lda, ldb, ldc, strideA, strideB, strideC, alpha, beta);
  typedef cutlass::gemm::WmmaGemmTraits<
    A_LAYOUT,
    B_LAYOUT,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    half,
    cutlass::gemm::LinearScaling<float>,
    float,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<typename cutlass::Shape<32, 16, 16> >::Shape,
    typename cutlass::Shape<16, 16, 16>,
    SRC_A,   //kScalarsPerLdgA_
    SRC_B,   //kScalarsPerLdgB_
    SRC_A,   //KScalarsPerLdsA_
    SRC_B,   //KScalarsPerLdsB_
    DST_C,   //kScalarsPerLdgCAndStgD_
    DST_C/2, //kScalarsPerStsD_
    DST_C/2  //kScalarsPerLdsD_
  >
    WmmaGemmTraits;

  typedef cutlass::gemm::Gemm<WmmaGemmTraits> Gemm;
  typename Gemm::Params params;


  int result = params.initialize(
    m,                  // M dimension for each batch
    n,                  // N dimension for each batch
    k,                  // K dimension for each batch
    alpha,              // scalar alpha
    a,
    lda,
    strideA,     // distance in memory between the first element of neighboring batch
    b,
    ldb,
    strideB,     // distance in memory between the first element of neighboring batch
    beta,               // scalar beta
    c,                  // source matrix C
    ldc,
    strideC,     // distance in memory between the first element of neighboring batch
    c,                  // destination matrix C (may be different memory than source C matrix)
    ldc,
    strideC,    // distance in memory between the first element of neighboring batch
    batchCount
  );

  AT_ASSERTM(result == 0, "Failed to initialize CUTLASS Gemm::Params object.");
  
  // Launch the CUTLASS GEMM kernel.
  THCudaCheck(Gemm::launch(params));

}

void gemm_switch_fp32accum(THCState *state, char transa, char transb, long m, long n, long k,
                           float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                           float beta, half *c, long ldc, long strideC, long batchCount) {
  //cudaStream_t stream = THCState_getCurrentStream(state);
  //printf("GEMM   -> %c%c M: %i N: %i K: %i Alpha: %f Beta: %f\n", (transa == 't' ? 'T' : 'N'), (transb =='t' ? 'T' : 'N'), m, n, k, alpha, beta);
  auto stream = c10::cuda::getCurrentCUDAStream();
  if        ( (transa == 't') && (transb == 'n') ) { 
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { CublasGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,8,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,8,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,4,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,8,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kRowMajor,cutlass::MatrixLayout::kColumnMajor,2,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else                                                   { CublasGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
  } else if ( (transa == 'n') && (transb == 'n') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { CublasGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,8,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,8,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,4,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,8,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kColumnMajor,2,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else                                                   { CublasGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
  } else if ( (transa == 'n') && (transb == 't') ) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) { CublasGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,8,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,8,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,4,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,8,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,8,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,8,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,4,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,4,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,4,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,2,8>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,2,4>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) { CutlassGemm_FP32Accum<cutlass::MatrixLayout::kColumnMajor,cutlass::MatrixLayout::kRowMajor,2,2,2>(stream, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
    else                                                   { CublasGemm(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount); }
  } else {
    AT_ASSERTM(false, "TransA and TransB are invalid");
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb, int64_t *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  if(n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if(transa_)
  {
    if(m <= 1)
      *lda = std::max<int64_t>(k, 1);
  }
  else
  {
    if(k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if(transb_)
  {
    if(k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  }
  else
  {
    if(n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }

}

void HgemmStridedBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             float alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                             float beta, half *c, long ldc, long strideC, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  //gemm_switch(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
  gemm_switch_fp32accum(state, transa, transb, m, n, k, alpha, a, lda, strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
}

at::Tensor strided_batched_gemm_cuda(
    float beta,
    at::Tensor in_result,
    float alpha,
    at::Tensor batch1,
    at::Tensor batch2) {

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  int64_t lda, ldb, ldc;
  at::Tensor result, input1, input2;
  if (in_result.stride(1) == 1)
  {
    transpose_result = false;
    result = in_result;
    ldc = result.stride(2);
  }
  else if (in_result.stride(2) == 1)
  {
    transpose_result = true;

    at::Tensor swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result = in_result;
    ldc = result.stride(1);
  } else { 
    AT_ASSERTM(false, "result should be contiguous");
  }

  if (batch1.stride(transpose_result ? 2 : 1) == 1 &&
      batch1.stride(transpose_result ? 1 : 2) != 0) {
    transpose_batch1 = 'n';
    input1 = batch1;
    lda = input1.stride(transpose_result ? 1 : 2);
  } else if (batch1.stride(transpose_result ? 1 : 2) == 1 &&
             batch1.stride(transpose_result ? 2 : 1) != 0) {
    transpose_batch1 = 't';
    input1 = batch1;
    lda = input1.stride(transpose_result ? 2 : 1);
  } else {
    AT_ASSERTM(false, "input1 should be contiguous");
  }

  if (batch2.stride(transpose_result ? 2 : 1) == 1 &&
      batch2.stride(transpose_result ? 1 : 2) != 0) {
    transpose_batch2 = 'n';
    input2 = batch2;
    ldb = input2.stride(transpose_result ? 1 : 2);
  } else if (batch2.stride(transpose_result ? 1 : 2) == 1 &&
             batch2.stride(transpose_result ? 2 : 1) != 0) {
    transpose_batch2 = 't';
    input2 = batch2;
    ldb = input2.stride(transpose_result ? 2 : 1);
  } else {
    AT_ASSERTM(false, "input2 should be contiguous");
  }
  int64_t num_batches = result.size(0);

  HgemmStridedBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result.size(transpose_result ? 2 : 1),
      result.size(transpose_result ? 1 : 2),
      input1.size(transpose_result ? 1 : 2),
      alpha,
      static_cast<const half*>(input1.data_ptr()), lda, input1.stride(0),
      static_cast<const half*>(input2.data_ptr()), ldb, input2.stride(0),
      beta,
      static_cast<half*>(result.data_ptr()), ldc, result.stride(0),
      num_batches);

  return in_result;
}


