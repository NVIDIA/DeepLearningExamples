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
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdexcept>
#include <map>
#include "stdio.h"

#define MAX_CONFIG_NUM 20
#define GEMM_NUM 6
#define COL32_ 32
#define ACTIVATION_AMAX_NUM 80
#define INT8O_GEMM_NUM 8
#define GEMM_CONFIG "gemm_config.in"
#define IGEMM_CONFIG "igemm_config.in"

#include "fastertransformer/gemm_test/encoder_gemm_func.h"
#include "fastertransformer/gemm_test/encoder_igemm_func.h"

namespace fastertransformer
{

enum class OperationType
{
  FP32,
  FP16
};
enum class AllocatorType
{
  CUDA,
  TF,
  TH
};

#define PRINT_FUNC_NAME_()                                          \
  do                                                                \
  {                                                                 \
    std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)

static const char *_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorString(error);
}

static inline __device__ int8_t float_to_int8_rn(float x)
{
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
               : "=r"(dst)
               : "f"(x));
  return reinterpret_cast<const int8_t &>(dst);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

//for int8 cublasLtMM with algo
//ATransform should be m*n, CUBLASLT_ORDER_COL32
//kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or CUBLASLT_ORDER_COL32_2R_4R4
//res is m*n, CUBLASLT_ORDER_COL32
template <typename T>
void cublasLtMM_withAlgo(int *res, int batchCount, int m, int n, int k,
                         int64_t stridea, int64_t strideb, int64_t stridec,
                         const int8_t *ATransform, const T *kernel, cublasLtHandle_t cublasLt_handle,
                         cudaStream_t stream, std::map<std::string, cublasLtMatmulAlgo_info> &cublasLtAlgoMap,
                         bool use_ORDER_COL32_2R_4R4, bool use_default_algo = false)
{
  cublasOperation_t opTranspose = CUBLAS_OP_T;
#ifdef CUDA11_MODE
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#ifdef CUDA11_MODE
  if (use_ORDER_COL32_2R_4R4)
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

    
  
  int ldaTransform = 32 * m;
  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4)
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
  int ldcTransform = 32 * m;

  // create matmulDesc
#ifdef CUDA11_MODE
  cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  if (batchCount > 1)
  {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
  }

  int alphaI = 1;
  int betaI = 0;

  //get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d", batchCount, m, n, k);
  std::string markStr(mark);
  int findAlgo = 0;
  if ((!use_default_algo) && cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() && cublasLtAlgoMap[markStr].workspaceSize == 0)
  {
    //printf("find algo %s\n", markStr.c_str());
    findAlgo = 1;

    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, cublasLtAlgoMap[markStr].algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(cublasLtAlgoMap[markStr].customOption), sizeof(cublasLtAlgoMap[markStr].customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasLtAlgoMap[markStr].tile), sizeof(cublasLtAlgoMap[markStr].tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(cublasLtAlgoMap[markStr].splitK_val), sizeof(cublasLtAlgoMap[markStr].splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(cublasLtAlgoMap[markStr].swizzle), sizeof(cublasLtAlgoMap[markStr].swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(cublasLtAlgoMap[markStr].reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasLtAlgoMap[markStr].stages), sizeof(cublasLtAlgoMap[markStr].stages));
#endif
  }
  else
  {
    findAlgo = 1; 
    int algoId;
    if (use_ORDER_COL32_2R_4R4)
    {
      algoId = 7;
    }
    else
    {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    int stages;
    if (use_ORDER_COL32_2R_4R4) 
      stages = 15;
    else
      stages = 13;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif       
  }

  cublasLtMatmul(cublasLt_handle,
                 matmulDesc,
                 &alphaI,
                 ATransform,
                 AtransformDesc,
                 kernel,
                 BtransformDesc,
                 &betaI,
                 res,
                 CtransformDesc,
                 res,
                 CtransformDesc,
                 (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
}

//for int8 IO cublasLtMM with algo
//ATransform should be m*k CUBLASLT_ORDER_COL32
//kernel should be n*k CUBLASLT_ORDER_COL4_4R2_8C
//res is m*n CUBLASLT_ORDER_COL32
template <typename T>
void cublasLtMM_withAlgo_int8IO(int8_t *res, int batchCount, int m, int n, int k,
                                int64_t stridea, int64_t strideb, int64_t stridec,
                                const float alpha, const int8_t *ATransform, const T *kernel,
                                cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                                std::map<std::string, cublasLtMatmulAlgo_info> &cublasLtAlgoMap,
                                bool use_ORDER_COL32_2R_4R4, bool use_default_algo=false)
{
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  //int8 gemm does not support CUBLAS_POINTER_MODE_DEVICE
  //cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO; 
  cudaDataType_t scaleType = CUDA_R_32F;
#ifdef CUDA11_MODE
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  
  cublasLtOrder_t order_matrixB;
#ifdef CUDA11_MODE
  if (use_ORDER_COL32_2R_4R4)
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

      
  int ldaTransform = 32 * m;
  
  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4)
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;


  int ldcTransform = 32 * m;

  // create matmulDesc
#ifdef CUDA11_MODE
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scaleType, sizeof(scaleType));
  //cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(cublasLtPointerMode_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
  if (batchCount > 1)
  {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
  }
  //get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  sprintf(mark, "%d_%d_%d_%d", batchCount, m, n, k);
  std::string markStr(mark);
  int findAlgo = 0;
  if ((!use_default_algo) && cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() && cublasLtAlgoMap[markStr].workspaceSize == 0)
  {
    findAlgo = 1;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, cublasLtAlgoMap[markStr].algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(cublasLtAlgoMap[markStr].customOption), sizeof(cublasLtAlgoMap[markStr].customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasLtAlgoMap[markStr].tile), sizeof(cublasLtAlgoMap[markStr].tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(cublasLtAlgoMap[markStr].splitK_val), sizeof(cublasLtAlgoMap[markStr].splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(cublasLtAlgoMap[markStr].swizzle), sizeof(cublasLtAlgoMap[markStr].swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(cublasLtAlgoMap[markStr].reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasLtAlgoMap[markStr].stages), sizeof(cublasLtAlgoMap[markStr].stages));
#endif
  }
  else
  {
    findAlgo = 1; 
    int algoId;
    if (use_ORDER_COL32_2R_4R4)
    {
      algoId = 7;
    }
    else
    {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme), sizeof(int));
#ifdef CUDA11_MODE
    int stages;
    if (use_ORDER_COL32_2R_4R4) 
      stages = 15;
    else
      stages = 13;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
#endif       
  }
  
  float beta = 0.0f;
  cublasLtMatmul(cublasLt_handle,
                 matmulDesc,
                 &alpha,
                 ATransform,
                 AtransformDesc,
                 kernel,
                 BtransformDesc,
                 &beta,
                 res,
                 CtransformDesc,
                 res,
                 CtransformDesc,
                 (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
  if (result)
  {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file +
                             ":" + std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void print_to_file(T *result, const int size, char *file)
{
  FILE *fd = fopen(file, "w");
  T *tmp = (T *)malloc(sizeof(T) * size);
  check_cuda_error(cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i)
  {
    float val;
    if (sizeof(T) == 2)
      val = (T)__half2float(tmp[i]);
    else
      val = (T)tmp[i];
    fprintf(fd, "%f\n", val);
  }
  free(tmp);
  fclose(fd);
}

template <typename T>
void print_to_screen(T *result, const int size)
{
  T *tmp = (T *)malloc(sizeof(T) * size);
  check_cuda_error(cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i)
    printf("%d, %f\n", i, (float)tmp[i]);
  free(tmp);
}

template <typename T>
void check_max_val(const T *result, const int size)
{
  T *tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float max_val = -100000;
  for (int i = 0; i < size; i++)
  {
    float val = (float)(tmp[i]);
    if (val > max_val)
      max_val = val;
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    cudaDeviceProp props;
    check_cuda_error(cudaGetDeviceProperties(&props, device));
    return props.major * 10 + props.minor;
}

template <typename T>
void check_abs_mean_val(const T *result, const int size)
{
  T *tmp = new T[size];
  cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    sum += abs((float)tmp[i]);
  }
  delete tmp;
  printf("[INFO][CUDA] addr %p abs mean val: %f \n", result, sum / size);
}

inline int div_up(int a, int n)
{
  return (a + n - 1) / n;
}

} //namespace fastertransformer
