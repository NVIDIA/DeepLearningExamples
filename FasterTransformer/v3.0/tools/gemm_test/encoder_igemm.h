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

#include <stdio.h>
#include <algorithm>
#include <time.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdlib.h>
#include <unistd.h>

int roundoff(int v, int d)
{
    return ((v+d-1)/d) * d;
}

static const char *showStatus(cublasStatus_t error)
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

/* Structure to store information about different run trials */
typedef struct {
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time;
    size_t workspaceSize;  // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

/* CAUTION : must match cublasLtMatmulTile_t */
const char * const matmulTileName[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8"   ,
    "8x32"   ,
    "16x16"  ,
    "32x8"   ,
    "8x64"   ,
    "16x32"  ,
    "32x16"  ,
    "64x8"   ,
    "32x32"  ,
    "32x64"  ,
    "64x32"  ,
    "32x128" ,
    "64x64"  ,
    "128x32" ,
    "64x128" ,
    "128x64" ,
    "64x256" ,
    "128x128",
    "256x64" ,
    "64x512" ,
    "128x256",
    "256x128",
    "512x64" ,
};
// Utility function to print customMatmulPerf_t structure
int printPerfStructure(int m, int n, int k, const customMatmulPerf_t &perf, char *outfilename, int hasPrint) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;
    
    const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d} status %d "
        "time %f workspace=%d mathMode=%d waves=%f\n",       
        algoId, tile, matmulTileName[tile],
        numSplitsK, reductionScheme,
        swizzle, customOption,
        perf.status,
        perf.time,
        (int)perf.workspaceSize,
        (int)perf.mathMode,
        perf.wavesCount);
    //chose the fastest algo that does not need workspace 
    if ((int)perf.workspaceSize == 0 && hasPrint == 0){
      FILE *fout;
      fout = fopen(outfilename, "a+");
      fprintf(fout, "1 %d %d %d %d %d %d %d %d %d %d\n", m, n, k, algoId, customOption, tile, numSplitsK, swizzle, reductionScheme, (int)perf.workspaceSize);
      fclose(fout);
      return 1;
    }
    else{
      return hasPrint;
    }
}

int printBatchPerfStructure(int batchCount, int m, int n, int k, const customMatmulPerf_t &perf, char *outfilename, int hasPrint) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;
    
    const cublasLtMatmulAlgo_t *matmulAlgo = &perf.algo;
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute( matmulAlgo,  CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);

    printf("algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d} status %d "
        "time %f workspace=%d mathMode=%d waves=%f\n",       
        algoId, tile, matmulTileName[tile],
        numSplitsK, reductionScheme,
        swizzle, customOption,
        perf.status,
        perf.time,
        (int)perf.workspaceSize,
        (int)perf.mathMode,
        perf.wavesCount);
    //chose the fastest algo that does not need workspace
    if ((int)perf.workspaceSize == 0 && hasPrint == 0){
      FILE *fout;
      fout = fopen(outfilename, "a+");
      fprintf(fout, "%d %d %d %d %d %d %d %d %d %d %d\n",batchCount, m, n, k, algoId, customOption, tile, numSplitsK, swizzle, reductionScheme, (int)perf.workspaceSize);
      fclose(fout);
      return 1;
    }
    else{
      return hasPrint;
    }
}

static inline bool
time_compare(const customMatmulPerf_t &perf_a, const customMatmulPerf_t &perf_b) {
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static cublasStatus_t 
customMatmulRun(cublasLtHandle_t ltHandle,  // to get the capabilities (required a GPU)
                 cublasLtMatmulDesc_t operationDesc,
                 const void *alpha, /* host or device pointer */
                 const void *A,
                 cublasLtMatrixLayout_t Adesc,
                 const void *B,
                 cublasLtMatrixLayout_t Bdesc,
                 const void *beta, /* host or device pointer */
                 const void *C,
                 cublasLtMatrixLayout_t Cdesc,
                 void *D,
                 cublasLtMatrixLayout_t Ddesc,
                 const cublasLtMatmulAlgo_t &algo,
                 int kernelRepeats,  
                 void *workSpace,
                 size_t workSpaceSizeInBytes,                 
                 customMatmulPerf_t &perfResults,                 
                 cudaStream_t stream,
                 cudaEvent_t &startEvent,
                 cudaEvent_t &stopEvent)
{
    cublasLtMatmulHeuristicResult_t heurResult;
    /* Looping over the Algo */
    int repeats = kernelRepeats;    
    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck( ltHandle,
                                                         operationDesc,
                                                         Adesc,
                                                         Bdesc,
                                                         Cdesc,
                                                         Ddesc,
                                                         &algo, 
                                                         &heurResult);     
    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
          cudaError_t err, err1, err2, err3;
          cublasStatus_t oneRunStatus;
          err  = cudaEventRecord(startEvent, stream);
          for (int loop = 0; loop < repeats; loop++) {
             oneRunStatus = cublasLtMatmul(ltHandle,
                                           operationDesc,
                                           alpha,
                                           A, Adesc,
                                           B, Bdesc,
                                           beta,
                                           C, Cdesc,
                                           D, Ddesc,
                                           &algo,
                                           workSpace,
                                           workSpaceSizeInBytes,
                                           stream);
          }
          if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
            algoStatus = oneRunStatus;
          }
          err1 = cudaEventRecord(stopEvent, stream);
          err2 = cudaEventSynchronize(stopEvent);
          float time;
          err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
          if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
            algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
          }                                     
          // For the moment only add successful findings
          if (algoStatus == CUBLAS_STATUS_SUCCESS) {
            perfResults.algo = algo;  
            perfResults.time = time/repeats;  
            perfResults.workspaceSize = heurResult.workspaceSize; 
            perfResults.wavesCount = heurResult.wavesCount;                                                                       
          }
        }
        else {
          printf("not enough workspace! %ld\n", heurResult.workspaceSize);
          algoStatus = CUBLAS_STATUS_NOT_SUPPORTED; //Not enough workspace
        }        
    }
    else{
      //printf("check fail!\n");
    }
    
    return algoStatus;
}

// Sample wrapper running through multiple algo and config attributes combination for INT8 gemm using cublasLt low-level API
int
LtIgemmCustomFind(cublasLtHandle_t ltHandle,
                  int m,
                  int n,
                  int k,
                  const int *alpha, /* host pointer */
                  const int8_t *A,
                  const int8_t *B,
                  const int *beta, /* host pointer */
                  int32_t *C,
                  void *workSpace,
                  size_t workSpaceSize,
                  char *outfilename,
                  cudaEvent_t startEvent,
                  cudaEvent_t stopEvent) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaStream_t stream = 0;
    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
     // Let try a fixed number of combinations
    #define ALGO_COMBINATIONS 50000
    int AlgoCombinations = ALGO_COMBINATIONS;
    int AlgoCount = 0;
    int kernelRepeats = 100; //number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    int nbAlgoIds = 0;
    #define ALGO_IDS 100
    int algoIdA[ALGO_IDS];
    cudaDataType_t computeType = CUDA_R_32I, scaleType = CUDA_R_32I, Atype = CUDA_R_8I, Btype = CUDA_R_8I, Ctype = CUDA_R_32I;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    int ldaTransform = 32 * m;
    int ldbTransform = 32 * roundoff(n,8);
    int ldcTransform = 32 * m;


    status = cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32I);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));    
    
    // Create matrix descriptors. 
    status = cublasLtMatrixLayoutCreate(&Adesc, Atype, m, k, ldaTransform);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutCreate(&Bdesc, Btype, n, k, ldbTransform);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldcTransform);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    
    // Request AlgoId available for IGEMM 
    status = cublasLtMatmulAlgoGetIds(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   
    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {   
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute( &algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int nbTiles = int(sizeWritten/sizeof(int));
        int *tileA = new int[ nbTiles == 0 ? 1:nbTiles];
        if(nbTiles == 0){
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int)*nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);        
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);
        /* Loop over the different tiles */        
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++) {
               cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
               /* Loop over the CTAs swizzling support */
               for (int k = 0; k <= swizzlingMax; k++) {
                    int splitK_trial = 0;
                    if (splitkSupport) {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in addtion to the case where splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                        /* Setup attribute of the algo to run */                                                
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                       int splitK_val = 0;
                       int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)); 
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)); 
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));  
                                                                        
                        if (l > 0) { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1 ; redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations); redScheme = redScheme << 1) {
                                if (redScheme & redMask) {
                                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme));
                                    status = customMatmulRun( ltHandle,
                                                              operationDesc,
                                                              alpha, /* host or device pointer */
                                                              A, Adesc,
                                                              B, Bdesc,
                                                              beta, /* host or device pointer */
                                                              C, Cdesc,
                                                              C, Cdesc,
                                                              algo,
                                                              kernelRepeats,  
                                                              workSpace,
                                                              workSpaceSize,                 
                                                              perfResults[AlgoCount],
                                                              stream,
                                                              startEvent, stopEvent);
                                    perfResults[AlgoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                                } // end if
                            } // end for
                        } else { // Non-splitK case
                            /* if user preference is ok with workspace */
                            if (AlgoCount < AlgoCombinations) {       
                                status = customMatmulRun( ltHandle,
                                                          operationDesc,
                                                          alpha, /* host or device pointer */
                                                          A, Adesc,
                                                          B, Bdesc,
                                                          beta, /* host or device pointer */
                                                          C, Cdesc,
                                                          C, Cdesc,
                                                          algo,
                                                          kernelRepeats,  
                                                          workSpace,
                                                          workSpaceSize,                 
                                                          perfResults[AlgoCount],
                                                          stream,
                                                          startEvent, stopEvent);
                                perfResults[AlgoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                            }
                        }
                    }  // end l
                }  // end k
            } //end customOption            
        } // end tileIdx
        delete [] tileA;
    } // end idx
    // Sort the results per run duration 
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details
    for (int i = 0, hasPrint = 0; i < AlgoCount; i++) {                
        printf( "result %03d : ", i);
        hasPrint = printPerfStructure(m, n, k, perfResults[i], outfilename, hasPrint);                          
    }

CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

int
LtBatchIgemmCustomFind(cublasLtHandle_t ltHandle,
                  int batchCount,
                  int m,
                  int n,
                  int k,
                  const int *alpha, /* host pointer */
                  const int8_t *A,
                  const int8_t *B,
                  const int *beta, /* host pointer */
                  int32_t *C,
                  void *workSpace,
                  size_t workSpaceSize,
                  char *outfilename,
                  cudaEvent_t startEvent,
                  cudaEvent_t stopEvent) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaStream_t stream = 0;
    // SplitK value that we are going to try when SplitK is supported for a given algo
    const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
     // Let try a fixed number of combinations
    #define ALGO_COMBINATIONS 50000
    int AlgoCombinations = ALGO_COMBINATIONS;
    int AlgoCount = 0;
    int kernelRepeats = 100; //number of time the CUDA kernels will be run back to back
    customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
    int nbAlgoIds = 0;
    #define ALGO_IDS 100
    int algoIdA[ALGO_IDS];
    cudaDataType_t computeType = CUDA_R_32I, scaleType = CUDA_R_32I, Atype = CUDA_R_8I, Btype = CUDA_R_8I, Ctype = CUDA_R_32I;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    int ldaTransform = 32 * m;
    int ldbTransform = 32 * roundoff(n,8);
    int ldcTransform = 32 * m;

    int64_t stridea, strideb, stridec;
    stridea = m*k;
    strideb = n*k;
    stridec = m*n;

    status = cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32I);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(cublasOperation_t));    
    
    // Create matrix descriptors. 
    status = cublasLtMatrixLayoutCreate(&Adesc, Atype, m, k, ldaTransform);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)); 
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea, sizeof(stridea));
    
    
    status = cublasLtMatrixLayoutCreate(&Bdesc, Btype, n, k, ldbTransform);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
     cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb, sizeof(strideb));
    
    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldcTransform);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,  &order_COL32, sizeof(order_COL32));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec, sizeof(stridec));
    
    // Request AlgoId available for IGEMM 
    status = cublasLtMatmulAlgoGetIds(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   
    // Loop over the Algo IDs
    for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations); idx++) {   
        cublasLtMatmulAlgo_t algo;
        size_t sizeWritten = 0;
        /* Initialize algo structure with given Algp ID */
        status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute( &algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int nbTiles = int(sizeWritten/sizeof(int));
        int *tileA = new int[ nbTiles == 0 ? 1:nbTiles];
        if(nbTiles == 0){
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
        }
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over the different combinations
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA, sizeof(int)*nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);        
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);
        /* Loop over the different tiles */        
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
            /* Loop over the different custom option if any */
            for (int customOption = 0; customOption <= customOptionMax; customOption++) {
               cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
               /* Loop over the CTAs swizzling support */
               for (int k = 0; k <= swizzlingMax; k++) {
                    int splitK_trial = 0;
                    if (splitkSupport) {
                        splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                    }
                    // Loop over the splitK value over a fixed sequence splitKSequenceA in addtion to the case where splitK is not enabled
                    for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                        /* Setup attribute of the algo to run */                                                
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                       int splitK_val = 0;
                       int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val)); 
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k)); 
                       cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));  
                                                                        
                        if (l > 0) { // Split-K case
                            splitK_val = splitKSequenceA[l - 1];
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                            /* Going over all the reduction scheme  */
                            for (redScheme = 1 ; redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations); redScheme = redScheme << 1) {
                                if (redScheme & redMask) {
                                    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(redScheme));
                                    status = customMatmulRun( ltHandle,
                                                              operationDesc,
                                                              alpha, /* host or device pointer */
                                                              A, Adesc,
                                                              B, Bdesc,
                                                              beta, /* host or device pointer */
                                                              C, Cdesc,
                                                              C, Cdesc,
                                                              algo,
                                                              kernelRepeats,  
                                                              workSpace,
                                                              workSpaceSize,                 
                                                              perfResults[AlgoCount],
                                                              stream,
                                                              startEvent, stopEvent);
                                    perfResults[AlgoCount].status = status;
                                    if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                                } // end if
                            } // end for
                        } else { // Non-splitK case
                            /* if user preference is ok with workspace */
                            if (AlgoCount < AlgoCombinations) {       
                                status = customMatmulRun( ltHandle,
                                                          operationDesc,
                                                          alpha, /* host or device pointer */
                                                          A, Adesc,
                                                          B, Bdesc,
                                                          beta, /* host or device pointer */
                                                          C, Cdesc,
                                                          C, Cdesc,
                                                          algo,
                                                          kernelRepeats,  
                                                          workSpace,
                                                          workSpaceSize,                 
                                                          perfResults[AlgoCount],
                                                          stream,
                                                          startEvent, stopEvent);
                                perfResults[AlgoCount].status = status;
                                if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                            }
                        }
                    }  // end l
                }  // end k
            } //end customOption            
        } // end tileIdx
        delete [] tileA;
    } // end idx
    // Sort the results per run duration 
    std::sort(perfResults, perfResults + AlgoCount, time_compare);
    // Print timing and perf details 
    for (int i = 0, hasPrint = 0; i < AlgoCount; i++) {                
        printf( "result %03d : ", i);
        hasPrint = printBatchPerfStructure(batchCount, m, n, k, perfResults[i], outfilename, hasPrint);            
    }

CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

// initialize matrix in column-major
void matInit(int rows, int cols, int8_t *p, int ld)
{
    srand(time(NULL)); 

    for (int c=0; c<cols; c++)
    {
        for (int r=0; r<rows; r++)
        {
            int index = r + c * ld;
            
            p[index] = rand()%255 - 127;
        }
    }
}

int batch_igemm_config(int batchCount, int m, int n, int k, char * outfilename, cudaEvent_t startEvent,cudaEvent_t stopEvent){   
    printf("batchCount %d m %d n %d k %d\n",batchCount, m ,n ,k);	
    int alpha = 1;
    int beta  = 0;

    int8_t  *h_A = NULL; // m * k, stored in column-major
    int8_t  *h_B = NULL; // k * n, stored in column-major
    int32_t *h_C = NULL; // m * n, stored in column-major

    int8_t  *d_A = NULL; // m * k, stored in column-major
    int8_t  *d_B = NULL; // k * n, stored in column-major
    int32_t *d_C = NULL; // m * n, stored in column-major
    int32_t *d_workspace = NULL;
    size_t workSpaceSize = sizeof(char*) * 64 * m * n;



    // allocate memory
    h_A = (int8_t* )malloc(sizeof(int8_t ) * m * k * batchCount);
    if (!h_A) printf("falied to allocate mem on CPU");
    h_B = (int8_t* )malloc(sizeof(int8_t ) * k * n * batchCount);   // B : k*n
    if (!h_B) printf("falied to allocate mem on CPU"); // BT: n*k, the transpose of B

    cudaMalloc((void **)&d_A, sizeof(int8_t ) * m * k * batchCount);
    cudaMalloc((void **)&d_B, sizeof(int8_t ) * k * n * batchCount);
    cudaMalloc((void **)&d_C, sizeof(int32_t) * m * n * batchCount);
    cudaMalloc((void **)&d_workspace, workSpaceSize);

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    for (int b = 0 ; b < batchCount ; b++){
        matInit(m, k, h_A+b*m*k, m);
        matInit(k, n, h_B+b*n*k, k);
    }

    cudaMemcpy(d_A, h_A, sizeof(int8_t) * m * k * batchCount,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int8_t) * k * n * batchCount,cudaMemcpyHostToDevice);


    LtBatchIgemmCustomFind(ltHandle,
                  batchCount,
                  m,
                  n,
                  k,
                  &alpha, /* host pointer */
                  d_A,
                  d_B,
                  &beta, /* host pointer */
                  d_C,
                  (void *)d_workspace,
                  workSpaceSize,
                  outfilename,
                  startEvent,
                  stopEvent);
    //free memory
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_workspace);

    cublasLtDestroy(ltHandle);
}

int igemm_config(int m, int n, int k, char * outfilename, cudaEvent_t startEvent,cudaEvent_t stopEvent){   
    printf("batchCount %d m %d n %d k %d\n", 1, m ,n ,k);	
    int alpha = 1;
    int beta  = 0;

    int8_t  *h_A = NULL; // m * k, stored in column-major
    int8_t  *h_B = NULL; // k * n, stored in column-major
    int32_t *h_C = NULL; // m * n, stored in column-major

    int8_t  *d_A = NULL; // m * k, stored in column-major
    int8_t  *d_B = NULL; // k * n, stored in column-major
    int32_t *d_C = NULL; // m * n, stored in column-major
    int32_t *d_workspace = NULL;
    size_t workSpaceSize = sizeof(char*) * 64 * m * n;

    // allocate memory
    h_A = (int8_t* )malloc(sizeof(int8_t ) * m * k);
    if (!h_A) printf("falied to allocate mem on CPU");
    h_B = (int8_t* )malloc(sizeof(int8_t ) * k * n);   // B : k*n
    if (!h_B) printf("falied to allocate mem on CPU"); // BT: n*k, the transpose of B

    cudaMalloc((void **)&d_A, sizeof(int8_t ) * m * k);
    cudaMalloc((void **)&d_B, sizeof(int8_t ) * k * n);
    cudaMalloc((void **)&d_C, sizeof(int32_t) * m * n);
    cudaMalloc((void **)&d_workspace, workSpaceSize);

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    matInit(m, k, h_A, m);
    matInit(k, n, h_B, k);

    cudaMemcpy(d_A, h_A, sizeof(int8_t) * m * k,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int8_t) * k * n,cudaMemcpyHostToDevice);


    LtIgemmCustomFind(ltHandle,
                      m,
                      n,
                      k,
                      &alpha, /* host pointer */
                      d_A,
                      d_B,
                      &beta, /* host pointer */
                      d_C,
                      (void *)d_workspace,
                      workSpaceSize,
                      outfilename,
                      startEvent,
                      stopEvent);
    //free memory
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_workspace);

    cublasLtDestroy(ltHandle);
}

int generate_encoder_igemm_config(int batch_size, int seqLen, int headNum, int size_per_head)
{
    int m;   // the batchsize
    int k;
    int n;
    int batchCount;
   
    //ensure program running on SM >= 7.5
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0));
    if (!(prop.major >= 8 || (prop.major >= 7 && prop.minor >= 5))){
      printf("[ERROR] INT8 mode > 0 is only supported on device with sm >= 7.5\n ");
      exit(-1);
    }
    printf("Device %s\n", prop.name);
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    // Create CUDA event to time the execution time of each algo
    cudaEventCreate(&startEvent, cudaEventBlockingSync);
    cudaEventCreate(&stopEvent, cudaEventBlockingSync);

    char outfilename[1000];
    sprintf(outfilename, "igemm_config.in");
    FILE* fout = fopen(outfilename, "w+");
    if(fout == NULL)
    {
      printf("Cannot write to file igemm_config.in\n");
      return -1;
    }

    fclose(fout);

    printf("***Encoder IGemm Testing***\n");
    printf("\n-----------------------------\n");
   
    batchCount = 3;
    m = batch_size*seqLen;
    k = headNum*size_per_head;
    n = k;
    batch_igemm_config(batchCount,m,n,k,outfilename, startEvent, stopEvent);
 
    printf("\n-----------------------------\n");
    m = seqLen;
    n = seqLen;
    k = size_per_head;
    batchCount = batch_size*headNum;
    batch_igemm_config(batchCount,m,n,k,outfilename, startEvent, stopEvent);


    printf("\n-----------------------------\n");
    m = seqLen;
    n = size_per_head;
    k = seqLen;
    batchCount = batch_size*headNum;
    batch_igemm_config(batchCount,m,n,k,outfilename, startEvent, stopEvent);


    printf("\n-----------------------------\n");
    m = batch_size*seqLen;
    n = headNum*size_per_head;
    k = headNum*size_per_head;
    
    igemm_config(m,n,k,outfilename, startEvent, stopEvent);


    printf("\n-----------------------------\n");
    n = 4*n;
    igemm_config(m,n,k,outfilename, startEvent, stopEvent);


    printf("\n-----------------------------\n");
    n = k;
    k = 4*n;
    igemm_config(m,n,k,outfilename, startEvent, stopEvent);

    if (startEvent) cudaEventDestroy(startEvent);
    if (stopEvent) cudaEventDestroy(stopEvent);
}
