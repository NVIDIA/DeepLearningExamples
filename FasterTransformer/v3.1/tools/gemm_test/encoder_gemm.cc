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

#include "fastertransformer/gemm_test/encoder_gemm_func.h"
#include "fastertransformer/gemm_test/encoder_igemm_func.h"
#include "common.h"

int main(int argc, char* argv[])
{
  if(argc != 7)
  {
    printf("[ERROR] encoder_gemm batch_size seq_len head_number size_per_head is_fp16 int8_mode. \n");
    printf("e.g. ./bin/encoder_gemm 1 32 12 64 0 0\n");
    return 0;
  }

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int head_num = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);
  const int is_fp16 = atoi(argv[5]); 
  const int int8_mode = atoi(argv[6]);
  
  char filename[128];
  if (int8_mode == 0)
  {
    sprintf(filename, "gemm_config.in");
  }
  else
  {
    sprintf(filename, "igemm_config.in");
  }
  
  void *gemm_test_buf;
  size_t buf_size_in_byte = fastertransformer::calGemmTestBufSizeInByte(batch_size, seq_len, head_num, size_per_head, int8_mode, is_fp16);
  size_t total, free;
  check_cuda_error(cudaMemGetInfo(&free, &total));
  if (free < buf_size_in_byte + 10*1024*1024)
  {
    printf("[ERROR] There is no enough device memory for gemm test!\n %ld Bytes is needed, but only %ld Bytes is free.\n", buf_size_in_byte, free);
    gemm_test_buf = NULL;
    return -1;
  }
  else
  {
    check_cuda_error(cudaMalloc((void**)&gemm_test_buf, buf_size_in_byte));
  }

  if (int8_mode != 0){
    fastertransformer::generate_encoder_igemm_config(batch_size, seq_len, head_num, 
                                                     size_per_head, gemm_test_buf, false);
  }
  else{
    if(is_fp16 == 0)
      fastertransformer::generate_encoder_gemm_config<float>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    else if(is_fp16 == 1)
      fastertransformer::generate_encoder_gemm_config<half>(batch_size, seq_len, head_num, size_per_head, gemm_test_buf, false);
    else
    {
      printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
      return -1;
    }
  }
  
  check_cuda_error(cudaFree(gemm_test_buf));
  return 0;
}

