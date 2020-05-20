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

#include "encoder_gemm.h"

int main(int argc, char* argv[])
{
  if(argc != 6)
  {
    printf("[ERROR] encoder_gemm batch_size seq_len head_number size_per_head is_fp16. \n");
    printf("e.g. ./bin/encoder_gemm 1 32 12 64 0\n");
    return 0;
  }

  const int batch_size = atoi(argv[1]);
  const int seq_len = atoi(argv[2]);
  const int head_num = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);

  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  if(atoi(argv[5]) == 0)
    generate_encoder_gemm_config<float>(batch_size, seq_len, head_num, size_per_head);
  else if(atoi(argv[5]) == 1)
    generate_encoder_gemm_config<half>(batch_size, seq_len, head_num, size_per_head);
  else
  {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }

  return 0;
}

