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

#include "decoding_gemm.h"

int main(int argc, char* argv[])
{
  if(argc != 9)
  {
    printf("[ERROR] decoding_gemm batch_size beam_width head_number size_per_head vocab_size seq_len memory_hidden_units is_fp16\n");
    printf("e.g. ./bin/decoding_gemm 32 4 8 64 30000 32 768 0\n");
    return 0;
  }
  const int batch_size = atoi(argv[1]);
  const int beam_width = atoi(argv[2]);
  const int head_number = atoi(argv[3]);
  const int size_per_head = atoi(argv[4]);
  const int vocab_size = atoi(argv[5]);
  const int seq_len = atoi(argv[6]);
  const int memory_hidden_units = atoi(argv[7]);

  struct cudaDeviceProp prop;
  check_cuda_error(cudaGetDeviceProperties(&prop, 0));
  printf("Device %s\n", prop.name);

  if(atoi(argv[8]) == 0)
    generate_decoding_gemm_config<float>(batch_size, beam_width, head_number, size_per_head, vocab_size, seq_len, memory_hidden_units);
  else if(atoi(argv[8]) == 1)
    generate_decoding_gemm_config<half>(batch_size, beam_width, head_number, size_per_head, vocab_size, seq_len, memory_hidden_units);
  else
  {
    printf("[ERROR] is_fp16 should be 0 (use float) or 1 (use half). \n");
    return -1;
  }

  return 0;
}

