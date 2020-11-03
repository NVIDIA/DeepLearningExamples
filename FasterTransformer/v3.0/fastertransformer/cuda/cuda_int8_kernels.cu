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

#include "fastertransformer/common.h"

#include "cuda_kernels.h"
#include "cuda_int8_kernels.h"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cfloat>
namespace fastertransformer{

template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));

}


template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}


template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax<T>(val);

  return val;
}

//transpose matrix
//for (m n) col-major
//grid((m+31)/32, (n+31)/32)
//block(32, 32)
template<typename T>
__global__
void transposeMatrix_kernel(T*dst, const T* src, const int m, const int n)
{
  __shared__ T tile[COL32_][COL32_+1];

  int blockx32 = blockIdx.x * COL32_;
  int blocky32 = blockIdx.y * COL32_;
  int x = blockx32 + threadIdx.x;
  int y = blocky32 + threadIdx.y;

  bool check = ((x < m) && (y < n));
  tile[threadIdx.y][threadIdx.x] = check ? __ldg(src+y*m+x) : T(0);

  __syncthreads();

  y = blockx32 + threadIdx.y;
  x = blocky32 + threadIdx.x;

  check = ((x < n) && (y < m));
  if (check)
    dst[y*n+x] = tile[threadIdx.x][threadIdx.y];
}

//for (m, n) col-major matrix
template <typename T>
void transposeMatrix_kernelLauncher(T* dst, const T* src, const int m, const int n, cudaStream_t stream)
{
  transposeMatrix_kernel<T><<<dim3((m+31)/32, (n+31)/32), dim3(32, 32), 0, stream>>>(dst, src, m, n);
}

template void transposeMatrix_kernelLauncher<float>(float* dst, const float* src, const int m, const int n, cudaStream_t stream);

template void transposeMatrix_kernelLauncher<half>(half *dst, const half* src, const int m, const int n, cudaStream_t stream);

template void transposeMatrix_kernelLauncher<int8_t>(int8_t* dst, const int8_t* src, const int m, const int n, cudaStream_t stream);

template void transposeMatrix_kernelLauncher<int>(int* dst, const int* src, const int m, const int n, cudaStream_t stream);

//add bias to matrix of m * n, CUBLASLT_ORDER_COL32
//grid, thread = (m), (n/4)
//using char4
//for per-axis-quantization weight
template <typename T>
__global__
void add_bias_act_COL32_int32I_int8O(int8_t *out, const int32_t* input, const T* bias, const int m, const int n, 
                                     const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{

  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
 
  int col_start = threadIdx.x << 2;
  char4 *outTmpPtr = (char4 *)out;
  char4 tmp;
  int inIdx = (col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start&31);
  int outIdx = inIdx >> 2;
  float val;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.x = float_to_int8_rn(val*out_scale);
 
  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.y = float_to_int8_rn(val*out_scale);
  
  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.z = float_to_int8_rn(val*out_scale);

  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start));
  val = gelu(val);
  tmp.w = float_to_int8_rn(val*out_scale);

  outTmpPtr[outIdx] = tmp;
}

template <>
__global__
void add_bias_act_COL32_int32I_int8O(int8_t *out, const int32_t* input, const half2* bias, const int m, const int n, 
                                     const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr)
{
  const float input_deQFactor_div127 = __ldg(input_deQFactor_div127_ptr);
  const float out_scale = __ldg(out_scale_ptr);
  int col_start = threadIdx.x << 2;
  int threadIdx2 = threadIdx.x << 1;
  char4 *outTmpPtr = (char4 *)out;
  char4 tmp;
  int inIdx = (col_start & 0xffffffe0) * m + (blockIdx.x << 5) + (col_start&31);
  int outIdx = inIdx >> 2;
  float val;
  
  half2 biasTmp = __ldg(bias+threadIdx2);

  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(biasTmp.x);
  val = gelu(val);
  tmp.x = float_to_int8_rn(out_scale * val);

  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(biasTmp.y);
  val = gelu(val);
  tmp.y = float_to_int8_rn(out_scale * val);
  
  biasTmp = __ldg(bias+threadIdx2+1);

  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(biasTmp.x);
  val = gelu(val);
  tmp.z = float_to_int8_rn(out_scale * val);

  col_start = col_start + 1;
  inIdx = inIdx + 1;
  val = static_cast<float>(__ldg(input+inIdx))*__ldg(weight_amax+col_start)*input_deQFactor_div127 + static_cast<float>(biasTmp.y);
  val = gelu(val);
  tmp.w = float_to_int8_rn(out_scale * val);

  outTmpPtr[outIdx] = tmp;
}


template <typename T>
void add_bias_act_COL32_int32I_int8O_kernelLauncher(int8_t *out, const int32_t* input, const T* bias, const int m, const int n, 
                                                    cudaStream_t stream, const float* weight_amax, const float* input_deQFactor_div127_ptr, const float* out_scale_ptr){
  dim3 grid(m);
  dim3 block(n/4);
  assert(block.x <= 1024);
  if (sizeof(T) == sizeof(half))
    add_bias_act_COL32_int32I_int8O<<<grid, block, 0, stream>>>(out, input, (const half2*)bias, m, n, weight_amax, input_deQFactor_div127_ptr, out_scale_ptr);
  else
    add_bias_act_COL32_int32I_int8O<T><<<grid, block, 0, stream>>>(out, input, bias, m, n, weight_amax, input_deQFactor_div127_ptr, out_scale_ptr);
}

template void add_bias_act_COL32_int32I_int8O_kernelLauncher<float>(int8_t *out, const int32_t* input, const float* bias, const int m, const int n, cudaStream_t stream, const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr);

template void add_bias_act_COL32_int32I_int8O_kernelLauncher<half>(int8_t *out, const int32_t* input, const half* bias, const int m, const int n, cudaStream_t stream, const float* weight_amax, const float *input_deQFactor_div127_ptr, const float *out_scale_ptr);

//input1/input2/out matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n/4)
//using char4
template <typename T>
__global__
void add_bias_input_layernorm_COL32_mixIntI_int8O(int8_t* output, const int32_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                  const T* beta, int m, int n, const float* weight_amax, const float *input1_deQFactor_div127_ptr, 
						  const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor_div127 = __ldg(input1_deQFactor_div127_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int input1Idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));
  int outIdx = input1Idx >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input2Tmp = __ldg(input2TmpPtr+outIdx);
  
  int col_start_tmp = col_start;
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx+1))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx+2))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(__ldg(input1+input1Idx+3))*__ldg(weight_amax+col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(__ldg(bias+col_start_tmp));


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(__ldg(gamma+col_start)) + static_cast<float>(__ldg(beta+col_start));
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}

template <>
__global__
void add_bias_input_layernorm_COL32_mixIntI_int8O(int8_t* output, const int32_t* input1, const int8_t* input2, const half2* bias, const half2* gamma, 
                                                  const half2* beta, int m, int n, const float* weight_amax, const float *input1_deQFactor_div127_ptr, 
                                                  const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  const float input1_deQFactor_div127 = __ldg(input1_deQFactor_div127_ptr);
  const float input2_deQFactor = __ldg(input2_deQFactor_ptr);
  const float output_scale = __ldg(output_scale_ptr);
  int col_start = threadIdx.x << 2;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out[4];
  int input1Idx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));
  int outIdx = input1Idx >> 2;
  char4 *outTmpPtr = (char4*)output;
  char4 *input2TmpPtr = (char4*)input2;
  char4 input2Tmp = __ldg(input2TmpPtr + outIdx);
  
  int col_start_tmp = col_start;
  half2 biasTmp = __ldg(bias + (col_start_tmp >> 1));  
  local_out[0] = static_cast<float>(input2Tmp.x)*input2_deQFactor + static_cast<float>(__ldg(input1 + input1Idx))*__ldg(weight_amax + col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[1] = static_cast<float>(input2Tmp.y)*input2_deQFactor + static_cast<float>(__ldg(input1 + input1Idx + 1))*__ldg(weight_amax + col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(biasTmp.y);
  
  col_start_tmp = col_start_tmp + 1;
  biasTmp = __ldg(bias + (col_start_tmp >> 1));
  local_out[2] = static_cast<float>(input2Tmp.z)*input2_deQFactor + static_cast<float>(__ldg(input1 + input1Idx + 2))*__ldg(weight_amax + col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(biasTmp.x);
  col_start_tmp = col_start_tmp + 1;
  local_out[3] = static_cast<float>(input2Tmp.w)*input2_deQFactor + static_cast<float>(__ldg(input1 + (input1Idx+3)))*__ldg(weight_amax + col_start_tmp)*input1_deQFactor_div127 + static_cast<float>(biasTmp.y);


  mean = blockReduceSum<float>(local_out[0] + local_out[1] + local_out[2] + local_out[3]);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();
  
  local_out[0] = local_out[0] - s_mean;
  local_out[1] = local_out[1] - s_mean;
  local_out[2] = local_out[2] - s_mean;
  local_out[3] = local_out[3] - s_mean;
  variance = blockReduceSum<float>(local_out[0] * local_out[0] +
                                   local_out[1] * local_out[1] +
                                   local_out[2] * local_out[2] +
                                   local_out[3] * local_out[3]
                                  );
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();
  
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  half2 betaTmp = __ldg(beta+col_start_tmp); 
  
  local_out[0] = (local_out[0] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.x = float_to_int8_rn(local_out[0] * output_scale);

  col_start = col_start+1;
  local_out[1] = (local_out[1] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.y = float_to_int8_rn(local_out[1] * output_scale);
  
  col_start = col_start+1;
  col_start_tmp = col_start >> 1;
  biasTmp = __ldg(gamma+col_start_tmp);
  betaTmp = __ldg(beta+col_start_tmp);
  local_out[2] = (local_out[2] * s_variance) * static_cast<float>(biasTmp.x) + static_cast<float>(betaTmp.x);
  input2Tmp.z = float_to_int8_rn(local_out[2] * output_scale);
  
  col_start = col_start+1;
  local_out[3] = (local_out[3] * s_variance) * static_cast<float>(biasTmp.y) + static_cast<float>(betaTmp.y);
  input2Tmp.w = float_to_int8_rn(local_out[3] * output_scale);
  
  outTmpPtr[outIdx] = input2Tmp;
}


template<typename T>
void add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher(int8_t* output, const int32_t* input1, const int8_t* input2, const T* bias, const T* gamma, 
                                                                 const T* beta, int m, int n, cudaStream_t stream, const float* weight_amax, 
                                                                 const float *input1_deQFactor_div127_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr)
{
  dim3 grid(m);
  dim3 block(n/4);
  assert(n <= 1024);
  if (sizeof(T) == sizeof(half)){
    add_bias_input_layernorm_COL32_mixIntI_int8O<<<grid, block, 0, stream>>>(output, input1, input2, (const half2*)bias, (const half2*)gamma, 
                                                                            (const half2*)beta, m, n, weight_amax, input1_deQFactor_div127_ptr, 
                                                                            input2_deQFactor_ptr, output_scale_ptr);
  }
  else{
    add_bias_input_layernorm_COL32_mixIntI_int8O<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, beta, 
                                                                                m, n, weight_amax, input1_deQFactor_div127_ptr, 
                                                                                input2_deQFactor_ptr, output_scale_ptr);
  }
}

template void add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher<float>(int8_t* output, const int32_t* input1, const int8_t* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float* weight_amax, const float *input1_deQFactor_div127_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);

template void add_bias_input_layernorm_COL32_mixIntI_int8O_kernelLauncher<half>(int8_t* output, const int32_t* input1, const int8_t* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float* weight_amax, const float *input1_deQFactor_div127_ptr, const float *input2_deQFactor_ptr, const float *output_scale_ptr);


//input1/input2/output matrix with layout of cublasLt CUBLASLT_ORDER_COL32 (m*n)
//(grid, block) must be (m, n)
//for per_axis_quantization for weight
template <typename T>
__global__
void add_bias_input_layernorm_COL32_int32I_DataTypeO(T* output, const int32_t* input1, const T* input2, const T* bias, const T* gamma, 
                                                     const T* beta, int m, int n, const float* weight_amax, const float *input1_amax_ptr)
{
  const float input1_amax = __ldg(input1_amax_ptr);
  int col_start = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out;
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));

  float tmp = static_cast<float>(__ldg(input1 + outIdx)) * static_cast<float>(__ldg(weight_amax + col_start)) * input1_amax * 0.000062f; //(1/127/127);
  float inputTmp = static_cast<float>(__ldg(input2 + outIdx));

  local_out = tmp + inputTmp + static_cast<float>(__ldg(bias + col_start));

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();

  local_out = local_out - s_mean;

  variance = blockReduceSum<float>(local_out * local_out);
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();

  local_out = (local_out * s_variance) * static_cast<float>(__ldg(gamma + col_start)) + static_cast<float>(__ldg(beta + col_start));

  output[outIdx] = local_out;
}

template <>
__global__
void add_bias_input_layernorm_COL32_int32I_DataTypeO(half2* output, const int32_t* input1, const half2* input2, const half2* bias, const half2* gamma, 
                                                     const half2* beta, int m, int n, const float* weight_amax, const float *input1_amax_ptr)
{
  int col_start = threadIdx.x << 1;

  const float input1_amax = __ldg(input1_amax_ptr);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out, local_out2;
  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));

  float tmp = static_cast<float>(__ldg(input1 + outIdx)) * __ldg(weight_amax + col_start) * input1_amax * 0.000062f; //(1/127/127);
  float tmp2 = static_cast<float>(__ldg(input1 + outIdx + 1)) * __ldg(weight_amax + col_start + 1) * input1_amax * 0.000062f; //(1/127/127);
  
  outIdx = outIdx >> 1;
  half2 inputTmp = __ldg(input2 + outIdx);

  half2 biasTmp = __ldg(bias + threadIdx.x);

  local_out = tmp + static_cast<float>(inputTmp.x) + static_cast<float>(biasTmp.x);
  local_out2 = tmp2 + static_cast<float>(inputTmp.y) + static_cast<float>(biasTmp.y);

  mean = blockReduceSum<float>(local_out + local_out2);
  if(threadIdx.x == 0)
    s_mean = mean * __fdividef(1.0f, n);
  __syncthreads();

  local_out = local_out - s_mean;
  local_out2 = local_out2 - s_mean;

  variance = blockReduceSum<float>(local_out*local_out + local_out2*local_out2);
  if(threadIdx.x == 0){
    s_variance = variance * __fdividef(1.0f, n) + 1e-6f;
    s_variance = rsqrtf(s_variance);
  }
  __syncthreads();

  float2 outputTmp;
  inputTmp = __ldg(gamma + threadIdx.x);
  biasTmp = __ldg(beta + threadIdx.x);

  outputTmp.x = (local_out * s_variance) * static_cast<float>(inputTmp.x) + static_cast<float>(biasTmp.x);
  outputTmp.y = (local_out2 * s_variance) * static_cast<float>(inputTmp.y) + static_cast<float>(biasTmp.y);

  inputTmp = __float22half2_rn(outputTmp);
  output[outIdx] = inputTmp;
}


template <typename T>
void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher(T* output, const int32_t* input1, const T* input2, const T* bias, const T* gamma, 
                                                                    const T* beta, int m, int n, cudaStream_t stream, const float* weight_amax, 
                                                                    const float* input1_amax_ptr){

  dim3 grid(m);
  dim3 block(n);
  if (sizeof(T) == sizeof(half)){
    block.x /= 2;
    assert(block.x <= 1024);
    add_bias_input_layernorm_COL32_int32I_DataTypeO<<<grid, block, 0, stream>>>((half2 *)output, input1, (const half2 *)input2, (const half2 *)bias, (const half2 *)gamma, 
                                                                                (const half2 *)beta, m, n, weight_amax, input1_amax_ptr);
  }
  else{
    assert(block.x <= 1024);
    add_bias_input_layernorm_COL32_int32I_DataTypeO<T><<<grid, block, 0, stream>>>(output, input1, input2, bias, gamma, 
                                                                                   beta, m, n, weight_amax, input1_amax_ptr);
  }
}

template void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher<float>(float* output, const int32_t* input1, const float* input2, const float* bias, const float* gamma, const float* beta, int m, int n, cudaStream_t stream, const float* weight_amax, const float *input1_amax_ptr);

template void add_bias_input_layernorm_COL32_int32I_DataTypeO_kernelLauncher<half>(half* output, const int32_t* input1, const half* input2, const half* bias, const half* gamma, const half* beta, int m, int n, cudaStream_t stream, const float* weight_amax, const float *input1_amax_ptr);

//input matrix is m*n column-major
//output matrix is m*n CUBLASLT_ORDER_COL32
//(grid, block) must be (m, n)
template <typename T>
__global__
void FT_transformA(T* dst, const T* src, int m, int n)
{
  int inIdx = threadIdx.x * m + blockIdx.x;
  int col_start = threadIdx.x;

  int outIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));
  
  dst[outIdx] = __ldg(&src[inIdx]);
}

template <typename T>
void FT_transformA_kernelLauncher(T* dst, const T* src, int m, int n, cudaStream_t stream){
  dim3 grid(m);
  dim3 block(n);
  assert(block.x <= 1024);
  FT_transformA<<<grid, block, 0, stream>>>(dst, src, m, n);
}

template void FT_transformA_kernelLauncher(float* dst, const float* src, int m, int n, cudaStream_t stream);

template void FT_transformA_kernelLauncher(half* dst, const half* src, int m, int n, cudaStream_t stream);


//input matrix is m*n CUBLASLT_ORDER_COL32
//output matrix is m*n column-major
//(grid, block) must be (m, n)
template <typename T>
__global__
void FT_transformC(T* dst, const T* src, int m, int n)
{
  int outIdx = threadIdx.x * m + blockIdx.x;
  int col_start = threadIdx.x;

  int inIdx = ((col_start & 0xffffffe0)*m+(blockIdx.x << 5) + (col_start&31));

  dst[outIdx] = __ldg(&src[inIdx]);
}

template <typename T>
void FT_transformC_kernelLauncher(T* dst, const T* src, int m, int n, cudaStream_t stream){
  dim3 grid(m);
  dim3 block(n);
  assert(block.x <= 1024);
  FT_transformC<<<grid, block, 0, stream>>>(dst, src, m, n);
}

template void FT_transformC_kernelLauncher(float* dst, const float* src, int m, int n, cudaStream_t stream);

template void FT_transformC_kernelLauncher(half* dst, const half* src, int m, int n, cudaStream_t stream);

template void FT_transformC_kernelLauncher(int8_t* dst, const int8_t* src, int m, int n, cudaStream_t stream);

//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = batch_size*seq_len, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_kernel(int8_t* dst, const int32_t* src, const int batch_size, const int seq_len, const int head_num, 
                            const int size_per_head, const float *v_buf_addBias_deQFactor, const float* qk_afterSM_deQFactor, const float* out_scale_ptr, 
                            const int batch_size_x_seq_len, const int seq_len_x_size_per_head)
{
  const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = batch_size*seq_len
  //k = head_num*size_per_head
  int mk_row = batch_id*seq_len + seq_id;
  int mk_col = head_id*size_per_head + threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m = 32*batch_size*seq_len
  int COL32_row = (mk_row << 5) + (mk_col&31);
  int COL32_col = mk_col >> 5;
  int outIdx = ((COL32_col << 5)*batch_size_x_seq_len + COL32_row) >> 2;

  //get the (row, col) input layout of m'*k'
  //m' = seq_len
  //k' = size_per_head
  mk_row = seq_id;
  mk_col = threadIdx4;
  //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
  COL32_row = (mk_row << 5) + (mk_col&31);
  COL32_col = mk_col >> 5;

  int inIdx = (batch_id*head_num + head_id)*seq_len_x_size_per_head + (COL32_col << 5 )*seq_len + COL32_row;
  char4 tmp;
  tmp.x = float_to_int8_rn(__ldg(src+inIdx)*scale);
  tmp.y = float_to_int8_rn(__ldg(src+inIdx+1)*scale);
  tmp.z = float_to_int8_rn(__ldg(src+inIdx+2)*scale);
  tmp.w = float_to_int8_rn(__ldg(src+inIdx+3)*scale);
  char4 *dst_ptr4 = (char4 *)dst;
  dst_ptr4[outIdx] = tmp;
}

void transpose_COL32_kernelLauncher(int8_t* dst, const int* src, const int batch_size, const int seq_len, const int head_num, 
                                    const int size_per_head, const float *v_buf_addBias_deQFactor, const float* qk_afterSM_deQFactor, 
                                    const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>(dst, src, batch_size, seq_len, head_num, size_per_head, v_buf_addBias_deQFactor, qk_afterSM_deQFactor, out_scale_ptr, batch_size*seq_len, seq_len*size_per_head);
}

//src is the result of batch MM, whose size is batch_size*head_num*(seq_len, size_per_head), CUBLASLT_ORDER_COL32
//dst is of m = valid_word_num, k(n) = head_num*size_per_head, CUBLASLT_ORDER_COL32
//grid(seq_len, batch_size)
//block(size_per_head/4, head_num)
//assume size_per_head is multiples of 32
__global__
void transpose_COL32_rebuild_padding_kernel(int8_t* dst, const int32_t* src, const int* sequence_id_map, const int valid_word_num, const int batch_size, 
                                            const int seq_len, const int head_num, const int size_per_head, const float *v_buf_addBias_deQFactor, 
                                            const float* qk_afterSM_deQFactor, const float* out_scale_ptr, const int seq_len_x_size_per_head)
{
  const float scale = __ldg(v_buf_addBias_deQFactor) * __ldg(qk_afterSM_deQFactor) * __ldg(out_scale_ptr);
  int threadIdx4 = threadIdx.x << 2;
  int batch_id = blockIdx.y;
  int seq_id = blockIdx.x;
  int head_id = threadIdx.y;
  //get the (row, col) output layout of m*k
  //m = valid_word_num
  //k = head_num*size_per_head
  int mk_row = __ldg(sequence_id_map + batch_id*seq_len + seq_id);
  if (mk_row >= 0){
    int mk_col = head_id*size_per_head + threadIdx4;
    //get the (row, col) layout of COL32; leading dimension = 32*m = 32*valid_word_num
    int COL32_row = (mk_row << 5) + (mk_col&31);
    int COL32_col = mk_col >> 5;
    int outIdx = ((COL32_col << 5)*valid_word_num + COL32_row) >> 2;

    //get the (row, col) input layout of m'*k'
    //m' = seq_len
    //k' = size_per_head
    mk_row = seq_id;
    mk_col = threadIdx4;
    //get the (row, col) layout of COL32; leading dimension = 32*m' = 32*seq_len
    COL32_row = (mk_row << 5) + (mk_col&31);
    COL32_col = mk_col >> 5;

    int inIdx = (batch_id*head_num + head_id)*seq_len_x_size_per_head + (COL32_col << 5 )*seq_len + COL32_row;
    char4 tmp;
    tmp.x = float_to_int8_rn(__ldg(src+inIdx)*scale);
    tmp.y = float_to_int8_rn(__ldg(src+inIdx+1)*scale);
    tmp.z = float_to_int8_rn(__ldg(src+inIdx+2)*scale);
    tmp.w = float_to_int8_rn(__ldg(src+inIdx+3)*scale);
    char4 *dst_ptr4 = (char4 *)dst;
    dst_ptr4[outIdx] = tmp;
  }
}

void transpose_COL32_rebuild_padding_kernelLauncher(int8_t* dst, const int* src, const int* sequence_id_map, const int valid_word_num, const int batch_size, 
                                                    const int seq_len, const int head_num, const int size_per_head, const float *v_buf_addBias_deQFactor, 
                                                    const float* qk_afterSM_deQFactor, const float* out_scale_ptr, cudaStream_t stream){
  assert(size_per_head%32==0);
  transpose_COL32_rebuild_padding_kernel<<<dim3(seq_len, batch_size), dim3(size_per_head/4, head_num), 0, stream>>>(dst, src, sequence_id_map, valid_word_num, batch_size, 
                                                                                                                    seq_len, head_num, size_per_head, v_buf_addBias_deQFactor, 
                                                                                                                    qk_afterSM_deQFactor, out_scale_ptr, seq_len*size_per_head);
}


template <typename T>
__global__
void quantized_kernel(int8_t *dst, const T* src, const int size, const float* scale_ptr)
{
  int tid = (blockIdx.x*blockDim.x + threadIdx.x) << 2;
  if (tid < size){
    const float scale = __ldg(scale_ptr);
    char4 tmp;
    tmp.x = float_to_int8_rn(static_cast<float>(__ldg(&src[tid]))*scale);
    tmp.y = float_to_int8_rn(static_cast<float>(__ldg(&src[tid+1]))*scale);
    tmp.z = float_to_int8_rn(static_cast<float>(__ldg(&src[tid+2]))*scale);
    tmp.w = float_to_int8_rn(static_cast<float>(__ldg(&src[tid+3]))*scale);
    char4 *dst_ptr4 = (char4 *)dst;
    dst_ptr4[tid >> 2] = tmp;
  }
}
template <typename T>
void quantized_kernelLauncher(int8_t* dst, const T * src, const int size, const float* scale_ptr, cudaStream_t stream)
{
   assert(size % (4 * 64) == 0);
   dim3 grid((size+255)/256);
   dim3 block(64);
   quantized_kernel<T><<<grid, block, 0, stream>>>(dst, src, size, scale_ptr);
}

template void quantized_kernelLauncher<float>(int8_t* dst, const float * src, const int size, const float* scale_ptr, cudaStream_t stream);

template void quantized_kernelLauncher<half>(int8_t* dst, const half * src, const int size, const float* scale_ptr, cudaStream_t stream);

template void quantized_kernelLauncher<int32_t>(int8_t* dst, const int32_t * src, const int size, const float* scale_ptr, cudaStream_t stream);

template <typename T>
__global__
void dequantized_kernel(T *dst, const int8_t* src, const int size, const float *scale_ptr)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid < size){
    float tmp = float(src[tid]);
    dst[tid] = T(float(tmp) *  __ldg(scale_ptr));
  }
}

template <typename T>
void dequantized_kernelLauncher(T* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream)
{
   dim3 grid((size+255)/256);
   dim3 block(256);
   dequantized_kernel<T><<<grid, block, 0, stream>>>(dst, src, size, scale_ptr);
}
template void dequantized_kernelLauncher<float>(float* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream);

template void dequantized_kernelLauncher<half>(half* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream);

template void dequantized_kernelLauncher<int32_t>(int32_t* dst, const int8_t * src, const int size, const float *scale_ptr, cudaStream_t stream);


}//namespace 
