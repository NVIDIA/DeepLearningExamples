#include <iostream>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#define CHK_CUDA(expression)                                                                                        \
  {                                                                                                                 \
    cudaError_t status = (expression);                                                                              \
    if (status != cudaSuccess) {                                                                                    \
      std::cerr << "Error in file: " << __FILE__ << ", on line: " << __LINE__ << ": " << cudaGetErrorString(status) \
                << std::endl;                                                                                       \
      std::exit(EXIT_FAILURE);                                                                                      \
    }                                                                                                               \
  }

// only 4 element vectorized types are implemented - can be done for other types
// load/store by "mask" vars
// assignments by "val" vars
template <class DTYPE>
struct VecType4{};

template <>
struct VecType4<__half> {
  typedef float2 Type;
  typedef struct __align__(8) {
    __half x;
    __half y;
    __half z;
    __half w;
  } half4;
  union Data {
    half4 val;
    Type mask;
  } data;

  __device__ VecType4() {
    data.mask = make_float2(0.0f, 0.0f);
  }

  __device__ VecType4& operator=(float4 &in) {
    data.val.x = __float2half(in.x);
    data.val.y = __float2half(in.y);
    data.val.z = __float2half(in.z);
    data.val.w = __float2half(in.w);

    return *this;
  }

  __device__ VecType4& operator=(half4 &in) {
    data.val = in;
    return *this;
  }
};

template <>
struct VecType4<float> {
  typedef float4 Type;
  union Data {
    Type val;
    Type mask;
  } data;

  __device__ VecType4() {
    data.val.x = 0.0f;
    data.val.y = 0.0f;
    data.val.z = 0.0f;
    data.val.w = 0.0f;
  }

  __device__ VecType4& operator=(VecType4<__half>::half4 &in) {
    data.val.x = __half2float(in.x);
    data.val.y = __half2float(in.y);
    data.val.z = __half2float(in.z);
    data.val.w = __half2float(in.w);

    return *this;
  }
  __device__ VecType4& operator=(float4 &in) {
    data.val = in;
    return *this;
  }
};

//  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
// above default build params to Torch extensions requires this extensive juggling around
template <typename ITYPE, typename OTYPE, typename std::enable_if<(std::is_same<ITYPE, float>::value &&
                                                                    std::is_same<OTYPE, __half>::value),
                                                                    ITYPE>::type * = nullptr>
__device__ __host__ __forceinline__  OTYPE fp_type_cast(ITYPE input) {
  return __float2half(input);
}

template <typename ITYPE, typename OTYPE, typename std::enable_if<(std::is_same<ITYPE, __half>::value &&
                                                                    std::is_same<OTYPE, float>::value),
                                                                    ITYPE>::type * = nullptr>
__device__ __host__ __forceinline__  OTYPE fp_type_cast(ITYPE input) {
  return __half2float(input);
}

template <typename ITYPE, typename OTYPE, typename std::enable_if<std::is_same<ITYPE, OTYPE>::value,
                                                                    ITYPE>::type * = nullptr>
__device__ __host__ __forceinline__  OTYPE fp_type_cast(ITYPE input) {
  return input;
}

// this kernel assumes embedding vector_width of 128
template <typename ITYPE, typename OTYPE>
__global__ void lookupEmbeddings(ITYPE *embeddingTable, int64_t *offsets,
                                    int64_t *indices, OTYPE *outLookup, int batch_size) {

  typedef typename VecType4<ITYPE>::Type invec4;
  typedef typename VecType4<OTYPE>::Type outvec4;

  int vector_width = 128;
  const int fea_count = 26;

  int lane_id = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int num_warps = blockDim.x / warpSize;
  int start_idx = warp_id * fea_count + lane_id + blockIdx.x * (num_warps * fea_count);

  int64_t lane_offset = 0;
  if (lane_id < fea_count)
    lane_offset = offsets[lane_id];

  while (1) {
    int64_t lookup_idx = -1;
    if (lane_id < fea_count && start_idx < (batch_size * fea_count)) {
      lookup_idx = indices[start_idx] + lane_offset;
  }

  if (__all_sync(0xffffffff, lookup_idx == -1))
    break;

  for (int i = 0; i < fea_count; i++) {
    int64_t table_idx = __shfl_sync(0xffffffff, lookup_idx, i);

    if (table_idx != -1) {
      invec4 *vec_embedding_table = reinterpret_cast<invec4*>(embeddingTable);
      outvec4 *vec_embedding_out = reinterpret_cast<outvec4*>(outLookup);

      int64_t out_idx = start_idx - lane_id + i;
      out_idx *= vector_width;

      int vector_inst_width = 4;    // 128 bit loads, 4-floats
      int64_t vec_in_idx = ((table_idx * vector_width) + (lane_id * vector_inst_width)) >> 2;
      int64_t vec_out_idx = (out_idx + (lane_id * vector_inst_width)) >> 2;

      VecType4<ITYPE> input_elements;
      input_elements.data.mask = vec_embedding_table[vec_in_idx];
      VecType4<OTYPE> output_elements;
      output_elements = input_elements.data.val;
      vec_embedding_out[vec_out_idx] = output_elements.data.mask;
    }
  }

  start_idx += (gridDim.x * num_warps * fea_count);
  }
}

__global__ void indices_offset_addition(int64_t *indices, int64_t *offsets, int64_t *output_indices,
                                          int batch_size) {
  const int fea_count = 26;
  __shared__ int64_t smem_offsets[fea_count];

  if (threadIdx.x < fea_count) {
    smem_offsets[threadIdx.x] = offsets[threadIdx.x];
  }
  __syncthreads();

  int start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = start_idx; i < (batch_size * fea_count); i+=(gridDim.x * blockDim.x)) {
    output_indices[i] = indices[i] + smem_offsets[i % fea_count];
  }
}

template <typename ITYPE, typename OTYPE>
__global__ void gradient_copy_kernel(ITYPE *input_gradient, OTYPE *output_gradient, int64_t num_elements) {
  typedef typename VecType4<ITYPE>::Type invec4;
  typedef typename VecType4<OTYPE>::Type outvec4;

  invec4 *vec_input_gradient = reinterpret_cast<invec4*>(input_gradient);
  outvec4 *vec_output_gradient = reinterpret_cast<outvec4*>(output_gradient);

  int64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = start_idx; i < num_elements / 4; i+= (gridDim.x * blockDim.x)) {
    VecType4<ITYPE> input_elements;
    input_elements.data.mask = vec_input_gradient[i];
    VecType4<OTYPE> output_elements;
    output_elements = input_elements.data.val;
    vec_output_gradient[i] = output_elements.data.mask;
  }
  int elements_left = num_elements % 4;

  if (threadIdx.x == 0 && elements_left != 0) {
    while(elements_left) {
      int64_t idx = num_elements - elements_left;
      output_gradient[idx] = fp_type_cast<ITYPE, OTYPE>(input_gradient[idx]);
      elements_left--;
    }
  }
}

// kernels are fully instantiation type compatible float<->float , float<->Half, half<->half
// but their runner functions are not instantiated for all types
template <typename ITYPE, typename OTYPE>
void gather_gpu_fused_fwd(ITYPE *embeddingTablePtr, int64_t *indices_offset, int64_t *lookup_indices,
                            OTYPE *outputPtr, int batch_size) {};

template <>
void gather_gpu_fused_fwd(float *embeddingTablePtr, int64_t *indices_offset, int64_t *lookup_indices,
                            c10::Half *outputPtr, int batch_size) {

  auto deviceProp = at::cuda::getCurrentDeviceProperties();
  dim3 block(deviceProp->maxThreadsPerBlock, 1, 1);
  dim3 grid((deviceProp->multiProcessorCount * deviceProp->maxThreadsPerMultiProcessor) / deviceProp->maxThreadsPerBlock,
              1, 1);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  lookupEmbeddings<float, __half><<<grid, block, 0, stream>>>(embeddingTablePtr, indices_offset, lookup_indices, (__half*)outputPtr, batch_size);
  CHK_CUDA(cudaGetLastError());
}

template <>
void gather_gpu_fused_fwd(float *embeddingTablePtr, int64_t *indices_offset, int64_t *lookup_indices,
                            float *outputPtr, int batch_size) {

  auto deviceProp = at::cuda::getCurrentDeviceProperties();
  dim3 block(deviceProp->maxThreadsPerBlock, 1, 1);
  dim3 grid((deviceProp->multiProcessorCount * deviceProp->maxThreadsPerMultiProcessor) / deviceProp->maxThreadsPerBlock,
              1, 1);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  lookupEmbeddings<float, float><<<grid, block, 0, stream>>>(embeddingTablePtr, indices_offset, lookup_indices, outputPtr, batch_size);
  CHK_CUDA(cudaGetLastError());
}

template <>
void gather_gpu_fused_fwd(c10::Half *embeddingTablePtr, int64_t *indices_offset, int64_t *lookup_indices,
                            c10::Half *outputPtr, int batch_size) {

  auto deviceProp = at::cuda::getCurrentDeviceProperties();
  dim3 block(deviceProp->maxThreadsPerBlock, 1, 1);
  dim3 grid((deviceProp->multiProcessorCount * deviceProp->maxThreadsPerMultiProcessor) / deviceProp->maxThreadsPerBlock,
              1, 1);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  lookupEmbeddings<__half, __half><<<grid, block, 0, stream>>>((__half*)embeddingTablePtr, indices_offset, lookup_indices, (__half*)outputPtr, batch_size);
  CHK_CUDA(cudaGetLastError());
}

template <typename ITYPE, typename OTYPE>
void gather_gpu_fused_bwd(ITYPE *input_gradient, int64_t *lookup_indices, int64_t *offsets, OTYPE *out_gradient,
                            int64_t *out_indices, int batch_size, int num_features, int embed_vector_dim) {};

template <>
void gather_gpu_fused_bwd(c10::Half *input_gradient, int64_t *lookup_indices, int64_t *offsets, float *out_gradient,
                            int64_t *out_indices, int batch_size, int num_features, int embed_vector_dim) {
  // offset addition to indices
  auto deviceProp = at::cuda::getCurrentDeviceProperties();
  dim3 block(deviceProp->maxThreadsPerBlock, 1, 1);
  dim3 grid((deviceProp->multiProcessorCount * deviceProp->maxThreadsPerMultiProcessor) / deviceProp->maxThreadsPerBlock,
              1, 1);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // indices - offset addition kernel
  indices_offset_addition<<<grid, block, 0, stream>>>(lookup_indices, offsets, out_indices, batch_size);
  CHK_CUDA(cudaGetLastError());

  gradient_copy_kernel<__half, float><<<grid, block, 0, stream>>>((__half *)input_gradient, out_gradient, (int64_t)batch_size * num_features * embed_vector_dim );
  CHK_CUDA(cudaGetLastError());
}

template <>
void gather_gpu_fused_bwd(float *input_gradient, int64_t *lookup_indices, int64_t *offsets, float *out_gradient,
                            int64_t *out_indices, int batch_size, int num_features, int embed_vector_dim) {
  // offset addition to indices
  auto deviceProp = at::cuda::getCurrentDeviceProperties();
  dim3 block(deviceProp->maxThreadsPerBlock, 1, 1);
  dim3 grid((deviceProp->multiProcessorCount * deviceProp->maxThreadsPerMultiProcessor) / deviceProp->maxThreadsPerBlock,
              1, 1);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // indices - offset addition kernel
  indices_offset_addition<<<grid, block, 0, stream>>>(lookup_indices, offsets, out_indices, batch_size);
  CHK_CUDA(cudaGetLastError());

  gradient_copy_kernel<float, float><<<grid, block, 0, stream>>>(input_gradient, out_gradient, (int64_t)batch_size * num_features * embed_vector_dim );
  CHK_CUDA(cudaGetLastError());
}
