#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cassert>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// For simplicity reason, boundry checks are removed
// All the  kernels MUST be launched with grid size = batch size and block size = embedding size

__global__ void GatherKernel(const float* params,
                             int64_t num_features,
                             int embed_size,
                             int batch_size,
                             int query_nnz,
                             const int64_t* indices,
                             float* ret) {
  int tid = threadIdx.x, bid = blockIdx.x;

  extern __shared__ int shmem_indices[];

  // each CTA load one row of indices in the mini batch into shared memory
  for (int i = tid; i < query_nnz; i += blockDim.x) {
    shmem_indices[i] = indices[query_nnz * bid + i];
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < query_nnz; ++i) {
    // printf("%d, %d, %d\n", bid, i, shmem_indices[i]);
    ret[(bid * query_nnz + i) * embed_size + tid] =
        params[(int64_t)shmem_indices[i] * embed_size + tid];
  }
}

__global__ void OneHotKernel(const float* params,
                             int64_t num_features,
                             int embed_size,
                             int batch_size,
                             const int64_t* indices,
                             float* ret) {
  int tid = threadIdx.x, bid = blockIdx.x;

  ret[bid * embed_size + tid] = params[(int64_t)indices[bid] * embed_size + tid];
}

// grads is used to update params directly by atomic instead of forming wgrad
// Only SGD without momentum and without weight decay is supported
__global__ void GatherBackwardFuseSgdKernel(const float* grads,
                                            int64_t num_features,
                                            int embed_size,
                                            int batch_size,
                                            int query_nnz,
                                            const int64_t* indices,
                                            float lr,
                                            float* params) {
  int tid = threadIdx.x, bid = blockIdx.x;

  extern __shared__ int shmem_indices[];

  for (int i = tid; i < query_nnz; i += blockDim.x) {
    shmem_indices[i] = indices[query_nnz * bid + i];
  }
  __syncthreads();

#pragma unroll
  for (int i = 0; i < query_nnz; ++i) {
    atomicAdd(&params[(int64_t)shmem_indices[i] * embed_size + tid],
              -lr * grads[(bid * query_nnz + i) * embed_size + tid]);
  }
}

// Keep the interface and argument name as torch.embedding()
// input is indices, and weight is embedding table
torch::Tensor gather_gpu_fwd(const torch::Tensor weight, const torch::Tensor indices) {
  AT_ASSERT(indices.is_cuda());
  AT_ASSERT(weight.is_cuda());
  AT_ASSERT(indices.scalar_type() == torch::ScalarType::Long);
  AT_ASSERT(weight.scalar_type() == torch::ScalarType::Float);
  AT_ASSERT(weight.is_contiguous());

  int batch_size = indices.size(0);
  int query_nnz = 1;
  if (indices.dim() > 1) {
    query_nnz = indices.size(1);
  }

  // Shared memory size limit. Larger nnz can also be supported by skipping shared memory if necessary
  TORCH_CHECK(query_nnz <= 12288, "Embedding width must be smaller than 48k");

  int num_features = weight.size(0);
  int embed_size = weight.size(1);

  // Block dimension limit. Large than 1024 width can be easily supported by letting each block read
  // from different strides if necessary.
  TORCH_CHECK(embed_size <= 1024, "Embedding width must be smaller than 1024");

  auto outputs =
      torch::empty(batch_size * query_nnz * embed_size, at::device(at::kCUDA).dtype(at::kFloat));

  if (query_nnz != 1) {
    GatherKernel<<<batch_size,
                   embed_size,
                   query_nnz * sizeof(int),
                   at::cuda::getCurrentCUDAStream()>>>(weight.data_ptr<float>(),
                                                       num_features,
                                                       embed_size,
                                                       batch_size,
                                                       query_nnz,
                                                       indices.contiguous().data_ptr<int64_t>(),
                                                       outputs.data_ptr<float>());
  } else {
    OneHotKernel<<<batch_size, embed_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        weight.data_ptr<float>(),
        num_features,
        embed_size,
        batch_size,
        indices.contiguous().data_ptr<int64_t>(),
        outputs.data_ptr<float>());
  }

  return outputs.reshape({batch_size, query_nnz, embed_size});
}

// Because complication of handling sparse tensor, use the native backward function is still faster
// TODO(haow): Figure out a way to write out sparse tensor directly to avoid addintional copy which makes
// customized implementation slower than Pytorch's own desipte kernels are more efficient
torch::Tensor gather_gpu_bwd(const torch::Tensor grad,
                             const torch::Tensor indices,
                             const int num_features) {
  return at::embedding_sparse_backward(grad, indices, num_features, /*padding_idx=*/-1, /*scale_grad_by_freq=*/false);
}

// Backward gather with fused plain SGD (no weight decay nor momentum)
void gather_gpu_bwd_fuse_sgd(const torch::Tensor grad,
                             const torch::Tensor indices,
                             float lr,
                             torch::Tensor weight) {
  AT_ASSERT(grad.is_cuda());
  AT_ASSERT(indices.is_cuda());
  AT_ASSERT(weight.is_cuda());
  AT_ASSERT(grad.scalar_type() == torch::ScalarType::Float);
  AT_ASSERT(indices.scalar_type() == torch::ScalarType::Long);
  AT_ASSERT(weight.scalar_type() == torch::ScalarType::Float);
  AT_ASSERT(weight.is_contiguous());

  int batch_size = indices.size(0);
  int query_nnz = 1;
  if (indices.dim() > 1) {
    query_nnz = indices.size(1);
  }
  int num_features = weight.size(0);
  int embed_size = weight.size(1);

  GatherBackwardFuseSgdKernel<<<batch_size,
                                embed_size,
                                query_nnz * sizeof(int),
                                at::cuda::getCurrentCUDAStream()>>>(
      grad.contiguous().data_ptr<float>(),
      num_features,
      embed_size,
      batch_size,
      query_nnz,
      indices.contiguous().data_ptr<int64_t>(),
      lr,
      weight.data_ptr<float>());
}
