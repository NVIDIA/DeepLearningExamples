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

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/cuda/detail/TensorInfo.cuh"
#include "curand_kernel.h"

#include <THC/THCGeneral.h>
#include <THC/THCTensorRandom.h>
#include <THC/THCGenerator.hpp>


THCGenerator* THCRandom_getGenerator(THCState* state);

// philox generates 128 bits of randomness at a time. Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
const int UNROLL = 4;

std::pair<uint64_t, uint64_t> next_philox_seed(uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}


template <
          typename scalar_t,
          typename accscalar_t,
          typename IndexType,
          int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(256,8)
#endif
__global__ void
fused_dropout_add_kernel(at::cuda::detail::TensorInfo<scalar_t, IndexType> input,
                      at::cuda::detail::TensorInfo<scalar_t, IndexType> input_add,
                      at::cuda::detail::TensorInfo<scalar_t, IndexType> ret,
                      at::cuda::detail::TensorInfo<uint8_t, IndexType> mask,
                      IndexType totalElements, accscalar_t prob, std::pair<uint64_t, uint64_t> seeds
                      ) {

  //accscalar_t pinv = accscalar_t(1)/prob;
  float pinv = 1.0/(float)prob;
  IndexType idx = blockIdx.x * blockDim.x*UNROLL + threadIdx.x;
  curandStatePhilox4_32_10_t state;
    curand_init(
        seeds.first,
        idx,
        seeds.second,
        &state);
  float4 rand = curand_uniform4(&state);
  scalar_t src[UNROLL];
  scalar_t src_add[UNROLL];
  rand.x = rand.x < prob;
  rand.y = rand.y < prob;
  rand.z = rand.z < prob;
  rand.w = rand.w < prob;
  IndexType offset = idx;
  for (int ii = 0; ii < UNROLL; ii++) {
      if (offset < totalElements) {
          src[ii] = input.data[offset];
          src_add[ii] = input_add.data[offset];
      }
      offset += blockDim.x; 
  }
  offset = idx;
  for (int ii = 0; ii < UNROLL; ii++) {
      if (offset < totalElements) {
          ret.data[offset] = src[ii]*(&rand.x)[ii]*pinv + src_add[ii];
          mask.data[offset] = (uint8_t)(&rand.x)[ii];
      }
      offset += blockDim.x; 
  }
}

template<typename scalar_t, typename accscalar_t>
void masked_scale_kernel(at::Tensor& ret, const at::Tensor src, const at::Tensor mask, accscalar_t scale){
   at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, uint8_t>(ret, src, mask, [scale]__device__(scalar_t& ret_val, const scalar_t& src_val, const uint8_t mask_val){
       ret_val = (float)mask_val * src_val * scale;
  });
}

std::vector<at::Tensor>
fused_dropout_add_cuda(const at::Tensor& input, const at::Tensor& input_add, double prob){
  at::Tensor ret = at::empty_like(input);
  at::Tensor mask = at::empty(input.sizes(), input.options().dtype(at::kByte));
  const int64_t nelem = input.numel();
  const int64_t block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((nelem + (block_size*UNROLL) -1)/(block_size*UNROLL));
//number of times random will be generated per thread, to offset philox counter in thc random state
  int64_t counter_offset = UNROLL ; //((nelem - 1)/(block_size*grid.x*UNROLL)+1)*UNROLL;
  if (at::cuda::detail::canUse32BitIndexMath(input)){
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_dropout_add", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      accscalar_t proba = (accscalar_t)(prob);
      auto input_info = at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(input);
      auto input_add_info = at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(input_add);
      auto ret_info = at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(ret);
      auto mask_info = at::cuda::detail::getTensorInfo<uint8_t, unsigned int>(mask);
      input_info.collapseDims();
      input_add_info.collapseDims();
      ret_info.collapseDims();
      mask_info.collapseDims(); //ret and mask are collapsed to 1d contiguous tensor
      switch (input_info.dims) {
        case 1:
            fused_dropout_add_kernel<scalar_t, accscalar_t, unsigned int, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(input_info, input_add_info, ret_info, mask_info, nelem, proba, next_philox_seed(counter_offset));
            break;
        default:
            fused_dropout_add_kernel<scalar_t, accscalar_t, unsigned int, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(input_info, input_add_info, ret_info, mask_info, nelem, proba, next_philox_seed(counter_offset));
      }
   });
  } else {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_dropout_add", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      accscalar_t proba = (accscalar_t)(prob);
      auto input_info = at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(input);
      auto input_add_info = at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(input_add);
      auto ret_info = at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(ret);
      auto mask_info = at::cuda::detail::getTensorInfo<uint8_t, uint64_t>(mask);
      input_info.collapseDims();
      input_add_info.collapseDims();
      ret_info.collapseDims();
      mask_info.collapseDims(); //ret and mask are collapsed to 1d contiguous tensor
      switch (input_info.dims) {
        case 1:
            fused_dropout_add_kernel<scalar_t, accscalar_t, uint64_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(input_info, input_add_info, ret_info, mask_info, nelem, proba, next_philox_seed(counter_offset));
            break;
        default:
            fused_dropout_add_kernel<scalar_t, accscalar_t, uint64_t, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(input_info, input_add_info, ret_info, mask_info, nelem, proba, next_philox_seed(counter_offset));
      }
   });
  }
  THCudaCheck(cudaGetLastError());
  return {ret, mask};
}

at::Tensor fused_dropout_add_backward_cuda(const at::Tensor& grad, const at::Tensor& mask, double scale){
   at::Tensor ret = at::empty_like(grad);
   AT_CHECK(mask.type().scalarType() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
   AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "masked_scale", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      accscalar_t proba = (accscalar_t)(scale);
    masked_scale_kernel<scalar_t, accscalar_t>(ret, grad, mask, proba);
  });
  return ret;
}
