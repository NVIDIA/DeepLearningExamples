// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
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

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

thread_local int multiProcessorCount=0;

#define ASSERT_UINT4_ALIGNED(PTR)                                              \
  AT_ASSERTM(is_aligned<uint4>(PTR), "Tensor " #PTR " is not uint4 aligned")

template <class T> bool is_aligned(const void *ptr) noexcept {
  auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  return !(iptr % alignof(T));
}

template <bool SMOOTHING, int ILP, typename scalar_t, typename labelscalar_t,
          typename accscalar_t, typename outscalar_t>
__global__ void focal_loss_forward_cuda_kernel(
    outscalar_t *loss, scalar_t *partial_grad,
    const scalar_t *__restrict__ cls_output,
    const labelscalar_t *__restrict__ cls_targets_at_level,
    const float *__restrict__ num_positives_sum, const int64_t num_examples,
    const int64_t num_classes, const int64_t num_real_classes,
    const float alpha, const float gamma, const float smoothing_factor) {
  extern __shared__ unsigned char shm[];
  accscalar_t *loss_shm = reinterpret_cast<accscalar_t *>(shm);
  loss_shm[threadIdx.x] = 0;
  accscalar_t loss_acc = 0;

  accscalar_t one = accscalar_t(1.0);
  accscalar_t K = accscalar_t(2.0);
  accscalar_t normalizer = one / static_cast<accscalar_t>(num_positives_sum[0]);
  accscalar_t nn_norm, np_norm, pn_norm, pp_norm;

  // *_norm is used for label smoothing only
  if (SMOOTHING) {
    nn_norm = one - smoothing_factor / K;
    np_norm = smoothing_factor / K;
    pn_norm = smoothing_factor - smoothing_factor / K;
    pp_norm = one - smoothing_factor + smoothing_factor / K;
  }

  uint4 p_vec, grad_vec;

  // Accumulate loss on each thread
  for (int64_t i = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;
       i < num_examples * num_classes; i += gridDim.x * blockDim.x * ILP) {
    int64_t idy = i / num_classes;
    labelscalar_t y = cls_targets_at_level[idy];
    int64_t base_yid = i % num_classes;

    int64_t pos_idx = idy * num_classes + y;
    p_vec = *(uint4 *)&cls_output[i];

    // Skip ignored matches
    if (y == -2) {
#pragma unroll
      for (int j = 0; j < ILP; j++) {
        *((scalar_t *)(&grad_vec) + j) = 0;
      }
      *(uint4 *)&partial_grad[i] = grad_vec;
      continue;
    }

#pragma unroll
    for (int j = 0; j < ILP; j++) {
      // Skip the pad classes
      if (base_yid + j >= num_real_classes) {
        *((scalar_t *)(&grad_vec) + j) = 0;
        continue;
      }

      accscalar_t p = static_cast<accscalar_t>(*((scalar_t *)(&p_vec) + j));
      accscalar_t exp_np = ::exp(-p);
      accscalar_t exp_pp = ::exp(p);
      accscalar_t sigma = one / (one + exp_np);
      accscalar_t logee = (p >= 0) ? exp_np : exp_pp;
      accscalar_t addee = (p >= 0) ? 0 : -p;
      accscalar_t off_a = addee + ::log(one + logee);

      // Negative matches
      accscalar_t base = SMOOTHING ? nn_norm * p : p;
      accscalar_t off_b = (SMOOTHING ? np_norm : 0) - sigma;
      accscalar_t coeff_f1 = one - alpha;
      accscalar_t coeff_f2 = sigma;
      accscalar_t coeff_b1 = gamma;
      accscalar_t coeff_b2 = one - sigma;

      // Positive matches
      if (y >= 0 && (i + j == pos_idx)) {
        base = SMOOTHING ? pn_norm * p : 0;
        off_b = (SMOOTHING ? pp_norm : one) - sigma;
        coeff_f1 = alpha;
        coeff_f2 = one - sigma;
        coeff_b1 = -gamma;
        coeff_b2 = sigma;
      }

      accscalar_t coeff_f = coeff_f1 * ::pow(coeff_f2, gamma);
      accscalar_t coeff_b = coeff_b1 * coeff_b2;

      accscalar_t loss_t = coeff_f * (base + off_a);
      accscalar_t grad = coeff_f * (coeff_b * (base + off_a) - off_b);

      // Delay the normalize of partial gradient by num_positives_sum to back
      // propagation because scalar_t reduces precision. Focal loss is very
      // sensitive to the small gradient. No worry on overflow here since
      // gradient has relative smaller range than input.
      loss_acc += loss_t;
      *((scalar_t *)(&grad_vec) + j) = static_cast<scalar_t>(grad);
    }

    // This can't ensure to generate stg.128 and may be two stg.64.
    *(uint4 *)&partial_grad[i] = grad_vec;
  }
  loss_shm[threadIdx.x] = loss_acc;

  // Intra-CTA reduction
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      loss_shm[threadIdx.x] += loss_shm[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Inter-CTA reduction
  if (threadIdx.x == 0) {
    loss_acc = loss_shm[0] * normalizer;
    atomicAdd(loss, loss_acc);
  }
}

template <int ILP, typename scalar_t, typename accscalar_t,
          typename outscalar_t>
__global__ void focal_loss_backward_cuda_kernel(
    scalar_t *partial_grad, const outscalar_t *__restrict__ grad_output,
    const float *__restrict__ num_positives_sum, const uint64_t numel) {
  int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ILP;

  accscalar_t normalizer = static_cast<accscalar_t>(grad_output[0]) /
                           static_cast<accscalar_t>(num_positives_sum[0]);

  // The input is enforced to pad to use vector load, thus there's no need to
  // check whether the last element of ILP can out of bound.
  if (idx >= numel)
    return;

  uint4 grad_vec;
  grad_vec = *(uint4 *)&partial_grad[idx];
#pragma unroll(ILP)
  for (int i = 0; i < ILP; i++) {
    auto grad = static_cast<accscalar_t>(*((scalar_t *)(&grad_vec) + i));
    grad *= normalizer;
    *((scalar_t *)(&grad_vec) + i) = static_cast<scalar_t>(grad);
  }
  *(uint4 *)&partial_grad[idx] = grad_vec;
}

std::vector<at::Tensor> focal_loss_forward_cuda(
    const at::Tensor &cls_output, const at::Tensor &cls_targets_at_level,
    const at::Tensor &num_positives_sum, const int64_t num_real_classes,
    const float alpha, const float gamma, const float smoothing_factor) {
  // Checks required for correctness
  AT_ASSERTM(cls_output.size(-1) >= num_real_classes,
             "Incorrect number of real classes.");
  AT_ASSERTM(cls_targets_at_level.scalar_type() == at::kLong,
             "Invalid label type.");
  AT_ASSERTM(
      (num_positives_sum.numel() == 1) &&
          (num_positives_sum.scalar_type() == at::kFloat),
      "Expect num_positives_sum to be a float32 tensor with only one element.");
  AT_ASSERTM(cls_output.dim() == cls_targets_at_level.dim() + 1,
             "Mis-matched dimensions between class output and label.");
  for (int64_t i = 0; i < cls_targets_at_level.dim(); i++)
    AT_ASSERTM(cls_output.size(i) == cls_targets_at_level.size(i),
               "Mis-matched shape between class output and label.");

  // Checks required for better performance
  const int ILP = sizeof(uint4) / cls_output.element_size();
  ASSERT_UINT4_ALIGNED(cls_output.data_ptr());
  AT_ASSERTM(cls_output.size(-1) % ILP == 0,
             "Pad number of classes first to take advantage of 128 bit load.");
  AT_ASSERTM(num_real_classes >= ILP, "Too few classes.");

  int64_t num_classes = cls_output.size(-1);
  int64_t num_examples = cls_output.numel() / num_classes;
  at::Tensor loss = at::zeros({}, cls_output.options().dtype(at::kFloat));

  // Compute the incompelete gradient during fprop since most of the heavy
  // functions of bprop are the same as fprop, thus trade memory for compute
  // helps with focal loss.
  at::Tensor partial_grad = at::empty_like(cls_output);

  // The grid contains 2 CTA per SM, each CTA loop on input with stride till the
  // last item.
  if (multiProcessorCount == 0) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, at::cuda::current_device());
    multiProcessorCount = props.multiProcessorCount;
  }
  dim3 block(512);
  dim3 grid(2 * multiProcessorCount);

  // Specialize on label smoothing or not to reduce redundant operations
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (smoothing_factor == 0.0f) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_fprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          const int ILP = sizeof(uint4) / sizeof(scalar_t);
          focal_loss_forward_cuda_kernel<false, ILP, scalar_t, labelscalar_t,
                                         accscalar_t, outscalar_t>
              <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                  loss.data_ptr<outscalar_t>(),
                  partial_grad.data_ptr<scalar_t>(),
                  cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum.data_ptr<float>(), num_examples,
                  num_classes, num_real_classes, alpha, gamma,
                  smoothing_factor);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        cls_output.scalar_type(), "focal_loss_fprop", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          using labelscalar_t = int64_t;
          using outscalar_t = float;
          const int ILP = sizeof(uint4) / sizeof(scalar_t);
          focal_loss_forward_cuda_kernel<true, ILP, scalar_t, labelscalar_t,
                                         accscalar_t, outscalar_t>
              <<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
                  loss.data_ptr<outscalar_t>(),
                  partial_grad.data_ptr<scalar_t>(),
                  cls_output.data_ptr<scalar_t>(),
                  cls_targets_at_level.data_ptr<labelscalar_t>(),
                  num_positives_sum.data_ptr<float>(), num_examples,
                  num_classes, num_real_classes, alpha, gamma,
                  smoothing_factor);
        });
  }

  THCudaCheck(cudaGetLastError());
  return {loss, partial_grad};
}

at::Tensor focal_loss_backward_cuda(const at::Tensor &grad_output,
                                    const at::Tensor &partial_grad,
                                    const at::Tensor &num_positives_sum) {
  // Each thread process ILP elements
  const int ILP = sizeof(uint4) / partial_grad.element_size();
  dim3 block(512);
  dim3 grid((partial_grad.numel() + block.x * ILP - 1) / (block.x * ILP));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      partial_grad.scalar_type(), "focal_loss_bprop", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        using outscalar_t = float;
        const int ILP = sizeof(uint4) / sizeof(scalar_t);
        focal_loss_backward_cuda_kernel<ILP, scalar_t, accscalar_t, outscalar_t>
            <<<grid, block, 0, stream>>>(partial_grad.data_ptr<scalar_t>(),
                                         grad_output.data_ptr<outscalar_t>(),
                                         num_positives_sum.data_ptr<float>(),
                                         partial_grad.numel());
      });

  THCudaCheck(cudaGetLastError());
  return partial_grad;
}
