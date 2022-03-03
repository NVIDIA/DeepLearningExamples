#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>
#include <iostream>

#include <assert.h>
#include <type_traits>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

typedef enum{
  MOMENT_MODE_0   =0, // L2 regularization mode
  MOMENT_MODE_1   =1  // Decoupled weight decay mode
} adamMode_t;

using MATH_T = float;

#include<cstdio>
template<typename T, typename param_t>
struct LAMBStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const float beta3,
    const int* step_ptr,
    const int bias_correction,
    const float epsilon,
    adamMode_t mode,
    const float decay,
    const float* global_grad_norm,
    const float* max_global_grad_norm,
    const float* found_inf,
    const float* inv_scale)
  {
    if (*noop_gmem) {
      return;
    }

    float beta1_correction = 1.0f;
    float beta2_correction = 1.0f;
    if (bias_correction == 1) {
      int step = *step_ptr;
      beta1_correction = 1 - std::pow(beta1, step);
      beta2_correction = 1 - std::pow(beta2, step);
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float clipped_global_grad_norm = (*global_grad_norm) > (*max_global_grad_norm) ? (*global_grad_norm) / (*max_global_grad_norm) : 1.0f;

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    param_t* p = (param_t*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    param_t* m = (param_t*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    param_t* v = (param_t*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    MATH_T r_g[ILP];
    MATH_T r_p[ILP];
    MATH_T r_m[ILP];
    MATH_T r_v[ILP];
    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 &&
       chunk_size % ILP == 0 &&
       is_aligned(g) &&
       is_aligned(p) &&
       is_aligned(m) &&
       is_aligned(v))
    {
      T l_g[ILP];
      param_t l_p[ILP];
      param_t l_m[ILP];
      param_t l_v[ILP];
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(l_g, g, 0, i_start);
        if (decay != 0)
          load_store(l_p, p, 0, i_start);
        load_store(l_m, m, 0, i_start);
        load_store(l_v, v, 0, i_start);
        // unpack
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_g[ii] = l_g[ii] * (*inv_scale);
          if (decay == 0) {
            r_p[ii] = MATH_T(0);
          }
          else {
            r_p[ii] = l_p[ii];
          }
          r_m[ii] = l_m[ii];
          r_v[ii] = l_v[ii];
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          if (mode == MOMENT_MODE_0) {
            MATH_T scaled_grad = r_g[ii] / clipped_global_grad_norm;
            // L2 on scaled grad
            scaled_grad = scaled_grad + decay*r_p[ii];
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = next_m_unbiased / denom;
          }
          else {
            MATH_T scaled_grad = r_g[ii] / clipped_global_grad_norm;
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased/denom) + (decay*r_p[ii]);
          }
          // printf("(%d %d) r_g: %f, r_p: %f, r_m: %f, r_v: %f\n", i_start, ii, r_g[ii], r_p[ii], r_m[ii], r_v[ii]);
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          l_p[ii] = r_p[ii];
          // Difference from APEX's LAMB kernel. `g` and `p` can be different dtypes.
          l_g[ii] = r_p[ii];
          l_m[ii] = r_m[ii];
          l_v[ii] = r_v[ii];
        }
        // store
        load_store(g, l_g, i_start, 0);
        load_store(m, l_m, i_start, 0);
        load_store(v, l_v, i_start, 0);
      }
    }
    else
    {
      // see note in multi_tensor_scale_kernel.cu
      for(int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP)
      {
        MATH_T r_g[ILP];
        MATH_T r_p[ILP];
        MATH_T r_m[ILP];
        MATH_T r_v[ILP];
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_g[ii] = g[i] * (*inv_scale);
            // special ?optimization? for lamb stage 1
            if (decay == 0) {
              r_p[ii] = MATH_T(0);
            }
            else {
              r_p[ii] = p[i];
            }
            r_m[ii] = m[i];
            r_v[ii] = v[i];
          } else {
            r_g[ii] = MATH_T(0);
            r_p[ii] = MATH_T(0);
            r_m[ii] = MATH_T(0);
            r_v[ii] = MATH_T(0);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          if (mode == MOMENT_MODE_0) {
            MATH_T scaled_grad = r_g[ii] / clipped_global_grad_norm;
            // L2 on scaled grad
            scaled_grad = scaled_grad + decay*r_p[ii];
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = next_m_unbiased / denom;
          }
          else {
            MATH_T scaled_grad = r_g[ii] / clipped_global_grad_norm;
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased/denom) + (decay*r_p[ii]);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            g[i] = r_p[ii];
            m[i] = r_m[ii];
            v[i] = r_v[ii];
          }
        }
      }
    }
  }
};

// Step 2 reads in 'update' value and per-tensor param_norm and update_norm.
// It computes new parameter value.
// N == 2: FP32 params, no master params
// N == 3: FP16 params, FP32 master params.
template<typename T, int N, typename param_t>
struct LAMBStage2Functor
{
  static_assert((N == 2 && std::is_same<T, param_t>::value) || (N == 3 && std::is_same<param_t, float>::value), "");
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<N>& tl,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_norm,
    const float* learning_rate,
    const float decay,
    bool use_nvlamb)
  {
    if (*noop_gmem) {
      return;
    }

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    MATH_T ratio = *learning_rate;
    // nvlamb: apply adaptive learning rate to all parameters
    // otherwise, only apply to those with non-zero weight decay
    if (use_nvlamb || (decay != 0.0))
    {
      float param_norm = per_tensor_param_norm[tensor_num];
      float update_norm = per_tensor_update_norm[tensor_num];
      ratio = (update_norm != 0.0f && param_norm != 0.0f) ? *learning_rate * (param_norm / update_norm) : *learning_rate;
    }

    T* update = (T*)tl.addresses[0][tensor_loc];
    update += chunk_idx*chunk_size;

    param_t* p = (param_t*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    T* out_p;
    if (N == 3) {
      out_p = (T*)tl.addresses[2][tensor_loc];
      out_p += chunk_idx*chunk_size;
    }

    n -= chunk_idx*chunk_size;

    // to make things simple, we put aligned case in a different code path
    bool can_use_aligned_path = n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(p) && is_aligned(update);
    if (N == 3) {
      can_use_aligned_path = can_use_aligned_path && is_aligned(out_p);
    }
    if(can_use_aligned_path)
    {
      param_t r_p[ILP];
      T r_update[ILP];
      T r_out_p[ILP];
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_p, p, 0, i_start);
        load_store(r_update, update, 0, i_start);
        if (N == 3) {
          load_store(r_out_p, out_p, 0, i_start);
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_p[ii] = static_cast<MATH_T>(r_p[ii]) - (ratio * static_cast<MATH_T>(r_update[ii]));
          if (N == 3) {
            r_out_p[ii] = r_p[ii];
          }
        }
        load_store(p, r_p, i_start, 0);
        if (N == 3) {
          load_store(out_p, r_out_p, i_start, 0);
        }
      }
    }
    else
    {
      for(int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP)
      {
        MATH_T r_p[ILP];
        MATH_T r_update[ILP];
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_p[ii] = p[i];
            r_update[ii] = update[i];
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_p[ii] = r_p[ii] - (ratio * r_update[ii]);
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            p[i] = r_p[ii];
            if (N == 3) {
              out_p[i] = p[i];
            }
          }
        }
      }
    }
  }
};


void multi_tensor_lamb_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const at::Tensor step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  at::Tensor global_grad_norm,
  at::Tensor max_grad_norm,
  at::optional<bool> use_nvlamb_python,
  at::Tensor found_inf,
  at::Tensor inv_scale)
{
  // n_tensors == 5: FP16 model params & FP32 master params
  // n_tensors == 4: FP32 model params & NO FP32 master params
  const auto n_tensors = tensor_lists.size();
  assert(n_tensors == 4 || n_tensors == 5);
  using namespace at;

  bool use_nvlamb = use_nvlamb_python.has_value() ? use_nvlamb_python.value() : false;

  // note(mkozuki): move bias handling below to functor
  // Handle bias correction mode
  // float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  // if (bias_correction == 1) {
  //   bias_correction1 = 1 - std::pow(beta1, step);
  //   bias_correction2 = 1 - std::pow(beta2, step);
  // }

  // Handle grad averaging mode
  float beta3 = 1.0f;
  if (grad_averaging == 1) beta3 = 1 - beta1;

  std::vector<std::vector<at::Tensor>> stage1_tensor_lists(tensor_lists.begin(), tensor_lists.begin() + 4);
  std::vector<std::vector<at::Tensor>> grad_list(tensor_lists.begin(), tensor_lists.begin()+1);
  std::vector<std::vector<at::Tensor>> param_list(tensor_lists.begin()+1, tensor_lists.begin()+2);

  // Compute per tensor param norm
  auto param_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, param_list, true);

  // We now in-place modify grad to store update before compute its norm
  // Generally this is not a issue since people modify grad in step() method all the time
  // We can also grab list of empty tensor to avoid this, but I'd like to save space/cpu code
  if (n_tensors == 4) {
    DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_1",
        multi_tensor_apply<4>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          stage1_tensor_lists,
          LAMBStage1Functor<scalar_t_0, scalar_t_0>(),
          beta1,
          beta2,
          beta3, // 1-beta1 or 1 depends on averaging mode
          // bias_correction1,
          // bias_correction2,
          step.data_ptr<int>(),
          bias_correction,
          epsilon,
          (adamMode_t) mode,
          weight_decay,
          global_grad_norm.data_ptr<float>(),
          max_grad_norm.data_ptr<float>(),
          found_inf.data_ptr<float>(),
          inv_scale.data_ptr<float>()); )
  } else {
    DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_1",
        multi_tensor_apply<4>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          stage1_tensor_lists,
          LAMBStage1Functor<scalar_t_0, float>(),
          beta1,
          beta2,
          beta3, // 1-beta1 or 1 depends on averaging mode
          // bias_correction1,
          // bias_correction2,
          step.data_ptr<int>(),
          bias_correction,
          epsilon,
          (adamMode_t) mode,
          weight_decay,
          global_grad_norm.data_ptr<float>(),
          max_grad_norm.data_ptr<float>(),
          found_inf.data_ptr<float>(),
          inv_scale.data_ptr<float>()); )
  }

  // Compute update norms
  auto update_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, grad_list, true);

  std::vector<std::vector<at::Tensor>> grad_param_list(tensor_lists.begin(), tensor_lists.begin()+2);
  if (n_tensors == 4) {
    DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_2",
        multi_tensor_apply<2>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          grad_param_list,
          LAMBStage2Functor<scalar_t_0, 2, scalar_t_0>(),
          std::get<1>(param_norm_tuple).data_ptr<float>(),
          std::get<1>(update_norm_tuple).data_ptr<float>(),
          lr.data_ptr<float>(),
      weight_decay,
      use_nvlamb); )
  } else {
    grad_param_list.push_back(tensor_lists[4]);
    DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_2",
        multi_tensor_apply<3>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          grad_param_list,
          LAMBStage2Functor<scalar_t_0, 3, float>(),
          std::get<1>(param_norm_tuple).data_ptr<float>(),
          std::get<1>(update_norm_tuple).data_ptr<float>(),
          lr.data_ptr<float>(),
      weight_decay,
      use_nvlamb); )
  }
  AT_CUDA_CHECK(cudaGetLastError());

}
