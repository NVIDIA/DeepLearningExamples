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

#include "fastertransformer/cuda/topk_kernels.cuh"
#include "cub/cub.cuh"

namespace fastertransformer
{

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void batch_topK_kernel(const int* __restrict topk_tmp_id_buf,
                        const T* __restrict topk_tmp_val_buf,
                        int* __restrict id_buf,
                        T* __restrict val_buf)
{
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    if (thread_id == 0)
    {
        for(int i = 0; i < MAX_K; ++i)
        {
            partial.p[i] = -1;
            partial.u[i] = -FLT_MAX;
        }

        int index = block_id * MAX_K * MAX_K;
        for(int i = 0; i < MAX_K * MAX_K; i++)
        {
            partial.insert( (T)topk_tmp_val_buf[index + i], topk_tmp_id_buf[index + i]);
        }

        index = block_id * MAX_K;
        for(int i = 0; i < MAX_K; i++)
        {
            id_buf[index + i] = partial.p[i];
            val_buf[index + i] = partial.u[i];
        }
    }
}


template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void batch_topk_kernel(
    const int * __restrict x,
    const T * __restrict y,
    int * __restrict z,
    T * __restrict v,
    int V,
    int K,
    T diversity_rate)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x, y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        int i = elem_id % K;
        T elem = y[elem_id] + diversity_rate * (T) i;
        int elem_idx = elem_id; //x[elem_id];
        partial.insert(elem, elem_idx);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        
        for(int i = 0; i < MAX_K; ++i)
        {
            if (i < K)
            {
                z[i] = x[total.p[i]];
                v[i] = y[total.p[i]];
            }
        }
    }
}

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<typename T, int MAX_K>
struct TopKMD
{
    MD md;
    TopK<T, MAX_K> topk;
};

template<typename T, int MAX_K>
__device__ __forceinline__ TopKMD<T, MAX_K> reduce_topk_md_op(const TopKMD<T, MAX_K>& a, const TopKMD<T, MAX_K>& b)
{
    TopKMD<T, MAX_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void beam_online_softmax_topk_kernel(
    const T * __restrict x,
    const float * __restrict b,
    const T * __restrict c,
    const bool  * __restrict finished,
    int * __restrict z,
    T * __restrict v,
    int V,
    int K,
    int E)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<float, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKMD<float, MAX_K> partial;
    bool finish = finished[vector_id];
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -FLT_MAX;
    }
    partial.md.m = -FLT_MAX;
    partial.md.d = 0.0F;

    if (finish)
    {
        for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == E) ? FLT_MAX : -FLT_MAX;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
            //if (elem_id > THREADBLOCK_SIZE * MAX_K && (elem_id == E)) break;
        }
    }
    else
    {
        for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        {
            float elem = x[elem_id] + b[elem_id];
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

    TopKMD<float, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<float, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        c += vector_id;
        
        //float d_total_inverse = __fdividef(1.0F, total.md.d);
        float d_total_log = logf(total.md.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            //float val = __expf(total.topk.u[i] - total.md.m) * d_total_inverse;
            float val = total.topk.u[i] - total.md.m - d_total_log;
            if (i < K)
            {
                z[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
                v[i] = val + c[0];
            }
        }
    }
}

template<typename T, int ITEMS_PER_THREAD, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void beam_online_softmax_topk_stage1_kernel(
    const T * __restrict x,
    const float * __restrict b,
    const bool  * __restrict finished,
    float * __restrict t,
    int V,
    int K,
    int E)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;

    // one will have multiple sections per V
    const int v_local = (V + gridDim.y - 1) / gridDim.y;
    const int section_start = v_local * blockIdx.y;
    int section_end = section_start + v_local;
    section_end = (section_end > V)? V : section_end;

    // reposition x to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float buf_s[PACKED_TOP_KMD_SIZE]; // save intermediate result

    TopKMD<T, MAX_K> partial;
    bool finish = finished[vector_id];
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -FLT_MAX;
    }
    partial.md.m = -FLT_MAX;
    partial.md.d = 0.0F;

    if (finish)
    {
        for(int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            float elem = (elem_id == E) ? FLT_MAX : -FLT_MAX;
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }
    else
    {
        for(int elem_id = section_start + thread_id; elem_id < section_end; elem_id += THREADBLOCK_SIZE)
        {
            T elem = x[elem_id] + b[elem_id];
            MD new_elem{elem, 1.0F};
            partial.md = reduce_md_op(partial.md, new_elem);
            partial.topk.insert(elem, elem_id);
        }
    }

    TopKMD<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);

    if (thread_id == 0)
    {
        for (int i = 0; i < K; i++)
        {
            reinterpret_cast<int *>(buf_s)[i] = total.topk.p[i] + vector_id * V; // faster transformer needs absolute id
            buf_s[MAX_K + i] = total.topk.u[i];
        }
        buf_s[2 * MAX_K] = total.md.d;
        buf_s[2 * MAX_K + 1] = total.md.m;
    }
    __syncthreads();
    if (threadIdx.x < PACKED_TOP_KMD_SIZE)
    {
        t[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y + blockIdx.y * PACKED_TOP_KMD_SIZE + threadIdx.x] = buf_s[threadIdx.x];
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void beam_online_softmax_topk_stage2_kernel(
    const float * __restrict x,
    const T * __restrict c,
    int * __restrict z,
    T * __restrict v,
    int K,
    int parts_per_beam)
{
    const int vector_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int PACKED_TOP_KMD_SIZE = 2 * MAX_K + 2;
    
    extern __shared__ char buf_s_[]; // intermediate result
    float * buf_s = reinterpret_cast<float *>(buf_s_);
    //__shared__ float buf_s[PACKED_TOP_KMD_SIZE * THREADBLOCK_SIZE]; // intermediate result

    typedef cub::BlockReduce<TopKMD<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    x += vector_id * PACKED_TOP_KMD_SIZE * parts_per_beam;

    TopKMD<T, MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.topk.p[i] = -1;
        partial.topk.u[i] = -FLT_MAX;
    }
    partial.md.m = -FLT_MAX;
    partial.md.d = 0.0F;

    // load and unpack into registers through smem
    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * parts_per_beam; idx += THREADBLOCK_SIZE)
    {
        buf_s[idx] = x[idx];
    }
    __syncthreads();

    if (threadIdx.x < parts_per_beam)
    {
        float * b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
        for (int i = 0; i < K; i++)
        {
            partial.topk.p[i] = reinterpret_cast<int *>(b_s)[i];
            partial.topk.u[i] = b_s[MAX_K + i];
        }
        partial.md.d = b_s[2 * MAX_K];
        partial.md.m = b_s[2 * MAX_K + 1];
    }
    __syncthreads();

    TopKMD<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<T, MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        c += vector_id;
        
        float d_total_log = logf(total.md.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            float val = total.topk.u[i] - total.md.m - d_total_log;
            if (i < K)
            {
                z[i] = total.topk.p[i];
                v[i] = val + c[0]; 
            }
        }
    }
}

template<typename T, int MAX_K>
void beam_online_softmax_topk_stage2_kernelLauncher(
    const float * temp_storage,
    const T * cum_log_probs,
    int * ids,
    T * vals,
    int batch_size,
    int beam_width,
    int parts_per_beam,
    cudaStream_t stream)
{
    // might rewrite beam_online_softmax_topk_stage2_kernel no to depend on constant block size
    // in oreder to reduce compilation time
    int smem_stage2_size = parts_per_beam * (2 * MAX_K + 2) * sizeof(T);

    if (parts_per_beam <= 32)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 32>
        <<<batch_size * beam_width, 32, smem_stage2_size, stream>>>
                (temp_storage, cum_log_probs, ids, vals,
                 beam_width, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 64)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 64>
        <<<batch_size * beam_width, 64, smem_stage2_size, stream>>>
                (temp_storage, cum_log_probs, ids, vals,
                 beam_width, parts_per_beam);
        return;
    }
    if (parts_per_beam <= 128)
    {
        beam_online_softmax_topk_stage2_kernel<T, MAX_K, 128>
        <<<batch_size * beam_width, 128, smem_stage2_size, stream>>>
                (temp_storage, cum_log_probs, ids, vals,
                 beam_width, parts_per_beam);
        return;
    }
    assert(0);
}

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T* log_probs,
                                 const float* bias,
                                 const bool* finished,
                                 T* cum_log_probs,
                                 int* ids,
                                 void* temp_storage,
                                 const int temp_storage_size,
                                 const int batch_size, 
                                 const int beam_width, 
                                 const int vocab_size, 
                                 const int end_id,
                                 T diversity_rate,
                                 cudaStream_t stream)
{
    const int items_per_thread = 1;
    const int block_sz = SMALL_TOP_K_SOFTMAX_THREADBLOCK_SIZE;

    assert(temp_storage_size % 2 == 0);
    assert(temp_storage_size >= 2 * batch_size * beam_width * beam_width);

    int* topk_tmp_id_buf = reinterpret_cast<int *>(temp_storage);
    T* topk_tmp_val_buf = reinterpret_cast<T *>(topk_tmp_id_buf + batch_size * beam_width * beam_width);
    float* tmp_buffer = reinterpret_cast<float *>(topk_tmp_val_buf + batch_size * beam_width * beam_width);

#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
    int voc_parts = 4;
    if (batch_size * beam_width < 256)
    {
        voc_parts = (256 + batch_size * beam_width - 1) / (batch_size * beam_width);
        voc_parts = std::min(128, voc_parts); // we implment up to 128
    }
    dim3 grid(batch_size * beam_width, voc_parts);
    beam_online_softmax_topk_stage1_kernel<T, items_per_thread, MAX_K, block_sz>
                            <<<grid, block_sz,0,stream>>>
                            (log_probs, bias, finished, tmp_buffer,
                            vocab_size, beam_width, end_id);
#endif
    if (beam_width > 1)
    {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K>
                                (tmp_buffer, cum_log_probs, topk_tmp_id_buf, topk_tmp_val_buf,
                                 batch_size, beam_width, voc_parts, stream);
#else
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
                        <<<batch_size * beam_width, block_sz, 0, stream>>>
                                (log_probs, bias, cum_log_probs, finished, topk_tmp_id_buf, 
                                topk_tmp_val_buf, vocab_size, beam_width, end_id);
#endif
#if 0
         // wrong result with diversity_rate != 0.f
         batch_topK_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>
                                (topk_tmp_id_buf, topk_tmp_val_buf, ids, cum_log_probs);
#else
         batch_topk_kernel<T, MAX_K, 32><<<batch_size, 32, 0, stream>>>
                                (topk_tmp_id_buf, topk_tmp_val_buf,
                                ids, cum_log_probs, beam_width * beam_width, beam_width, diversity_rate);
#endif
    }
    else
    {
#ifdef DO_SPLIT_SMALL_TOP_K_SOFTMAX
        beam_online_softmax_topk_stage2_kernelLauncher<T, MAX_K>
                                (tmp_buffer, cum_log_probs, ids, cum_log_probs,
                                batch_size, beam_width, voc_parts, stream);
#else
        beam_online_softmax_topk_kernel<T, items_per_thread, MAX_K, block_sz>
                            <<<batch_size * beam_width, block_sz, 0, stream>>>
                                   (log_probs, bias, cum_log_probs, finished, ids, 
                                    cum_log_probs, vocab_size, beam_width, end_id);
#endif
    }
}

#define CASE_K(K) \
  case K : \
    topK_softMax_kernelLauncher<T, K> \
      (log_probs, bias, finished, cum_log_probs, ids, temp_storage, temp_storage_size, \
      batch_size, beam_width, vocab_size, end_id, diversity_rate, stream); \
  break; \

template <typename T>
void topK_softMax(const T* log_probs, 
                  const float* bias, 
                  const bool* finished, 
                  T* cum_log_probs,
                  int* ids,
                  void* temp_storage,
                  DecodingBeamsearchArguments args,
                  cudaStream_t stream)
{
    const int temp_storage_size = args.temp_storage_size_;
    const int batch_size = args.batch_size_;
    const int beam_width = args.beam_width_;
    const int vocab_size = args.vocab_size_;
    const int end_id = args.end_id_;
    const T diversity_rate = args.beam_search_diversity_rate_;

    switch(beam_width)
    {
        CASE_K(1);
        CASE_K(2);
        CASE_K(4);
        default :
            printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
            exit(0);
            break;
    }
}
#undef CASE_K

template void topK_softMax<float>(const float* log_probs, 
                                const float* bias, 
                                const bool* finished, 
                                float* cum_log_probs,
                                int* ids, 
                                void * tmp_storage,
                                DecodingBeamsearchArguments args,
                                cudaStream_t stream);
} // end of namespace fastertransformer