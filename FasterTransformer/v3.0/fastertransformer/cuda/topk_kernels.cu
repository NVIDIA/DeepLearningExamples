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
void beam_topK_kernel(const T* log_probs, 
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        const int vocab_size,
                        T diversity_rate)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    TopK<T, MAX_K> partial;
    
    #pragma unroll
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    #pragma unroll
    for(int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + block_id * vocab_size;        
        partial.insert(log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        int index = block_id * MAX_K;
        
        #pragma unroll
        for(int i = 0; i < MAX_K; ++i)
        {
            topk_tmp_id_buf[index + i] = total.p[i];
            topk_tmp_val_buf[index + i] = total.u[i] + diversity_rate * (T)i;
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void batch_topK_kernel(int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* id_buf)
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
        }
    }
}

template<typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__
void batch_topK_kernel_v2(int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* id_buf)
{
    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    TopK<T, MAX_K> partial;
    #pragma unroll
    for(int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -FLT_MAX;
    }

    int ite = MAX_K * MAX_K / THREADBLOCK_SIZE;
    #pragma unroll
    for(int i = 0; i < ite; i++)
    {
        int index = bid * MAX_K * MAX_K + i * THREADBLOCK_SIZE + tid;
        partial.insert( (T)topk_tmp_val_buf[index], topk_tmp_id_buf[index]);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if(tid == 0)
    {
        #pragma unroll
        for(int i = 0; i < MAX_K; i++)
            id_buf[bid * MAX_K + i] = total.p[i];
    }
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_1_opt3(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const int k,
    const int vocab_size
)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM_; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM_; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size; 
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM_ * k + block_lane * k;
    TopK_2<T> partial;

    for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index]; 
    }


    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage_2_opt3(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM_; 
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);
    
    TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE_)
        {
            partial.insert(s_val[i], i);
        }
    
        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);
    
        if(tid == 0) 
        {
            s_id[ite] = total.p;
            s_val[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
    if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_1_opt2_general(
    const T* __restrict log_probs,
    T* tmp_log_probs,
    int* topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    const int k,
    const int vocab_size
)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_id = bid / BLOCKS_PER_BEAM; // row id for log_probs
    const int block_lane = bid % BLOCKS_PER_BEAM; // block id for a beam 
    const int tmp_log_buf_index = row_id * vocab_size; 
    const int tmp_topk_buf_index = row_id * BLOCKS_PER_BEAM * k + block_lane * k;
    TopK_2<T> partial;

    for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
    {
        int index = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index]; 
    }


    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_BEAM)
        {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0)
        {
            const int index = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index] = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_BEAM>
__global__ void topk_stage_2_opt2_general(
    const int* __restrict topk_tmp_id_buf,
    T* topk_tmp_val_buf,
    int* ids,
    const int k)
{
    const int size = k * k * BLOCKS_PER_BEAM; 
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;

    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    extern __shared__ char array[];
    T *s_val = topk_tmp_val_buf + batch_id * size;
    int *s_id = (int*)(array);
    
    TopK_2<T> partial;

    for(int ite = 0; ite < k; ite++)
    {
        partial.init();
        #pragma unroll
        for(int i = tid; i < size; i+= BLOCK_SIZE)
        {
            partial.insert(s_val[i], i);
        }
    
        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);
    
        if(tid == 0) 
        {
            s_id[ite] = total.p;
            s_val[total.p] = -FLT_MAX;
        }
        __syncthreads();
    }
    if(tid < k) ids[batch_id * k + tid] = topk_tmp_id_buf[batch_id * size + s_id[tid]];
}

#define CASE_K_DIV(K,BLOCK_SIZE_1, BLOCK_SIZE_2) \
  case K: \
    beam_topK_kernel<T, K, BLOCK_SIZE_2><<<batch_size * beam_width, BLOCK_SIZE_2, 0, stream>>>(log_probs, \
        topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, diversity_rate); \
    if (K < 10) \
      batch_topK_kernel<T, K, BLOCK_SIZE_1><<<batch_size, BLOCK_SIZE_1, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids); \
    else \
      batch_topK_kernel_v2<T, K, 32><<<batch_size, 32, 0, stream>>>(topk_tmp_id_buf, topk_tmp_val_buf, ids); \
  break; \

#define CASE_K(K,BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_) \
  case K: \
    topk_stage_1_opt3<float, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<batch_size * K * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>( \
        log_probs, \
        temp_log_probs, \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        beam_width, vocab_size); \
    topk_stage_2_opt3<float, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_><<<batch_size, BLOCK_SIZE_2_, K * sizeof(int), stream>>>( \
        topk_tmp_id_buf, \
        topk_tmp_val_buf, \
        ids, \
        beam_width); \
  break; \

template <typename T>
void topK_kernelLauncher(T* log_probs,
                        T* temp_log_probs,
                        int* topk_tmp_id_buf,
                        T* topk_tmp_val_buf,
                        int* ids,
                        DecodingBeamsearchArguments args,
                        cudaStream_t stream)
{
    const int batch_size = args.batch_size_;
    const int beam_width = args.beam_width_;
    const int vocab_size = args.vocab_size_;
    const T diversity_rate = args.beam_search_diversity_rate_;
    if(diversity_rate == 0.0f)
    {
        switch(beam_width)
        {
            CASE_K(1,128,128,8);
            CASE_K(4,128,128,8);
            CASE_K(10,128,128,8);
            CASE_K(16,128,128,5);
            CASE_K(32,256,128,1);
            CASE_K(64,256,256,1);
            default:
                topk_stage_1_opt2_general<T, 128, 1><<<batch_size * beam_width * 1, 128, 0, stream>>>(
                    log_probs,
                    temp_log_probs,
                    topk_tmp_id_buf,
                    topk_tmp_val_buf,
                    beam_width, vocab_size);
                topk_stage_2_opt2_general<T, 128, 1><<<batch_size, 128, beam_width*beam_width*1*sizeof(float) + beam_width * sizeof(int), stream>>>(
                    topk_tmp_id_buf,
                    topk_tmp_val_buf,
                    ids,
                    beam_width);
                break;
        }
    }
    else
    {
        switch(beam_width)
        {
            CASE_K_DIV(1,256,256);
            CASE_K_DIV(4,256,256);
            CASE_K_DIV(16,256,64);
            CASE_K_DIV(64,256,64);
            default:
                printf("[ERROR] Topk kernel does not support beamwidth = %d \n", beam_width);
                exit(0);
                break;
        }
    }

}
#undef CASE_K
#undef CASE_K_DIV

template void topK_kernelLauncher<float>(float* log_probs,
                                          float* temp_log_probs,
                                          int* topk_tmp_id_buf,
                                          float* topk_tmp_val_buf,
                                          int* ids,
                                          DecodingBeamsearchArguments args,
                                          cudaStream_t stream);

// Sampling kernels
template<typename T>
__global__ void sampling(int* topk_tmp_id_buf, 
                        T* topk_tmp_val_buf, 
                        int* ids, 
                        int* sequence_length, 
                        bool* finished_buf,
                        const int candidate_num, 
                        int random_num,
                        const int end_id,
                        const int vocab_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ T sum;
    __shared__ T rand_num;

    if(tid < candidate_num)
    {
        T max_val = topk_tmp_val_buf[bid * candidate_num];
        topk_tmp_val_buf[bid * candidate_num + tid] = __expf(topk_tmp_val_buf[bid * candidate_num + tid] - max_val);
    }
    
    if(tid == 0)
    {
        sum = 0.0f;
        for(int i = 0; i < candidate_num; i++)
        {
            sum = sum + topk_tmp_val_buf[bid * candidate_num + i];
        }
        
        curandState_t local_state;
        curand_init((T)random_num, bid, 0, &local_state);
        rand_num = (T)curand_uniform(&local_state) * sum;

        ids[bid] = topk_tmp_id_buf[bid * candidate_num + candidate_num - 1] % vocab_size;
        for(int i = 0; i < candidate_num; i++)
        {
            rand_num = rand_num - topk_tmp_val_buf[bid * candidate_num + i];
            if(rand_num <= 0.0f){
                ids[bid] = topk_tmp_id_buf[bid * candidate_num + i] % vocab_size;
                break;
            }
        }

        sequence_length[bid] = finished_buf[bid] ? sequence_length[bid] : sequence_length[bid] + 1;
        finished_buf[bid] = ids[bid] == end_id ? 1 : 0;
    }
}

#define CASE_K(K) \
  case K : \
    beam_topK_kernel<T, K, block_size><<<batch_size, block_size, 0, stream>>>(log_probs, \
        topk_tmp_id_buf, topk_tmp_val_buf, vocab_size, 0.0f); \
  break; \

template <typename T>
void topK_sampling_kernel_kernelLauncher(T* log_probs,
                                        int* topk_tmp_id_buf,
                                        T* topk_tmp_val_buf,
                                        int* ids,
                                        int* sequence_length,
                                        bool* finished_buf,
                                        int random_num,
                                        DecodingSamplingArguments args,
                                        cudaStream_t stream)
{
    const int batch_size = args.batch_size_;
    const int vocab_size = args.vocab_size_;
    const int candidate_num = args.candidate_num_;
    const int end_id = args.end_id_;
    const int block_size = 256;
    switch(candidate_num)
    {
        CASE_K(1);
        CASE_K(2);
        CASE_K(4);
        default:
            printf("[ERROR] Topk kernel does not support candidate_num = %d \n", candidate_num);
            exit(0);
            break;
    }
    sampling<T> <<< batch_size, candidate_num, 0, stream>>> (topk_tmp_id_buf, topk_tmp_val_buf, 
                                                            ids, sequence_length, finished_buf,
                                                            candidate_num, random_num, end_id, vocab_size);
}

template void topK_sampling_kernel_kernelLauncher(float* log_probs,
                                                int* topk_tmp_id_buf,
                                                float* topk_tmp_val_buf,
                                                int* ids,
                                                int* sequence_length,
                                                bool* finished_buf,
                                                int random_num,
                                                DecodingSamplingArguments args,
                                                cudaStream_t stream);

__global__ void init_topp_id_val(int* topp_id_val_buf, 
                                int* topp_offset_buf,
                                const int batch_size,
                                const int vocab_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if(bid == 0)
    {
        for(int i = tid; i < batch_size + 1; i+= blockDim.x)
        {
            topp_offset_buf[i] = i * vocab_size;
        }
    }

    while(tid < vocab_size)
    {
        topp_id_val_buf[bid * vocab_size + tid] = tid;
        tid += blockDim.x;
    }
}


void init_topp_id_val_kernel_kernelLauncher(int* topp_id_val_buf,
                                            int* topp_offset_buf,
                                            const int batch_size,
                                            const int vocab_size,
                                            cudaStream_t stream)
{
    init_topp_id_val<<<batch_size, 512, 0, stream>>>(topp_id_val_buf,
                                                    topp_offset_buf,
                                                    batch_size,
                                                    vocab_size);
}

// Sampling kernels
template<typename T>
__global__ void top_p_sampling(T* sorted_log_probs, 
                                int* sorted_id_vals,
                                int* ids,
                                int* sequence_length,
                                bool* finished_buf,
                                const int vocab_size,
                                const int random_num,
                                const float prob_threshold, 
                                const int end_id)
{
    int tid = threadIdx.x;
    curandState_t local_state;
    curand_init((T)random_num, tid, 0, &local_state);
    T rand_num = (T)curand_uniform(&local_state) * prob_threshold;
    ids[tid] = sorted_id_vals[vocab_size - 1];

    for(int i = tid * vocab_size; i < tid * vocab_size + vocab_size; i++)
    {
        rand_num = rand_num - sorted_log_probs[i];
        if(rand_num <= 0)
        {
            ids[tid] = sorted_id_vals[i];
            break;
        }
    }

    sequence_length[tid] = finished_buf[tid] ? sequence_length[tid] : sequence_length[tid] + 1;
    finished_buf[tid] = ids[tid] == end_id ? 1 : 0;
}

template <typename T>
__global__ void sort_kernel(const T* log_probs, 
                            const int* id_vals,
                            T* sorted_log_probs,
                            int* sorted_id_vals,
                            const int vocab_size)
{
    typedef cub::BlockRadixSort<T, 256, 32, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    // Obtain a segment of consecutive items that are blocked across threads
    T thread_keys[32];
    int thread_values[32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for(int i = 0; i < 32; i++)
    {
        int index = tid + 256 * i + bid * vocab_size;
        thread_keys[i] = log_probs[index];
        thread_values[i] = id_vals[index];
    }
    BlockRadixSort(temp_storage).SortDescending(thread_keys, thread_values);

    for(int i = 0; i < 32; i++)
    {
        int index = tid + 256 * i + bid * vocab_size;
        sorted_log_probs[index] = thread_keys[i];
        sorted_id_vals[index] = thread_values[i];
    }
}

template<typename T>
void topP_sampling_kernel_kernelLauncher(const T* log_probs,
                                        const int* id_vals,
                                        T* sorted_log_probs,
                                        int* sorted_id_vals, 
                                        int* topp_offset_buf,
                                        void* temp_storage,
                                        bool* finished_buf,
                                        int step,
                                        DecodingSamplingArguments args,
                                        int* output_ids, 
                                        int* sequence_length, 
                                        cudaStream_t stream)
{
    // sort_kernel<<<batch_size, 256, 0, stream>>>(log_probs, 
    //                                             id_vals,
    //                                             sorted_log_probs,
    //                                             sorted_id_vals,
    //                                             vocab_size);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage, 
                                                        args.temp_storage_size_,
                                                        log_probs, 
                                                        sorted_log_probs,
                                                        id_vals, 
                                                        sorted_id_vals, 
                                                        args.vocab_size_ * args.batch_size_,
                                                        args.batch_size_, 
                                                        topp_offset_buf, topp_offset_buf + 1);
                                                        
    
    top_p_sampling<<<1, args.batch_size_, 0, stream>>>(sorted_log_probs, 
                                                        sorted_id_vals,
                                                        output_ids + (step - 1) * args.batch_size_,
                                                        sequence_length,
                                                        finished_buf,
                                                        args.vocab_size_, 
                                                        step,
                                                        args.probability_threshold_,
                                                        args.end_id_);
}

template void topP_sampling_kernel_kernelLauncher(const float* log_probs,
                                        const int* id_vals,
                                        float* sorted_log_probs,
                                        int* sorted_id_vals, 
                                        int* topp_offset_buf,
                                        void* temp_storage,
                                        bool* finished_buf,
                                        int step,
                                        DecodingSamplingArguments args,
                                        int* output_ids, 
                                        int* sequence_length, 
                                        cudaStream_t stream);

} // end of namespace fastertransformer