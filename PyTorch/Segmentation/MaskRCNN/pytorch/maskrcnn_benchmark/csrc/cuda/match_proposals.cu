/**
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>

__launch_bounds__(256) static __global__
    void max_along_gt_idx(float *match, unsigned char *pred_forgiven, long *max_gt_idx, long long gt,long long preds,
                          bool include_low_quality, float low_th, float high_th) {
    
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < preds){
        float max_iou = 0.0f;
        int max_idx = 0;
        float iou;
        for(long long i = 0;i < gt; i++){
            iou = match[i * preds + tid]; 
            if (iou > max_iou) {max_iou = iou; max_idx = i;}
        }

        if (max_iou >= high_th) max_gt_idx[tid] = max_idx;
        else if ((pred_forgiven[tid] == 1 && include_low_quality)) max_gt_idx[tid] = max_idx;
        else if (max_iou < low_th) max_gt_idx[tid] = -1;
        else if (max_iou < high_th) max_gt_idx[tid] = -2;
    }
}


__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] = fmax(sdata[tid],sdata[tid + 32]);
    sdata[tid] = fmax(sdata[tid],sdata[tid + 16]);
    sdata[tid] = fmax(sdata[tid],sdata[tid + 8]);
    sdata[tid] = fmax(sdata[tid],sdata[tid + 4]);
    sdata[tid] = fmax(sdata[tid],sdata[tid + 2]);
    sdata[tid] = fmax(sdata[tid],sdata[tid + 1]);
}


static __global__
    void max_along_preds(float* match, float* inter_gt, long long gt,long long preds) {
    int gt_idx = blockIdx.x;
    int chunk_idx = blockIdx.y;
    int gt_offset = chunk_idx * 2048;
    int start_idx = gt_idx * preds + gt_offset;
    int idx = threadIdx.x;
    __shared__ float shbuf[1024]; 
   shbuf[idx] = 0.0f;
    __syncthreads();
    if(gt_offset + idx + 1024 < preds) shbuf[idx] = fmax(match[start_idx + idx], match[start_idx + idx + 1024]);
    else if (gt_offset + idx < preds) shbuf[idx] = match[start_idx + idx];
    __syncthreads();
    if(idx < 512) shbuf[idx] = fmax(shbuf[idx],shbuf[idx + 512]);
    __syncthreads();
    if(idx < 256) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 256]);
    __syncthreads();
    if(idx < 128) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 128]);
    __syncthreads();
    if(idx < 64) shbuf[idx] = fmax(shbuf[idx], shbuf[idx + 64]);
    __syncthreads();
    if(idx < 32) warpReduce(shbuf, idx);
    if (idx == 0) inter_gt[((preds + 2047) / 2048) * gt_idx + chunk_idx] = shbuf[idx];
}



__launch_bounds__(256) static __global__
    void max_along_preds_reduced(float *match, float *max_preds, long long gt,long long preds) {
    
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < gt){
        float max_iou = 0.0f;
        float iou;
        for(long long i = 0; i < preds; i++){
            iou = match[tid * preds + i]; 
            if (iou > max_iou) max_iou = iou;
        }
        max_preds[tid] = max_iou;
    }
}

__launch_bounds__(256) static __global__
    void forgive_preds(float *match_quality_data, float *d_best_pred_per_gt, unsigned char *d_pred_forgiven, 
                       long gt, long preds) {
        long tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < preds){
            unsigned char forgiven = 0;
            float iou;
            for(int i = 0; i < gt; i++){
                iou = match_quality_data[i * preds + tid];
                if(iou == d_best_pred_per_gt[i]){
                    forgiven = 1;
                    break;
                }            
            }
            d_pred_forgiven[tid] = forgiven;
        }    
    } 

at::Tensor match_proposals_cuda(at::Tensor match_quality_matrix, bool allow_low_quality_matches, 
                                float low_th, float high_th){
    
    int gt = match_quality_matrix.size(0);
    long long preds = match_quality_matrix.size(1);
    float *match_quality_data = match_quality_matrix.data_ptr<float>();
    using namespace at;
    //predictions are reduced by chunks of 2048 elements per block
    int num_chunks = (preds + 2047) / 2048;
    auto result = torch::ones({preds}, torch::CUDA(at::kLong)); 
    at::Tensor best_pred_per_gt = torch::zeros({gt}, torch::CUDA(at::kFloat));
    at::Tensor pred_forgiven = torch::zeros({preds}, torch::CUDA(at::kByte));    
    at::Tensor intergt = torch::zeros({gt * num_chunks}, torch::CUDA(at::kFloat));
    
    auto stream = at::cuda::getCurrentCUDAStream();
    
    //do an intermediate reduction along all predictions for each gt
    dim3 block(1024, 1, 1);
    dim3 grid(gt, num_chunks, 1);
    if (allow_low_quality_matches) max_along_preds<<<grid, block, 0, stream.stream()>>>(
                                                        match_quality_matrix.data_ptr<float>(), 
                                                        intergt.data_ptr<float>(), 
                                                        gt, 
                                                        preds);  
    //final reduction to find best iou per gt  
    int numThreads = 256;
    int numBlocks=(gt + numThreads - 1) / numThreads;    

    if (allow_low_quality_matches) max_along_preds_reduced<<<numBlocks, numThreads, 0, stream.stream()>>>(
                                                        intergt.data_ptr<float>(), 
                                                        best_pred_per_gt.data_ptr<float>(), 
                                                        gt, 
                                                        num_chunks); 
    numBlocks=(preds + numThreads - 1) / numThreads;
    //if low_quality_matches are allowed, mark some predictions to keep their best matching gt even though
    //iou < threshold
    if (allow_low_quality_matches) forgive_preds<<<numBlocks, numThreads, 0, stream.stream()>>>(
                                                        match_quality_matrix.data_ptr<float>(), 
                                                        best_pred_per_gt.data_ptr<float>(), 
                                                        pred_forgiven.data_ptr<unsigned char>(), 
                                                        gt, 
                                                        preds); 
    //compute resulting tensor of indices
    max_along_gt_idx<<<numBlocks, numThreads, 0, stream.stream()>>>(match_quality_matrix.data_ptr<float>(), 
                                                                    pred_forgiven.data_ptr<unsigned char>(), 
                                                                    result.data_ptr<long>(), 
                                                                    gt, 
                                                                    preds, 
                                                                    allow_low_quality_matches, 
                                                                    low_th, 
                                                                    high_th);       
    return result;

}

