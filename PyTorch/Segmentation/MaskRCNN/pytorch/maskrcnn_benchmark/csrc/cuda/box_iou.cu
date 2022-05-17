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
#include <iostream>

__global__ void box_iou_cuda_kernel(float *box_iou, float4 *box1, float4 *box2, long M, 
                                    long N, int idxJump) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    size_t b1_idx, b2_idx, b1_row_offset, b2_row_offset; 
    float xmin1, xmin2, xmax1, xmax2, ymin1, ymin2, ymax1, ymax2;
    float x_tl, y_tl, x_br, y_br, w, h, inter, area1, area2, iou;
          
    for (long i = idx; i < M * N; i += idxJump){
        
        b1_idx = i / N;
        b2_idx = i % N;
        b1_row_offset = b1_idx;
        b2_row_offset = b2_idx;

        xmin1 = box1[b1_row_offset].x;
        ymin1 = box1[b1_row_offset].y;
        xmax1 = box1[b1_row_offset].z;
        ymax1 = box1[b1_row_offset].w;
        xmin2 = box2[b2_row_offset].x;
        ymin2 = box2[b2_row_offset].y;
        xmax2 = box2[b2_row_offset].z;
        ymax2 = box2[b2_row_offset].w;

        x_tl = fmaxf(xmin1, xmin2);
        y_tl = fmaxf(ymin1, ymin2);

        x_br = fminf(xmax1, xmax2);
        y_br = fminf(ymax1, ymax2);                                
        w = (x_br - x_tl + 1) < 0 ? 0.0f : (x_br - x_tl + 1);
        h = (y_br - y_tl + 1) < 0 ? 0.0f : (y_br - y_tl + 1);

        inter = w * h;
        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1);
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1);
        iou = inter / (area1 + area2 - inter);
        box_iou[b1_idx * N + b2_idx] = iou;
    }  

}

at::Tensor box_iou_cuda(at::Tensor box1, at::Tensor box2){

    int minGridSize;
    int blockSize;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                       &blockSize,
                                       (void*) box_iou_cuda_kernel,
                                       0,  // dynamic memory
                                       0); // maximum utilized threads
    
    long M = box1.size(0);
    long N = box2.size(0);
    auto box_iou = torch::ones({M, N}, torch::CUDA(at::kFloat));
    
    dim3 gridDim(minGridSize);
    dim3 blockDim(blockSize);
    int idxJump = minGridSize * blockSize;
    auto stream = at::cuda::getCurrentCUDAStream();
    box_iou_cuda_kernel<<<gridDim, blockDim, 0, stream.stream()>>>(box_iou.data_ptr<float>(), 
                                                                  (float4*) box1.data_ptr<float>(), 
                                                                  (float4*) box2.data_ptr<float>(), 
                                                                  M, N, 
                                                                  idxJump);
    return box_iou;
}

