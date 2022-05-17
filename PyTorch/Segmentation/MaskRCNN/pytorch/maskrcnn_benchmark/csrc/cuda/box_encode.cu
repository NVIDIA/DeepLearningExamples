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

__global__ void box_encode_kernel(float *targets_dx, float *targets_dy, float *targets_dw, float *targets_dh,  
                                  float4 *boxes, float4 *anchors, float wx, float wy, float ww, float wh, 
                                  size_t gt, size_t idxJump) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row_offset; 
    float anchors_x1, anchors_x2, anchors_y1, anchors_y2, 
        boxes_x1, boxes_x2, boxes_y1, boxes_y2, ex_w, ex_h, 
        ex_ctr_x, ex_ctr_y, gt_w, gt_h, gt_ctr_x, gt_ctr_y;
          
    for (int i = idx; i < gt; i += idxJump){
        row_offset = i;
        anchors_x1 = anchors[row_offset].x;
        anchors_y1 = anchors[row_offset].y;
        anchors_x2 = anchors[row_offset].z;
        anchors_y2 = anchors[row_offset].w;        

        boxes_x1 = boxes[row_offset].x;
        boxes_y1 = boxes[row_offset].y;
        boxes_x2 = boxes[row_offset].z;
        boxes_y2 = boxes[row_offset].w; 
        
        ex_w = anchors_x2 - anchors_x1 + 1;
        ex_h = anchors_y2 - anchors_y1 + 1;
        ex_ctr_x = anchors_x1 + 0.5 * ex_w; 
        ex_ctr_y = anchors_y1 + 0.5 * ex_h;
               
        gt_w = boxes_x2 - boxes_x1 + 1;
        gt_h = boxes_y2 -  boxes_y1 + 1; 
        gt_ctr_x = boxes_x1 + 0.5 * gt_w; 
        gt_ctr_y = boxes_y1 + 0.5 * gt_h;        
        
        targets_dx[i] = wx * (gt_ctr_x - ex_ctr_x) / ex_w; 
        targets_dy[i] = wy * (gt_ctr_y - ex_ctr_y) / ex_h; 
        targets_dw[i] = ww * log(gt_w / ex_w); 
        targets_dh[i] = wh * log(gt_h / ex_h);          
    }  

}


std::vector<at::Tensor> box_encode_cuda(at::Tensor boxes, at::Tensor anchors, float wx, float wy, float ww, float wh){
   
    int minGridSize;
    int blockSize;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                       &blockSize,
                                       (void*) box_encode_kernel,
                                       0,  // dynamic memory
                                       0); // maximum utilized threads    
    long size = boxes.size(0);
    auto targets_dx = torch::ones({size}, torch::CUDA(at::kFloat)); 
    auto targets_dy = torch::ones({size}, torch::CUDA(at::kFloat));
    auto targets_dw = torch::ones({size}, torch::CUDA(at::kFloat));
    auto targets_dh = torch::ones({size}, torch::CUDA(at::kFloat));
    
    dim3 gridDim(minGridSize);
    dim3 blockDim(blockSize);
    int idxJump = minGridSize * blockSize;
    auto stream = at::cuda::getCurrentCUDAStream();
    box_encode_kernel<<<gridDim,blockDim,0,stream.stream()>>>(targets_dx.data_ptr<float>(), 
                                                              targets_dy.data_ptr<float>(), 
                                                              targets_dw.data_ptr<float>(), 
                                                              targets_dh.data_ptr<float>(), 
                                                              (float4*) boxes.data_ptr<float>(), 
                                                              (float4*) anchors.data_ptr<float>(), 
                                                              wx, wy, ww, wh, 
                                                              size, idxJump);
     
    std::vector<at::Tensor> result;
    result.push_back(targets_dx);
    result.push_back(targets_dy);
    result.push_back(targets_dw);
    result.push_back(targets_dh);  
    return result;
}
