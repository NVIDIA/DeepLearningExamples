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

#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include "cpu/vision.h"


/*rle cuda kernels are cuda version of the corresponding cpu functions here 
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c 
these are only a subset of rle kernels.*/

typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;

//6144 is based on minimum shared memory size per SM 
//across all pytorch-supported GPUs. Need to use blocking
//to avoid this restriction
const int BUFFER_SIZE=6144;
const int CNTS_SIZE=6144;

__global__ void crop_and_scale_cuda_kernel(double *dense_poly_data, int *per_anchor_poly_idx, int *poly_rel_idx, 
                                           int poly_count, int anchor_count, float4 *anchor_data, int mask_size){
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    int poly_id = blockIdx.x;
    int anchor_idx;
    for (anchor_idx = 0; anchor_idx < anchor_count; anchor_idx++){
        if (poly_id < per_anchor_poly_idx[anchor_idx + 1]) break;    
    }
    float w = anchor_data[anchor_idx].z - anchor_data[anchor_idx].x;
  	float h = anchor_data[anchor_idx].w - anchor_data[anchor_idx].y;
  	w = fmaxf(w, 1.0f);
  	h = fmaxf(h, 1.0f);  	
    float ratio_h = ((float) mask_size) / h;
    float ratio_w = ((float) mask_size) / w;

    int poly_ptr_idx_start = poly_rel_idx[poly_id];
    int poly_ptr_idx_end = poly_rel_idx[poly_id + 1];

    double *poly_data_buf = dense_poly_data + poly_ptr_idx_start;
    int len = poly_ptr_idx_end - poly_ptr_idx_start;
    
    for (int j = tid; j < len; j += block_jump){
        if (j % 2 == 0) poly_data_buf[j] = ratio_w*((float) poly_data_buf[j]- anchor_data[anchor_idx].x);
        if (j % 2 == 1) poly_data_buf[j] = ratio_h*((float) poly_data_buf[j]- anchor_data[anchor_idx].y);
    }
}

//merging masks happens on mask format, not RLE format.     
__global__ void merge_masks_cuda_kernel(byte *masks_in, float *masks_out, const int mask_size, 
                                        int *per_anchor_poly_idx, int anchor_count){

    int anchor_idx = blockIdx.x;
    int tid = threadIdx.x;
    int jump_block = blockDim.x;
    int mask_start_idx = per_anchor_poly_idx[anchor_idx];
    int num_of_masks_to_merge = per_anchor_poly_idx[anchor_idx + 1]-per_anchor_poly_idx[anchor_idx];
    
    for(int j = tid; j < mask_size * mask_size; j += jump_block){
        int transposed_pixel = (j % mask_size) * mask_size + j / mask_size;
        byte pixel = 0;
        for(int k = 0; k < num_of_masks_to_merge; k++){
            if (masks_in[(mask_start_idx + k) * mask_size * mask_size + j] == 1) pixel = 1;
            if (pixel == 1) break;
        }
        masks_out[anchor_idx * mask_size * mask_size + transposed_pixel] = (float) pixel;       
    }
}

/*cuda version of rleDecode function in this API: 
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c*/

__global__ void decode_rle_cuda_kernel(const int *num_of_cnts, uint *cnts, long h, long w, byte *mask)
{
    int poly_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    
    int m = num_of_cnts[poly_id];
    uint *cnts_buf = cnts + CNTS_SIZE * poly_id;
    byte *mask_ptr = mask + poly_id * h * w;
    
    __shared__ uint shbuf1[CNTS_SIZE];
    __shared__ uint shbuf2[CNTS_SIZE];

    
    //initialize shbuf for scan. first element is 0 (exclusive scan)
    for (long i = tid; i < CNTS_SIZE; i += block_jump){
        shbuf1[i] = (i <= m & i > 0) ? cnts_buf[i - 1]:0;
        shbuf2[i] = (i <= m & i > 0) ? cnts_buf[i - 1]:0;
    }
    __syncthreads();
    
    //double buffering for scan
    int switch_buf = 0;
    for (int offset = 1; offset <= m; offset *= 2){
        switch_buf = 1 - switch_buf;
        if(switch_buf == 0){
            for(int j = tid;j <= m;j += block_jump){
                if(j >= offset) shbuf2[j] = shbuf1[j]+shbuf1[j - offset];
                else shbuf2[j] = shbuf1[j];
            }
        }else if (switch_buf == 1){
            for(int j = tid;j <= m;j += block_jump){
                if(j >= offset) shbuf1[j] = shbuf2[j] + shbuf2[j - offset];
                else shbuf1[j] = shbuf2[j];
            }        
        
        }
        __syncthreads();
    }
    uint *scanned_buf = switch_buf == 0 ? shbuf2 : shbuf1;
       
    //find which bin pixel j falls into , which determines the pixel value
    //use binary search
    for(int j = tid; j < h * w; j += block_jump){
        int bin = 0;
        int min_idx = 0;
        int max_idx = m;
        int mid_idx = m / 2;        
        while(max_idx > min_idx){
            if(j > scanned_buf[mid_idx]) {
                min_idx = mid_idx+1; 
                mid_idx = (min_idx + max_idx) / 2;
            }
            else if (j < scanned_buf[mid_idx]) {
                max_idx = mid_idx; 
                mid_idx = (min_idx + max_idx) / 2;
            }
            else {
                mid_idx++; 
                break;
            }    
        }         
        int k = mid_idx;
        byte pixel = k % 2 == 0 ? 1 : 0;
        mask_ptr[j] = pixel; 
    }
}

/*cuda version of rleFrPoly function in this API: 
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c*/

__global__ void rle_fr_poly_cuda_kernel(const double *dense_coordinates, int *poly_rel_idx, long h, long w, 
                                        uint *cnts, int *x_in, int *y_in, int *u_in, int *v_in, uint *a_in, 
                                        uint *b_in, int *num_of_cnts) {   
    int poly_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    long cnts_offset = poly_id * CNTS_SIZE;
    long k = (poly_rel_idx[poly_id + 1] - poly_rel_idx[poly_id]) / 2;
    
    const double *xy = dense_coordinates + poly_rel_idx[poly_id];
    int *x = x_in + poly_id * BUFFER_SIZE;
    int *y = y_in + poly_id * BUFFER_SIZE;
    int *u = u_in + poly_id * BUFFER_SIZE;
    int *v = v_in + poly_id * BUFFER_SIZE;
    uint *a = a_in + poly_id * BUFFER_SIZE;
    uint *b = b_in + poly_id * BUFFER_SIZE;
    /* upsample and get discrete points densely along entire boundary */
    long j, m = 0;
    double scale = 5;
    __shared__ int shbuf1[BUFFER_SIZE];
    __shared__ int shbuf2[BUFFER_SIZE];
    for(long j = tid; j < BUFFER_SIZE; j += block_jump) {
        shbuf1[j] = 0; 
        shbuf2[j] = 0;
    }
    for(long j = tid; j <= k; j += block_jump) 
        x[j] = j < k ? ((int) (scale * xy[2 * j + 0] + 0.5)) : ((int) (scale * xy[0] + 0.5));
    for(long j = tid; j <= k; j += block_jump) 
        y[j] = j < k ? ((int) (scale * xy[2 * j + 1] + 0.5)) : ((int) (scale * xy[1] + 0.5));        
    __syncthreads();
        
    for(int j = tid; j < k; j += block_jump){
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d, dist;
        int flip; 
        double s; 
        dx = abs(xe - xs); 
        dy = abs(ys - ye);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t;}
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        dist = dx >= dy ? dx + 1 : dy + 1;
        shbuf1[j + 1] = dist; 
        shbuf2[j + 1] = dist;
    }
    __syncthreads();
    //block-wide exclusive prefix scan
    int switch_buf = 0;
    for (int offset = 1; offset <= k; offset *= 2){
        switch_buf = 1 - switch_buf;
        if (switch_buf == 0){
            for(int j = tid; j <= k; j += block_jump){
                if (j >= offset) shbuf2[j] = shbuf1[j] + shbuf1[j - offset];
                else shbuf2[j] = shbuf1[j];                
            }
        }
        else if (switch_buf == 1){
            for(int j = tid; j <= k; j += block_jump){
                if (j >= offset) shbuf1[j] = shbuf2[j] + shbuf2[j - offset];
                else shbuf1[j] = shbuf2[j];                
            }                  
        } 
        __syncthreads();
    }
      
    for (int j = tid; j < k; j += block_jump){
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d, dist;
        int flip; 
        double s; 
        dx = __sad(xe, xs, 0); 
        dy = __sad(ys, ye, 0);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t;}
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        m = switch_buf == 0 ? shbuf2[j] : shbuf1[j];
        if (dx >= dy) for (d = 0; d <= dx; d++) {
          /*the multiplication statement 's*t' causes nvcc to optimize with flush-to-zero=True for 
          double precision multiply, which we observe produces different results than CPU occasionally. 
          To force flush-to-zero=False, we use __dmul_rn intrinsics function */
          t = flip ? dx - d : d; 
          u[m] = t + xs; 
          v[m] = (int) (ys + __dmul_rn(s, t) + .5); 
          m++; 
        } 
        else for (d = 0; d <= dy; d++) {
          t = flip ? dy - d : d; 
          v[m] = t + ys; 
          u[m] = (int) (xs + __dmul_rn(s, t) + .5); 
          m++;
        }
    }    
    __syncthreads();
    m = switch_buf == 0 ? shbuf2[k] : shbuf1[k];
    int k2 = m;
    __syncthreads();
    double xd, yd;
    if (tid == 0) {
        shbuf1[tid] = 0; 
        shbuf2[tid] = 0;
    }     
    /* get points along y-boundary and downsample */
    for (int j = tid; j < k2; j += block_jump){
        if (j > 0){
            if (u[j] != u[j - 1]){
                xd = (double) (u[j] < u[j-1] ? u[j] : u[j] - 1); 
                xd = (xd + .5) / scale - .5;   
                if (floor(xd) != xd || xd < 0 || xd > w - 1 ) {
                    shbuf1[j] = 0; 
                    shbuf2[j] = 0; 
                    continue;
                }
                yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]); yd = (yd + .5) / scale - .5;
                if (yd < 0) yd = 0; 
                else if (yd > h) yd = h; yd = ceil(yd);                
                shbuf1[j] = 1; 
                shbuf2[j] = 1;               
            } else {
                shbuf1[j] = 0; 
                shbuf2[j] = 0;
            }                    
        }    
    }
    __syncthreads(); 
    //exclusive prefix scan
    switch_buf = 0;
    for (int offset = 1; offset < k2; offset *= 2){
        switch_buf = 1 - switch_buf;
        if (switch_buf == 0){
            for (int j = tid; j < k2; j += block_jump){
                if (j >= offset) shbuf2[j] = shbuf1[j - offset] + shbuf1[j];
                else shbuf2[j] = shbuf1[j];                
            }
        }
        else if (switch_buf == 1){
            for (int j = tid; j < k2; j += block_jump){
                if (j >= offset) shbuf1[j] = shbuf2[j - offset] + shbuf2[j];
                else shbuf1[j] = shbuf2[j];                
            }                  
        } 
        __syncthreads();             
    }
  
    for (int j = tid; j < k2; j += block_jump){
        if (j > 0){
            if(u[j] != u[j - 1]){
                xd = (double) (u[j] < u[j - 1] ? u[j] : u[j] - 1); 
                xd = (xd + .5) / scale - .5;
                if (floor(xd) != xd || xd < 0 || xd > w - 1) {continue;}
                yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]); 
                yd = (yd + .5) / scale - .5;
                if (yd < 0) yd = 0; 
                else if (yd > h) yd = h; yd = ceil(yd);                
                m = switch_buf == 0 ? shbuf2[j - 1]:shbuf1[j - 1];
                x[m] = (int) xd; 
                y[m] = (int) yd; 
                m++;                
            }                   
        }    
    }
    __syncthreads(); 
    
    /* compute rle encoding given y-boundary points */
    m = switch_buf == 0 ? shbuf2[k2 - 1] : shbuf1[k2 - 1]; 
    int k3 = m;
    for (int j = tid; j <= k3; j += block_jump){
       if (j < k3) a[j] = (uint) (x[j] * (int) (h) + y[j]); 
       else a[j] = (uint)(h * w);        
    }    
    k3++;
    __syncthreads();
    
    //run brick sort on a for k3+1 element
    //load k3+1 elements of a into shared memory
    for(long j = tid; j < k3; j += block_jump) shbuf1[j]=a[j];
    __syncthreads();
    uint a_temp;
    for (int r = 0; r <= k3 / 2; r++){
        int evenCas = k3 / 2;
        int oddCas = (k3 - 1) / 2;
        //start with 0, need (k3+1)/2 CAS
        for (int j = tid; j < evenCas; j += block_jump){
            if (shbuf1[2 * j] > shbuf1[2 * j + 1]){
                a_temp = shbuf1[2 * j];
                shbuf1[2 * j]=shbuf1[2 * j + 1];
                shbuf1[2 * j + 1] = a_temp;
            }            
        }
        __syncthreads();
        //start with 1
        for (int j = tid; j < oddCas; j += block_jump){
            if (shbuf1[2 * j + 1] > shbuf1[2 * j + 2]){
                a_temp=shbuf1[2 * j + 1];
                shbuf1[2 * j + 1] = shbuf1[2 * j + 2];
                shbuf1[2 * j + 2]=a_temp;
            }            
        }
        __syncthreads();    
    }
    
    for(long j = tid; j < k3; j += block_jump) {
        if(j>0) shbuf2[j] = shbuf1[j - 1];
        else shbuf2[j] = 0;        
    } 
     __syncthreads();  
    for(int j = tid; j < k3; j += block_jump){
        shbuf1[j] -= shbuf2[j];       
    }      
    __syncthreads();   
    uint *cnts_buf = cnts + cnts_offset;
    if (tid == 0){
        j = m = 0; 
        cnts_buf[m++] = shbuf1[j++];
        while (j < k3) if (shbuf1[j] > 0) cnts_buf[m++] = shbuf1[j++]; else {
            j++; if (j < k3) cnts_buf[m - 1] += shbuf1[j++]; } 
        num_of_cnts[poly_id] = m;               
    }
    __syncthreads();
}

at::Tensor generate_mask_targets_cuda(at::Tensor dense_vector, const std::vector<std::vector<at::Tensor>> polygons, 
                                      const at::Tensor anchors, const int mask_size){    
    const int M = mask_size;
    assert (M < 32); 
    //if M >=32, shared memory buffer size may not be
    //sufficient. Need to fix this by blocking    
    float *d_anchor_data = anchors.data_ptr<float>();
    int num_of_anchors = anchors.size(0);  
    auto per_anchor_poly_idx = at::empty({num_of_anchors + 1}, at::CPU(at::kInt));
    int num_of_poly = 0;
    for (int i = 0; i < num_of_anchors; i++){
  	    *(per_anchor_poly_idx.data_ptr<int>() + i) = num_of_poly;
  	    num_of_poly += polygons[i].size();
    }
    *(per_anchor_poly_idx.data_ptr<int>() + num_of_anchors) = num_of_poly;
  
    auto poly_rel_idx = at::empty({num_of_poly + 1}, at::CPU(at::kInt));
    double *dense_poly_data = dense_vector.data_ptr<double>();
    int start_idx = 0;
    int poly_count = 0;
  
    for(int i = 0; i < polygons.size(); i++){
  	    for(int j=0; j < polygons[i].size(); j++) {
  		    *(poly_rel_idx.data_ptr<int>() + poly_count) = start_idx;
  		    start_idx += polygons[i][j].size(0);
  		    poly_count++;
  	    }
    }    
    *(poly_rel_idx.data_ptr<int>() + poly_count) = start_idx;

    at::Tensor d_x_t = torch::empty({BUFFER_SIZE * num_of_poly}, torch::CUDA(at::kInt));
    at::Tensor d_y_t = torch::empty({BUFFER_SIZE * num_of_poly}, torch::CUDA(at::kInt));
    at::Tensor d_u_t = torch::empty({BUFFER_SIZE * num_of_poly}, torch::CUDA(at::kInt));
    at::Tensor d_v_t = torch::empty({BUFFER_SIZE * num_of_poly}, torch::CUDA(at::kInt));
    at::Tensor d_a_t = torch::empty({BUFFER_SIZE * num_of_poly}, torch::CUDA(at::kInt));//used with uint* pointer
    at::Tensor d_b_t = torch::empty({BUFFER_SIZE * num_of_poly}, torch::CUDA(at::kInt)); //used with uint* pointer
    at::Tensor d_mask_t = torch::empty({M * M * num_of_poly}, torch::CUDA(at::kByte)); 
    auto result =  torch::empty({num_of_anchors, M, M}, torch::CUDA(at::kFloat));   
    at::Tensor d_num_of_counts_t = torch::empty({num_of_poly}, torch::CUDA(at::kInt));
    at::Tensor d_cnts_t = torch::empty({CNTS_SIZE * num_of_poly}, torch::CUDA(at::kInt));     
    auto d_dense_vector = dense_vector.cuda();   
    auto d_per_anchor_poly_idx = per_anchor_poly_idx.cuda();
    auto d_poly_rel_idx = poly_rel_idx.cuda();
    auto stream = at::cuda::getCurrentCUDAStream();  
    
    crop_and_scale_cuda_kernel<<<num_of_poly, 256, 0, stream.stream()>>>(d_dense_vector.data_ptr<double>(), 
                                                                      d_per_anchor_poly_idx.data_ptr<int>(), 
                                                                      d_poly_rel_idx.data_ptr<int>(), 
                                                                      poly_count, 
                                                                      num_of_anchors, 
                                                                      (float4*) d_anchor_data, 
                                                                      M);       
                                                                      
    //TODO: larger threads-per-block might be better here, because each CTA uses 32 KB of shmem,
    //and occupancy is likely shmem capacity bound                                                                                
    rle_fr_poly_cuda_kernel<<<num_of_poly, 1024, 0, stream.stream()>>>(d_dense_vector.data_ptr<double>(), 
                                                                   d_poly_rel_idx.data_ptr<int>(), 
                                                                   M, M, 
                                                                   (uint*) d_cnts_t.data_ptr<int>(), 
                                                                   d_x_t.data_ptr<int>(), 
                                                                   d_y_t.data_ptr<int>(), 
                                                                   d_u_t.data_ptr<int>(), 
                                                                   d_v_t.data_ptr<int>(), 
                                                                   (uint*) d_a_t.data_ptr<int>(), 
                                                                   (uint*) d_b_t.data_ptr<int>(), 
                                                                   d_num_of_counts_t.data_ptr<int>());
                                                                 
    decode_rle_cuda_kernel<<<num_of_poly, 256, 0, stream.stream()>>>(d_num_of_counts_t.data_ptr<int>(), 
                                                                 (uint*) d_cnts_t.data_ptr<int>(), 
                                                                 M, M, 
                                                                 d_mask_t.data_ptr<byte>());
                                                                 
    merge_masks_cuda_kernel<<<num_of_anchors, 256, 0, stream.stream()>>>(d_mask_t.data_ptr<byte>(), result.data_ptr<float>(), 
                                                                      M, d_per_anchor_poly_idx.data_ptr<int>(), 
                                                                      num_of_anchors); 
    return result;
}

