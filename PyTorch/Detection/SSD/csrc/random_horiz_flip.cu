/******************************************************************************
*
* Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
*

 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCNumerics.cuh>
#include <THC/THC.h>

#include <cuda.h>

/**
 * Each block will handle one channel of each image
 **/
template <typename T>
__global__
void HorizFlipImagesAndBoxes(
                             const int N,
                             const int C,
                             const int H,
                             const int W,
                             const T* img_in,
                             float* bboxes,
                             const int* offsets,
                             const float p,
                             const float* flip,
                             T* img_out,
                             const bool nhwc) {
  // early return if not flipping
  if (flip[blockIdx.x] < p) return;

  // pointer offset into images
  const int img_offset = blockIdx.x * C * H * W;
  const T* img = &img_in[img_offset];
  T* img_o = &img_out[img_offset];

  // flip bboxes
  auto bbox_offset_begin = offsets[blockIdx.x];
  auto bbox_offset_end   = offsets[blockIdx.x + 1];
  auto num_bboxes = bbox_offset_end - bbox_offset_begin;

  const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

  // bboxes in ltrb format, scaled to [0, 1]
  for (int i = thread_idx; i < num_bboxes; i += blockDim.x * blockDim.y) {
    float *bbox = &bboxes[(bbox_offset_begin + thread_idx) * 4];
    // Could do this inplace, but not register constrained
    auto bbox_0 = bbox[0];
    auto bbox_2 = bbox[2];
    bbox[0] = 1. - bbox_2;
    bbox[2] = 1. - bbox_0;
  }

  if (nhwc) {
    // loop over float3 pixels, handle 3 values / thread
    for (int h = threadIdx.y; h < H; h += blockDim.y) {
      for (int w = threadIdx.x; w < W; w += blockDim.x) {
        const T* img_hw = &img[h * W * C + w * C];
        T * img_out_hw = &img_o[h * W * C + (W - 1 - w) * C];

        for (int c = 0; c < C; ++c) {
          img_out_hw[c] = img_hw[c];
        }
      }
    }
  } else {
    // loop over channels
    for (int c = 0; c < C; ++c) {
      const T* img_c = &img[c * H * W];
      T *img_out_c = &img_o[c * H * W];

      // handle tiles of (h, w) at a time
      for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int w = threadIdx.x; w < W; w += blockDim.x) {
          const int input_idx = h * W + w;
          const int output_idx = h * W + (W - 1 - w);


          img_out_c[output_idx] = img_c[input_idx];
        }
      }
    }
  }
}

/**
  * Take images and their bboxes, randomly flip on horizontal axis
  * In/Out: img: NCHW tensor of N, C-channel images of constant (H, W)
  * In/Out: bboxes: [N_i, 4] tensor of original bboxes in ltrb format
  * In: bbox_offsets: [N] offset values into bboxes
  * In: p \in [0, 1): probability of flipping each (img, bbox) pair
  * In: nhwc: Tensor in NHWC format
  * ----
  * Note: allocate temp memory, but effectively do this inplace
  */
std::vector<at::Tensor> random_horiz_flip(
                             at::Tensor& img,
                             at::Tensor& bboxes,
                             const at::Tensor& bbox_offsets,
                             const float p,
                             const bool nhwc) {
  // dimensions
  const int N = img.size(0);
  int C, H, W;
  if (nhwc) {
    C = img.size(3);
    H = img.size(1);
    W = img.size(2);

  } else {
    C = img.size(1);
    H = img.size(2);
    W = img.size(3);
  }

  assert(img.type().is_cuda());
  assert(bboxes.type().is_cuda());
  assert(bbox_offsets.type().is_cuda());

  // printf("%d %d %d %d\n", N, C, H, W);
  // Need temp storage of size img
  at::Tensor tmp_img = img.clone();
  at::Tensor flip = at::zeros({N}, at::CUDA(at::kFloat)).uniform_(0., 1.);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      img.type(),
      "HorizFlipImagesAndBoxes",
      [&] {
        HorizFlipImagesAndBoxes<scalar_t><<<N, dim3(16, 16), 0, stream.stream()>>>(
          N,
          C,
          H,
          W,
          img.data<scalar_t>(),
          bboxes.data<float>(),
          bbox_offsets.data<int>(),
          p,
          flip.data<float>(),
          tmp_img.data<scalar_t>(),
          nhwc);
        THCudaCheck(cudaGetLastError());
      });

  // copy tmp_img -> img
  // img = tmp_img;

  return {tmp_img, bboxes};
}

