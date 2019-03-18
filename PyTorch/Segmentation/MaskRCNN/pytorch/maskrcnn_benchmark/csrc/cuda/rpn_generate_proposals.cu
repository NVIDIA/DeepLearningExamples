/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>

namespace {

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
    i += blockDim.x * gridDim.x)

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
// 1D grid
constexpr int CUDA_NUM_THREADS = 128;

constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int GetBlocks(const int N) {
  return std::max(
      std::min(
        (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
        MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

/**
 * d_sorted_score_keys -- indexes into _original_ scores
 * nboxes_to_generate -- pre_nms_topn
 */
__global__ void GeneratePreNMSUprightBoxesKernel(
    const long *d_sorted_scores_keys,
		const int nboxes_to_generate,
		const float *d_bbox_deltas,   // [N, A*4, H, W]
		const float4 *d_anchors,
		const int H,
		const int W,
		const int K, // K = H*W
		const int A,
		const int KA, // KA = K*A
		const float min_size,
		const float *d_img_info_vec,
		const int num_images,
		const float bbox_xform_clip,
		const bool correct_transform,
		float4 *d_out_boxes,
		const int prenms_nboxes, // leading dimension of out_boxes
		float *d_inout_scores, // [N, A, H, W]
		uint8_t *d_boxes_keep_flags) {
  // Going to generate pre_nms_nboxes boxes per image
  for (int ibox = blockIdx.x * blockDim.x + threadIdx.x; ibox < nboxes_to_generate;
    ibox += blockDim.x * gridDim.x) {
    for (int image_index = blockIdx.y * blockDim.y + threadIdx.y; image_index < num_images;
        image_index += blockDim.y * gridDim.y) {
      // box_conv_index : # of the same box, but indexed in
      // the scores from the conv layer, of shape (A,H,W)
      // the num_images dimension was already removed
      // box_conv_index = a*K + h*W + w
      // Note: PyT code takes topK, so need to adjust the indexing for multi-image
      // box_conv_index is _local_ to the image_index, need to adjust into global arrays
      const int box_conv_index = d_sorted_scores_keys[image_index * prenms_nboxes + ibox];

      // We want to decompose box_conv_index in (a,h,w)
      // such as box_conv_index = a*K + h*W + w
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int dA = K; // stride of A
      const int a = remaining / dA;
      remaining -= a*dA;
      const int dH = W; // stride of H
      const int h = remaining / dH;
      remaining -= h*dH;
      const int w = remaining; // dW = 1

      // Order of anchors is [N, H, W, A, 4]
      const int a_idx = h * W * A + w * A + a;
      const float4 anchor = d_anchors[image_index * KA + a_idx];

      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)

      float x1 = anchor.x;
      float x2 = anchor.z;
      float y1 = anchor.y;
      float y2 = anchor.w;

      // Deltas for that box
      // Deltas of shape (num_images,4*A,K)
      // We're going to compute 4 scattered reads
      // better than the alternative, ie transposing the complete deltas
      // array first
      int deltas_idx = image_index * (KA*4) + a*4*K+h*W+w;
      const float dx = d_bbox_deltas[deltas_idx];
      // Stride of K between each dimension
      deltas_idx += K; const float dy = d_bbox_deltas[deltas_idx];
      deltas_idx += K; float dw = d_bbox_deltas[deltas_idx];
      deltas_idx += K; float dh = d_bbox_deltas[deltas_idx];

      // Upper bound on dw,dh
      dw = fmin(dw, bbox_xform_clip);
      dh = fmin(dh, bbox_xform_clip);

      // Applying the deltas
      float width = x2 - x1 + 1.0f;
      const float ctr_x = x1 + 0.5f*width;
      const float pred_ctr_x = ctr_x + width*dx;
      const float pred_w = width*expf(dw);
      x1 = pred_ctr_x - 0.5f*pred_w;
      x2 = pred_ctr_x + 0.5f*pred_w;

      float height = y2 - y1 + 1.0f;
      const float ctr_y = y1 + 0.5f*height;
      const float pred_ctr_y = ctr_y + height*dy;
      const float pred_h = height*expf(dh);
      y1 = pred_ctr_y - 0.5f*pred_h;
      y2 = pred_ctr_y + 0.5f*pred_h;

      if(correct_transform) {
        x2 -= 1.0f;
        y2 -= 1.0f;
      }

      // End of box_coder.decode(..) part

      // Clipping box to image
      // p = _clip_box_to_image(proposal, height, width)
      const float img_height = d_img_info_vec[2*image_index+1];
      const float img_width = d_img_info_vec[2*image_index+0];
      const float min_size_scaled = min_size;
      x1 = fmax(fmin(x1, img_width-1.0f), 0.0f);
      y1 = fmax(fmin(y1, img_height-1.0f), 0.0f);
      x2 = fmax(fmin(x2, img_width-1.0f), 0.0f);
      y2 = fmax(fmin(y2, img_height-1.0f), 0.0f);

      // Filter boxes
      // Removing boxes with one dim < min_size
      // (center of box is in image, because of previous step)
      // keep = _filter_boxes(p, self.min_size, im_shape)
      width = x2 - x1 + 1.0f;
      height = y2 - y1 + 1.0f;
      bool keep_box = fmin(width, height) >= min_size_scaled;
      // We are not deleting the box right now even if !keep_box
      // we want to keep the relative order of the elements stable
      // we'll do it in such a way later
      // d_boxes_keep_flags size: (num_images,prenms_nboxes)
      // d_out_boxes size: (num_images,prenms_nboxes)
      const int out_index = image_index * prenms_nboxes + ibox;
      d_boxes_keep_flags[out_index] = keep_box;
      d_out_boxes[out_index] = {x1,y1,x2,y2};
    }
  }
}

} // namespace


/**
 * Generate boxes associated to topN pre-NMS scores
 */
std::vector<at::Tensor> GeneratePreNMSUprightBoxes(
        const int num_images,
        const int A,
        const int H,
        const int W,
        at::Tensor& sorted_indices, // topK sorted pre_nms_topn indices
        at::Tensor& sorted_scores,  // topK sorted pre_nms_topn scores [N, A, H, W]
        at::Tensor& bbox_deltas,    // [N, A*4, H, W] (full, unsorted / sliced)
        at::Tensor& anchors,        // input (full, unsorted, unsliced)
        at::Tensor& image_shapes,   // (h, w) of images
        const int pre_nms_nboxes,
        const int rpn_min_size,
        const float bbox_xform_clip_default,
        const bool correct_transform_coords) {
  // constants
  constexpr int box_dim = 4;
  const int K = H * W;

  // temp Tensors
  at::Tensor boxes = at::zeros({num_images, box_dim * pre_nms_nboxes}, sorted_scores.options()).to(at::kFloat);
  at::Tensor boxes_keep_flags = at::empty({num_images, pre_nms_nboxes}, sorted_scores.options()).to(at::kByte);
  boxes_keep_flags.zero_();

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Call kernel
  GeneratePreNMSUprightBoxesKernel<<<
      (GetBlocks(pre_nms_nboxes), num_images),
      CUDA_NUM_THREADS, // blockDim.y == 1
      0, stream>>>(
          sorted_indices.data<long>(),
          pre_nms_nboxes,
          bbox_deltas.data<float>(),
          reinterpret_cast<float4*>(anchors.data<float>()),
          H,
          W,
          K,
          A,
          K * A,
          rpn_min_size,
          image_shapes.data<float>(), // image size vec
          num_images,
          bbox_xform_clip_default, // utils::BBOX_XFORM_CLIP_DEFAULT
          correct_transform_coords,
          reinterpret_cast<float4*>(boxes.data<float>()),
          pre_nms_nboxes,
          sorted_scores.data<float>(),
          boxes_keep_flags.data<uint8_t>());
  THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>{boxes, sorted_scores, boxes_keep_flags};
}


