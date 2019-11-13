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

//#define DEBUG

// calculate the IoU of a single box against another box
__device__
float calc_single_iou(const float4 b1, const float4 b2) {
  // (lt), (rb)
  float l = max(b1.x, b2.x);
  float t = max(b1.y, b2.y);
  float r = min(b1.z, b2.z);
  float b = min(b1.w, b2.w);

  float first = (r - l);
  first = (first < 0) ? 0 : first;
  float second = (b - t);
  second = (second < 0) ? 0 : second;

  float intersection = first * second;

  float area1 = (b1.w - b1.y) * (b1.z - b1.x);
  float area2 = (b2.w - b2.y) * (b2.z - b2.x);

  return intersection / (area1 + area2 - intersection);
}

__global__
// boxes1 : [N x 4]
// boxes2 : [M x 4]
//   ious : [N x M]
void calc_ious_kernel(const int N_img, const float4 *box1, const int *box1_offsets,
                      const int M, const float4 *boxes2, float *ious) {

  // launch N_img blocks
  const int img = blockIdx.x;

  // each block, i will run over the box1_N[i] source and M target boxes
  // generating box1_N[i] x M outputs

  // alias to start of boxes for this image
  const float4 *b1 = &box1[box1_offsets[img]];

  if (threadIdx.x == 0) {
    //printf("offset for img %d : %d\n", img, box1_offsets[img]);
  }

  // number of boxes for this image from offsets
  int N = box1_offsets[img+1] - box1_offsets[img];

  for (int i = 0; i < N; ++i) {
    // if (threadIdx.x == 0) printf("i : %d\n", i);
    const float4 source = b1[i];
    // for each source, loop over targets
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
      const float4 target = boxes2[j];

      float iou = calc_single_iou(source, target);

      // store the calculated IoU in the correct spot
      int out_idx = box1_offsets[img] * M + i * M + j;
      ious[out_idx] = iou;

    }
  }
}

__device__
void reduce_val_idx(int N, volatile float *vals, volatile int *idx) {
  // naive: single thread for now
  if (threadIdx.x == 0) {
    float max_val = vals[0];
    int max_idx = idx[0];

    for (int i = 1; i < N; ++i) {
      if (vals[i] > max_val) {
        max_val = vals[i];
        max_idx = idx[i];
      }
    }

    vals[0] = max_val;
    idx[0] = max_idx;
  }
}

/**
 * perform remaining parts, storing temporary values in global workspace
 * workspace needs N_img * M values, each of 8 bytes (float, int)
 **/
template <int BLOCK_SIZE, int MAX_BBOXES_PER_BLOCK>
__global__
void encode(const int N_img, const float4 *bbox_in, const long *labels_in, const int *offsets,
            const int M, const float4 *dboxes, // const float *ious,
            const float criteria, uint8_t *workspace, float4 *bbox_out, long *label_out) {

  // Each block will take a single image's IoU set
  const int img = blockIdx.x;

  // shared memory for intermediate results
  __shared__ volatile float best_bbox_iou_tmp[BLOCK_SIZE];
  __shared__ volatile int best_bbox_idx_tmp[BLOCK_SIZE];

  // shared memory for final best_bbox_{iou, idx} values
  __shared__ volatile float best_bbox_iou[MAX_BBOXES_PER_BLOCK];
  __shared__ volatile int best_bbox_idx[MAX_BBOXES_PER_BLOCK];

  // index into the global workspace - each image needs (float + int) * M values
  volatile float *best_dbox_iou = (float *)&workspace[img * M * 8];
  volatile int *best_dbox_idx = (int *)&workspace[img * M * 8 + M * 4];

  // number of input bboxes for this image
  const int N_rows = offsets[img+1] - offsets[img];

  // Check for potential crash
  assert(N_rows <= MAX_BBOXES_PER_BLOCK);
#ifdef DEBUG
  if (threadIdx.x == 0)
    printf("N rows: %d %d to %d (%p - %p)\n", N_rows, offsets[img], offsets[img+1], best_dbox_iou, best_dbox_idx);
#endif

  for (int i = threadIdx.x; i < MAX_BBOXES_PER_BLOCK; i += blockDim.x) {
    best_bbox_iou[i] = -FLT_MAX;
    best_bbox_idx[i] = -1;
  }
  __syncthreads();

  // loop serially over the rows of the IoU set that correspond to this image
  int row_num = 0;
  for (int i = offsets[img]; i < offsets[img+1]; ++i) {
    // reset shmem tallies
    best_bbox_iou_tmp[threadIdx.x] = -FLT_MAX;
    best_bbox_idx_tmp[threadIdx.x] = -1;

    // index into the input buffer
    // const float *row = &ious[i * M];
    const float4 input_bbox = bbox_in[i];
#ifdef DEBUG
    if (threadIdx.x == 0)
      printf("%d - %p\n", img, &input_bbox);
#endif

    // loop by threads over the columns
    for (int j = threadIdx.x; j < M; j += blockDim.x) {

      // check and store new max if necessary
      const float4 input_dbox = dboxes[j];
      // float new_val = row[j];
      float new_val = calc_single_iou(input_bbox, input_dbox);

      // handle per-row max in shared memory
      if (new_val > best_bbox_iou_tmp[threadIdx.x]) {
        best_bbox_iou_tmp[threadIdx.x] = new_val;
        best_bbox_idx_tmp[threadIdx.x] = j;
      }

      // handle per-col max in global workspace
      if (new_val > best_dbox_iou[j]) {
        best_dbox_iou[j] = new_val;
        best_dbox_idx[j] = row_num;

#ifdef DEBUG
        assert(best_dbox_idx[j] >= 0);
        assert(best_dbox_idx[j] < N_rows);
#endif
      }
    }

    // Now we have all the values for this row -- reduce
    __syncthreads();

    // reduce - output is in max_{val, idx}_row[0]
    reduce_val_idx(blockDim.x, best_bbox_iou_tmp, best_bbox_idx_tmp);
#ifdef DEBUG
    __syncthreads();
#endif


    // store output for row i
    if (threadIdx.x == 0) {
      best_bbox_iou[row_num] = best_bbox_iou_tmp[0];
      best_bbox_idx[row_num] = best_bbox_idx_tmp[0];

#ifdef DEBUG
      assert(best_bbox_idx[row_num] >= 0);
      assert(best_bbox_idx[row_num] < M);
#endif
    }
    __syncthreads();

    // keep track of _local_ row
    row_num++;
  }

#ifdef DEBUG
  if (threadIdx.x == 0) {
    for (int i = 0; i < N_rows; ++i) {
      printf("%d - row : %d : best bbox_idx: %d\n", img, i, best_bbox_idx[i]);
    }
  }
#endif

#ifdef DEBUG
  // make sure all best_bbox_{iou, val} are seen by everyone
  __syncthreads();
#endif
  // At this point we have the maximum values & indices for both bbox and dbox
  /*
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx
  */
  for (int i = threadIdx.x; i < N_rows; i += blockDim.x) {
    int idx = best_bbox_idx[i];

#ifdef DEBUG
    assert(idx < M);
    assert(idx >= 0);
#endif

    best_dbox_iou[idx] = 2.;
    best_dbox_idx[idx] = i;
#ifdef DEBUG
    printf("%d - set best dbox_idx[%d] to %d\n", img, best_bbox_idx[i], i);
#endif
  }

  /**
        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        #print(maxloc.shape, labels_in.shape, labels_out.shape)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        # Transform format to xywh format
        x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out
  **/
  __syncthreads();
  for (int i = threadIdx.x; i < M; i += blockDim.x) {
    // offset into output arrays: M values per image
    // int output_idx = offsets[img] * M + i;
    int output_idx = img * M + i;

    // reset output labels to background
    // NOTE: bbox_out is already cloned from dbox outside of this kernel
    label_out[output_idx] = 0;

    // Filter IoU > 0.5
    bool mask = best_dbox_iou[i] > criteria;

    float4 bbox = bbox_out[output_idx];
    // copy some labels and bboxes
    if (mask) {
      // copy label
#ifdef DEBUG
      printf("%d : label: local input idx: %d, value: %d\n", i, best_dbox_idx[i], labels_in[offsets[img] + best_dbox_idx[i]]);
      // printf("%d : label: local input idx: %d, value: %d\n", i, best_dbox_idx[i], labels_in[offsets[img] + i]);
#endif
      label_out[output_idx] = labels_in[offsets[img] + best_dbox_idx[i]];

      // grab original box
      bbox = bbox_in[offsets[img] + best_dbox_idx[i]];
#ifdef DEBUG
      printf("mask %d : %d : %f %f %f %f\n", i, best_dbox_idx[i], bbox.x, bbox.y, bbox.z, bbox.w);
#endif
    }

    // transfer to xywh
    float4 bbox_tmp;
    bbox_tmp.x = 0.5 * (bbox.x + bbox.z);
    bbox_tmp.y = 0.5 * (bbox.y + bbox.w);
    bbox_tmp.z = bbox.z - bbox.x;
    bbox_tmp.w = bbox.w - bbox.y;

    // write out
    bbox_out[output_idx] = bbox_tmp;
  }
}

/**
    def encode(self, bboxes_in, labels_in, criteria = 0.5):

        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        #print(maxloc.shape, labels_in.shape, labels_out.shape)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        # Transform format to xywh format
        x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out
**/
std::vector<at::Tensor> box_encoder(const int N_img,
                                    const at::Tensor& bbox_input,
                                    const at::Tensor& bbox_offsets,
                                    const at::Tensor& labels_input,
                                    const at::Tensor& dbox,
                                    float criteria) {
  // Check everything is on the device
  AT_ASSERTM(bbox_input.type().is_cuda(), "bboxes must be a CUDA tensor");
  AT_ASSERTM(bbox_offsets.type().is_cuda(), "bbox offsets must be a CUDA tensor");
  AT_ASSERTM(labels_input.type().is_cuda(), "labels must be a CUDA tensor");
  AT_ASSERTM(dbox.type().is_cuda(), "dboxes must be a CUDA tensor");

  // Check at least offsets, bboxes and labels are consistent
  // Note: offsets is N+1 vs. N for labels
  AT_ASSERTM(N_img + 1 == bbox_offsets.numel(), "must have N_img+1 offsets");


  auto num_bbox_total = bbox_offsets[bbox_offsets.numel()-1].item<int>();
#ifdef DEBUG
  printf("%d : bboxes: %d\n", (int)bbox_offsets.numel(), num_bbox_total);
#endif
  AT_ASSERTM(num_bbox_total <= 2048, "total num bboxes must be <= 2048");

  AT_ASSERTM(bbox_input.size(0) == labels_input.size(0), "bbox and labels must have same leading dimension");

  const int N = bbox_input.size(0);
  const int M = dbox.size(0);

  auto stream = at::cuda::getCurrentCUDAStream();

  // allocate final outputs (known size)
#ifdef DEBUG
  printf("%d x %d\n", N_img * M, 4);
  // at::Tensor bbox_out = dbox.type().tensor({N_img * M, 4});
  printf("allocating %lu bytes for output labels\n", N_img*M*sizeof(long));
#endif
  at::Tensor labels_out = at::empty({N_img * M}, labels_input.options());
  THCudaCheck(cudaGetLastError());

  // copy default boxes to outputs
#ifdef DEBUG
  printf("allocating %lu bytes for output bboxes\n", N_img*M*4*sizeof(float));
#endif
  at::Tensor bbox_out = dbox.repeat({N_img, 1});
  THCudaCheck(cudaGetLastError());

  // need to allocate some workspace
#ifdef DEBUG
  printf("allocating %lu bytes for workspace\n", 8*M*N_img);
#endif
  // at::Tensor workspace = at::CUDA(at::kByte).zeros({8 * M * N_img});
  at::Tensor workspace = at::zeros({8 * M * N_img}, at::CUDA(at::kByte));
  THCudaCheck(cudaGetLastError());

  // Encode the inputs
  const int THREADS_PER_BLOCK = 256;
  encode<THREADS_PER_BLOCK, 256><<<N_img, THREADS_PER_BLOCK, 0, stream.stream()>>>(N_img,
                      (float4*)bbox_input.data<float>(),
                      labels_input.data<long>(),
                      bbox_offsets.data<int>(),
                      M,
                      (float4*)dbox.data<float>(),
                      criteria,
                      workspace.data<uint8_t>(),
                      (float4*)bbox_out.data<float>(),
                      labels_out.data<long>());

  THCudaCheck(cudaGetLastError());
  return {bbox_out, labels_out};
}

at::Tensor calc_ious(const int N_img,
                     const at::Tensor& boxes1,
                     const at::Tensor& boxes1_offsets,
                     const at::Tensor& boxes2) {

  const int N = boxes1.size(0);
  const int M = boxes2.size(0);

  auto stream = at::cuda::getCurrentCUDAStream();

  // at::Tensor ious = at::CUDA(at::kFloat).zeros({N, M});
  // at::Tensor ious = at::ones(at::CUDA(at::kFloat), {N, M});
  at::Tensor ious = at::empty({N, M}, boxes1.options());

  // Get IoU of all source x default box pairs
  calc_ious_kernel<<<N_img, 256, 0, stream.stream()>>>(
                        N_img,
                        (float4*)boxes1.data<float>(),
                        boxes1_offsets.data<int>(),
                        M,
                        (float4*)boxes2.data<float>(),
                        ious.data<float>());

  THCudaCheck(cudaGetLastError());
  return ious;
}
