/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "generate_mask_targets.h"
#include "box_iou.h"
#include "box_encode.h"
#include "match_proposals.h"

#ifdef WITH_CUDA
#include "cuda/rpn_generate_proposals.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("generate_mask_targets", &generate_mask_targets, "generate_mask_targets");
  m.def("box_iou", &box_iou, "box_iou");
  m.def("box_encode", &box_encode, "box_encode");
  m.def("match_proposals", &match_proposals, "match_proposals");
#ifdef WITH_CUDA
  m.def("GeneratePreNMSUprightBoxes", &rpn::GeneratePreNMSUprightBoxes, "RPN Proposal Generation");
#endif
}
