""" PyTorch Soft-NMS
This code was adapted from a PR for detectron2 submitted by https://github.com/alekseynp
https://github.com/facebookresearch/detectron2/pull/1183/files
Detectron2 is licensed Apache 2.0, Copyright Facebook Inc.
"""
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import List
import time

import effdet_ext._C as _C

def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1[:, 4]  # [N,]
    area2 = boxes2[:, 4]  # [M,]

    width_height = torch.min(boxes1[:, None, 2:4], boxes2[:, 2:4]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def soft_nms(
    boxes,
    scores,
    method_gaussian: bool = True,
    sigma: float = 0.5,
    iou_threshold: float = .5,
    score_threshold: float = 0.005
):
    """
    Soft non-max suppression algorithm.
    Implementation of [Soft-NMS -- Improving Object Detection With One Line of Codec]
    (https://arxiv.org/abs/1704.04503)
    Args:
        boxes_remain (Tensor[N, ?]):
           boxes where NMS will be performed
           if Boxes, in (x1, y1, x2, y2) format
           if RotatedBoxes, in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores_remain (Tensor[N]):
           scores for each one of the boxes
        method_gaussian (bool): use gaussian method if True, otherwise linear        
        sigma (float):
           parameter for Gaussian penalty function
        iou_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        score_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    # st = time.perf_counter()
    device = boxes.device
    boxes_remain = boxes.clone()
    scores_remain = scores.clone()
    num_elem = scores_remain.size()[0]
    idxs = torch.arange(num_elem)
    idxs_out = torch.zeros(num_elem, dtype=torch.int64, device=device)
    scores_out = torch.zeros(num_elem, dtype=torch.float32, device=device)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes_remain = torch.cat((boxes_remain, area.unsqueeze(1)), dim=1) # [N, 5] BS, x1, y1, x2, y2, area
    count: int = 0
    # print("[SOFTMAX] before loop starts in softnms {}".format(time.perf_counter() - st))
    while scores_remain.numel() > 0:
        # st1 = time.perf_counter()
        top_idx = 0 # torch.argmax(scores_remain)
        idxs_out[count] = idxs[top_idx]
        scores_out[count] = scores_remain[top_idx]
        count += 1

        top_box = boxes_remain[top_idx]
        ious = pairwise_iou(top_box.unsqueeze(0), boxes_remain)[0]
        # st2 = time.perf_counter()
        # print("[SOFTMAX] Before gaussian in softnms {}".format(st2 - st1))

        if method_gaussian:
            decay = torch.exp(-torch.pow(ious, 2) / sigma)
        else:
            decay = torch.ones_like(ious)
            decay_mask = ious > iou_threshold
            decay[decay_mask] = 1 - ious[decay_mask]
        # st3 = time.perf_counter()
        # print("[SOFTMAX] Gaussian in softnms {}".format(st3 - st2))
        scores_remain *= decay
        keep = scores_remain > score_threshold
        keep[top_idx] = torch.tensor(False, device=device)

        boxes_remain = boxes_remain[keep]
        scores_remain = scores_remain[keep]
        idxs = idxs[keep]
        # st4 = time.perf_counter()
        # print("[SOFTMAX] Remaining in softnms {}".format(st4 - st3))
        # print("[SOFTMAX] Entire loop takes in softnms {}".format(st4 - st1))
    # st5 = time.perf_counter()
    # print("[SOFTMAX] Remaining in softnms {}".format(st5 - st))
    return idxs_out[:count], scores_out[:count]


def batched_nms(
    boxes, scores, idxs,
    iou_threshold: float = .5,):
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return _C.nms(
        boxes_for_nms, scores, iou_threshold
    )


def batched_soft_nms(
    boxes, scores, idxs,
    method_gaussian: bool = True,
    sigma: float = 0.5,
    iou_threshold: float = .5,
    score_threshold: float = 0.001):

    """
    Performs soft non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Args:
        boxes (Tensor[N, 4]):
           boxes where NMS will be performed. They
           are expected to be in (x1, y1, x2, y2) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        sigma (float):
           parameter for Gaussian penalty function
        iou_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        score_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return soft_nms(
        boxes_for_nms, scores, method_gaussian=method_gaussian, sigma=sigma,
        iou_threshold=iou_threshold, score_threshold=score_threshold
    )
