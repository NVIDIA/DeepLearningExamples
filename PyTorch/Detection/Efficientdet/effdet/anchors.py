""" RetinaNet / EfficientDet Anchor Gen

Adapted for PyTorch from Tensorflow impl at
    https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py

Hacked together by Ross Wightman, original copyright below
"""
# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================

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

"""Anchor definition.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""
import collections
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops.boxes import remove_small_boxes, batched_nms

from effdet.object_detection import ArgMaxMatcher, FasterRcnnBoxCoder, BoxList, IouSimilarity, TargetAssigner
from .layers.nms_layer import batched_soft_nms

# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5

# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 5000

# The maximum number of detections per image.
MAX_DETECTIONS_PER_IMAGE = 100


def decode_box_outputs(rel_codes, anchors, output_xyxy: bool=False):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.

    """
    ycenter_a = (anchors[:, 0] + anchors[:, 2]) / 2
    xcenter_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]

    ty, tx, th, tw = rel_codes.unbind(dim=1)

    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    if output_xyxy:
        out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    return out


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.

    A configuration is a tuple of (num_anchors, scale, aspect_ratio).

    Args:
        min_level: integer number of minimum level of the output feature pyramid.

        max_level: integer number of maximum level of the output feature pyramid.

        num_scales: integer number representing intermediate scales added on each level.
            For instances, num_scales=2 adds two additional anchor scales [2^0, 2^0.5] on each level.

        aspect_ratios: list of tuples representing the aspect ratio anchors added on each level.
            For instances, aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

    Returns:
        anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
    """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append((2 ** level, scale_octave / float(num_scales), aspect))
    return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    """Generates multiscale anchor boxes.

    Args:
        image_size: integer number of input image size. The input image has the same dimension for
            width and height. The image_size should be divided by the largest feature stride 2^max_level.

        anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.

        anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

    Returns:
        anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all feature levels.

    Raises:
        ValueError: input size must be the multiple of largest feature stride.
    """
    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            if image_size % stride != 0:
                raise ValueError("input size must be divided by the stride.")
            base_anchor_size = anchor_scale * stride * 2 ** octave_scale
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes


def clip_boxes_xyxy(boxes: torch.Tensor, size: torch.Tensor):
    boxes = boxes.clamp(min=0)
    size = torch.cat([size, size], dim=0)
    boxes = boxes.min(size)
    return boxes


def generate_detections(
        cls_outputs, box_outputs, anchor_boxes, indices, classes, img_scale, img_size,
        max_det_per_image: int = MAX_DETECTIONS_PER_IMAGE, soft_nms: bool = False):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels. (k being MAX_DETECTION_POINTS)

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [MAX_DETECTION_POINTS, 6],
            each row representing [x, y, width, height, score, class]
    """
    anchor_boxes = anchor_boxes[indices, :]

    # apply bounding box regression to anchors
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes, output_xyxy=True)
    boxes = clip_boxes_xyxy(boxes, img_size / img_scale)  # clip before NMS better?

    scores = cls_outputs.sigmoid().squeeze(1).float()
    if soft_nms:
        top_detection_idx, soft_scores = batched_soft_nms(
            boxes, scores, classes, method_gaussian=True, iou_threshold=0.3, score_threshold=.001)
        scores[top_detection_idx] = soft_scores
    else:
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=0.5)

    # keep only topk scoring predictions
    top_detection_idx = top_detection_idx[:max_det_per_image]
    boxes = boxes[top_detection_idx]
    scores = scores[top_detection_idx, None]
    classes = classes[top_detection_idx, None]

    # xyxy to xywh & rescale to original image
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes *= img_scale

    classes += 1  # back to class idx with background class = 0

    # stack em and pad out to MAX_DETECTIONS_PER_IMAGE if necessary
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if len(top_detection_idx) < max_det_per_image:
        detections = torch.cat([
            detections,
            torch.zeros(
                (max_det_per_image - len(top_detection_idx), 6), device=detections.device, dtype=detections.dtype)
        ], dim=0)
    return detections


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: integer number of input image size. The input image has the
                same dimension for width and height. The image_size should be divided by
                the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        return _generate_anchor_configs(self.min_level, self.max_level, self.num_scales, self.aspect_ratios)

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale, self.config)
        boxes = torch.from_numpy(boxes).float()
        return boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


#@torch.jit.script
class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes.
    """

    def __init__(self, anchors, num_classes: int, match_threshold: float = 0.5):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            anchors: an instance of class Anchors.

            num_classes: integer number representing number of classes in the dataset.

            match_threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        similarity_calc = IouSimilarity()
        matcher = ArgMaxMatcher(
            match_threshold,
            unmatched_threshold=match_threshold,
            negatives_lower_than_unmatched=True,
            force_match_for_each_row=True)
        box_coder = FasterRcnnBoxCoder()

        self.target_assigner = TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.match_threshold = match_threshold
        self.num_classes = num_classes
        self.feat_size = {}
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            self.feat_size[level] = int(self.anchors.image_size / 2 ** level)
        self.indices_cache = {}

    def label_anchors(self, gt_boxes, gt_labels):
        """Labels anchors with ground truth inputs.

        Args:
            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
                For each row, it stores [y0, x0, y1, x1] for four corners of a box.

            gt_labels: A integer tensor with shape [N, 1] representing groundtruth classes.

        Returns:
            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.

            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at l-th level.

            num_positives: scalar tensor storing number of positives in an image.
        """
        cls_targets_out = dict()
        box_targets_out = dict()

        gt_box_list = BoxList(gt_boxes)
        anchor_box_list = BoxList(self.anchors.boxes)

        # cls_weights, box_weights are not used
        cls_targets, _, box_targets, _, matches = self.target_assigner.assign(anchor_box_list, gt_box_list, gt_labels)

        # class labels start from 1 and the background class = -1
        cls_targets -= 1
        cls_targets = cls_targets.long()

        # Unpack labels.
        """Unpacks an array of cls/box into multiple scales."""
        count = 0
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            feat_size = self.feat_size[level]
            steps = feat_size ** 2 * self.anchors.get_anchors_per_location()
            indices = torch.arange(count, count + steps, device=cls_targets.device)
            count += steps
            cls_key = 'cls_targets_%d' % (level - self.anchors.min_level + 1)
            bbox_key = 'box_targets_%d' % (level - self.anchors.min_level + 1)
            cls_targets_out[cls_key] = torch.index_select(cls_targets, 0, indices).view([feat_size, feat_size, -1])
            box_targets_out[bbox_key] = torch.index_select(box_targets, 0, indices).view([feat_size, feat_size, -1])

        num_positives = (matches.match_results != -1).float().sum()

        return cls_targets_out, box_targets_out, num_positives

    def _build_indices(self, device):
        anchors_per_loc = self.anchors.get_anchors_per_location()
        indices_dict = {}
        count = 0
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            feat_size = self.feat_size[level]
            steps = feat_size ** 2 * anchors_per_loc
            indices = torch.arange(count, count + steps, device=device)
            indices_dict[level] = indices
            count += steps
        return indices_dict

    def _get_indices(self, device, level):
        if device not in self.indices_cache:
            self.indices_cache[device] = self._build_indices(device)
        return self.indices_cache[device][level]

    def batch_label_anchors(self, batch_size: int, gt_boxes, gt_classes):
        num_levels = self.anchors.max_level - self.anchors.min_level + 1
        cls_targets_out = [[] for _ in range(num_levels)]
        box_targets_out = [[] for _ in range(num_levels)]
        num_positives_out = []

        # FIXME this may be a bottleneck, would be faster if batched, or should be done in loader/dataset?
        anchor_box_list = BoxList(self.anchors.boxes)
        for i in range(batch_size):
            last_sample = i == batch_size - 1
            # cls_weights, box_weights are not used
            cls_targets, _, box_targets, _, matches = self.target_assigner.assign(
                anchor_box_list, BoxList(gt_boxes[i]), gt_classes[i])

            # class labels start from 1 and the background class = -1
            cls_targets -= 1
            cls_targets = cls_targets.long()

            # Unpack labels.
            """Unpacks an array of cls/box into multiple scales."""
            for level in range(self.anchors.min_level, self.anchors.max_level + 1):
                level_index = level - self.anchors.min_level
                feat_size = self.feat_size[level]
                indices = self._get_indices(cls_targets.device, level)
                cls_targets_out[level_index].append(
                    torch.index_select(cls_targets, 0, indices).view([feat_size, feat_size, -1]))
                box_targets_out[level_index].append(
                    torch.index_select(box_targets, 0, indices).view([feat_size, feat_size, -1]))
                if last_sample:
                    cls_targets_out[level_index] = torch.stack(cls_targets_out[level_index])
                    box_targets_out[level_index] = torch.stack(box_targets_out[level_index])

            num_positives_out.append((matches.match_results != -1).float().sum())
            if last_sample:
                num_positives_out = torch.stack(num_positives_out)

        return cls_targets_out, box_targets_out, num_positives_out

