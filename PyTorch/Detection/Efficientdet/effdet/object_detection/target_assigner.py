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

"""Base target assigner module.

The job of a TargetAssigner is, for a given set of anchors (bounding boxes) and
groundtruth detections (bounding boxes), to assign classification and regression
targets to each anchor as well as weights to each anchor (specifying, e.g.,
which anchors should not contribute to training loss).

It assigns classification/regression targets by performing the following steps:
1) Computing pairwise similarity between anchors and groundtruth boxes using a
  provided RegionSimilarity Calculator
2) Computing a matching based on the similarity matrix using a provided Matcher
3) Assigning regression targets based on the matching and a provided BoxCoder
4) Assigning classification targets based on the matching and groundtruth labels

Note that TargetAssigners only operate on detections from a single
image at a time, so any logic for applying a TargetAssigner to multiple
images must be handled externally.
"""
import torch

from . import box_list
from .region_similarity_calculator import IouSimilarity
from .argmax_matcher import ArgMaxMatcher
from .matcher import Match
from .box_list import BoxList
from .box_coder import FasterRcnnBoxCoder

KEYPOINTS_FIELD_NAME = 'keypoints'


#@torch.jit.script
class TargetAssigner(object):
    """Target assigner to compute classification and regression targets."""

    def __init__(self, similarity_calc: IouSimilarity, matcher: ArgMaxMatcher, box_coder: FasterRcnnBoxCoder,
                 negative_class_weight: float = 1.0, unmatched_cls_target=None,
                 keypoints_field_name: str = KEYPOINTS_FIELD_NAME):
        """Construct Object Detection Target Assigner.

        Args:
            similarity_calc: a RegionSimilarityCalculator

            matcher: Matcher used to match groundtruth to anchors.

            box_coder: BoxCoder used to encode matching groundtruth boxes with respect to anchors.

            negative_class_weight: classification weight to be associated to negative
                anchors (default: 1.0). The weight must be in [0., 1.].

            unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
                which is consistent with the classification target for each
                anchor (and can be empty for scalar targets).  This shape must thus be
                compatible with the groundtruth labels that are passed to the "assign"
                function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
                If set to None, unmatched_cls_target is set to be [0] for each anchor.

        Raises:
            ValueError: if similarity_calc is not a RegionSimilarityCalculator or
                if matcher is not a Matcher or if box_coder is not a BoxCoder
        """
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_coder = box_coder
        self._negative_class_weight = negative_class_weight
        self._unmatched_cls_target = unmatched_cls_target
        self._keypoints_field_name = keypoints_field_name

    def assign(self, anchors: BoxList, groundtruth_boxes: BoxList, groundtruth_labels=None, groundtruth_weights=None):
        """Assign classification and regression targets to each anchor.

        For a given set of anchors and groundtruth detections, match anchors
        to groundtruth_boxes and assign classification and regression targets to
        each anchor as well as weights based on the resulting match (specifying,
        e.g., which anchors should not contribute to training loss).

        Anchors that are not matched to anything are given a classification target
        of self._unmatched_cls_target which can be specified via the constructor.

        Args:
            anchors: a BoxList representing N anchors

            groundtruth_boxes: a BoxList representing M groundtruth boxes

            groundtruth_labels:  a tensor of shape [M, d_1, ... d_k]
                with labels for each of the ground_truth boxes. The subshape
                [d_1, ... d_k] can be empty (corresponding to scalar inputs).  When set
                to None, groundtruth_labels assumes a binary problem where all
                ground_truth boxes get a positive label (of 1).

            groundtruth_weights: a float tensor of shape [M] indicating the weight to
                assign to all anchors match to a particular groundtruth box. The weights
                must be in [0., 1.]. If None, all weights are set to 1.

            **params: Additional keyword arguments for specific implementations of the Matcher.

        Returns:
            cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
                where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
                which has shape [num_gt_boxes, d_1, d_2, ... d_k].

            cls_weights: a float32 tensor with shape [num_anchors]

            reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]

            reg_weights: a float32 tensor with shape [num_anchors]

            match: a matcher.Match object encoding the match between anchors and groundtruth boxes,
                with rows corresponding to groundtruth boxes and columns corresponding to anchors.

        Raises:
            ValueError: if anchors or groundtruth_boxes are not of type box_list.BoxList
        """
        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')

        device = anchors.device()

        if groundtruth_labels is None:
            groundtruth_labels = torch.ones(groundtruth_boxes.num_boxes(), device=device).unsqueeze(0)
            groundtruth_labels = groundtruth_labels.unsqueeze(-1)

        if groundtruth_weights is None:
            num_gt_boxes = groundtruth_boxes.num_boxes()
            if not num_gt_boxes:
                num_gt_boxes = groundtruth_boxes.num_boxes()
            groundtruth_weights = torch.ones([num_gt_boxes], device=device)

        match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes, anchors)
        match = self._matcher.match(match_quality_matrix)
        reg_targets = self._create_regression_targets(anchors, groundtruth_boxes, match)
        cls_targets = self._create_classification_targets(groundtruth_labels, match)
        reg_weights = self._create_regression_weights(match, groundtruth_weights)
        cls_weights = self._create_classification_weights(match, groundtruth_weights)

        return cls_targets, cls_weights, reg_targets, reg_weights, match

    def _create_regression_targets(self, anchors: BoxList, groundtruth_boxes: BoxList, match: Match):
        """Returns a regression target for each anchor.

        Args:
            anchors: a BoxList representing N anchors

            groundtruth_boxes: a BoxList representing M groundtruth_boxes

            match: a matcher.Match object

        Returns:
            reg_targets: a float32 tensor with shape [N, box_code_dimension]
        """
        device = anchors.device()
        zero_box = torch.zeros(4, device=device)
        matched_gt_boxes = match.gather_based_on_match(
            groundtruth_boxes.boxes(), unmatched_value=zero_box, ignored_value=zero_box)
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
        if groundtruth_boxes.has_field(self._keypoints_field_name):
            groundtruth_keypoints = groundtruth_boxes.get_field(self._keypoints_field_name)
            zero_kp = torch.zeros(groundtruth_keypoints.shape[1:], device=device)
            matched_keypoints = match.gather_based_on_match(
                groundtruth_keypoints, unmatched_value=zero_kp, ignored_value=zero_kp)
            matched_gt_boxlist.add_field(self._keypoints_field_name, matched_keypoints)
        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)

        unmatched_ignored_reg_targets = self._default_regression_target(device).repeat(match.match_results.shape[0], 1)

        matched_anchors_mask = match.matched_column_indicator()
        reg_targets = torch.where(matched_anchors_mask.unsqueeze(1), matched_reg_targets, unmatched_ignored_reg_targets)
        return reg_targets

    def _default_regression_target(self, device: torch.device):
        """Returns the default target for anchors to regress to.

        Default regression targets are set to zero (though in this implementation what
        these targets are set to should not matter as the regression weight of any box
        set to regress to the default target is zero).

        Returns:
            default_target: a float32 tensor with shape [1, box_code_dimension]
        """
        return torch.zeros(1, self._box_coder.code_size(), device=device)

    def _create_classification_targets(self, groundtruth_labels, match: Match):
        """Create classification targets for each anchor.

        Assign a classification target of for each anchor to the matching
        groundtruth label that is provided by match.  Anchors that are not matched
        to anything are given the target self._unmatched_cls_target

        Args:
            groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
                with labels for each of the ground_truth boxes. The subshape
                [d_1, ... d_k] can be empty (corresponding to scalar labels).
            match: a matcher.Match object that provides a matching between anchors
                and groundtruth boxes.

        Returns:
            a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
            subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
            shape [num_gt_boxes, d_1, d_2, ... d_k].
        """
        if self._unmatched_cls_target is not None:
            uct = self._unmatched_cls_target
        else:
            uct = torch.scalar_tensor(0, device=groundtruth_labels.device)
        return match.gather_based_on_match(groundtruth_labels, unmatched_value=uct, ignored_value=uct)

    def _create_regression_weights(self, match: Match, groundtruth_weights):
        """Set regression weight for each anchor.

        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.

        Args:
            match: a matcher.Match object that provides a matching between anchors and groundtruth boxes.
            groundtruth_weights: a float tensor of shape [M] indicating the weight to
                assign to all anchors match to a particular groundtruth box.

        Returns:
            a float32 tensor with shape [num_anchors] representing regression weights.
        """
        zs = torch.scalar_tensor(0, device=groundtruth_weights.device)
        return match.gather_based_on_match(groundtruth_weights, ignored_value=zs, unmatched_value=zs)

    def _create_classification_weights(self, match: Match, groundtruth_weights):
        """Create classification weights for each anchor.

        Positive (matched) anchors are associated with a weight of
        positive_class_weight and negative (unmatched) anchors are associated with
        a weight of negative_class_weight. When anchors are ignored, weights are set
        to zero. By default, both positive/negative weights are set to 1.0,
        but they can be adjusted to handle class imbalance (which is almost always
        the case in object detection).

        Args:
            match: a matcher.Match object that provides a matching between anchors and groundtruth boxes.
            groundtruth_weights: a float tensor of shape [M] indicating the weight to
                assign to all anchors match to a particular groundtruth box.

        Returns:
            a float32 tensor with shape [num_anchors] representing classification weights.
        """
        ignored = torch.scalar_tensor(0, device=groundtruth_weights.device)
        ncw = torch.scalar_tensor(self._negative_class_weight, device=groundtruth_weights.device)
        return match.gather_based_on_match(groundtruth_weights, ignored_value=ignored, unmatched_value=ncw)

    def box_coder(self):
        """Get BoxCoder of this TargetAssigner.

        Returns:
            BoxCoder object.
        """
        return self._box_coder
