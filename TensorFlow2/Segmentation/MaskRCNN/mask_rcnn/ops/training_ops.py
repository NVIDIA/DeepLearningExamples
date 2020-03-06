#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Training specific ops, including sampling, building targets, etc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mask_rcnn.utils import box_utils
from mask_rcnn.ops import spatial_transform_ops

from mask_rcnn.object_detection import balanced_positive_negative_sampler

_EPSILON = 1e-8


def _add_class_assignments(iou, gt_boxes, gt_labels):
    """Computes object category assignment for each box.

  Args:
    iou: a tensor for the iou matrix with a shape of
      [batch_size, K, MAX_NUM_INSTANCES]. K is the number of post-nms RoIs
      (i.e., rpn_post_nms_topn).
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4].
      This tensor might have paddings with negative values. The coordinates
      of gt_boxes are in the pixel coordinates of the scaled image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
  Returns:
    max_boxes: a tensor with a shape of [batch_size, K, 4], representing
      the ground truth coordinates of each roi.
    max_classes: a int32 tensor with a shape of [batch_size, K], representing
      the ground truth class of each roi.
    max_overlap: a tensor with a shape of [batch_size, K], representing
      the maximum overlap of each roi.
    argmax_iou: a tensor with a shape of [batch_size, K], representing the iou
      argmax.
  """
    with tf.name_scope('add_class_assignments'):
        batch_size, _, _ = iou.get_shape().as_list()

        argmax_iou = tf.argmax(input=iou, axis=2, output_type=tf.int32)

        indices = tf.reshape(
            argmax_iou + tf.expand_dims(tf.range(batch_size) * tf.shape(input=gt_labels)[1], 1),
            shape=[-1]
        )

        max_classes = tf.reshape(tf.gather(tf.reshape(gt_labels, [-1, 1]), indices), [batch_size, -1])

        max_overlap = tf.reduce_max(input_tensor=iou, axis=2)

        bg_mask = tf.equal(max_overlap, tf.zeros_like(max_overlap))

        max_classes = tf.where(bg_mask, tf.zeros_like(max_classes), max_classes)

        max_boxes = tf.reshape(
            tf.gather(tf.reshape(gt_boxes, [-1, 4]), indices),
            [batch_size, -1, 4]
        )

        max_boxes = tf.where(
            tf.tile(tf.expand_dims(bg_mask, axis=2), [1, 1, 4]),
            tf.zeros_like(max_boxes),
            max_boxes
        )

    return max_boxes, max_classes, max_overlap, argmax_iou


def encode_box_targets(boxes, gt_boxes, gt_labels, bbox_reg_weights):
    """Encodes predicted boxes with respect to ground truth boxes."""
    with tf.name_scope('encode_box_targets'):
        box_targets = box_utils.encode_boxes(boxes=gt_boxes, anchors=boxes, weights=bbox_reg_weights)
        # If a target is background, the encoded box target should be zeros.
        mask = tf.tile(tf.expand_dims(tf.equal(gt_labels, tf.zeros_like(gt_labels)), axis=2), [1, 1, 4])
        box_targets = tf.where(mask, tf.zeros_like(box_targets), box_targets)

    return box_targets


def proposal_label_op(boxes, gt_boxes, gt_labels,
                      batch_size_per_im=512, fg_fraction=0.25, fg_thresh=0.5,
                      bg_thresh_hi=0.5, bg_thresh_lo=0.):
    """Assigns the proposals with ground truth labels and performs subsmpling.

    Given proposal `boxes`, `gt_boxes`, and `gt_labels`, the function uses the
    following algorithm to generate the final `batch_size_per_im` RoIs.
    1. Calculates the IoU between each proposal box and each gt_boxes.
    2. Assigns each proposal box with a ground truth class and box label by
     choosing the largest overlap.
    3. Samples `batch_size_per_im` boxes from all proposal boxes, and returns
     box_targets, class_targets, and RoIs.
    The reference implementations of #1 and #2 are here:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py
    The reference implementation of #3 is here:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py

    Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates of scaled images in
      [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a value of -1. The coordinates of gt_boxes
      are in the pixel coordinates of the scaled image.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
    batch_size_per_im: a integer represents RoI minibatch size per image.
    fg_fraction: a float represents the target fraction of RoI minibatch that
      is labeled foreground (i.e., class > 0).
    fg_thresh: a float represents the overlap threshold for an RoI to be
      considered foreground (if >= fg_thresh).
    bg_thresh_hi: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    bg_thresh_lo: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    Returns:
    box_targets: a tensor with a shape of [batch_size, K, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi. K is the number of sample RoIs (e.g., batch_size_per_im).
    class_targets: a integer tensor with a shape of [batch_size, K]. The tensor
      contains the ground truth class for each roi.
    rois: a tensor with a shape of [batch_size, K, 4], representing the
      coordinates of the selected RoI.
    proposal_to_label_map: a tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels. proposal_to_label_map[i]
      means the index of the ground truth instance for the i-th proposal.
    """
    with tf.name_scope('proposal_label'):
        batch_size = boxes.shape[0]

        # The reference implementation intentionally includes ground truth boxes in
        # the proposals.
        # see https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py#L359
        boxes = tf.concat([boxes, gt_boxes], axis=1)
        iou = box_utils.bbox_overlap(boxes, gt_boxes)

        (pre_sample_box_targets, pre_sample_class_targets, max_overlap,
         proposal_to_label_map) = _add_class_assignments(iou, gt_boxes, gt_labels)

        # Generates a random sample of RoIs comprising foreground and background
        # examples.
        # reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py#L132
        positives = tf.greater(max_overlap,
                               fg_thresh * tf.ones_like(max_overlap))
        negatives = tf.logical_and(
            tf.greater_equal(max_overlap, bg_thresh_lo * tf.ones_like(max_overlap)),
            tf.less(max_overlap, bg_thresh_hi * tf.ones_like(max_overlap))
        )

        pre_sample_class_targets = tf.where(
            negatives,
            tf.zeros_like(pre_sample_class_targets),
            pre_sample_class_targets
        )

        proposal_to_label_map = tf.where(
            negatives,
            tf.zeros_like(proposal_to_label_map),
            proposal_to_label_map
        )

        # Handles ground truth paddings.
        ignore_mask = tf.less(tf.reduce_min(input_tensor=iou, axis=2), tf.zeros_like(max_overlap))

        # indicator includes both positive and negative labels.
        # labels includes only positives labels.
        # positives = indicator & labels.
        # negatives = indicator & !labels.
        # ignore = !indicator.
        labels = positives
        pos_or_neg = tf.logical_or(positives, negatives)
        indicator = tf.logical_and(pos_or_neg, tf.logical_not(ignore_mask))

        all_samples = []
        sampler = balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=fg_fraction,
            is_static=True
        )

        # Batch-unroll the sub-sampling process.
        for i in range(batch_size):
            samples = sampler.subsample(indicator[i], batch_size_per_im, labels[i])
            all_samples.append(samples)

        all_samples = tf.stack([all_samples], axis=0)[0]
        # A workaround to get the indices from the boolean tensors.
        _, samples_indices = tf.nn.top_k(tf.cast(all_samples, dtype=tf.int32), k=batch_size_per_im, sorted=True)

        # Contructs indices for gather.
        samples_indices = tf.reshape(
            samples_indices + tf.expand_dims(tf.range(batch_size) * tf.shape(input=boxes)[1], 1),
            [-1]
        )

        rois = tf.reshape(
            tf.gather(tf.reshape(boxes, [-1, 4]), samples_indices),
            [batch_size, -1, 4]
        )

        class_targets = tf.reshape(
            tf.gather(tf.reshape(pre_sample_class_targets, [-1, 1]), samples_indices),
            [batch_size, -1]
        )

        sample_box_targets = tf.reshape(
            tf.gather(tf.reshape(pre_sample_box_targets, [-1, 4]), samples_indices),
            [batch_size, -1, 4]
        )

        sample_proposal_to_label_map = tf.reshape(
            tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), samples_indices),
            [batch_size, -1]
        )

    return sample_box_targets, class_targets, rois, sample_proposal_to_label_map


def select_fg_for_masks(class_targets, box_targets, boxes, proposal_to_label_map, max_num_fg=128):
    """Selects the fore ground objects for mask branch during training.

    Args:
    class_targets: a tensor of shape [batch_size, num_boxes]  representing the
      class label for each box.
    box_targets: a tensor with a shape of [batch_size, num_boxes, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi.
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    proposal_to_label_map: a tensor with a shape of [batch_size, num_boxes].
      This tensor keeps the mapping between proposal to labels.
      proposal_to_label_map[i] means the index of the ground truth instance for
      the i-th proposal.
    max_num_fg: a integer represents the number of masks per image.
    Returns:
    class_targets, boxes, proposal_to_label_map, box_targets that have
    foreground objects.
    """

    # Masks are for positive (fg) objects only.
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py
    batch_size = boxes.shape[0]
    _, fg_indices = tf.nn.top_k(tf.cast(tf.greater(class_targets, 0), dtype=tf.float32), k=max_num_fg)

    # Contructs indices for gather.
    indices = tf.reshape(fg_indices + tf.expand_dims(tf.range(batch_size) * tf.shape(input=class_targets)[1], 1), [-1])

    fg_class_targets = tf.reshape(
        tf.gather(tf.reshape(class_targets, [-1, 1]), indices),
        [batch_size, -1]
    )

    fg_box_targets = tf.reshape(
        tf.gather(tf.reshape(box_targets, [-1, 4]), indices),
        [batch_size, -1, 4]
    )

    fg_box_rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), indices), [batch_size, -1, 4]
    )

    fg_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), indices),
        [batch_size, -1]
    )

    return (fg_class_targets, fg_box_targets, fg_box_rois,
            fg_proposal_to_label_map)


def get_mask_targets(fg_boxes, fg_proposal_to_label_map, fg_box_targets, mask_gt_labels, output_size=28):
    """Crop and resize on multilevel feature pyramid.

    Args:
    fg_boxes: A 3-D tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    fg_proposal_to_label_map: A tensor of shape [batch_size, num_masks].
    fg_box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_masks, 4].
    mask_gt_labels: A tensor with a shape of [batch_size, M, H+4, W+4]. M is
      NUM_MAX_INSTANCES (i.e., 100 in this implementation) in each image, while
      H and W are ground truth mask size. The `+4` comes from padding of two
      zeros in both directions of height and width dimension.
    output_size: A scalar to indicate the output crop size.

    Returns:
    A 4-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size].
    """

    _, _, max_feature_height, max_feature_width = mask_gt_labels.get_shape().as_list()

    # proposal_to_label_map might have a -1 paddings.
    levels = tf.maximum(fg_proposal_to_label_map, 0)

    # Projects box location and sizes to corresponding cropped ground truth
    # mask coordinates.
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(value=fg_boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(value=fg_box_targets, num_or_size_splits=4, axis=2)

    valid_feature_width = max_feature_width - 4
    valid_feature_height = max_feature_height - 4

    y_transform = (bb_y_min - gt_y_min) * valid_feature_height / (gt_y_max - gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * valid_feature_width / (gt_x_max - gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * valid_feature_height / (gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * valid_feature_width / (gt_x_max - gt_x_min + _EPSILON)

    boundaries = tf.concat(
        [
            tf.cast(tf.ones_like(y_transform) * (max_feature_height - 1), dtype=tf.float32),
            tf.cast(tf.ones_like(x_transform) * (max_feature_width - 1), dtype=tf.float32)
        ],
        axis=-1
    )

    features_per_box = spatial_transform_ops.selective_crop_and_resize(
        tf.expand_dims(mask_gt_labels, -1),
        tf.concat([y_transform, x_transform, h_transform, w_transform], -1),
        tf.expand_dims(levels, -1),
        boundaries,
        output_size
    )

    features_per_box = tf.squeeze(features_per_box, axis=-1)

    # Masks are binary outputs.
    features_per_box = tf.where(
        tf.greater_equal(features_per_box, 0.5),
        tf.ones_like(features_per_box),
        tf.zeros_like(features_per_box)
    )

    # mask_targets depend on box RoIs, which have gradients. This stop_gradient
    # prevents the flow of gradient to box RoIs.
    features_per_box = tf.stop_gradient(features_per_box)

    return features_per_box
