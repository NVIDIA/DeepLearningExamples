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
"""Ops used to post-process raw detections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mask_rcnn.utils import box_utils


def generate_detections_per_image_tpu(cls_outputs,
                                      box_outputs,
                                      anchor_boxes,
                                      image_info,
                                      pre_nms_num_detections=1000,
                                      post_nms_num_detections=100,
                                      nms_threshold=0.3,
                                      bbox_reg_weights=(10., 10., 5., 5.)):
    """Generate the final detections per image given the model outputs.

  Args:
    cls_outputs: a tensor with shape [N, num_classes], which stacks class
      logit outputs on all feature levels. The N is the number of total anchors
      on all levels. The num_classes is the number of classes predicted by the
      model. Note that the cls_outputs should be the output of softmax().
    box_outputs: a tensor with shape [N, num_classes*4], which stacks box
      regression outputs on all feature levels. The N is the number of total
      anchors on all levels.
    anchor_boxes: a tensor with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of total anchors on all levels.
    image_info: a tensor of shape [5] which encodes the input image's [height,
      width, scale, original_height, original_width]
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

  Returns:
    detections: Tuple of tensors corresponding to number of valid boxes,
    box coordinates, object categories for each boxes, and box scores
    -- respectively.
  """
    num_boxes, num_classes = cls_outputs.get_shape().as_list()

    # Remove background class scores.
    cls_outputs = cls_outputs[:, 1:num_classes]
    top_k_scores, top_k_indices_with_classes = tf.nn.top_k(
        tf.reshape(cls_outputs, [-1]),
        k=pre_nms_num_detections,
        sorted=False
    )

    classes = tf.math.mod(top_k_indices_with_classes, num_classes - 1)
    top_k_indices = tf.math.floordiv(top_k_indices_with_classes, num_classes - 1)

    anchor_boxes = tf.gather(anchor_boxes, top_k_indices)
    box_outputs = tf.reshape(box_outputs, [num_boxes, num_classes, 4])[:, 1:num_classes, :]

    class_indices = classes

    box_outputs = tf.gather_nd(box_outputs, tf.stack([top_k_indices, class_indices], axis=1))

    # apply bounding box regression to anchors
    boxes = box_utils.decode_boxes(box_outputs, anchor_boxes, bbox_reg_weights)
    boxes = box_utils.clip_boxes(boxes, image_info[0], image_info[1])

    list_of_all_boxes = []
    list_of_all_scores = []
    list_of_all_classes = []

    # Skip background class.
    for class_i in range(num_classes):
        # Compute bitmask for the given classes.
        class_i_bitmask = tf.cast(tf.equal(classes, class_i), top_k_scores.dtype)
        # This works because score is in [0, 1].
        class_i_scores = top_k_scores * class_i_bitmask

        # The TPU and CPU have different behaviors for
        # tf.image.non_max_suppression_padded (b/116754376).
        class_i_post_nms_indices, class_i_nms_num_valid = tf.image.non_max_suppression_padded(
            tf.cast(boxes, dtype=tf.float32),
            tf.cast(class_i_scores, dtype=tf.float32),
            post_nms_num_detections,
            iou_threshold=nms_threshold,
            score_threshold=0.05,
            pad_to_max_output_size=True,
            name='nms_detections_' + str(class_i)
        )

        class_i_post_nms_boxes = tf.gather(boxes, class_i_post_nms_indices)
        class_i_post_nms_scores = tf.gather(class_i_scores, class_i_post_nms_indices)

        mask = tf.less(tf.range(post_nms_num_detections), [class_i_nms_num_valid])

        class_i_post_nms_scores = tf.where(
            mask, class_i_post_nms_scores, tf.zeros_like(class_i_post_nms_scores)
        )

        class_i_classes = tf.fill(tf.shape(input=class_i_post_nms_scores), class_i + 1)
        list_of_all_boxes.append(class_i_post_nms_boxes)
        list_of_all_scores.append(class_i_post_nms_scores)
        list_of_all_classes.append(class_i_classes)

    post_nms_boxes = tf.concat(list_of_all_boxes, axis=0)
    post_nms_scores = tf.concat(list_of_all_scores, axis=0)
    post_nms_classes = tf.concat(list_of_all_classes, axis=0)

    # sort all results.
    post_nms_scores, sorted_indices = tf.nn.top_k(
        tf.cast(post_nms_scores, dtype=tf.float32),
        k=post_nms_num_detections,
        sorted=True
    )

    post_nms_boxes = tf.gather(post_nms_boxes, sorted_indices)
    post_nms_classes = tf.gather(post_nms_classes, sorted_indices)

    valid_mask = tf.where(
        tf.greater(post_nms_scores, 0), tf.ones_like(post_nms_scores),
        tf.zeros_like(post_nms_scores)
    )

    num_valid_boxes = tf.reduce_sum(input_tensor=valid_mask, axis=-1)
    box_classes = tf.cast(post_nms_classes, dtype=tf.float32)

    return num_valid_boxes, post_nms_boxes, box_classes, post_nms_scores


def generate_detections_tpu(class_outputs,
                            box_outputs,
                            anchor_boxes,
                            image_info,
                            pre_nms_num_detections=1000,
                            post_nms_num_detections=100,
                            nms_threshold=0.3,
                            bbox_reg_weights=(10., 10., 5., 5.)
                            ):
    """Generate the final detections given the model outputs (TPU version).

    Args:
    class_outputs: a tensor with shape [batch_size, N, num_classes], which
      stacks class logit outputs on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    box_outputs: a tensor with shape [batch_size, N, num_classes*4], which
      stacks box regression outputs on all feature levels. The N is the number
      of total anchors on all levels.
    anchor_boxes: a tensor with shape [batch_size, N, 4], which stacks anchors
      on all feature levels. The N is the number of total anchors on all levels.
    image_info: a tensor of shape [batch_size, 5] which encodes each image's
      [height, width, scale, original_height, original_width].
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

    Returns:
    a tuple of tensors corresponding to number of valid boxes,
    box coordinates, object categories for each boxes, and box scores stacked
    in batch_size.
    """

    with tf.name_scope('generate_detections'):

        batch_size, _, _ = class_outputs.get_shape().as_list()
        softmax_class_outputs = tf.nn.softmax(class_outputs)

        num_valid_boxes, box_coordinates, box_classes, box_scores = ([], [], [], [])

        for i in range(batch_size):
            result = generate_detections_per_image_tpu(
                softmax_class_outputs[i], box_outputs[i], anchor_boxes[i],
                image_info[i], pre_nms_num_detections, post_nms_num_detections,
                nms_threshold, bbox_reg_weights)

            num_valid_boxes.append(result[0])
            box_coordinates.append(result[1])
            box_classes.append(result[2])
            box_scores.append(result[3])

        num_valid_boxes = tf.stack(num_valid_boxes)
        box_coordinates = tf.stack(box_coordinates)
        box_classes = tf.stack(box_classes)
        box_scores = tf.stack(box_scores)

    return num_valid_boxes, box_coordinates, box_classes, box_scores


def generate_detections_gpu(class_outputs,
                            box_outputs,
                            anchor_boxes,
                            image_info,
                            pre_nms_num_detections=1000,
                            post_nms_num_detections=100,
                            nms_threshold=0.3,
                            bbox_reg_weights=(10., 10., 5., 5.)
                            ):
    """Generate the final detections given the model outputs (GPU version).

    Args:
    class_outputs: a tensor with shape [batch_size, N, num_classes], which
      stacks class logit outputs on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    box_outputs: a tensor with shape [batch_size, N, num_classes*4], which
      stacks box regression outputs on all feature levels. The N is the number
      of total anchors on all levels.
    anchor_boxes: a tensor with shape [batch_size, N, 4], which stacks anchors
      on all feature levels. The N is the number of total anchors on all levels.
    image_info: a tensor of shape [batch_size, 5] which encodes each image's
      [height, width, scale, original_height, original_width].
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

    Returns:
    a tuple of tensors corresponding to number of valid boxes,
    box coordinates, object categories for each boxes, and box scores stacked
    in batch_size.
    """
    with tf.name_scope('generate_detections'):

        batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list()
        softmax_class_outputs = tf.nn.softmax(class_outputs)

        # Remove background
        scores = tf.slice(softmax_class_outputs, [0, 0, 1], [-1, -1, -1])
        boxes = tf.slice(
            tf.reshape(box_outputs, [batch_size, num_boxes, num_classes, 4]),
            [0, 0, 1, 0], [-1, -1, -1, -1]
        )

        anchor_boxes = tf.expand_dims(anchor_boxes, axis=2) * tf.ones([1, 1, num_classes - 1, 1])

        num_detections = num_boxes * (num_classes - 1)

        boxes = tf.reshape(boxes, [batch_size, num_detections, 4])
        scores = tf.reshape(scores, [batch_size, num_detections, 1])
        anchor_boxes = tf.reshape(anchor_boxes, [batch_size, num_detections, 4])

        # Decode
        boxes = box_utils.decode_boxes(boxes, anchor_boxes, bbox_reg_weights)

        # Clip boxes
        height = tf.expand_dims(image_info[:, 0:1], axis=-1)
        width = tf.expand_dims(image_info[:, 1:2], axis=-1)
        boxes = box_utils.clip_boxes(boxes, height, width)

        # NMS
        pre_nms_boxes = box_utils.to_normalized_coordinates(boxes, height, width)
        pre_nms_boxes = tf.reshape(pre_nms_boxes, [batch_size, num_boxes, num_classes - 1, 4])
        pre_nms_scores = tf.reshape(scores, [batch_size, num_boxes, num_classes - 1])

        post_nms_boxes, post_nms_scores, post_nms_classes, \
        post_nms_num_valid_boxes = tf.image.combined_non_max_suppression(
            pre_nms_boxes,
            pre_nms_scores,
            max_output_size_per_class=pre_nms_num_detections,
            max_total_size=post_nms_num_detections,
            iou_threshold=nms_threshold,
            score_threshold=0.0,
            pad_per_class=False
        )

        post_nms_classes = post_nms_classes + 1

        post_nms_boxes = box_utils.to_absolute_coordinates(post_nms_boxes, height, width)

    return post_nms_num_valid_boxes, post_nms_boxes, tf.cast(post_nms_classes, dtype=tf.float32), post_nms_scores
