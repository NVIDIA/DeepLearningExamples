# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
"""ROI-related ops."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from mrcnn_tf2.utils import box_utils


def _propose_rois_gpu(scores,
                      boxes,
                      anchor_boxes,
                      height,
                      width,
                      scale,
                      rpn_pre_nms_topn,
                      rpn_post_nms_topn,
                      rpn_nms_threshold,
                      rpn_min_size,
                      bbox_reg_weights):
    """Proposes RoIs giva group of candidates (GPU version).

    Args:
    scores: a tensor with a shape of [batch_size, num_boxes].
    boxes: a tensor with a shape of [batch_size, num_boxes, 4],
      in the encoded form.
    anchor_boxes: an Anchors object that contains the anchors with a shape of
      [batch_size, num_boxes, 4].
    height: a tensor of shape [batch_size, 1, 1] representing the image height.
    width: a tensor of shape [batch_size, 1, 1] representing the image width.
    scale: a tensor of shape [batch_size, 1, 1] representing the image scale.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    bbox_reg_weights: None or a list of four integer specifying the weights used
      when decoding the box.

    Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals. It has same dtype as input
      scores.
    boxes: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      represneting the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax]. It has same dtype as
      input boxes.
    """
    batch_size, num_boxes = scores.get_shape().as_list()

    topk_limit = min(num_boxes, rpn_pre_nms_topn)

    boxes = box_utils.decode_boxes(boxes, anchor_boxes, bbox_reg_weights)

    boxes = box_utils.clip_boxes(boxes, height, width)

    if rpn_min_size > 0.0:
        boxes, scores = box_utils.filter_boxes(
            boxes,
            tf.expand_dims(scores, axis=-1),
            rpn_min_size,
            height,
            width,
            scale
        )

        scores = tf.squeeze(scores, axis=-1)

    post_nms_topk_limit = topk_limit if topk_limit < rpn_post_nms_topn else rpn_post_nms_topn

    if rpn_nms_threshold > 0:
        # Normalize coordinates as combined_non_max_suppression currently
        # only support normalized coordinates.
        pre_nms_boxes = box_utils.to_normalized_coordinates(boxes, height, width)
        pre_nms_boxes = tf.reshape(pre_nms_boxes, [batch_size, num_boxes, 1, 4])
        pre_nms_scores = tf.reshape(scores, [batch_size, num_boxes, 1])

        # fixed problems when running with Keras AMP
        pre_nms_boxes = tf.cast(pre_nms_boxes, dtype=tf.float32)
        pre_nms_scores = tf.cast(pre_nms_scores, dtype=tf.float32)

        with tf.device('CPU:0'):
            boxes, scores, _, _ = tf.image.combined_non_max_suppression(
                pre_nms_boxes,
                pre_nms_scores,
                max_output_size_per_class=topk_limit,
                max_total_size=post_nms_topk_limit,
                iou_threshold=rpn_nms_threshold,
                score_threshold=0.0,
                pad_per_class=False
            )

        boxes = box_utils.to_absolute_coordinates(boxes, height, width)

    else:
        scores, boxes = box_utils.top_k(scores, k=post_nms_topk_limit, boxes_list=[boxes])
        boxes = boxes[0]

    return scores, boxes


def multilevel_propose_rois(scores_outputs,
                            box_outputs,
                            all_anchors,
                            image_info,
                            rpn_pre_nms_topn,
                            rpn_post_nms_topn,
                            rpn_nms_threshold,
                            rpn_min_size,
                            bbox_reg_weights):
    """Proposes RoIs given a group of candidates from different FPN levels.

    Args:
    scores_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4]
    all_anchors: an Anchors object that contains the all anchors.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    bbox_reg_weights: None or a list of four integer specifying the weights used
      when decoding the box.

    Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals.
    rois: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      representing the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('multilevel_propose_rois'):

        levels = scores_outputs.keys()
        scores = []
        rois = []
        anchor_boxes = all_anchors.get_unpacked_boxes()

        height = tf.expand_dims(image_info[:, 0:1], axis=-1)
        width = tf.expand_dims(image_info[:, 1:2], axis=-1)
        scale = tf.expand_dims(image_info[:, 2:3], axis=-1)

        for level in levels:

            with tf.name_scope('level_%d' % level) as scope:

                batch_size, feature_h, feature_w, num_anchors_per_location = scores_outputs[level].get_shape().as_list()
                num_boxes = feature_h * feature_w * num_anchors_per_location

                this_level_scores = tf.reshape(scores_outputs[level], [batch_size, num_boxes])
                this_level_scores = tf.sigmoid(this_level_scores)
                this_level_boxes = tf.reshape(box_outputs[level], [batch_size, num_boxes, 4])

                this_level_anchors = tf.cast(
                    tf.reshape(
                        tf.expand_dims(anchor_boxes[level], axis=0) *
                        tf.ones([batch_size, 1, 1, 1]),
                        [batch_size, num_boxes, 4]
                    ),
                    dtype=this_level_scores.dtype
                )

                this_level_scores, this_level_boxes = _propose_rois_gpu(
                    this_level_scores,
                    this_level_boxes,
                    this_level_anchors,
                    height,
                    width,
                    scale,
                    rpn_pre_nms_topn,
                    rpn_post_nms_topn,
                    rpn_nms_threshold,
                    rpn_min_size,
                    bbox_reg_weights
                )

                scores.append(this_level_scores)
                rois.append(this_level_boxes)

    scores = tf.concat(scores, axis=1)
    rois = tf.concat(rois, axis=1)

    with tf.name_scope('roi_post_nms_topk'):

        post_nms_num_anchors = scores.shape[1]
        post_nms_topk_limit = min(post_nms_num_anchors, rpn_post_nms_topn)

        top_k_scores, top_k_rois = box_utils.top_k(
            scores,
            k=post_nms_topk_limit,
            boxes_list=[rois]
        )

        top_k_rois = top_k_rois[0]

    return top_k_scores, top_k_rois
