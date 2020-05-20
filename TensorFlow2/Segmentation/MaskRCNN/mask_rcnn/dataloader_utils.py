#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""

import tensorflow as tf

from mask_rcnn import anchors
from mask_rcnn.utils import coco_utils
from mask_rcnn.ops import preprocess_ops

from mask_rcnn.object_detection import tf_example_decoder

MAX_NUM_INSTANCES = 100
MAX_NUM_VERTICES_PER_INSTANCE = 1500
MAX_NUM_POLYGON_LIST_LEN = 2 * MAX_NUM_VERTICES_PER_INSTANCE * MAX_NUM_INSTANCES
POLYGON_PAD_VALUE = coco_utils.POLYGON_PAD_VALUE

__all__ = [
    # dataset parser
    "dataset_parser",
    # common functions
    "preprocess_image",
    "process_groundtruth_is_crowd",
    "process_source_id",
    # eval
    "prepare_labels_for_eval",
    # training
    "augment_image",
    "process_boxes_classes_indices_for_training",
    "process_gt_masks_for_training",
    "process_labels_for_training",
    "process_targets_for_training"
]


###############################################################################################################

def dataset_parser(value, mode, params, use_instance_mask, seed=None, regenerate_source_id=False):
    """Parse data to a fixed dimension input image and learning targets.

    Args:
    value: A dictionary contains an image and groundtruth annotations.

    Returns:
    features: a dictionary that contains the image and auxiliary
      information. The following describes {key: value} pairs in the
      dictionary.
      image: Image tensor that is preproessed to have normalized value and
        fixed dimension [image_size, image_size, 3]
      image_info: image information that includes the original height and
        width, the scale of the proccessed image to the original image, and
        the scaled height and width.
      source_ids: Source image id. Default value -1 if the source id is
        empty in the groundtruth annotation.
    labels: a dictionary that contains auxiliary information plus (optional)
      labels. The following describes {key: value} pairs in the dictionary.
      `labels` is only for training.
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of objectiveness score at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      gt_boxes: Groundtruth bounding box annotations. The box is represented
         in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
         fixed dimension [MAX_NUM_INSTANCES, 4].
      gt_classes: Groundtruth classes annotations. The tennsor is padded
        with -1 to the fixed dimension [MAX_NUM_INSTANCES].
      cropped_gt_masks: groundtrugh masks cropped by the bounding box and
        resized to a fixed size determined by params['gt_mask_size']
      regenerate_source_id: `bool`, if True TFExampleParser will use hashed
        value of `image/encoded` for `image/source_id`.
    """
    if mode not in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        raise ValueError("Unknown execution mode received: %s" % mode)

    def create_example_decoder():
        return tf_example_decoder.TfExampleDecoder(
            use_instance_mask=use_instance_mask,
            regenerate_source_id=regenerate_source_id
    )

    example_decoder = create_example_decoder()

    with tf.xla.experimental.jit_scope(compile_ops=True):

        with tf.name_scope('parser'):

            data = example_decoder.decode(value)

            data['groundtruth_is_crowd'] = process_groundtruth_is_crowd(data)

            image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)

            source_id = process_source_id(data['source_id'])

            if mode == tf.estimator.ModeKeys.PREDICT:

                features = {
                    'source_ids': source_id,
                }

                if params['visualize_images_summary']:
                    features['orig_images'] = tf.image.resize(image, params['image_size'])

                features["images"], features["image_info"], _, _ = preprocess_image(
                    image,
                    boxes=None,
                    instance_masks=None,
                    image_size=params['image_size'],
                    max_level=params['max_level'],
                    augment_input_data=False,
                    seed=seed
                )

                if params['include_groundtruth_in_features']:
                    labels = prepare_labels_for_eval(
                        data,
                        target_num_instances=MAX_NUM_INSTANCES,
                        target_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN,
                        use_instance_mask=params['include_mask']
                    )
                    return {'features': features, 'labels': labels}

                else:
                    return {'features': features}

            elif mode == tf.estimator.ModeKeys.TRAIN:

                labels = {}
                features = {
                    'source_ids': source_id
                }

                boxes, classes, indices, instance_masks = process_boxes_classes_indices_for_training(
                    data,
                    skip_crowd_during_training=params['skip_crowd_during_training'],
                    use_category=params['use_category'],
                    use_instance_mask=use_instance_mask
                )

                image, image_info, boxes, instance_masks = preprocess_image(
                    image,
                    boxes=boxes,
                    instance_masks=instance_masks,
                    image_size=params['image_size'],
                    max_level=params['max_level'],
                    augment_input_data=params['augment_input_data'],
                    seed=seed
                )

                features.update({
                    'images': image,
                    'image_info': image_info,
                })

                padded_image_size = image.get_shape().as_list()[:2]

                # Pads cropped_gt_masks.
                if use_instance_mask:
                    labels['cropped_gt_masks'] = process_gt_masks_for_training(
                        instance_masks,
                        boxes,
                        gt_mask_size=params['gt_mask_size'],
                        padded_image_size=padded_image_size,
                        max_num_instances=MAX_NUM_INSTANCES
                    )

                with tf.xla.experimental.jit_scope(compile_ops=False):
                    # Assign anchors.
                    (score_targets, box_targets), input_anchor = process_targets_for_training(
                        padded_image_size=padded_image_size,
                        boxes=boxes,
                        classes=classes,
                        params=params
                    )

                additional_labels = process_labels_for_training(
                    image_info, boxes, classes, score_targets, box_targets,
                    max_num_instances=MAX_NUM_INSTANCES,
                    min_level=params["min_level"],
                    max_level=params["max_level"]
                )

                labels.update(additional_labels)
                # labels["input_anchor"] = input_anchor

                # Features
                # {
                #   'source_ids': <tf.Tensor 'parser/StringToNumber:0' shape=() dtype=float32>,
                #   'images': <tf.Tensor 'parser/pad_to_bounding_box/Squeeze:0' shape=(1024, 1024, 3) dtype=float32>,
                #   'image_info': <tf.Tensor 'parser/stack_1:0' shape=(5,) dtype=float32>
                # }

                FAKE_FEATURES = False

                if FAKE_FEATURES:
                    labels["source_ids"] = tf.ones(shape=(), dtype=tf.float32)
                    labels["images"] = tf.ones(shape=(1024, 1024, 3), dtype=tf.float32)
                    labels["image_info"] = tf.ones(shape=(5,), dtype=tf.float32)

                # Labels
                # {
                #   'cropped_gt_masks': <tf.Tensor 'parser/Reshape_4:0' shape=(100, 116, 116) dtype=float32>,
                #   'score_targets_2': <tf.Tensor 'parser/Reshape_9:0' shape=(256, 256, 3) dtype=int32>,
                #   'box_targets_2': <tf.Tensor 'parser/Reshape_14:0' shape=(256, 256, 12) dtype=float32>,
                #   'score_targets_3': <tf.Tensor 'parser/Reshape_10:0' shape=(128, 128, 3) dtype=int32>,
                #   'box_targets_3': <tf.Tensor 'parser/Reshape_15:0' shape=(128, 128, 12) dtype=float32>,
                #   'score_targets_4': <tf.Tensor 'parser/Reshape_11:0' shape=(64, 64, 3) dtype=int32>,
                #   'box_targets_4': <tf.Tensor 'parser/Reshape_16:0' shape=(64, 64, 12) dtype=float32>,
                #   'score_targets_5': <tf.Tensor 'parser/Reshape_12:0' shape=(32, 32, 3) dtype=int32>,
                #   'box_targets_5': <tf.Tensor 'parser/Reshape_17:0' shape=(32, 32, 12) dtype=float32>,
                #   'score_targets_6': <tf.Tensor 'parser/Reshape_13:0' shape=(16, 16, 3) dtype=int32>,
                #   'box_targets_6': <tf.Tensor 'parser/Reshape_18:0' shape=(16, 16, 12) dtype=float32>,
                #   'gt_boxes': <tf.Tensor 'parser/Reshape_20:0' shape=(100, 4) dtype=float32>,
                #   'gt_classes': <tf.Tensor 'parser/Reshape_22:0' shape=(100, 1) dtype=float32>
                # }

                FAKE_LABELS = False

                if FAKE_LABELS:
                    labels["cropped_gt_masks"] = tf.ones(shape=(100, 116, 116), dtype=tf.float32)
                    labels["gt_boxes"] = tf.ones(shape=(100, 4), dtype=tf.float32)
                    labels["gt_classes"] = tf.ones(shape=(100, 1), dtype=tf.float32)

                    idx = 1
                    for dim in [256, 128, 64, 32, 16]:
                        idx += 1  # Starts at 2

                        labels["score_targets_%d" % idx] = tf.ones(shape=(dim, dim, 3), dtype=tf.float32)
                        labels["box_targets_%d" % idx] = tf.ones(shape=(dim, dim, 12), dtype=tf.float32)

                return features, labels

###############################################################################################################

# common functions


def preprocess_image(image, boxes, instance_masks, image_size, max_level, augment_input_data=False, seed=None):
    image = preprocess_ops.normalize_image(image)

    if augment_input_data:
        image, boxes, instance_masks = augment_image(image=image, boxes=boxes, instance_masks=instance_masks, seed=seed)

    # Scaling and padding.
    image, image_info, boxes, instance_masks = preprocess_ops.resize_and_pad(
        image=image,
        target_size=image_size,
        stride=2 ** max_level,
        boxes=boxes,
        masks=instance_masks
    )
    return image, image_info, boxes, instance_masks


def process_groundtruth_is_crowd(data):
    return tf.cond(
        pred=tf.greater(tf.size(input=data['groundtruth_is_crowd']), 0),
        true_fn=lambda: data['groundtruth_is_crowd'],
        false_fn=lambda: tf.zeros_like(data['groundtruth_classes'], dtype=tf.bool)
    )


# def process_source_id(data):
#     source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1', source_id)
#     source_id = tf.strings.to_number(source_id)
#     return source_id


def process_source_id(source_id):
    """Processes source_id to the right format."""
    if source_id.dtype == tf.string:
        source_id = tf.cast(tf.strings.to_number(source_id), tf.int64)

    with tf.control_dependencies([source_id]):
        source_id = tf.cond(
            tf.equal(tf.size(source_id), 0),
            lambda: tf.cast(tf.constant(-1), tf.int64),
            lambda: tf.identity(source_id)
        )

    return source_id


# eval
def prepare_labels_for_eval(
        data,
        target_num_instances=MAX_NUM_INSTANCES,
        target_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN,
        use_instance_mask=False
):

    """Create labels dict for infeed from data of tf.Example."""
    image = data['image']

    height, width = tf.shape(input=image)[:2]

    boxes = data['groundtruth_boxes']

    classes = tf.cast(data['groundtruth_classes'], dtype=tf.float32)

    num_labels = tf.shape(input=classes)[0]

    boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [target_num_instances, 4])
    classes = preprocess_ops.pad_to_fixed_size(classes, -1, [target_num_instances, 1])

    is_crowd = tf.cast(data['groundtruth_is_crowd'], dtype=tf.float32)
    is_crowd = preprocess_ops.pad_to_fixed_size(is_crowd, 0, [target_num_instances, 1])

    labels = dict()

    labels['width'] = width
    labels['height'] = height
    labels['groundtruth_boxes'] = boxes
    labels['groundtruth_classes'] = classes
    labels['num_groundtruth_labels'] = num_labels
    labels['groundtruth_is_crowd'] = is_crowd

    if use_instance_mask:
        data['groundtruth_polygons'] = preprocess_ops.pad_to_fixed_size(
            data=data['groundtruth_polygons'],
            pad_value=POLYGON_PAD_VALUE,
            output_shape=[target_polygon_list_len, 1]
        )

        if 'groundtruth_area' in data:
            labels['groundtruth_area'] = preprocess_ops.pad_to_fixed_size(
                data=labels['groundtruth_area'],
                pad_value=0,
                output_shape=[target_num_instances, 1]
            )

    return labels


# training
def augment_image(image, boxes, instance_masks, seed):
    flipped_results = preprocess_ops.random_horizontal_flip(
        image,
        boxes=boxes,
        masks=instance_masks,
        seed=seed
    )

    if instance_masks is not None:
        image, boxes, instance_masks = flipped_results

    else:
        image, boxes = flipped_results

    # image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
    # image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=seed)
    # image = tf.image.random_saturation(image, lower=0.9, upper=1.1, seed=seed)
    # image = tf.image.random_jpeg_quality(image, min_jpeg_quality=80, max_jpeg_quality=100, seed=seed)

    return image, boxes, instance_masks


def process_boxes_classes_indices_for_training(data, skip_crowd_during_training, use_category, use_instance_mask):
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
    indices = None
    instance_masks = None

    if not use_category:
        classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

    if skip_crowd_during_training:
        indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
        classes = tf.gather_nd(classes, indices)
        boxes = tf.gather_nd(boxes, indices)

        if use_instance_mask:
            instance_masks = tf.gather_nd(data['groundtruth_instance_masks'], indices)

    return boxes, classes, indices, instance_masks


def process_gt_masks_for_training(instance_masks, boxes, gt_mask_size, padded_image_size, max_num_instances):
    cropped_gt_masks = preprocess_ops.crop_gt_masks(
        instance_masks=instance_masks,
        boxes=boxes,
        gt_mask_size=gt_mask_size,
        image_size=padded_image_size
    )

    # cropped_gt_masks = tf.reshape(cropped_gt_masks, [max_num_instances, -1])

    cropped_gt_masks = preprocess_ops.pad_to_fixed_size(
        data=cropped_gt_masks,
        pad_value=-1,
        output_shape=[max_num_instances, (gt_mask_size + 4) ** 2]
    )

    return tf.reshape(cropped_gt_masks, [max_num_instances, gt_mask_size + 4, gt_mask_size + 4])


def process_labels_for_training(
    image_info, boxes, classes,
    score_targets, box_targets,
    max_num_instances, min_level, max_level
):
    labels = {}

    # Pad groundtruth data.
    # boxes *= image_info[2]
    boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [max_num_instances, 4])

    classes = preprocess_ops.pad_to_fixed_size(classes, -1, [max_num_instances, 1])

    for level in range(min_level, max_level + 1):
        labels['score_targets_%d' % level] = score_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]

    labels['gt_boxes'] = boxes
    labels['gt_classes'] = classes

    return labels


def process_targets_for_training(padded_image_size, boxes, classes, params):
    input_anchors = anchors.Anchors(
        params['min_level'],
        params['max_level'],
        params['num_scales'],
        params['aspect_ratios'],
        params['anchor_scale'],
        padded_image_size
    )

    anchor_labeler = anchors.AnchorLabeler(
        input_anchors,
        params['num_classes'],
        params['rpn_positive_overlap'],
        params['rpn_negative_overlap'],
        params['rpn_batch_size_per_im'],
        params['rpn_fg_fraction']
    )

    return anchor_labeler.label_anchors(boxes, classes), input_anchors
