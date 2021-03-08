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
"""Preprocessing ops."""
import math
import tensorflow as tf

from mask_rcnn.object_detection import preprocessor


def normalize_image(image):
    """Normalize the image.

    Args:
    image: a tensor of shape [height, width, 3] in dtype=tf.float32.

    Returns:
    normalized_image: a tensor which has the same shape and dtype as image,
      with pixel values normalized.
    """
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.reshape(offset, shape=(1, 1, 3))

    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.reshape(scale, shape=(1, 1, 3))

    normalized_image = (image - offset) / scale

    return normalized_image


def random_horizontal_flip(image, boxes=None, masks=None, seed=None):
    """Random horizontal flip the image, boxes, and masks.

    Args:
    image: a tensor of shape [height, width, 3] representing the image.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    masks: (Optional) a tensor of shape [num_masks, height, width]
      representing the object masks. Note that the size of the mask is the
      same as the image.

    Returns:
    image: the processed image tensor after being randomly flipped.
    boxes: None or the processed box tensor after being randomly flipped.
    masks: None or the processed mask tensor after being randomly flipped.
    """
    return preprocessor.random_horizontal_flip(image, boxes, masks, seed=seed)


def resize_and_pad(image, target_size, stride, boxes=None, masks=None):
    """Resize and pad images, boxes and masks.

    Resize and pad images, (optionally boxes and masks) given the desired output
    size of the image and stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `target_size`.
    2. Pad the rescaled image such that the height and width of the image become
     the smallest multiple of the stride that is larger or equal to the desired
     output diemension.

    Args:
    image: an image tensor of shape [original_height, original_width, 3].
    target_size: a tuple of two integers indicating the desired output
      image size. Note that the actual output size could be different from this.
    stride: the stride of the backbone network. Each of the output image sides
      must be the multiple of this.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    masks: (Optional) a tensor of shape [num_masks, height, width]
      representing the object masks. Note that the size of the mask is the
      same as the image.

    Returns:
    image: the processed image tensor after being resized and padded.
    image_info: a tensor of shape [5] which encodes the height, width before
      and after resizing and the scaling factor.
    boxes: None or the processed box tensor after being resized and padded.
      After the processing, boxes will be in the absolute coordinates w.r.t.
      the scaled image.
    masks: None or the processed mask tensor after being resized and padded.
    """

    input_height, input_width, _ = tf.unstack(
        tf.cast(tf.shape(input=image), dtype=tf.float32),
        axis=0
    )

    target_height, target_width = target_size

    scale_if_resize_height = target_height / input_height
    scale_if_resize_width = target_width / input_width

    scale = tf.minimum(scale_if_resize_height, scale_if_resize_width)

    scaled_height = tf.cast(scale * input_height, dtype=tf.int32)
    scaled_width = tf.cast(scale * input_width, dtype=tf.int32)

    image = tf.image.resize(image, [scaled_height, scaled_width], method=tf.image.ResizeMethod.BILINEAR)

    padded_height = int(math.ceil(target_height * 1.0 / stride) * stride)
    padded_width = int(math.ceil(target_width * 1.0 / stride) * stride)

    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_height, padded_width)
    image.set_shape([padded_height, padded_width, 3])

    image_info = tf.stack([
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32),
        1.0 / scale,
        input_height,
        input_width]
    )

    if boxes is not None:
        normalized_box_list = preprocessor.box_list.BoxList(boxes)
        scaled_boxes = preprocessor.box_list_scale(normalized_box_list, scaled_height, scaled_width).get()

    else:
        scaled_boxes = None

    if masks is not None:
        scaled_masks = tf.image.resize(
            tf.expand_dims(masks, -1),
            [scaled_height, scaled_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        # Check if there is any instance in this image or not.
        num_masks = tf.shape(input=scaled_masks)[0]
        scaled_masks = tf.cond(
            pred=tf.greater(num_masks, 0),
            true_fn=lambda: tf.image.pad_to_bounding_box(scaled_masks, 0, 0, padded_height, padded_width),
            false_fn=lambda: tf.zeros([0, padded_height, padded_width, 1])
        )

    else:
        scaled_masks = None

    return image, image_info, scaled_boxes, scaled_masks


def crop_gt_masks(instance_masks, boxes, gt_mask_size, image_size):
    """Crops the ground truth binary masks and resize to fixed-size masks."""
    num_masks = tf.shape(input=instance_masks)[0]

    scale_sizes = tf.convert_to_tensor(value=[image_size[0], image_size[1]] * 2, dtype=tf.float32)

    boxes = boxes / scale_sizes

    cropped_gt_masks = tf.image.crop_and_resize(
        image=instance_masks,
        boxes=boxes,
        box_indices=tf.range(num_masks, dtype=tf.int32),
        crop_size=[gt_mask_size, gt_mask_size],
        method='bilinear')[:, :, :, 0]

    cropped_gt_masks = tf.pad(
        tensor=cropped_gt_masks,
        paddings=tf.constant([[0, 0], [2, 2], [2, 2]]),
        mode='CONSTANT',
        constant_values=0.
    )

    return cropped_gt_masks


def pad_to_fixed_size(data, pad_value, output_shape):
    """Pad data to a fixed length at the first dimension.

    Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

    Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
    """
    max_num_instances = output_shape[0]
    dimension = output_shape[1]

    data = tf.reshape(data, [-1, dimension])
    num_instances = tf.shape(input=data)[0]

    pad_length = max_num_instances - num_instances

    paddings = pad_value * tf.ones([pad_length, dimension])

    padded_data = tf.reshape(tf.concat([data, paddings], axis=0), output_shape)
    return padded_data
