#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
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

import tensorflow as tf

_RESIZE_MIN = 256
_DEFAULT_IMAGE_SIZE = 224

__all__ = ['preprocess_image_record', 'preprocess_image_file']


def _deserialize_image_record(record):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        bbox = tf.stack([obj['image/object/bbox/%s' % x].values for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
        text = obj['image/class/text']
        return imgdata, label, bbox, text


def _decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels, fancy_upscaling=False, dct_method='INTEGER_FAST')


def _crop_and_filp(image, bbox, num_channels):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )

    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    cropped = tf.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.

    Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

    Returns:
    resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

    Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
    """
    return tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)


def preprocess_image_record(record, height, width, num_channels, is_training=False):

    imgdata, label, bbox, text = _deserialize_image_record(record)
    label -= 1
    
    try:
        image = _decode_jpeg(imgdata, channels=3)
    except:
        image = tf.image.decode_image(imgdata, channels=3)

    if is_training:
        # For training, we want to randomize some of the distortions.
        image = _crop_and_filp(image, bbox, num_channels)
        image = _resize_image(image, height, width)
    else:
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, height, width)

    return image, label


def preprocess_image_file(filename, height, width, num_channels, is_training=False):
    
    imgdata = tf.read_file(filename)

    try:
        image = _decode_jpeg(imgdata, channels=3)
    except:
        image = tf.image.decode_image(imgdata, channels=3)
        
    if is_training:
        # For training, we want to randomize some of the distortions.
        image = _crop_and_filp(image, bbox, num_channels)
        image = _resize_image(image, height, width)
    else:
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, height, width)

    return image, filename
