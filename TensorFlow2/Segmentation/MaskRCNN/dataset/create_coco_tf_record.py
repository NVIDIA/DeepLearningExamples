# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Convert raw COCO dataset to TFRecord for object_detection.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import, division, print_function

import collections
import hashlib
import io
import json
import logging
import multiprocessing
import os

import PIL.Image
import numpy as np
import tensorflow as tf
from absl import app, flags
from pycocotools import mask
from research.object_detection.utils import dataset_util, label_map_util

flags.DEFINE_boolean('include_masks', False,
                     'Whether to include instance segmentations masks '
                     '(PNG encoded) in the result. default: False.')
flags.DEFINE_string('train_image_dir', '', 'Training image directory.')
flags.DEFINE_string('val_image_dir', '', 'Validation image directory.')
flags.DEFINE_string('test_image_dir', '', 'Test image directory.')
flags.DEFINE_string('train_object_annotations_file', '', '')
flags.DEFINE_string('val_object_annotations_file', '', '')
flags.DEFINE_string('train_caption_annotations_file', '', '')
flags.DEFINE_string('val_caption_annotations_file', '', '')
flags.DEFINE_string('testdev_annotations_file', '',
                    'Test-dev annotations JSON file.')
flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS


def create_tf_example(image,
                      bbox_annotations,
                      caption_annotations,
                      image_dir,
                      category_index,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.

    Args:
      image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
      bbox_annotations:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
      image_dir: directory containing the image files.
      category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in bbox_annotations:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])

        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    captions = []
    for caption_annotation in caption_annotations:
        captions.append(caption_annotation['caption'].encode('utf8'))

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/caption':
            dataset_util.bytes_list_feature(captions),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def _pool_create_tf_example(args):
    return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
    with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)

    images = obj_annotations['images']
    category_index = label_map_util.create_category_index(
        obj_annotations['categories'])

    img_to_obj_annotation = collections.defaultdict(list)
    logging.info('Building bounding box index.')
    for annotation in obj_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_obj_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1

    logging.info('%d images are missing bboxes.', missing_annotation_count)

    return images, img_to_obj_annotation, category_index


def _load_caption_annotations(caption_annotations_file):
    with tf.io.gfile.GFile(caption_annotations_file, 'r') as fid:
        caption_annotations = json.load(fid)

    img_to_caption_annotation = collections.defaultdict(list)
    logging.info('Building caption index.')
    for annotation in caption_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_caption_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    images = caption_annotations['images']
    for image in images:
        image_id = image['id']
        if image_id not in img_to_caption_annotation:
            missing_annotation_count += 1

    logging.info('%d images are missing captions.', missing_annotation_count)

    return img_to_caption_annotation


def _create_tf_record_from_coco_annotations(
        object_annotations_file,
        caption_annotations_file,
        image_dir, output_path, include_masks, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
      object_annotations_file: JSON file containing bounding box annotations.
      caption_annotations_file: JSON file containing caption annotations.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
      num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)
    writers = [
        tf.io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                             (i, num_shards)) for i in range(num_shards)
    ]

    images, img_to_obj_annotation, category_index = (
        _load_object_annotations(object_annotations_file))
    img_to_caption_annotation = (
        _load_caption_annotations(caption_annotations_file))

    pool = multiprocessing.Pool()
    total_num_annotations_skipped = 0
    for idx, (_, tf_example, num_annotations_skipped) in enumerate(
            pool.imap(_pool_create_tf_example,
                      [(image,
                        img_to_obj_annotation[image['id']],
                        img_to_caption_annotation[image['id']],
                        image_dir,
                        category_index,
                        include_masks)
                       for image in images])):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(images))

        total_num_annotations_skipped += num_annotations_skipped
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()

    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    assert FLAGS.test_image_dir, '`test_image_dir` missing.'

    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'train')
    val_output_path = os.path.join(FLAGS.output_dir, 'val')
    testdev_output_path = os.path.join(FLAGS.output_dir, 'test-dev')

    _create_tf_record_from_coco_annotations(
        FLAGS.train_object_annotations_file,
        FLAGS.train_caption_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        FLAGS.include_masks,
        num_shards=256)
    _create_tf_record_from_coco_annotations(
        FLAGS.val_object_annotations_file,
        FLAGS.val_caption_annotations_file,
        FLAGS.val_image_dir,
        val_output_path,
        FLAGS.include_masks,
        num_shards=32)


if __name__ == '__main__':
    app.run(main)
