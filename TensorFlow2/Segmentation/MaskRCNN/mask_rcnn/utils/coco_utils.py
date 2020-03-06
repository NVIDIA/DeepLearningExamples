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
"""Util functions to manipulate masks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pycocotools.mask as coco_mask

POLYGON_PAD_VALUE = -3
POLYGON_SEPARATOR = -1
MASK_SEPARATOR = -2


def _np_array_split(a, v):
  """Split numpy array by separator value.

  Args:
    a: 1-D numpy.array.
    v: number. Separator value. e.g -1.

  Returns:
    2-D list of clean separated arrays.

  Example:
    a = [1, 2, 3, 4, -1, 5, 6, 7, 8]
    b = _np_array_split(a, -1)
    # Output: b = [[1, 2, 3, 4], [5, 6, 7, 8]]
  """
  a = np.array(a)
  arrs = np.split(a, np.where(a[:] == v)[0])
  return [e if (len(e) <= 0 or e[0] != v) else e[1:] for e in arrs]


def _unflat_polygons(x):
  """Unflats/recovers 1-d padded polygons to 3-d polygon list.

  Args:
    x: numpay.array. shape [num_elements, 1], num_elements = num_obj *
      num_vertex + padding.

  Returns:
    A list of three dimensions: [#obj, #polygon, #vertex]
  """
  num_segs = _np_array_split(x, MASK_SEPARATOR)
  polygons = []
  for s in num_segs:
    polygons.append(_np_array_split(s, POLYGON_SEPARATOR))
  polygons = [[polygon.tolist() for polygon in obj] for obj in polygons]  # pylint: disable=g-complex-comprehension
  return polygons


def _denormalize_to_coco_bbox(bbox, height, width):
  """Denormalize bounding box.

  Args:
    bbox: numpy.array[float]. Normalized bounding box. Format: ['ymin', 'xmin',
      'ymax', 'xmax'].
    height: int. image height.
    width: int. image width.

  Returns:
    [x, y, width, height]
  """
  y1, x1, y2, x2 = bbox
  y1 *= height
  x1 *= width
  y2 *= height
  x2 *= width
  box_height = y2 - y1
  box_width = x2 - x1
  return [float(x1), float(y1), float(box_width), float(box_height)]


def _extract_image_info(prediction, b):
  return {
      'id': int(prediction['source_id'][b]),
      'width': int(prediction['width'][b]),
      'height': int(prediction['height'][b]),
  }


def _extract_bbox_annotation(prediction, b, obj_i):
  """Constructs COCO format bounding box annotation."""
  height = prediction['height'][b]
  width = prediction['width'][b]

  bbox = _denormalize_to_coco_bbox(
    prediction['groundtruth_boxes'][b][obj_i, :], height, width)

  if 'groundtruth_area' in prediction:
    area = float(prediction['groundtruth_area'][b][obj_i])

  else:
    # Using the box area to replace the polygon area. This value will not affect
    # real evaluation but may fail the unit test.
    area = bbox[2] * bbox[3]

  annotation = {
      'id': b * 1000 + obj_i,  # place holder of annotation id.
      'image_id': int(prediction['source_id'][b]),  # source_id,
      'category_id': int(prediction['groundtruth_classes'][b][obj_i]),
      'bbox': bbox,
      'iscrowd': int(prediction['groundtruth_is_crowd'][b][obj_i]),
      'area': area,
      'segmentation': [],
  }
  return annotation


def _extract_polygon_info(prediction, polygons, b, obj_i):
  """Constructs 'area' and 'segmentation' fields.

  Args:
    prediction: dict[str, numpy.array]. Model outputs. The value dimension is
      [batch_size, #objects, #features, ...]
    polygons: list[list[list]]. Dimensions are [#objects, #polygon, #vertex].
    b: batch index.
    obj_i: object index.

  Returns:
    dict[str, numpy.array]. COCO format annotation with 'area' and
    'segmentation'.
  """
  annotation = {}
  if 'groundtruth_area' in prediction:
    groundtruth_area = float(prediction['groundtruth_area'][b][obj_i])
  else:
    height = prediction['height'][b]
    width = prediction['width'][b]
    rles = coco_mask.frPyObjects(polygons[obj_i], height, width)
    groundtruth_area = coco_mask.area(rles)
  annotation['area'] = groundtruth_area

  annotation['segmentation'] = polygons[obj_i]

  # Add dummy polygon to is_crowd instance.
  if not annotation['segmentation'][0]:
    # Adds a dummy polygon in case there is no segmentation.
    # Note that this could affect eval number in a very tiny amount since
    # for the instance without masks, it creates a fake single pixel mask
    # in the center of the box.
    height = prediction['height'][b]
    width = prediction['width'][b]
    bbox = _denormalize_to_coco_bbox(
      prediction['groundtruth_boxes'][b][obj_i, :], height, width)
    xcenter = bbox[0] + bbox[2] / 2.0
    ycenter = bbox[1] + bbox[3] / 2.0

    annotation['segmentation'] = [[
      xcenter, ycenter, xcenter, ycenter, xcenter, ycenter, xcenter, ycenter
    ]]

  return annotation


def _extract_categories(annotations):
  """Extract categories from annotations."""
  categories = {}
  for anno in annotations:
    category_id = int(anno['category_id'])
    categories[category_id] = {'id': category_id}
  return list(categories.values())


def extract_coco_groundtruth(prediction, include_mask=False):
  """Extract COCO format groundtruth.

  Args:
    prediction: dictionary of batch of prediction result. the first dimension
      each element is the batch.
    include_mask: True for including masks in the output annotations.

  Returns:
    Tuple of (images, annotations).
    images: list[dict].Required keys: 'id', 'width' and 'height'. The values are
      image id, width and height.
    annotations: list[dict]. Required keys: {'id', 'source_id', 'category_id',
      'bbox', 'iscrowd'} when include_mask=False. If include_mask=True, also
      required {'area', 'segmentation'}. The 'id' value is the annotation id
      and can be any **positive** number (>=1).
      Refer to http://cocodataset.org/#format-data for more details.
  Raises:
    ValueError: If any groundtruth fields is missing.
  """
  required_fields = [
      'source_id', 'width', 'height', 'num_groundtruth_labels',
      'groundtruth_boxes', 'groundtruth_classes'
  ]
  if include_mask:
    required_fields += ['groundtruth_polygons', 'groundtruth_area']
  for key in required_fields:
    if key not in prediction.keys():
      raise ValueError('Missing groundtruth field: "{}" keys: {}'.format(
          key, prediction.keys()))

  images = []
  annotations = []
  for b in range(prediction['source_id'].shape[0]):
    # Constructs image info.
    image = _extract_image_info(prediction, b)
    images.append(image)

    if include_mask:
      flatten_padded_polygons = prediction['groundtruth_polygons'][b]
      flatten_polygons = np.delete(
          flatten_padded_polygons,
          np.where(flatten_padded_polygons[:] == POLYGON_PAD_VALUE)[0])
      polygons = _unflat_polygons(flatten_polygons)

    # Constructs annotations.
    num_labels = prediction['num_groundtruth_labels'][b]
    for obj_i in range(num_labels):
      annotation = _extract_bbox_annotation(prediction, b, obj_i)

      if include_mask:
        polygon_info = _extract_polygon_info(prediction, polygons, b, obj_i)
        annotation.update(polygon_info)

      annotations.append(annotation)
  return images, annotations


def create_coco_format_dataset(images,
                               annotations,
                               regenerate_annotation_id=True):
  """Creates COCO format dataset with COCO format images and annotations."""
  if regenerate_annotation_id:
    for i in range(len(annotations)):
      # WARNING: The annotation id must be positive.
      annotations[i]['id'] = i + 1

  categories = _extract_categories(annotations)
  dataset = {
      'images': images,
      'annotations': annotations,
      'categories': categories,
  }
  return dataset
