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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import atexit

import copy
import tempfile
import numpy as np

import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

import cv2


class MaskCOCO(COCO):
  """COCO object for mask evaluation.
  """

  def reset(self, dataset):
    """Reset the dataset and groundtruth data index in this object.

    Args:
      dataset: dict of groundtruth data. It should has similar structure as the
        COCO groundtruth JSON file. Must contains three keys: {'images',
          'annotations', 'categories'}.
        'images': list of image information dictionary. Required keys: 'id',
          'width' and 'height'.
        'annotations': list of dict. Bounding boxes and segmentations related
          information. Required keys: {'id', 'image_id', 'category_id', 'bbox',
            'iscrowd', 'area', 'segmentation'}.
        'categories': list of dict of the category information.
          Required key: 'id'.
        Refer to http://cocodataset.org/#format-data for more details.

    Raises:
      AttributeError: If the dataset is empty or not a dict.
    """
    assert dataset, 'Groundtruth should not be empty.'
    assert isinstance(dataset,
                      dict), 'annotation file format {} not supported'.format(
                          type(dataset))
    self.anns, self.cats, self.imgs = dict(), dict(), dict()
    self.dataset = copy.deepcopy(dataset)
    self.createIndex()

  def loadRes(self, detection_results, include_mask, is_image_mask=False):
    """Load result file and return a result api object.

    Args:
      detection_results: a dictionary containing predictions results.
      include_mask: a boolean, whether to include mask in detection results.
      is_image_mask: a boolean, where the predict mask is a whole image mask.

    Returns:
      res: result MaskCOCO api object
    """
    res = MaskCOCO()
    res.dataset['images'] = [img for img in self.dataset['images']]
    logging.info('Loading and preparing results...')
    predictions = self.load_predictions(
        detection_results,
        include_mask=include_mask,
        is_image_mask=is_image_mask)
    assert isinstance(predictions, list), 'results in not an array of objects'
    if predictions:
      image_ids = [pred['image_id'] for pred in predictions]
      assert set(image_ids) == (set(image_ids) & set(self.getImgIds())), \
             'Results do not correspond to current coco set'

      if (predictions and 'bbox' in predictions[0] and predictions[0]['bbox']):
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for idx, pred in enumerate(predictions):
          bb = pred['bbox']
          x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
          if 'segmentation' not in pred:
            pred['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
          pred['area'] = bb[2] * bb[3]
          pred['id'] = idx + 1
          pred['iscrowd'] = 0
      elif 'segmentation' in predictions[0]:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
        for idx, pred in enumerate(predictions):
          # now only support compressed RLE format as segmentation results
          pred['area'] = maskUtils.area(pred['segmentation'])
          if 'bbox' not in pred:
            pred['bbox'] = maskUtils.toBbox(pred['segmentation'])
          pred['id'] = idx + 1
          pred['iscrowd'] = 0

      res.dataset['annotations'] = predictions

    res.createIndex()
    return res

  def load_predictions(self,
                       detection_results,
                       include_mask,
                       is_image_mask=False):
    """Create prediction dictionary list from detection and mask results.

    Args:
      detection_results: a dictionary containing numpy arrays which corresponds
        to prediction results.
      include_mask: a boolean, whether to include mask in detection results.
      is_image_mask: a boolean, where the predict mask is a whole image mask.

    Returns:
      a list of dictionary including different prediction results from the model
        in numpy form.
    """
    predictions = []
    num_detections = detection_results['detection_scores'].size
    current_index = 0
    for i, image_id in enumerate(detection_results['source_id']):

      if include_mask:
        box_coorindates_in_image = detection_results['detection_boxes'][i]
        segments = generate_segmentation_from_masks(
            detection_results['detection_masks'][i],
            box_coorindates_in_image,
            int(detection_results['image_info'][i][3]),
            int(detection_results['image_info'][i][4]),
            is_image_mask=is_image_mask
        )

        # Convert the mask to uint8 and then to fortranarray for RLE encoder.
        encoded_masks = [
            maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
            for instance_mask in segments
        ]

      for box_index in range(int(detection_results['num_detections'][i])):
        if current_index % 1000 == 0:
          logging.info('{}/{}'.format(current_index, num_detections))

        current_index += 1

        prediction = {
            'image_id': int(image_id),
            'bbox': detection_results['detection_boxes'][i][box_index].tolist(),
            'score': detection_results['detection_scores'][i][box_index],
            'category_id': int(
                detection_results['detection_classes'][i][box_index]),
        }

        if include_mask:
          prediction['segmentation'] = encoded_masks[box_index]

        predictions.append(prediction)

    return predictions


def generate_segmentation_from_masks(masks,
                                     detected_boxes,
                                     image_height,
                                     image_width,
                                     is_image_mask=False):
  """Generates segmentation result from instance masks.

  Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.
    is_image_mask: bool. True: input masks are whole-image masks. False: input
      masks are bounding-box level masks.

  Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
  """

  def expand_boxes(boxes, scale):
    """Expands an array of boxes by a given scale."""
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227
    # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
    # whereas `boxes` here is in [x1, y1, w, h] form
    w_half = boxes[:, 2] * .5
    h_half = boxes[:, 3] * .5
    x_c = boxes[:, 0] + w_half
    y_c = boxes[:, 1] + h_half

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp

  # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812
  # To work around an issue with cv2.resize (it seems to automatically pad
  # with repeated border values), we manually zero-pad the masks by 1 pixel
  # prior to resizing back to the original image resolution. This prevents
  # "top hat" artifacts. We therefore need to expand the reference boxes by an
  # appropriate factor.

  _, mask_height, mask_width = masks.shape
  scale = max((mask_width + 2.0) / mask_width,
              (mask_height + 2.0) / mask_height)

  ref_boxes = expand_boxes(detected_boxes, scale)
  ref_boxes = ref_boxes.astype(np.int32)
  padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
  segms = []
  for mask_ind, mask in enumerate(masks):
    im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if is_image_mask:
      # Process whole-image masks.
      im_mask[:, :] = mask[:, :]
    else:
      # Process mask inside bounding boxes.
      padded_mask[1:-1, 1:-1] = mask[:, :]

      ref_box = ref_boxes[mask_ind, :]
      w = ref_box[2] - ref_box[0] + 1
      h = ref_box[3] - ref_box[1] + 1
      w = np.maximum(w, 1)
      h = np.maximum(h, 1)

      mask = cv2.resize(padded_mask, (w, h))
      mask = np.array(mask > 0.5, dtype=np.uint8)

      x_0 = max(ref_box[0], 0)
      x_1 = min(ref_box[2] + 1, image_width)
      y_0 = max(ref_box[1], 0)
      y_1 = min(ref_box[3] + 1, image_height)

      im_mask[y_0:y_1, x_0:x_1] = mask[(y_0 - ref_box[1]):(y_1 - ref_box[1]), (
          x_0 - ref_box[0]):(x_1 - ref_box[0])]
    segms.append(im_mask)

  segms = np.array(segms)
  assert masks.shape[0] == segms.shape[0]
  return segms


class EvaluationMetric(object):
  """COCO evaluation metric class."""

  def __init__(self, filename, include_mask):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _evaluate() loads a JSON file in COCO annotation format as the
    groundtruths and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation.
      include_mask: boolean to indicate whether or not to include mask eval.
    """
    if filename:
      if filename.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.io.gfile.remove(local_val_json)

        tf.io.gfile.copy(filename, local_val_json)
        atexit.register(tf.io.gfile.remove, local_val_json)
      else:
        local_val_json = filename
      self.coco_gt = MaskCOCO(local_val_json)
    self.filename = filename
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self._include_mask = include_mask
    if self._include_mask:
      mask_metric_names = ['mask_' + x for x in self.metric_names]
      self.metric_names.extend(mask_metric_names)

    self._reset()

  def _reset(self):
    """Reset COCO API object."""
    if self.filename is None and not hasattr(self, 'coco_gt'):
      self.coco_gt = MaskCOCO()

  def predict_metric_fn(self,
                        predictions,
                        is_predict_image_mask=False,
                        groundtruth_data=None):
    """Generates COCO metrics."""
    image_ids = list(set(predictions['source_id']))
    if groundtruth_data is not None:
      self.coco_gt.reset(groundtruth_data)
    coco_dt = self.coco_gt.loadRes(
        predictions, self._include_mask, is_image_mask=is_predict_image_mask)
    coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats

    if self._include_mask:
      # Create another object for instance segmentation metric evaluation.
      mcoco_eval = COCOeval(self.coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    if self._include_mask:
      metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
      metrics = coco_metrics

    # clean up after evaluation is done.
    self._reset()
    metrics = metrics.astype(np.float32)

    metrics_dict = {}
    for i, name in enumerate(self.metric_names):
      metrics_dict[name] = metrics[i]
    return metrics_dict
