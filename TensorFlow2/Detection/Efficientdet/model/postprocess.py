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
# =============================================================================
"""Postprocessing for anchor-based detection."""
import functools
from typing import List, Tuple

from absl import logging
import tensorflow as tf

from model import nms_np
from utils import model_utils
from model import anchors
T = tf.Tensor  # a shortcut for typing check.
CLASS_OFFSET = 1


def to_list(inputs):
  if isinstance(inputs, dict):
    return [inputs[k] for k in sorted(inputs.keys())]
  if isinstance(inputs, list):
    return inputs
  raise ValueError('Unrecognized inputs : {}'.format(inputs))


def batch_map_fn(map_fn, inputs, *args):
  """Apply map_fn at batch dimension."""
  if isinstance(inputs[0], (list, tuple)):
    batch_size = len(inputs[0])
  else:
    batch_size = inputs[0].shape.as_list()[0]

  if not batch_size:
    # handle dynamic batch size: tf.vectorized_map is faster than tf.map_fn.
    return tf.vectorized_map(map_fn, inputs, *args)

  outputs = []
  for i in range(batch_size):
    outputs.append(map_fn([x[i] for x in inputs]))
  return [tf.stack(y) for y in zip(*outputs)]


def clip_boxes(boxes: T, image_size: int) -> T:
  """Clip boxes to fit the image size."""
  image_size = model_utils.parse_image_size(image_size) * 2
  return tf.clip_by_value(boxes, [0], image_size)


def merge_class_box_level_outputs(params, cls_outputs: List[T],
                                  box_outputs: List[T]) -> Tuple[T, T]:
  """Concatenates class and box of all levels into one tensor."""
  cls_outputs_all, box_outputs_all = [], []
  batch_size = tf.shape(cls_outputs[0])[0]
  for level in range(0, params['max_level'] - params['min_level'] + 1):
    if params['data_format'] == 'channels_first':
      cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
      box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])
    cls_outputs_all.append(
        tf.reshape(cls_outputs[level], [batch_size, -1, params['num_classes']]))
    box_outputs_all.append(tf.reshape(box_outputs[level], [batch_size, -1, 4]))
  return tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)


def topk_class_boxes(params, cls_outputs: T,
                     box_outputs: T) -> Tuple[T, T, T, T]:
  """Pick the topk class and box outputs."""
  batch_size = tf.shape(cls_outputs)[0]
  num_classes = params['num_classes']

  max_nms_inputs = params['nms_configs'].get('max_nms_inputs', 0)
  if max_nms_inputs > 0:
    # Prune anchors and detections to only keep max_nms_inputs.
    # Due to some issues, top_k is currently slow in graph model.
    logging.info('use max_nms_inputs for pre-nms topk.')
    cls_outputs_reshape = tf.reshape(cls_outputs, [batch_size, -1])
    _, cls_topk_indices = tf.math.top_k(
        cls_outputs_reshape, k=max_nms_inputs, sorted=False)
    indices = cls_topk_indices // num_classes
    classes = cls_topk_indices % num_classes
    cls_indices = tf.stack([indices, classes], axis=2)

    cls_outputs_topk = tf.gather_nd(cls_outputs, cls_indices, batch_dims=1)
    box_outputs_topk = tf.gather_nd(
        box_outputs, tf.expand_dims(indices, 2), batch_dims=1)
  else:
    logging.info('use max_reduce for pre-nms topk.')
    # Keep all anchors, but for each anchor, just keep the max probablity for
    # each class.
    cls_outputs_idx = tf.math.argmax(cls_outputs, axis=-1, output_type=tf.int32)
    num_anchors = tf.shape(cls_outputs)[1]

    classes = cls_outputs_idx
    indices = tf.tile(
        tf.expand_dims(tf.range(num_anchors), axis=0), [batch_size, 1])
    cls_outputs_topk = tf.reduce_max(cls_outputs, -1)
    box_outputs_topk = box_outputs

  return cls_outputs_topk, box_outputs_topk, classes, indices


def pre_nms(params, cls_outputs, box_outputs, topk=True):
  """Detection post processing before nms.

  It takes the multi-level class and box predictions from network, merge them
  into unified tensors, and compute boxes, scores, and classes.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors].
    topk: if True, select topk before nms (mainly to speed up nms).

  Returns:
    A tuple of (boxes, scores, classes).
  """
  # get boxes by apply bounding box regression to anchors.
  eval_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                 params['num_scales'], params['aspect_ratios'],
                                 params['anchor_scale'], params['image_size'])

  cls_outputs, box_outputs = merge_class_box_level_outputs(
      params, cls_outputs, box_outputs)

  if topk:
    # select topK purely based on scores before NMS, in order to speed up nms.
    cls_outputs, box_outputs, classes, indices = topk_class_boxes(
        params, cls_outputs, box_outputs)
    anchor_boxes = tf.gather(eval_anchors.boxes, indices)
  else:
    anchor_boxes = eval_anchors.boxes
    classes = None

  boxes = anchors.decode_box_outputs(box_outputs, anchor_boxes)
  # convert logits to scores.
  scores = tf.math.sigmoid(cls_outputs)
  return boxes, scores, classes


def nms(params, boxes: T, scores: T, classes: T,
        padded: bool) -> Tuple[T, T, T, T]:
  """Non-maximum suppression.

  Args:
    params: a dict of parameters.
    boxes: a tensor with shape [N, 4], where N is the number of boxes. Box
      format is [y_min, x_min, y_max, x_max].
    scores: a tensor with shape [N].
    classes: a tensor with shape [N].
    padded: a bool vallue indicating whether the results are padded.

  Returns:
    A tuple (boxes, scores, classes, valid_lens), where valid_lens is a scalar
    denoting the valid length of boxes/scores/classes outputs.
  """
  nms_configs = params['nms_configs']
  method = nms_configs['method']
  max_output_size = nms_configs['max_output_size']

  if method == 'hard' or not method:
    # hard nms.
    sigma = 0.0
    iou_thresh = nms_configs['iou_thresh'] or 0.5
    score_thresh = nms_configs['score_thresh'] or float('-inf')
  elif method == 'gaussian':
    sigma = nms_configs['sigma'] or 0.5
    iou_thresh = 1.0
    score_thresh = nms_configs['score_thresh'] or 0.001
  else:
    raise ValueError('Inference has invalid nms method {}'.format(method))

  # TF API's sigma is twice as the paper's value, so here we divide it by 2:
  # https://github.com/tensorflow/tensorflow/issues/40253.
  nms_top_idx, nms_scores, nms_valid_lens = tf.raw_ops.NonMaxSuppressionV5(
      boxes=boxes,
      scores=scores,
      max_output_size=max_output_size,
      iou_threshold=iou_thresh,
      score_threshold=score_thresh,
      soft_nms_sigma=(sigma / 2),
      pad_to_max_output_size=padded)

  nms_boxes = tf.gather(boxes, nms_top_idx)
  nms_classes = tf.cast(
      tf.gather(classes, nms_top_idx) + CLASS_OFFSET, tf.float32)
  return nms_boxes, nms_scores, nms_classes, nms_valid_lens


def postprocess_combined(params, cls_outputs, box_outputs, image_scales=None):
  """Post processing with combined NMS.

  Leverage the tf combined NMS. It is fast on TensorRT, but slow on CPU/GPU.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [y_min,
      x_min, y_max, x_man].
    image_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  cls_outputs = to_list(cls_outputs)
  box_outputs = to_list(box_outputs)
  # Don't filter any outputs because combine_nms need the raw information.
  boxes, scores, _ = pre_nms(params, cls_outputs, box_outputs, topk=False)

  max_output_size = params['nms_configs']['max_output_size']
  score_thresh = params['nms_configs']['score_thresh'] or float('-inf')
  nms_boxes, nms_scores, nms_classes, nms_valid_len = (
      tf.image.combined_non_max_suppression(
          tf.expand_dims(boxes, axis=2),
          scores,
          max_output_size,
          max_output_size,
          score_threshold=score_thresh,
          clip_boxes=False))
  nms_classes += CLASS_OFFSET
  nms_boxes = clip_boxes(nms_boxes, params['image_size'])
  if image_scales is not None:
    scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
    nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
  return nms_boxes, nms_scores, nms_classes, nms_valid_len


def postprocess_global(params, cls_outputs, box_outputs, image_scales=None):
  """Post processing with global NMS.

  A fast but less accurate version of NMS. The idea is to treat the scores for
  different classes in a unified way, and perform NMS globally for all classes.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [y_min,
      x_min, y_max, x_man].
    image_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  cls_outputs = to_list(cls_outputs)
  box_outputs = to_list(box_outputs)
  boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)

  def single_batch_fn(element):
    return nms(params, element[0], element[1], element[2], True)

  nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
      single_batch_fn, [boxes, scores, classes])
  nms_boxes = clip_boxes(nms_boxes, params['image_size'])
  if image_scales is not None:
    scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
    nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
  return nms_boxes, nms_scores, nms_classes, nms_valid_len


def per_class_nms(params, boxes, scores, classes, image_scales=None):
  """Per-class nms, a utility for postprocess_per_class.

  Args:
    params: a dict of parameters.
    boxes: A tensor with shape [N, K, 4], where N is batch_size, K is num_boxes.
      Box format is [y_min, x_min, y_max, x_max].
    scores: A tensor with shape [N, K].
    classes: A tensor with shape [N, K].
    image_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  def single_batch_fn(element):
    """A mapping function for a single batch."""
    boxes_i, scores_i, classes_i = element[0], element[1], element[2]
    nms_boxes_cls, nms_scores_cls, nms_classes_cls = [], [], []
    nms_valid_len_cls = []
    for cid in range(params['num_classes']):
      indices = tf.where(tf.equal(classes_i, cid))
      if indices.shape[0] == 0:
        continue
      classes_cls = tf.gather_nd(classes_i, indices)
      boxes_cls = tf.gather_nd(boxes_i, indices)
      scores_cls = tf.gather_nd(scores_i, indices)

      nms_boxes, nms_scores, nms_classes, nms_valid_len = nms(
          params, boxes_cls, scores_cls, classes_cls, False)
      nms_boxes_cls.append(nms_boxes)
      nms_scores_cls.append(nms_scores)
      nms_classes_cls.append(nms_classes)
      nms_valid_len_cls.append(nms_valid_len)

    # Pad zeros and select topk.
    max_output_size = params['nms_configs'].get('max_output_size', 100)
    nms_boxes_cls = tf.pad(
        tf.concat(nms_boxes_cls, 0), [[0, max_output_size], [0, 0]])
    nms_scores_cls = tf.pad(
        tf.concat(nms_scores_cls, 0), [[0, max_output_size]])
    nms_classes_cls = tf.pad(
        tf.concat(nms_classes_cls, 0), [[0, max_output_size]])
    nms_valid_len_cls = tf.stack(nms_valid_len_cls)

    _, indices = tf.math.top_k(nms_scores_cls, k=max_output_size, sorted=True)

    return tuple((
        tf.gather(nms_boxes_cls, indices),
        tf.gather(nms_scores_cls, indices),
        tf.gather(nms_classes_cls, indices),
        tf.minimum(max_output_size, tf.reduce_sum(nms_valid_len_cls))))
    # end of single_batch_fn

  nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
      single_batch_fn, [boxes, scores, classes])
  if image_scales is not None:
    scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
    nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
  return nms_boxes, nms_scores, nms_classes, nms_valid_len


def postprocess_per_class(params, cls_outputs, box_outputs, image_scales=None):
  """Post processing with per class NMS.

  An accurate but relatively slow version of NMS. The idea is to perform NMS for
  each class, and then combine them.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [y_min,
      x_min, y_max, x_man].
    image_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  cls_outputs = to_list(cls_outputs)
  box_outputs = to_list(box_outputs)
  boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)
  return per_class_nms(params, boxes, scores, classes, image_scales)


def generate_detections(params,
                        cls_outputs,
                        box_outputs,
                        image_scales,
                        image_ids,
                        flip=False):
  """A legacy interface for generating [id, x, y, w, h, score, class]."""
  _, width = model_utils.parse_image_size(params['image_size'])

  original_image_widths = tf.expand_dims(image_scales, -1) * width

  if params['nms_configs'].get('pyfunc', True):
    # numpy based soft-nms gives better accuracy than the tensorflow builtin
    # the reason why is unknown
    detections_bs = []
    boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)
    for index in range(boxes.shape[0]):
      nms_configs = params['nms_configs']
      detections = tf.numpy_function(
          functools.partial(nms_np.per_class_nms, nms_configs=nms_configs), [
              boxes[index],
              scores[index],
              classes[index],
              tf.slice(image_ids, [index], [1]),
              tf.slice(image_scales, [index], [1]),
              params['num_classes'],
              nms_configs['max_output_size'],
          ], tf.float32)

      if flip:
        detections = tf.stack([
            detections[:, 0],
            # the mirrored location of the left edge is the image width
            # minus the position of the right edge
            original_image_widths[index] - detections[:, 3],
            detections[:, 2],
            # the mirrored location of the right edge is the image width
            # minus the position of the left edge
            original_image_widths[index] - detections[:, 1],
            detections[:, 4],
            detections[:, 5],
            detections[:, 6],
        ], axis=-1)
      detections_bs.append(detections)
    return tf.stack(detections_bs, axis=0, name='detnections')

  nms_boxes_bs, nms_scores_bs, nms_classes_bs, _ = postprocess_per_class(
      params, cls_outputs, box_outputs, image_scales)

  image_ids_bs = tf.cast(tf.expand_dims(image_ids, -1), nms_scores_bs.dtype)
  if flip:
    detections_bs = [
        image_ids_bs * tf.ones_like(nms_scores_bs),
        # the mirrored location of the left edge is the image width
        # minus the position of the right edge
        original_image_widths - nms_boxes_bs[:, :, 3],
        nms_boxes_bs[:, :, 0],
        # the mirrored location of the right edge is the image width
        # minus the position of the left edge
        original_image_widths - nms_boxes_bs[:, :, 1],
        nms_boxes_bs[:, :, 2],
        nms_scores_bs,
        nms_classes_bs,
    ]
  else:
    detections_bs = [
        image_ids_bs * tf.ones_like(nms_scores_bs),
        nms_boxes_bs[:, :, 1],
        nms_boxes_bs[:, :, 0],
        nms_boxes_bs[:, :, 3],
        nms_boxes_bs[:, :, 2],
        nms_scores_bs,
        nms_classes_bs,
    ]
  return tf.stack(detections_bs, axis=-1, name='detnections')


def transform_detections(detections):
  """A transforms detections in [id, x1, y1, x2, y2, score, class] form to [id, x, y, w, h, score, class]."""
  return tf.stack([
      detections[:, :, 0],
      detections[:, :, 1],
      detections[:, :, 2],
      detections[:, :, 3] - detections[:, :, 1],
      detections[:, :, 4] - detections[:, :, 2],
      detections[:, :, 5],
      detections[:, :, 6],
  ],
                  axis=-1)
