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
"""Functions to perform COCO evaluation."""
import numpy as np

from mrcnn_tf2.utils import coco_utils, coco_metric


def process_predictions(predictions):
    """ Process the model predictions for COCO eval.
    Converts boxes from [y1, x1, y2, x2] to [x1, y1, w, h] and scales them by image scale.
    Flattens source_ids

    Args:
        predictions (dict): Predictions returned by model

    Returns:
        Converted prediction.
    """
    image_info = predictions['image_info']
    detection_boxes = predictions['detection_boxes']

    for pred_id, box_id in np.ndindex(*detection_boxes.shape[:2]):
        # convert from [y1, x1, y2, x2] to [x1, y1, w, h] * scale
        scale = image_info[pred_id, 2]
        y1, x1, y2, x2 = detection_boxes[pred_id, box_id, :]

        new_box = np.array([x1, y1, x2 - x1, y2 - y1]) * scale

        detection_boxes[pred_id, box_id, :] = new_box

    # flatten source ids
    predictions['source_ids'] = predictions['source_ids'].flatten()

    return predictions


def evaluate(predictions, eval_file=None, include_mask=True):
    """ Evaluates given iterable of predictions.

    Args:
        predictions (Iterable): Iterable of predictions returned from.
        eval_file (Optional(str)): Path to file with eval annotations.
            If None then groundtruth from feature will be used.
        include_mask (bool): Indicates if eval mask should be included.

    Returns:

    """
    # convert from [y1, x1, y2, x2] to [x1, y1, w, h] * scale
    predictions = process_predictions(predictions)

    # create evaluation metric
    eval_metric = coco_metric.EvaluationMetric(filename=eval_file, include_mask=include_mask)

    # eval using the file or groundtruth from features
    if eval_file is not None:
        eval_results = eval_metric.predict_metric_fn(predictions)
    else:
        images, annotations = coco_utils.extract_coco_groundtruth(predictions, include_mask)
        coco_dataset = coco_utils.create_coco_format_dataset(images, annotations)
        eval_results = eval_metric.predict_metric_fn(predictions, groundtruth_data=coco_dataset)

    return eval_results
