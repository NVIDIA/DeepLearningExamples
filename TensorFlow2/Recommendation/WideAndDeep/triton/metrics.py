#!/usr/bin/env python3

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from triton.deployment_toolkit.core import BaseMetricsCalculator


class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self, *, output_used_for_metrics: str):
        self.output_used_for_metrics = output_used_for_metrics
        self._ids = None
        self._y_pred = None
        self._y_real = None

    def update(
        self,
        ids: List[Any],
        y_pred: Dict[str, np.ndarray],
        x: Optional[Dict[str, np.ndarray]],
        y_real: Optional[Dict[str, np.ndarray]],
    ):
        y_real = y_real[self.output_used_for_metrics]
        y_pred = y_pred[self.output_used_for_metrics]

        def _concat_batches(b1, b2):
            if b1 is None:
                return b2
            else:
                return np.concatenate([b1, b2], axis=0)

        self._ids = _concat_batches(self._ids, ids)
        self._y_real = _concat_batches(self._y_real, y_real)
        self._y_pred = _concat_batches(self._y_pred, y_pred)

    @property
    def metrics(self) -> Dict[str, Any]:
        metrics = {"map12": self.get_map12(self._ids, self._y_pred, self._y_real)}

        return metrics

    def get_map12(self, ids, y_pred, y_real):
        with tf.device("/cpu:0"):
            predictions = tf.reshape(y_pred, [-1])
            predictions = tf.cast(predictions, tf.float64)
            display_ids = tf.reshape(ids, [-1])
            labels = tf.reshape(y_real, [-1])
            sorted_ids = tf.argsort(display_ids)
            display_ids = tf.gather(display_ids, indices=sorted_ids)
            predictions = tf.gather(predictions, indices=sorted_ids)
            labels = tf.gather(labels, indices=sorted_ids)
            _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(display_ids, out_idx=tf.int64)
            pad_length = 30 - tf.reduce_max(display_ids_ads_count)
            preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
            labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()

            labels_mask = tf.math.reduce_max(labels, 1)
            preds_masked = tf.boolean_mask(preds, labels_mask)
            labels_masked = tf.boolean_mask(labels, labels_mask)
            labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
            labels_masked = tf.reshape(labels_masked, [-1, 1])

            preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
            _, predictions_idx = tf.math.top_k(preds_masked, 12)
            indices = tf.math.equal(predictions_idx, labels_masked)
            indices_mask = tf.math.reduce_any(indices, 1)
            masked_indices = tf.boolean_mask(indices, indices_mask)

            res = tf.argmax(masked_indices, axis=1)
            ap_matrix = tf.divide(1, tf.add(res, 1))
            ap_sum = tf.reduce_sum(ap_matrix)
            shape = tf.cast(tf.shape(indices)[0], tf.float64)

            return (ap_sum / shape).numpy()
