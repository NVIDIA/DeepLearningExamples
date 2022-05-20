# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import horovod.tensorflow as hvd
import tensorflow as tf


class Dice(tf.keras.metrics.Metric):
    def __init__(self, n_class, **kwargs):
        super().__init__(**kwargs)
        self.n_class = n_class
        self.steps = self.add_weight(name="steps", initializer="zeros", aggregation=tf.VariableAggregation.SUM)
        self.dice = self.add_weight(
            name="dice",
            shape=(n_class,),
            initializer="zeros",
            aggregation=tf.VariableAggregation.SUM,
        )

    def update_state(self, y_pred, y_true):
        self.steps.assign_add(1)
        self.dice.assign_add(self.compute_stats(y_pred, y_true))

    def result(self):
        dice_sum = hvd.allreduce(self.dice, op=hvd.mpi_ops.Sum)
        steps_sum = hvd.allreduce(self.steps, op=hvd.mpi_ops.Sum)
        return dice_sum / steps_sum

    def compute_stats(self, y_pred, y_true):
        scores = tf.TensorArray(tf.float32, size=self.n_class)
        pred_classes = tf.argmax(y_pred, axis=-1)
        if tf.rank(pred_classes) < tf.rank(y_true) and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=[-1])
        for i in range(0, self.n_class):
            if tf.math.count_nonzero(y_true == i) == 0:
                scores = scores.write(i, 1 if tf.math.count_nonzero(pred_classes == i) == 0 else 0)
                continue
            true_pos, false_neg, false_pos = self.get_stats(pred_classes, y_true, i)
            denom = tf.cast(2 * true_pos + false_pos + false_neg, dtype=tf.float32)
            score_cls = tf.cast(2 * true_pos, tf.float32) / denom
            scores = scores.write(i, score_cls)
        return scores.stack()

    @staticmethod
    def get_stats(preds, target, class_idx):
        true_pos = tf.math.count_nonzero(tf.logical_and(preds == class_idx, target == class_idx))
        false_neg = tf.math.count_nonzero(tf.logical_and(preds != class_idx, target == class_idx))
        false_pos = tf.math.count_nonzero(tf.logical_and(preds == class_idx, target != class_idx))
        return true_pos, false_neg, false_pos


class Max(tf.keras.metrics.Metric):
    def __init__(self, name="max", dtype=tf.float32):
        super().__init__(name=name)
        self.value = self.add_weight(
            name=f"{name}_weight",
            shape=(),
            initializer="zeros",
            dtype=dtype,
        )

    def update_state(self, new_value):
        self.value.assign(tf.math.maximum(self.value, new_value))

    def result(self):
        return self.value


class MetricAggregator:
    def __init__(self, name, dtype=tf.float32, improvement_metric="max"):
        self.name = name
        self.improvement_metric = improvement_metric
        self.metrics = {
            "value": tf.Variable(0, dtype=dtype),
            "max": Max(name=f"{name}_max", dtype=dtype),
        }

    def logger_metrics(self):
        return {
            f"{self.name}_value": float(self.metrics["value"]),
            f"{self.name}_max": float(self.metrics["max"].result()),
        }

    def checkpoint_metrics(self):
        return {f"{self.name}_{k}": v for k, v in self.metrics.items()}

    def update(self, value):
        old_metric_value = self.metrics[self.improvement_metric].result()
        self.metrics["value"] = value
        self.metrics["max"].update_state(value)
        new_metric_value = self.metrics[self.improvement_metric].result()
        return old_metric_value < new_metric_value


def make_class_logger_metrics(scores, name="val_dice"):
    metrics = {}
    for i, v in enumerate(scores):
        metrics[f"L{i}"] = float(v)
    return metrics


if __name__ == "__main__":
    n_class = 3
    pred = tf.one_hot([0, 1, 1, 1, 2], depth=n_class, dtype=tf.float32)
    target = tf.constant([0, 1, 1, 2, 2])

    metric = Dice(n_class)
    metric.update_state(pred, target)
    # metric.result() == [1.  0.8  0.66]

    # Check Dice computing with zero denominator
    pred = tf.one_hot([0, 2], depth=n_class, dtype=tf.float32)
    target = tf.constant([0, 2])
    metric.update_state(pred, target)
    # metric.result() == [1.  0.9  0.83]
