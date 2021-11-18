# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import dllogger
import horovod.tensorflow as hvd
import tensorflow as tf
from data.outbrain.features import DISPLAY_ID_COLUMN
from horovod.tensorflow.mpi_ops import Sum, Average


class Evaluator:
    def __init__(
        self,
        model,
        throughput_calculator,
        eval_dataset,
        compiled_loss,
        steps,
        args,
    ):

        self.model = model
        self.steps = steps
        self.args = args
        self.throughput_calculator = throughput_calculator
        self.compiled_loss = compiled_loss
        self.eval_loss = tf.keras.metrics.Mean()
        self.metrics = []
        self.eval_dataset = eval_dataset

        with tf.device("/CPU:0"):
            self.current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
            self.display_id_counter = tf.Variable(
                0.0, trainable=False, dtype=tf.float64
            )
            self.streaming_map = tf.Variable(
                0.0, name="STREAMING_MAP", trainable=False, dtype=tf.float64
            )

    def _reset_states(self):
        for metric in self.metrics:
            metric.reset_states()

        self.eval_loss.reset_states()
        self.display_id_counter.assign(1)
        self.current_step_var.assign(1)
        self.streaming_map.assign(1)

    @tf.function
    def _calculate_map(self, x, y, predictions):
        predictions = tf.reshape(predictions, [-1])
        predictions = tf.cast(predictions, tf.float64)
        display_ids = x[DISPLAY_ID_COLUMN]
        display_ids = tf.reshape(display_ids, [-1])
        labels = tf.reshape(y, [-1])
        sorted_ids = tf.argsort(display_ids)
        display_ids = tf.gather(display_ids, indices=sorted_ids)
        predictions = tf.gather(predictions, indices=sorted_ids)
        labels = tf.gather(labels, indices=sorted_ids)
        _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(
            display_ids, out_idx=tf.int64
        )
        pad_length = 30 - tf.reduce_max(display_ids_ads_count)
        preds = tf.RaggedTensor.from_value_rowids(
            predictions, display_ids_idx
        ).to_tensor()
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
        self.display_id_counter.assign_add(shape)
        self.streaming_map.assign_add(ap_sum)

    @tf.function
    def _execute_step_calculations(self, x, y):
        predictions = self.model(x, training=False)

        with tf.device("/CPU:0"):
            loss = self.compiled_loss(y, predictions)
            for metric in self.metrics:
                metric.update_state(y, predictions)
            self.eval_loss.update_state(loss)
            self._calculate_map(x, y, predictions)

        return loss

    @tf.function
    def _reduce_results(self):
        if not self.args.cpu:
            all_streaming_map = hvd.allreduce(self.streaming_map, op=Sum)
            all_display_id_counter = hvd.allreduce(self.display_id_counter, op=Sum)
            eval_loss = hvd.allreduce(
                self.eval_loss.result(), op=Average
            )
        else:
            all_streaming_map = self.streaming_map
            all_display_id_counter = self.display_id_counter
            eval_loss = self.eval_loss.result()

        map_metric = tf.divide(all_streaming_map, all_display_id_counter)
        eval_loss = eval_loss

        return map_metric, eval_loss

    @staticmethod
    def log(eval_data, step, steps):
        dllogger.log(data=eval_data, step=(step, steps))

    def eval_step(self, x, y):
        self._execute_step_calculations(x, y)

        if self.args.benchmark:
            self.throughput_calculator(y.shape[0], eval_benchmark=True)

    def eval(self, step):

        eval_data = {}
        self._reset_states()
        range_val = 1 if not self.args.benchmark else 100

        # Graph mode part
        for _ in range(range_val):
            for x, y in self.eval_dataset:
                self.eval_step(x, y)

        map_metric, eval_loss = self._reduce_results()

        if self.args.cpu or hvd.rank() == 0:
            with tf.device("/CPU:0"):
                # Eager mode part
                current_step = int(step.numpy())
                eval_data = {
                    "loss_val": f"{eval_loss.numpy():.4f}",
                    "streaming_map_val": f"{map_metric.numpy():.4f}",
                }

                self.log(eval_data, current_step, self.steps)

        return eval_data
