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

import dllogger
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from data.outbrain.dataloader import make_padding_function
from horovod.tensorflow.mpi_ops import Average, Sum


class MapMetric:
    def __init__(self, map_column, cpu):
        self.map_column = map_column
        self.cpu = cpu
        with tf.device("/CPU:0"):
            self.current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
            self.map_id_counter = tf.Variable(
                0.0, trainable=False, dtype=tf.float64
            )
            self.streaming_map = tf.Variable(
                0.0, name="STREAMING_MAP", trainable=False, dtype=tf.float64
            )

    def reset_states(self):
        self.map_id_counter.assign(1)
        self.current_step_var.assign(1)
        self.streaming_map.assign(1)

    @tf.function
    def calculate_map(self, y, predictions, map_ids):

        #flatten all arrays
        predictions = tf.reshape(predictions, [-1])
        predictions = tf.cast(predictions, tf.float64)
        map_ids = tf.reshape(map_ids, [-1])
        labels = tf.reshape(y, [-1])

        # sort map_ids and reorder other arrays to match
        sorted_ids = tf.argsort(map_ids)
        map_ids = tf.gather(map_ids, indices=sorted_ids)
        predictions = tf.gather(predictions, indices=sorted_ids)
        labels = tf.gather(labels, indices=sorted_ids)

        # renumber map ids to 0...n and get counts for each occurence
        _, map_ids_idx, map_ids_count = tf.unique_with_counts(
            map_ids, out_idx=tf.int64
        )

        # get how many times the most common ad id occurs and calculate the padding
        pad_length = 30 - tf.reduce_max(map_ids_count)

        # group predictions into rows based on map id idx and turn into tensor
        preds = tf.RaggedTensor.from_value_rowids(
            predictions, map_ids_idx
        ).to_tensor()
        # ditto for labels
        labels = tf.RaggedTensor.from_value_rowids(labels, map_ids_idx).to_tensor()


        #get only rows for which there is a positive label
        labels_mask = tf.math.reduce_max(labels, 1)
        preds_masked = tf.boolean_mask(preds, labels_mask)
        labels_masked = tf.boolean_mask(labels, labels_mask)

        # get the position of the positive label
        labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
        labels_masked = tf.reshape(labels_masked, [-1, 1])

        # add pad_length zeros to each row of the predictions tensor
        preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])

        # get indices of the top 12 predictions for each map  id
        _, predictions_idx = tf.math.top_k(preds_masked, 12)

        # get rows in which the true positive is among our top 12

        #indicators of our hits
        indices = tf.math.equal(predictions_idx, labels_masked)

        #indicators of hits per row
        indices_mask = tf.math.reduce_any(indices, 1)
        masked_indices = tf.boolean_mask(indices, indices_mask)

        res = tf.argmax(masked_indices, axis=1)
        ap_matrix = tf.divide(1, tf.add(res, 1))
        ap_sum = tf.reduce_sum(ap_matrix)
        shape = tf.cast(tf.shape(indices)[0], tf.float64)
        self.map_id_counter.assign_add(shape)
        self.streaming_map.assign_add(ap_sum)

    @tf.function
    def reduce_results(self):
        if not self.cpu:
            all_streaming_map = hvd.allreduce(self.streaming_map, op=Sum)
            all_map_id_counter = hvd.allreduce(self.map_id_counter, op=Sum)

        else:
            all_streaming_map = self.streaming_map
            all_map_id_counter = self.map_id_counter

        map_metric = tf.divide(all_streaming_map, all_map_id_counter)

        return map_metric


class Evaluator:
    def __init__(
            self,
            model,
            throughput_calculator,
            eval_dataset,
            compiled_loss,
            args,
            maybe_map_column,
            multihot_hotnesses_dict,
            num_auc_thresholds
    ):

        self.model = model
        self.steps_per_epoch = len(eval_dataset)
        self.args = args
        self.throughput_calculator = throughput_calculator
        self.compiled_loss = compiled_loss
        self.eval_loss = tf.keras.metrics.Mean()
        self.metrics = [tf.keras.metrics.AUC(num_thresholds=num_auc_thresholds,
                                          curve='ROC', summation_method='interpolation',
                                          from_logits=True)]
        self.map_enabled=False
        self.map_column = None
        if maybe_map_column is not None:
            self.map_metric=MapMetric(maybe_map_column, cpu=args.cpu)
            self.map_enabled=True
            self.map_column=maybe_map_column

        self.metric_names = ["auc_roc"]
        self.eval_dataset = eval_dataset
        self.multihot_hotnesses_dict = multihot_hotnesses_dict
        self.padding_function = make_padding_function(multihot_hotnesses_dict)

    def _reset_states(self):
        for metric in self.metrics:
            metric.reset_states()

        if self.map_enabled:
            self.map_metric.reset_states()
        self.eval_loss.reset_states()

    def prepare_dataset(self, current_epoch):
        benchmark_needed_steps = self.args.benchmark_steps // self.steps_per_epoch + 1
        n = 1 if self.args.evaluate and not self.args.benchmark else self.args.num_epochs - current_epoch \
            if not self.args.benchmark else max(benchmark_needed_steps, self.args.num_epochs)
        self.eval_dataset = self.eval_dataset.epochs(n)

    #todo find a nicer way to do this
    @tf.function(experimental_relax_shapes=True)
    def _execute_step_calculations_with_map(self, x, y, map_ids):
        predictions = self.model(x, training=False)

        with tf.device("/CPU:0"):
            loss = self.compiled_loss(y, predictions)
            for metric in self.metrics:
                metric.update_state(y, predictions)
            self.eval_loss.update_state(loss)
            self.map_metric.calculate_map(y, predictions, map_ids)

        return loss

    @tf.function(experimental_relax_shapes=True)
    def _execute_step_calculations_no_map(self, x, y):
        predictions = self.model(x, training=False)

        with tf.device("/CPU:0"):
            loss = self.compiled_loss(y, predictions)
            for metric in self.metrics:
                metric.update_state(y, predictions)
            self.eval_loss.update_state(loss)

        return loss


    @tf.function
    def _reduce_results(self):
        if not self.args.cpu:
            eval_loss = hvd.allreduce(self.eval_loss.result(), op=Average)
        else:
            eval_loss = self.eval_loss.result()

        return eval_loss

    def _reduce_metrics(self):
        if self.args.cpu:
            return self.metrics

        hand_reduced_metrics = []
        for metric in self.metrics:
            # as of 6.2022, hvd.allgather_object() cannot gather tf.Variable when amp is enabled
            # this is a workaround that instead gathers the tensors that merge_state uses
            # verified to be equivalent to just allgather and merge_state for keras.AUC
            to_gather = list(x.value() for x in metric.weights)
            gathered_weights = hvd.allgather_object(to_gather)
            if hvd.rank() == 0:
                hand_gather_root = metric
                hand_gather_root.reset_state()
                for list_of_weights in gathered_weights:
                    for (base_weight, new_weight) in zip(hand_gather_root.weights, list_of_weights):
                        base_weight.assign_add(new_weight)
                hand_reduced_metrics.append(hand_gather_root)

        return hand_reduced_metrics

    @staticmethod
    def log(eval_data, step):
        dllogger.log(data=eval_data, step=(step,))

    def eval_step(self, x, y):
        if self.map_enabled:
            map_ids = x.pop(self.map_column)
            self._execute_step_calculations_with_map(x, y, map_ids)
        else:
            self._execute_step_calculations_no_map(x,y)

        if self.args.benchmark:
            self.throughput_calculator(y.shape[0], eval_benchmark=True)

    def eval(self, step):

        eval_data = {}
        self._reset_states()

        # Graph mode part
        for i, (x, y) in enumerate(self.eval_dataset, 1):
            x = self.padding_function(x)
            self.eval_step(x, y)
            if i == self.steps_per_epoch and not self.args.benchmark:
                break

        eval_loss = self._reduce_results().numpy()
        hand_reduced_metrics = self._reduce_metrics()

        map_value = None
        if self.map_enabled:
            map_value = self.map_metric.reduce_results().numpy()

        if self.args.cpu or hvd.rank() == 0:
            with tf.device("/CPU:0"):
                # Eager mode part
                current_step = int(step.numpy())

                eval_data = {
                    "loss_val": np.around(eval_loss.astype(np.float64), 4)
                }
                if map_value is not None:
                    eval_data["streaming_map_val"] = np.around(map_value, 4)

                for metric_name, metric in zip(self.metric_names, hand_reduced_metrics):
                    eval_data[metric_name] = np.around(metric.result().numpy().astype(np.float64), 4)

                self.log(eval_data, current_step)

        return eval_data
