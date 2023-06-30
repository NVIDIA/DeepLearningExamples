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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import tensorflow as tf
import time

from .nn_utils import create_inputs_dict

class Evaluator:
    def __init__(self, model, timer, auc_thresholds, max_steps=None, cast_dtype=None, distributed=False):
        self.model = model
        self.timer = timer
        self.max_steps = max_steps
        self.cast_dtype = cast_dtype
        self.distributed = distributed

        if self.distributed:
            import horovod.tensorflow as hvd
            self.hvd = hvd
        else:
            self.hvd = None

        self.auc_metric = tf.keras.metrics.AUC(num_thresholds=auc_thresholds, curve='ROC',
                                               summation_method='interpolation', from_logits=True)
        self.bce_op = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)

    def _reset(self):
        self.latencies, self.all_test_losses = [], []
        self.auc_metric.reset_state()

    @tf.function
    def update_auc_metric(self, labels, y_pred):
        self.auc_metric.update_state(labels, y_pred)

    @tf.function
    def compute_bce_loss(self, labels, y_pred):
        return self.bce_op(labels, y_pred)

    def _step(self, pipe):
        begin = time.time()

        batch = pipe.get_next()
        (numerical_features, categorical_features), labels = batch

        if self.cast_dtype is not None:
            numerical_features = tf.cast(numerical_features, self.cast_dtype)

        inputs = create_inputs_dict(numerical_features, categorical_features)
        y_pred = self.model(inputs, sigmoid=False, training=False)

        end = time.time()
        self.latencies.append(end - begin)

        if self.distributed:
            y_pred = self.hvd.allgather(y_pred)
            labels = self.hvd.allgather(labels)

        self.timer.step_test()
        if not self.distributed or self.hvd.rank() == 0:
            self.update_auc_metric(labels, y_pred)
            test_loss = self.compute_bce_loss(labels, y_pred)
            self.all_test_losses.append(test_loss)

    def __call__(self, validation_pipeline):
        self._reset()
        auc, test_loss = 0, 0
        pipe = iter(validation_pipeline.op())

        num_steps = len(validation_pipeline)
        if self.max_steps is not None and self.max_steps >= 0:
            num_steps = min(num_steps, self.max_steps)

        for _ in range(num_steps):
            self._step(pipe)

        if not self.distributed or self.hvd.rank() == 0:
            auc = self.auc_metric.result().numpy().item()
            test_loss = tf.reduce_mean(self.all_test_losses).numpy().item()

        return auc, test_loss, self.latencies
