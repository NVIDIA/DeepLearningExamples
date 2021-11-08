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

import logging
import os

import dllogger
import horovod.tensorflow as hvd
import tensorflow as tf


class Trainer:
    def __init__(
        self,
        model,
        scheduler,
        deep_optimizer,
        wide_optimizer,
        throughput_calculator,
        compiled_loss,
        steps,
        args,
        train_dataset,
        evaluator,
    ):
        self.model = model
        self.scheduler = scheduler
        self.deep_optimizer = deep_optimizer
        self.wide_optimizer = wide_optimizer
        self.throughput_calculator = throughput_calculator
        self.steps = steps
        self.args = args
        self.train_dataset = train_dataset
        self.evaluator = evaluator
        self.compiled_loss = compiled_loss
        self.logger = logging.getLogger("tensorflow")

        with tf.device("/CPU:0"):
            self.current_step_var = tf.Variable(0, trainable=False, dtype=tf.int64)
            self.display_id_counter = tf.Variable(
                0.0, trainable=False, dtype=tf.float64
            )

        self._init_checkpoint_manager()

    def _init_checkpoint_manager(self):
        self.checkpoint = tf.train.Checkpoint(
            deep_optimizer=self.deep_optimizer,
            wide_optimizer=self.wide_optimizer,
            model=self.model,
            current_step=self.current_step_var,
        )
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(self.args.model_dir, "checkpoint"),
            max_to_keep=1,
        )

    def maybe_restore_checkpoint(self):
        if self.args.use_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
            if self.manager.latest_checkpoint:
                self.logger.warning(
                    f"Model restored from checkpoint {self.args.model_dir}"
                )
                if self.args.benchmark:
                    self.current_step_var.assign(0)
            else:
                self.logger.warning(
                    f"Failed to restore model from checkpoint {self.args.model_dir}"
                )

    @tf.function
    def __call__(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            linear_loss = (
                self.wide_optimizer.get_scaled_loss(loss) if self.args.amp else loss
            )
            deep_loss = (
                self.deep_optimizer.get_scaled_loss(loss) if self.args.amp else loss
            )

        if not self.args.cpu:
            tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True)

        linear_vars = self.model.linear_model.trainable_variables
        dnn_vars = self.model.dnn_model.trainable_variables
        linear_grads = tape.gradient(linear_loss, linear_vars)
        dnn_grads = tape.gradient(deep_loss, dnn_vars)
        if self.args.amp:
            linear_grads = self.wide_optimizer.get_unscaled_gradients(linear_grads)
            dnn_grads = self.deep_optimizer.get_unscaled_gradients(dnn_grads)

        self.wide_optimizer.apply_gradients(zip(linear_grads, linear_vars))
        self.deep_optimizer.apply_gradients(zip(dnn_grads, dnn_vars))

        if self.current_step_var == 0:
            hvd.broadcast_variables(self.model.linear_model.variables, root_rank=0)
            hvd.broadcast_variables(self.model.dnn_model.variables, root_rank=0)
            hvd.broadcast_variables(self.wide_optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(self.deep_optimizer.variables(), root_rank=0)

        return loss

    @tf.function
    def _execute_step_calculations(self, x, y):
        loss = self(x, y)
        with tf.device("/CPU:0"):
            self.scheduler(tf.cast(self.current_step_var + 1, tf.float32))
            self.current_step_var.assign_add(1)

        return loss

    def log(self, current_step, loss):
        train_data = {"loss": f"{loss:.4f}"}
        dllogger.log(data=train_data, step=(current_step, self.steps))

    def train_step(self, x, y):

        # Graph mode part
        loss = self._execute_step_calculations(x, y)

        # Eager mode part
        current_step = int(self.current_step_var.numpy()) - 1
        if self.args.benchmark:
            self.throughput_calculator(y.shape[0])
        elif (self.args.cpu or hvd.rank() == 0) and current_step % 100 == 0:
            self.log(current_step, loss.numpy())

    def join_and_broadcast(self):
        hvd.join()
        if not self.args.benchmark:
            hvd.broadcast_variables(self.model.linear_model.variables, root_rank=0)
            hvd.broadcast_variables(self.model.dnn_model.variables, root_rank=0)
            hvd.broadcast_variables(self.wide_optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(self.deep_optimizer.variables(), root_rank=0)

    def run_loop(self):
        eval_data = {}
        current_epoch = int(self.current_step_var.numpy()) // len(self.train_dataset) + 1

        for _ in range(current_epoch, self.args.num_epochs + 1):
            range_val = 1 if not self.args.benchmark else 100

            # Graph mode part
            for _ in range(range_val):
                for x, y in self.train_dataset:
                    self.train_step(x, y)
                self.join_and_broadcast()

            eval_data = self.evaluator.eval(self.current_step_var)

            if self.args.cpu or hvd.rank() == 0:
                self.manager.save()

        if self.args.cpu or hvd.rank() == 0:
            dllogger.log(data=eval_data, step=tuple())
