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

import tensorflow as tf


class LearningRateScheduler:
    def __init__(self, args, steps_per_epoch, optimizer):
        assert args.deep_warmup_epochs <= args.num_epochs, \
            "Number of warmup epochs cannot be higher than training epochs"
        self.base_lr = args.deep_learning_rate
        self.warmup_steps = args.deep_warmup_epochs * steps_per_epoch
        bound_epoch = args.deep_warmup_epochs + (args.num_epochs - args.deep_warmup_epochs) / 2

        self.boundaries = [bound_epoch * steps_per_epoch]
        self.values = [self.base_lr / 4, self.base_lr / 8]
        self.optimizer = optimizer

    @tf.function
    def __call__(self, step):
        if step < self.warmup_steps:
            warmup_lr = self.base_lr * step / self.warmup_steps
            self.optimizer.lr.assign(warmup_lr)
        else:
            index = tf.reduce_sum(tf.cast(step > self.boundaries, tf.int64))
            value = tf.gather(self.values, index)
            self.optimizer.lr.assign(value)
