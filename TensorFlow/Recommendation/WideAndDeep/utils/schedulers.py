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

import tensorflow as tf

__all__ = ['learning_rate_scheduler']


def learning_rate_scheduler(lr_init, warmup_steps, global_step):
    warmup_lr = (lr_init * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr_init)
