# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


def build_sim_loss_fn(alpha=1.0, beta=1.0):
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def sim_loss_fn(targets, gsu_logits, esu_logits):
        gsu_loss = cross_entropy_loss(targets, gsu_logits)
        esu_loss = cross_entropy_loss(targets, esu_logits)
        return 0.5 * (alpha * gsu_loss + beta * esu_loss)

    return sim_loss_fn


@tf.function
def dien_auxiliary_loss_fn(click_probs, noclick_probs, mask=None):
    if mask is None:
        mask = tf.ones_like(click_probs)
    click_loss_term = -tf.math.log(click_probs) * mask
    noclick_loss_term = -tf.math.log(1.0 - noclick_probs) * mask

    return tf.reduce_mean(click_loss_term + noclick_loss_term)
