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

import tensorflow as tf


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, y_one_hot=True, reduce_batch=False, eps=1e-6, include_background=False):
        super().__init__()
        self.y_one_hot = y_one_hot
        self.reduce_batch = reduce_batch
        self.eps = eps
        self.include_background = include_background

    def dice_coef(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=1)
        pred_sum = tf.reduce_sum(y_pred, axis=1)
        true_sum = tf.reduce_sum(y_true, axis=1)
        dice = (2.0 * intersection + self.eps) / (pred_sum + true_sum + self.eps)
        return tf.reduce_mean(dice, axis=0)

    @tf.function
    def call(self, y_true, y_pred):
        n_class = y_pred.shape[-1]
        if self.reduce_batch:
            flat_shape = (1, -1, n_class)
        else:
            flat_shape = (y_pred.shape[0], -1, n_class)
        if self.y_one_hot:
            y_true = tf.one_hot(y_true, n_class)

        flat_pred = tf.reshape(tf.cast(y_pred, tf.float32), flat_shape)
        flat_true = tf.reshape(y_true, flat_shape)

        dice_coefs = self.dice_coef(flat_true, tf.keras.activations.softmax(flat_pred, axis=-1))
        if not self.include_background:
            dice_coefs = dice_coefs[1:]
        dice_loss = tf.reduce_mean(1 - dice_coefs)

        return dice_loss


class DiceCELoss(tf.keras.losses.Loss):
    def __init__(self, y_one_hot=True, **dice_kwargs):
        super().__init__()
        self.y_one_hot = y_one_hot
        self.dice_loss = DiceLoss(y_one_hot=False, **dice_kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        n_class = y_pred.shape[-1]
        if self.y_one_hot:
            y_true = tf.one_hot(y_true, n_class)
        dice_loss = self.dice_loss(y_true, y_pred)
        ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred,
            )
        )
        return dice_loss + ce_loss


class WeightDecay:
    def __init__(self, factor):
        self.factor = factor

    @tf.function
    def __call__(self, model):
        # TODO: add_n -> accumulate_n ?
        return self.factor * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if "norm" not in v.name])
