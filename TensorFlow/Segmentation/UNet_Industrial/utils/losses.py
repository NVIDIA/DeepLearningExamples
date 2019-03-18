# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# ==============================================================================

import tensorflow as tf

__all__ = ["regularization_l2loss", "reconstruction_l2loss", "reconstruction_x_entropy", "adaptive_loss"]


def regularization_l2loss(weight_decay):

    def loss_filter_fn(name):
        """we don't need to compute L2 loss for BN"""

        return all(
            [tensor_name not in name.lower() for tensor_name in ["batchnorm", "batch_norm", "batch_normalization"]]
        )

    filtered_params = [tf.cast(v, tf.float32) for v in tf.trainable_variables() if loss_filter_fn(v.name)]

    if len(filtered_params) != 0:

        l2_loss_per_vars = [tf.nn.l2_loss(v) for v in filtered_params]
        l2_loss = tf.multiply(tf.add_n(l2_loss_per_vars), weight_decay)

    else:
        l2_loss = tf.zeros(shape=(), dtype=tf.float32)

    return l2_loss


def reconstruction_l2loss(y_pred, y_true):
    reconstruction_err = tf.subtract(y_pred, y_true)
    return tf.reduce_mean(tf.nn.l2_loss(reconstruction_err), name='reconstruction_loss_l2_loss')


def reconstruction_x_entropy(y_pred, y_true, from_logits=False):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=from_logits))


def dice_coe(y_pred, y_true, loss_type='jaccard', smooth=1.):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    y_true : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_pred : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
            dice = ```smooth/(small_value + smooth)``,
            then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
            so in this case, higher smooth can have a higher dice.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)


def adaptive_loss(y_pred, y_pred_logits, y_true, switch_at_threshold=0.3, loss_type='jaccard'):

    dice_loss = 1 - dice_coe(y_pred=y_pred, y_true=y_true, loss_type=loss_type, smooth=1.)

    return tf.cond(
        dice_loss < switch_at_threshold,
        true_fn=lambda: dice_loss,
        false_fn=lambda: reconstruction_x_entropy(y_pred=y_pred_logits, y_true=y_true, from_logits=True)
    )
