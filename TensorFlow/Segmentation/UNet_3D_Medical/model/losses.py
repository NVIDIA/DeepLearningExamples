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

""" Different losses for UNet3D """
import tensorflow as tf


def make_loss(params, y_true, y_pred):
    """ Factory method for loss functions

    :param params: Dict with additional parameters
    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :return: Loss
    """
    if params.loss == 'dice':
        return _dice(y_true, y_pred)
    if params.loss == 'ce':
        return _ce(y_true, y_pred)
    if params.loss == 'dice+ce':
        return tf.add(_ce(y_true, y_pred), _dice(y_true, y_pred), name="total_loss_ref")

    raise ValueError('Unknown loss: {}'.format(params.loss))


def _ce(y_true, y_pred):
    """ Crossentropy

    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :return: loss
    """
    return tf.reduce_sum(
        tf.reduce_mean(tf.keras.backend.binary_crossentropy(tf.cast(y_true, tf.float32), y_pred), axis=[0, 1, 2, 3]),
        name='crossentropy_loss_ref')


def _dice(y_true, y_pred):
    """ Training dice

    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :return: loss
    """
    return tf.reduce_sum(dice_loss(predictions=y_pred, targets=y_true), name='dice_loss_ref')


def eval_dice(y_true, y_pred):
    """ Evaluation dice

    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :return: loss
    """
    return 1 - dice_loss(predictions=y_pred, targets=y_true)


def dice_loss(predictions,
              targets,
              squared_pred=False,
              smooth=1e-5,
              top_smooth=0.0):
    """ Dice

    :param predictions: Predicted labels
    :param targets: Ground truth labels
    :param squared_pred: Square the predicate
    :param smooth: Smooth term for denominator
    :param top_smooth: Smooth term for numerator
    :return: loss
    """
    is_channels_first = False

    n_len = len(predictions.get_shape())
    reduce_axis = list(range(2, n_len)) if is_channels_first else list(range(1, n_len - 1))
    intersection = tf.reduce_sum(targets * predictions, axis=reduce_axis)

    if squared_pred:
        targets = tf.square(targets)
        predictions = tf.square(predictions)

    y_true_o = tf.reduce_sum(targets, axis=reduce_axis)
    y_pred_o = tf.reduce_sum(predictions, axis=reduce_axis)

    denominator = y_true_o + y_pred_o

    dice = (2.0 * intersection + top_smooth) / (denominator + smooth)

    return 1 - tf.reduce_mean(dice, axis=0)


def total_dice(predictions,
               targets,
               smooth=1e-5,
               top_smooth=0.0):
    """ Total Dice

    :param predictions: Predicted labels
    :param targets: Ground truth labels
    :param smooth: Smooth term for denominator
    :param top_smooth: Smooth term for numerator
    :return: loss
    """
    n_len = len(predictions.get_shape())
    reduce_axis = list(range(1, n_len-1))
    targets = tf.reduce_sum(targets, axis=-1)
    predictions = tf.reduce_sum(predictions, axis=-1)
    intersection = tf.reduce_sum(targets * predictions, axis=reduce_axis)

    y_true_o = tf.reduce_sum(targets, axis=reduce_axis)
    y_pred_o = tf.reduce_sum(predictions, axis=reduce_axis)

    denominator = y_true_o + y_pred_o

    return tf.reduce_mean((2.0 * intersection + top_smooth) / (denominator + smooth))
