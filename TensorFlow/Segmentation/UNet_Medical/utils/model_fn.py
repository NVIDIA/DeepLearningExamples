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

""" Runner class encapsulating the training

This module provides the functionality to initialize a run with hyper-parameters
which can be later used for training and inference.

Example:
    Runner can be created with a parameter dictionary, and those parameters
    are reused for training and inference::

        params = {...}

        runner = Runner(params)
        runner.train()
        runner.predict()

"""
import os

import horovod.tensorflow as hvd
import tensorflow as tf

from model.unet import unet_v1


# Class Dice coefficient averaged over batch
def dice_coef(predict, target, axis=1, eps=1e-6):
    intersection = tf.reduce_sum(predict * target, axis=axis)
    union = tf.reduce_sum(predict * predict + target * target, axis=axis)
    dice = (2. * intersection + eps) / (union + eps)
    return tf.reduce_mean(dice, axis=0)  # average over batch


def regularization_l2loss(weight_decay):
    def loss_filter_fn(name):
        """we don't need to compute L2 loss for BN"""

        return all([
            tensor_name not in name.lower()
            for tensor_name in ["batchnorm", "batch_norm", "batch_normalization"]
        ])

    filtered_params = [tf.cast(v, tf.float32) for v in tf.trainable_variables() if loss_filter_fn(v.name)]

    if len(filtered_params) != 0:

        l2_loss_per_vars = [tf.nn.l2_loss(v) for v in filtered_params]
        l2_loss = tf.multiply(tf.add_n(l2_loss_per_vars), weight_decay)

    else:
        l2_loss = tf.zeros(shape=(), dtype=tf.float32)

    return l2_loss


def is_using_hvd():
    env_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]

    if all([var in os.environ for var in env_vars]):
        return True
    else:
        return False


def unet_fn(features, labels, mode, params):
    """ Model function for tf.Estimator

    Controls how the training is performed by specifying how the
    total_loss is computed and applied in the backward pass.

    Args:
        features (tf.Tensor): Tensor samples
        labels (tf.Tensor): Tensor labels
        mode (tf.estimator.ModeKeys): Indicates if we train, evaluate or predict
        params (dict): Additional parameters supplied to the estimator

    Returns:
        Appropriate tf.estimator.EstimatorSpec for the current mode

    """
    dtype = tf.float32

    device = '/gpu:0'

    global_step = tf.compat.v1.train.get_global_step()

    if mode == tf.estimator.ModeKeys.TRAIN:
        lr_init = params.learning_rate

    with tf.device(device):
        features = tf.cast(features, dtype)

        output_map = unet_v1(features=features, mode=mode)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'logits': tf.nn.softmax(output_map, axis=-1)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        n_classes = output_map.shape[-1].value

        flat_logits = tf.reshape(tf.cast(output_map, tf.float32),
                                 [tf.shape(output_map)[0], -1, n_classes])
        flat_labels = tf.reshape(labels,
                                 [tf.shape(output_map)[0], -1, n_classes])

        crossentropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                       labels=flat_labels), name='cross_loss_ref')
        dice_loss = tf.reduce_mean(1 - dice_coef(tf.keras.activations.softmax(flat_logits, axis=-1),
                                                 flat_labels), name='dice_loss_ref')
        total_loss = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {"eval_ce_loss": tf.compat.v1.metrics.mean(crossentropy_loss),
                               "eval_dice_loss": tf.compat.v1.metrics.mean(dice_loss),
                               "eval_total_loss": tf.compat.v1.metrics.mean(total_loss),
                               "eval_dice_score": tf.compat.v1.metrics.mean(1.0 - dice_loss)}
            return tf.estimator.EstimatorSpec(mode=mode, loss=dice_loss, eval_metric_ops=eval_metric_ops)

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_init)

        if is_using_hvd():
            opt = hvd.DistributedOptimizer(opt, device_dense='/gpu:0')

        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            deterministic = True
            gate_gradients = (
                tf.compat.v1.train.Optimizer.GATE_OP
                if deterministic
                else tf.compat.v1.train.Optimizer.GATE_NONE)

            train_op = opt.minimize(total_loss, gate_gradients=gate_gradients, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op,
                                      eval_metric_ops={})
