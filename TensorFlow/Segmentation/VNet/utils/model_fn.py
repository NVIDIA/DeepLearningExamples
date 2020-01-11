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
import os

import horovod.tensorflow as hvd
import tensorflow as tf

from model.vnet import Builder
from utils.var_storage import model_variable_scope


def dice_coef(predict, target, dice_type, axis=1, eps=1e-6):
    intersection = tf.reduce_sum(predict * target, axis=axis)

    if dice_type == 'sorensen':
        union = tf.reduce_sum(predict + target, axis=axis)
    else:
        raise ValueError("dice_type must be either sorensen")

    dice = (2 * intersection + eps) / (union + eps)
    return tf.reduce_mean(dice, axis=0)  # average over batch


def vnet_v2(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_eval = (mode == tf.estimator.ModeKeys.EVAL)
    is_predict = (mode == tf.estimator.ModeKeys.PREDICT)
    num_classes = len(params.labels)
    channel_axis = -1

    with model_variable_scope(
            'vnet',
            reuse=tf.AUTO_REUSE,
            dtype=tf.float16,
            debug_mode=False
    ):
        features = tf.reshape(features,
                              [params.batch_size] + params.input_shape + [1])
        if labels is not None:
            labels = tf.reshape(labels,
                                [params.batch_size] + params.input_shape + [1])

        logits = Builder(kernel_size=params.convolution_size,
                         n_classes=num_classes,
                         downscale_blocks=params.downscale_blocks,
                         upscale_blocks=params.upscale_blocks,
                         upsampling=params.upsampling,
                         pooling=params.pooling,
                         normalization=params.normalization_layer,
                         activation=params.activation,
                         mode=mode)(features)

        softmax = tf.nn.softmax(logits=logits, axis=channel_axis)

        if is_predict:
            prediction = tf.argmax(input=softmax, axis=channel_axis)
            predictions = {'prediction': prediction}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Flattened logits and softmax - in FP32
        flattened_softmax = tf.reshape(softmax, [tf.shape(logits)[0], -1, num_classes])
        flattened_softmax = tf.cast(flattened_softmax, tf.float32)

        # One hot encoding
        flattened_labels = tf.layers.flatten(labels)
        one_hot_labels = tf.one_hot(indices=flattened_labels,
                                    depth=num_classes,
                                    dtype=tf.float32)

        with tf.name_scope("loss"):
            if params.loss == 'dice':
                loss = dice_coef(predict=tf.cast(flattened_softmax, tf.float32),
                                 target=one_hot_labels,
                                 dice_type='sorensen')
                total_loss = tf.identity(tf.reduce_sum(1. - loss),
                                         name='total_loss_ref')
            else:
                raise NotImplementedError

        train_op = None
        if is_training:
            global_step = tf.train.get_or_create_global_step()

            with tf.name_scope("optimizer"):
                if params.optimizer == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=params.base_lr,
                                                          momentum=params.momentum,
                                                          centered=True)
                else:
                    raise NotImplementedError

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gradients, variables = zip(*optimizer.compute_gradients(total_loss))
                if params.gradient_clipping == 'global_norm':
                    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                    tf.logging.info('clipping: global_norm')
                else:
                    return NotImplementedError

                optimizer = hvd.DistributedOptimizer(optimizer)

                try:
                    amp_envar_enabled = (int(os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']) == 1)
                except KeyError:
                    amp_envar_enabled = False

                if params.use_amp and not amp_envar_enabled:
                    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                        optimizer,
                        loss_scale='dynamic'
                    )

                train_op = optimizer.minimize(total_loss, global_step=global_step)

        eval_metric_ops = None
        if is_eval:
            dice_loss = dice_coef(predict=tf.cast(flattened_softmax, tf.float32),
                                  target=one_hot_labels,
                                  dice_type='sorensen')
            eval_loss = tf.identity(dice_loss, name='eval_loss_ref')
            eval_metric_ops = {}
            for i in range(num_classes):
                eval_metric_ops['%s dice' % params.labels[str(i)]] = tf.metrics.mean(eval_loss[i])

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op,
        eval_metric_ops=eval_metric_ops)
