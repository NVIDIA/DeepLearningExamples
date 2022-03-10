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

""" Model function in charge to collect metrics and feed them to the optimizer """
import horovod.tensorflow as hvd
import tensorflow as tf

from model.unet3d import Builder
from model.losses import make_loss, eval_dice, total_dice
from dataset.data_loader import CLASSES


def unet_3d(features, labels, mode, params):
    """ Gather loss and feed it to the optimizer

    :param features: Input features
    :param labels: Input labels
    :param mode: Estimator's execution mode
    :param params: Dict with additional parameters
    :return: Estimator spec
    """
    # TODO: Find a better way to handle the empty params namespace
    try:
        normalization = params.normalization
    except:
        normalization = 'instancenorm'

    input_node = tf.identity(features, name='input_node')

    logits = Builder(n_classes=4, normalization=normalization, mode=mode)(input_node)

    logits = tf.identity(logits, name='output_node')

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = tf.argmax(input=logits, axis=-1, output_type=tf.dtypes.int32, name="predictions")
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={'predictions': tf.cast(prediction, tf.int8)})

    labels = tf.cast(labels, tf.float32)

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(input=logits, axis=-1, output_type=tf.dtypes.int32)
        prediction = tf.one_hot(prediction, 4)
        if not params.include_background:
            labels = labels[..., 1:]
            prediction = prediction[..., 1:]
        prediction = tf.cast(prediction, tf.float32)
        eval_acc = eval_dice(y_true=labels, y_pred=prediction)
        total_eval_acc = total_dice(prediction, labels)
        metrics = {CLASSES[i]: tf.compat.v1.metrics.mean(eval_acc[i]) for i in range(eval_acc.shape[-1])}
        metrics['whole_tumor'] = tf.compat.v1.metrics.mean(total_eval_acc)
        return tf.estimator.EstimatorSpec(mode=mode, loss=tf.reduce_mean(eval_acc),
                                          eval_metric_ops=metrics)

    if not params.include_background:
        labels = labels[..., 1:]
        logits = logits[..., 1:]

    loss = make_loss(params, y_pred=logits, y_true=labels)
    loss = tf.identity(loss, name="total_loss_ref")

    global_step = tf.compat.v1.train.get_or_create_global_step()
    boundaries = [params.max_steps // (2 * hvd.size()),
                  params.max_steps // (2 * hvd.size()),
                  3 * params.max_steps // (4 * hvd.size())]

    lr = params.learning_rate
    values = [lr / 4, lr, lr / 5, lr / 20]
    learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    if params.use_amp:
        loss_scale = tf.train.experimental.DynamicLossScale()
        optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer, loss_scale)

    optimizer = hvd.DistributedOptimizer(optimizer)

    with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)
