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
import time
import os
import pickle

from PIL import Image
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from dllogger.logger import LOGGER

from model.unet import unet_v1
from utils.data_loader import Dataset
from utils.hooks.profiler_hook import ProfilerHook
from utils.var_storage import model_variable_scope


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


def _model_fn(features, labels, mode, params):
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
    dtype = params['dtype']
    max_steps = params['max_steps']
    lr_init = params['learning_rate']
    momentum = params['momentum']

    device = '/gpu:0'

    global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(lr_init, global_step,
                                               decay_steps=max_steps,
                                               decay_rate=0.96)

    with tf.device(device):
        features = tf.cast(features, dtype)

        with model_variable_scope(
                'UNet',
                reuse=tf.AUTO_REUSE,
                dtype=tf.float16,
                debug_mode=False
        ):
            output_map = unet_v1(features, mode)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'logits': tf.nn.softmax(output_map, axis=-1)}
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            n_classes = output_map.shape[-1].value

            flat_logits = tf.reshape(tf.cast(output_map, tf.float32),
                                     [tf.shape(output_map)[0], -1, n_classes])
            flat_labels = tf.reshape(labels,
                                     [tf.shape(output_map)[0], -1, n_classes])

            crossentropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                          labels=flat_labels),
                                               name='cross_loss_ref')
            dice_loss = tf.reduce_mean(1 - dice_coef(flat_logits, flat_labels), name='dice_loss_ref')

            total_loss = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")

            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

            if is_using_hvd():
                opt = hvd.DistributedOptimizer(opt, device_dense='/gpu:0')

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                deterministic = True
                gate_gradients = (
                    tf.train.Optimizer.GATE_OP
                    if deterministic
                    else tf.train.Optimizer.GATE_NONE)

                train_op = opt.minimize(total_loss, gate_gradients=gate_gradients, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op,
                                      eval_metric_ops={})


class Runner():
    """ Runner class for encapsulating hyperparameters

    This class is constructed with a set of hyper-parameters
    which are later reused for training and prediction

    Args:
        params (dict): Provides the parametrization for training and prediction

    Attributes:
        _max_steps (int): Number of steps for training
        _classifier (tf.estimator.Estimator): Estimator used for training and validation
        _dataset (tf.data.Dataset): Source of sample and label pairs
        _training_hooks (tf.train.SessionRunHook): Parallel training, and benchmarking utils

    """

    def __init__(self, params):
        hvd.init()

        LOGGER.log(str(params))

        data_dir = params['data_dir']
        batch_size = params['batch_size']
        augment = params['augment']
        benchmark = params['benchmark']
        seed = params['seed']

        self._model_dir = params['model_dir']
        self._max_steps = params['max_steps']

        self._classifier = tf.estimator.Estimator(
            model_fn=_model_fn,
            model_dir=self._model_dir,
            params=params,
            config=tf.estimator.RunConfig(
                tf_random_seed=None,
                session_config=self._get_session_config(),
                save_checkpoints_steps=self._max_steps if hvd.rank() == 0 else None,
                keep_checkpoint_max=1))

        self._dataset = Dataset(data_dir=data_dir,
                                batch_size=batch_size,
                                augment=augment,
                                gpu_id=hvd.rank(),
                                num_gpus=hvd.size(),
                                seed=seed)

        self._training_hooks = [hvd.BroadcastGlobalVariablesHook(0)]

        if benchmark and hvd.rank() == 0:
            self._training_hooks.append(ProfilerHook(self._model_dir, batch_size, log_every=params['log_every'],
                                                     warmup_steps=params['warmup_steps']))

    def _get_session_config(self):
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        config.gpu_options.force_gpu_compatible = True
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = max(2, 40 // hvd.size() - 2)
        return config

    def train(self):
        """Perform training with the runner's classifier"""
        LOGGER.log("Begin training...")

        try:
            self._classifier.train(
                input_fn=self._dataset.train_fn,
                steps=self._max_steps,
                hooks=self._training_hooks)
        except KeyboardInterrupt:
            print("Keyboard interrupt")

        LOGGER.log("Training finished")

    def predict(self):
        """Perform prediction with the runner's classifier """

        if hvd.rank() == 0:
            LOGGER.log("Begin predict...")

            begin = time.time()

            pred = self._classifier.predict(input_fn=self._dataset.test_fn)

            predictions = [p['logits'] for p in pred]

            print('Inference took: {} sec'.format(time.time() - begin))

            binary_masks = [np.argmax(p, axis=-1).astype(np.uint8) * 255 for p in predictions]
            multipage_tif = [Image.fromarray(mask).resize(size=(512, 512), resample=Image.BILINEAR)
                             for mask in binary_masks]

            output_dir = os.path.join(self._model_dir, 'pred')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            multipage_tif[0].save(os.path.join(output_dir, 'test-masks.tif'),
                                  compression="tiff_deflate",
                                  save_all=True,
                                  append_images=multipage_tif[1:])

            pickle.dump(predictions, open(os.path.join(output_dir, 'predictions.pkl'), 'wb'))

            LOGGER.log("Predict finished")

    def benchmark(self):
        if hvd.rank() == 0:
            self._classifier.evaluate(input_fn=self._dataset.synth_fn,
                                      steps=self._max_steps,
                                      hooks=[self._training_hooks[-1]])
