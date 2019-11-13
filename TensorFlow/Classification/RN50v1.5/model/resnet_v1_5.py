#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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


from __future__ import print_function

import tensorflow as tf

import horovod.tensorflow as hvd

from model import layers
from model import blocks

from utils import var_storage
from utils import hvd_utils

from utils.data_utils import normalized_inputs

from utils.learning_rate import learning_rate_scheduler
from utils.optimizers import FixedLossScalerOptimizer

from dllogger.logger import LOGGER

__all__ = [
    'ResnetModel',
]


class ResnetModel(object):
    """Resnet cnn network configuration."""

    def __init__(
        self,
        model_name,
        n_classes,
        compute_format='NCHW',
        input_format='NHWC',
        dtype=tf.float32,
        use_dali=False,
    ):

        self.model_hparams = tf.contrib.training.HParams(
            n_classes=n_classes,
            compute_format=compute_format,
            input_format=input_format,
            dtype=dtype,
            layer_counts=(3, 4, 6, 3),
            model_name=model_name,
            use_dali=use_dali
        )

        self.batch_norm_hparams = tf.contrib.training.HParams(
            decay=0.9,
            epsilon=1e-5,
            scale=True,
            center=True,
            param_initializers={
                'beta': tf.constant_initializer(0.0),
                'gamma': tf.constant_initializer(1.0),
                'moving_mean': tf.constant_initializer(0.0),
                'moving_variance': tf.constant_initializer(1.0)
            },
        )

        self.conv2d_hparams = tf.contrib.training.HParams(
            kernel_initializer=tf.variance_scaling_initializer(
                scale=2.0, distribution='truncated_normal', mode='fan_out'
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )

        self.dense_hparams = tf.contrib.training.HParams(
            kernel_initializer=tf.variance_scaling_initializer(
                scale=2.0, distribution='truncated_normal', mode='fan_out'
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )
        if hvd.rank() == 0:
            LOGGER.log("Model HParams:")
            LOGGER.log("Name", model_name)
            LOGGER.log("Number of classes", n_classes)
            LOGGER.log("Compute_format", compute_format)
            LOGGER.log("Input_format", input_format)
            LOGGER.log("dtype", str(dtype))


    def __call__(self, features, labels, mode, params):

        if mode == tf.estimator.ModeKeys.TRAIN:

            if "batch_size" not in params.keys():
                raise RuntimeError("Parameter `batch_size` is missing...")

            if "lr_init" not in params.keys():
                raise RuntimeError("Parameter `lr_init` is missing...")

            if "num_gpus" not in params.keys():
                raise RuntimeError("Parameter `num_gpus` is missing...")

            if "steps_per_epoch" not in params.keys():
                raise RuntimeError("Parameter `steps_per_epoch` is missing...")

            if "momentum" not in params.keys():
                raise RuntimeError("Parameter `momentum` is missing...")

            if "weight_decay" not in params.keys():
                raise RuntimeError("Parameter `weight_decay` is missing...")

            if "loss_scale" not in params.keys():
                raise RuntimeError("Parameter `loss_scale` is missing...")
            
            if "label_smoothing" not in params.keys():
                raise RuntimeError("Parameter `label_smoothing` is missing...")
                
        if mode == tf.estimator.ModeKeys.TRAIN and not self.model_hparams.use_dali:

            with tf.device('/cpu:0'):
                # Stage inputs on the host
                cpu_prefetch_op, (features, labels) = ResnetModel._stage([features, labels])

            with tf.device('/gpu:0'):
                # Stage inputs to the device
                gpu_prefetch_op, (features, labels) = ResnetModel._stage([features, labels])

        with tf.device("/gpu:0"):

            if features.dtype != self.model_hparams.dtype:
                features = tf.cast(features, self.model_hparams.dtype)

            # Subtract mean per channel
            # and enforce values between [-1, 1]
            if not self.model_hparams.use_dali:
                features = normalized_inputs(features)

            mixup = 0
            eta = 0
            
            if mode == tf.estimator.ModeKeys.TRAIN:        
                eta = params['label_smoothing']
                mixup = params['mixup']
                
            if mode != tf.estimator.ModeKeys.PREDICT: 
                one_hot_smoothed_labels = tf.one_hot(labels, 1001, 
                                                     on_value = 1 - eta + eta/1001,
                                                     off_value = eta/1001)
                if mixup != 0:

                    LOGGER.log("Using mixup training with beta=", params['mixup'])
                    beta_distribution = tf.distributions.Beta(params['mixup'], params['mixup'])

                    feature_coefficients = beta_distribution.sample(sample_shape=[params['batch_size'], 1, 1, 1])      

                    reversed_feature_coefficients = tf.subtract(tf.ones(shape=feature_coefficients.shape), feature_coefficients)

                    rotated_features = tf.reverse(features, axis=[0])      

                    features = feature_coefficients * features + reversed_feature_coefficients * rotated_features

                    label_coefficients = tf.squeeze(feature_coefficients, axis=[2, 3])

                    rotated_labels = tf.reverse(one_hot_smoothed_labels, axis=[0])    

                    reversed_label_coefficients = tf.subtract(tf.ones(shape=label_coefficients.shape), label_coefficients)

                    one_hot_smoothed_labels = label_coefficients * one_hot_smoothed_labels + reversed_label_coefficients * rotated_labels
                
                
            # Update Global Step
            global_step = tf.train.get_or_create_global_step()
            tf.identity(global_step, name="global_step_ref")

            tf.identity(features, name="features_ref")
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.identity(labels, name="labels_ref")

            probs, logits = self.build_model(
                features,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                reuse=False
            )

            y_preds = tf.argmax(logits, axis=1, output_type=tf.int32)

            # Check the output dtype, shall be FP32 in training
            assert (probs.dtype == tf.float32)
            assert (logits.dtype == tf.float32)
            assert (y_preds.dtype == tf.int32)

            tf.identity(logits, name="logits_ref")
            tf.identity(probs, name="probs_ref")
            tf.identity(y_preds, name="y_preds_ref")

            if mode == tf.estimator.ModeKeys.TRAIN:
                
                assert (len(tf.trainable_variables()) == 161)

            else:
                
                assert (len(tf.trainable_variables()) == 0)


        if mode == tf.estimator.ModeKeys.PREDICT:

            predictions = {'classes': y_preds, 'probabilities': probs}

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)}
            )

        else:

            with tf.device("/gpu:0"):

                if mode == tf.estimator.ModeKeys.TRAIN:
                    acc_top1 = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
                    acc_top5 = tf.nn.in_top_k(predictions=logits, targets=labels, k=5)

                else:
                    acc_top1, acc_top1_update_op = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=labels, k=1))
                    acc_top5, acc_top5_update_op = tf.metrics.mean(tf.nn.in_top_k(predictions=logits, targets=labels, k=5))

                tf.identity(acc_top1, name="acc_top1_ref")
                tf.identity(acc_top5, name="acc_top5_ref")

                predictions = {
                    'classes': y_preds,
                    'probabilities': probs,
                    'accuracy_top1': acc_top1,
                    'accuracy_top5': acc_top5
                }
                
                cross_entropy = tf.losses.softmax_cross_entropy(
                    logits=logits, onehot_labels=one_hot_smoothed_labels)

                assert (cross_entropy.dtype == tf.float32)
                tf.identity(cross_entropy, name='cross_entropy_loss_ref')

                def loss_filter_fn(name):
                    """we don't need to compute L2 loss for BN and bias (eq. to add a cste)"""
                    return all([
                        tensor_name not in name.lower()
                        # for tensor_name in ["batchnorm", "batch_norm", "batch_normalization", "bias"]
                        for tensor_name in ["batchnorm", "batch_norm", "batch_normalization"]
                    ])

                filtered_params = [tf.cast(v, tf.float32) for v in tf.trainable_variables() if loss_filter_fn(v.name)]

                if len(filtered_params) != 0:

                    l2_loss_per_vars = [tf.nn.l2_loss(v) for v in filtered_params]
                    l2_loss = tf.multiply(tf.add_n(l2_loss_per_vars), params["weight_decay"])

                else:
                    l2_loss = tf.zeros(shape=(), dtype=tf.float32)

                assert (l2_loss.dtype == tf.float32)
                tf.identity(l2_loss, name='l2_loss_ref')

                total_loss = tf.add(cross_entropy, l2_loss, name="total_loss")

                assert (total_loss.dtype == tf.float32)
                tf.identity(total_loss, name='total_loss_ref')

                tf.summary.scalar('cross_entropy', cross_entropy)
                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('total_loss', total_loss)
                
                if mode == tf.estimator.ModeKeys.TRAIN:

                    with tf.device("/cpu:0"):

                        learning_rate = learning_rate_scheduler(
                            lr_init=params["lr_init"],
                            lr_warmup_epochs=params["lr_warmup_epochs"],
                            global_step=global_step,
                            batch_size=params["batch_size"],
                            num_batches_per_epoch=params["steps_per_epoch"],
                            num_decay_steps=params["num_decay_steps"],
                            num_gpus=params["num_gpus"],
                            use_cosine_lr=params["use_cosine_lr"]
                        )

                    tf.identity(learning_rate, name='learning_rate_ref')
                    tf.summary.scalar('learning_rate', learning_rate)

                    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params["momentum"])

                    if params["apply_loss_scaling"]:

                        optimizer = FixedLossScalerOptimizer(optimizer, scale=params["loss_scale"])

                    if hvd_utils.is_using_hvd():
                        
                        optimizer = hvd.DistributedOptimizer(optimizer)

                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    if mode != tf.estimator.ModeKeys.TRAIN:
                        update_ops += [acc_top1_update_op, acc_top5_update_op]
                    
                    deterministic = True
                    gate_gradients = (tf.train.Optimizer.GATE_OP if deterministic else tf.train.Optimizer.GATE_NONE)

                    backprop_op = optimizer.minimize(total_loss, gate_gradients=gate_gradients, global_step=global_step)

                    
                    if self.model_hparams.use_dali:
                    
                        train_ops = tf.group(backprop_op, update_ops, name='train_ops')
                    
                    else:
                        train_ops = tf.group(backprop_op, cpu_prefetch_op, gpu_prefetch_op, update_ops, name='train_ops')
                    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_ops)

                elif mode == tf.estimator.ModeKeys.EVAL:

                    eval_metrics = {
                        "top1_accuracy": (acc_top1, acc_top1_update_op),
                        "top5_accuracy": (acc_top5, acc_top5_update_op)
                    }

                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions=predictions,
                        loss=total_loss,
                        eval_metric_ops=eval_metrics
                    )

                else:
                    raise NotImplementedError('Unknown mode {}'.format(mode))

                
    @staticmethod
    def _stage(tensors):
        """Stages the given tensors in a StagingArea for asynchronous put/get.
        """
        stage_area = tf.contrib.staging.StagingArea(
            dtypes=[tensor.dtype for tensor in tensors],
            shapes=[tensor.get_shape() for tensor in tensors]
        )

        put_op = stage_area.put(tensors)
        get_tensors = stage_area.get()

        tf.add_to_collection('STAGING_AREA_PUTS', put_op)

        return put_op, get_tensors


    def build_model(self, inputs, training=True, reuse=False):
        
        with var_storage.model_variable_scope(
            self.model_hparams.model_name,
            reuse=reuse,
            dtype=self.model_hparams.dtype
        ):

            with tf.variable_scope("input_reshape"):

                if self.model_hparams.input_format == 'NHWC' and self.model_hparams.compute_format == 'NCHW':
                    # Reshape inputs: NHWC => NCHW
                    inputs = tf.transpose(inputs, [0, 3, 1, 2])

                elif self.model_hparams.input_format == 'NCHW' and self.model_hparams.compute_format == 'NHWC':

                    # Reshape inputs: NCHW => NHWC
                    inputs = tf.transpose(inputs, [0, 2, 3, 1])

            if self.model_hparams.dtype != inputs.dtype:
                inputs = tf.cast(inputs, self.model_hparams.dtype)

            net = blocks.conv2d_block(
                inputs,
                n_channels=64,
                kernel_size=(7, 7),
                strides=(2, 2),
                mode='SAME_RESNET',
                use_batch_norm=True,
                activation='relu',
                is_training=training,
                data_format=self.model_hparams.compute_format,
                conv2d_hparams=self.conv2d_hparams,
                batch_norm_hparams=self.batch_norm_hparams,
                name='conv2d'
            )

            net = layers.max_pooling2d(
                net,
                pool_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                data_format=self.model_hparams.compute_format,
                name="max_pooling2d",
            )

            for block_id, _ in enumerate(range(self.model_hparams.layer_counts[0])):

                net = blocks.bottleneck_block(
                    inputs=net,
                    depth=256,
                    depth_bottleneck=64,
                    stride=1,
                    training=training,
                    data_format=self.model_hparams.compute_format,
                    conv2d_hparams=self.conv2d_hparams,
                    batch_norm_hparams=self.batch_norm_hparams,
                    block_name="btlnck_block_1_%d" % (block_id + 1)
                )

            for block_id, i in enumerate(range(self.model_hparams.layer_counts[1])):

                stride = 2 if i == 0 else 1
                
                net = blocks.bottleneck_block(
                    inputs=net,
                    depth=512,
                    depth_bottleneck=128,
                    stride=stride,
                    training=training,
                    data_format=self.model_hparams.compute_format,
                    conv2d_hparams=self.conv2d_hparams,
                    batch_norm_hparams=self.batch_norm_hparams,
                    block_name="btlnck_block_2_%d" % (block_id + 1)
                )

            for block_id, i in enumerate(range(self.model_hparams.layer_counts[2])):

                block_id += 1
                stride = 2 if i == 0 else 1

                net = blocks.bottleneck_block(
                    inputs=net,
                    depth=1024,
                    depth_bottleneck=256,
                    stride=stride,
                    training=training,
                    data_format=self.model_hparams.compute_format,
                    conv2d_hparams=self.conv2d_hparams,
                    batch_norm_hparams=self.batch_norm_hparams,
                    block_name="btlnck_block_3_%d" % (block_id + 1)
                )

            for block_id, i in enumerate(range(self.model_hparams.layer_counts[3])):

                stride = 2 if i == 0 else 1

                net = blocks.bottleneck_block(
                    inputs=net,
                    depth=2048,
                    depth_bottleneck=512,
                    stride=stride,
                    training=training,
                    data_format=self.model_hparams.compute_format,
                    conv2d_hparams=self.conv2d_hparams,
                    batch_norm_hparams=self.batch_norm_hparams,
                    block_name="btlnck_block_4_%d" % (block_id + 1)
                )

            with tf.variable_scope("output"):

                net = layers.reduce_mean(
                    net, keepdims=False, data_format=self.model_hparams.compute_format, name='spatial_mean'
                )

                logits = layers.dense(
                    inputs=net,
                    units=self.model_hparams.n_classes,
                    use_bias=True,
                    trainable=training,
                    kernel_initializer=self.dense_hparams.kernel_initializer,
                    bias_initializer=self.dense_hparams.bias_initializer
                )

                if logits.dtype != tf.float32:
                    logits = tf.cast(logits, tf.float32)

                probs = layers.softmax(logits, name="softmax", axis=1)

            return probs, logits
