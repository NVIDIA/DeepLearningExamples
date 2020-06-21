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
import dllogger

from model import layers
from model import blocks

from utils import var_storage
from utils import hvd_utils

from utils.data_utils import normalized_inputs

from utils.learning_rate import learning_rate_scheduler
from utils.optimizers import FixedLossScalerOptimizer


__all__ = [
    'ResnetModel',
]


class ResnetModel(object):
    """Resnet cnn network configuration."""

    def __init__(
        self,
        model_name,
        n_classes,
        layers_count,
        layers_depth,
        expansions,
        compute_format='NCHW',
        input_format='NHWC',
        weight_init='fan_out',
        dtype=tf.float32,
        use_dali=False,
        cardinality=1,
        use_se=False,
        se_ratio=1,
    ):

        self.model_hparams = tf.contrib.training.HParams(
            n_classes=n_classes,
            compute_format=compute_format,
            input_format=input_format,
            dtype=dtype,
            layers_count=layers_count,
            layers_depth=layers_depth,
            expansions=expansions,
            model_name=model_name,
            use_dali=use_dali,
            cardinality=cardinality,
            use_se=use_se,
            se_ratio=se_ratio
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
                scale=2.0, distribution='truncated_normal', mode=weight_init
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )

        self.dense_hparams = tf.contrib.training.HParams(
            kernel_initializer=tf.variance_scaling_initializer(
                scale=2.0, distribution='truncated_normal', mode=weight_init
            ),
            bias_initializer=tf.constant_initializer(0.0)
        )
        if hvd.rank() == 0:
            print("Model HParams:")
            print("Name", model_name)
            print("Number of classes", n_classes)
            print("Compute_format", compute_format)
            print("Input_format", input_format)
            print("dtype", str(dtype))


    def __call__(self, features, labels, mode, params):

        if mode == tf.estimator.ModeKeys.TRAIN:
            mandatory_params = ["batch_size", "lr_init", "num_gpus", "steps_per_epoch",
                                "momentum", "weight_decay", "loss_scale", "label_smoothing"]
            for p in mandatory_params:
                if p not in params:
                    raise RuntimeError("Parameter {} is missing.".format(p))

        if mode == tf.estimator.ModeKeys.TRAIN and not self.model_hparams.use_dali:

            with tf.device('/cpu:0'):
                # Stage inputs on the host
                cpu_prefetch_op, (features, labels) = self._stage([features, labels])

            with tf.device('/gpu:0'):
                # Stage inputs to the device
                gpu_prefetch_op, (features, labels) = self._stage([features, labels])

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

                    print("Using mixup training with beta=", params['mixup'])
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
                reuse=False,
                use_final_conv=params['use_final_conv']
            )
            
            if mode!=tf.estimator.ModeKeys.PREDICT:
                logits = tf.squeeze(logits)

            y_preds = tf.argmax(logits, axis=1, output_type=tf.int32)

            # Check the output dtype, shall be FP32 in training
            assert (probs.dtype == tf.float32)
            assert (logits.dtype == tf.float32)
            assert (y_preds.dtype == tf.int32)

            tf.identity(logits, name="logits_ref")
            tf.identity(probs, name="probs_ref")
            tf.identity(y_preds, name="y_preds_ref")
            
            if mode == tf.estimator.ModeKeys.TRAIN and params['quantize']:
                dllogger.log(data={"QUANTIZATION AWARE TRAINING ENABLED": True}, step=tuple())
                if params['symmetric']:
                    dllogger.log(data={"MODE":"USING SYMMETRIC MODE"}, step=tuple())
                    tf.contrib.quantize.experimental_create_training_graph(tf.get_default_graph(), symmetric=True, use_qdq=params['use_qdq'] ,quant_delay=params['quant_delay'])
                else:
                    dllogger.log(data={"MODE":"USING ASSYMETRIC MODE"}, step=tuple())
                    tf.contrib.quantize.create_training_graph(tf.get_default_graph(), quant_delay=params['quant_delay'], use_qdq=params['use_qdq'])
            
            # Fix for restoring variables during fine-tuning of Resnet-50
            if 'finetune_checkpoint' in params.keys():
                train_vars = tf.trainable_variables()
                train_var_dict = {}
                for var in train_vars:
                    train_var_dict[var.op.name] = var
                dllogger.log(data={"Restoring variables from checkpoint": params['finetune_checkpoint']}, step=tuple())
                tf.train.init_from_checkpoint(params['finetune_checkpoint'], train_var_dict)
                
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



    def build_model(self, inputs, training=True, reuse=False, use_final_conv=False):
        
        with var_storage.model_variable_scope(
            self.model_hparams.model_name,
            reuse=reuse,
            dtype=self.model_hparams.dtype):

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
                mode='SAME',
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

            model_bottlenecks = self.model_hparams.layers_depth
            for block_id, block_bottleneck in enumerate(model_bottlenecks):
                for layer_id in range(self.model_hparams.layers_count[block_id]):
                    stride = 2 if (layer_id == 0 and block_id != 0) else 1

                    net = blocks.bottleneck_block(
                        inputs=net,
                        depth=block_bottleneck * self.model_hparams.expansions,
                        depth_bottleneck=block_bottleneck,
                        cardinality=self.model_hparams.cardinality,
                        stride=stride,
                        training=training,
                        data_format=self.model_hparams.compute_format,
                        conv2d_hparams=self.conv2d_hparams,
                        batch_norm_hparams=self.batch_norm_hparams,
                        block_name="btlnck_block_%d_%d" % (block_id, layer_id),
                        use_se=self.model_hparams.use_se,
                        ratio=self.model_hparams.se_ratio)

            with tf.variable_scope("output"):
                net = layers.reduce_mean(
                    net, keepdims=use_final_conv, data_format=self.model_hparams.compute_format, name='spatial_mean')

                if use_final_conv:
                    logits = layers.conv2d(
                                    net,
                                    n_channels=self.model_hparams.n_classes,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding='SAME',
                                    data_format=self.model_hparams.compute_format,
                                    dilation_rate=(1, 1),
                                    use_bias=True,
                                    kernel_initializer=self.dense_hparams.kernel_initializer,
                                    bias_initializer=self.dense_hparams.bias_initializer,
                                    trainable=training,
                                    name='dense'
                                )
                else:
                    logits = layers.dense(
                        inputs=net,
                        units=self.model_hparams.n_classes,
                        use_bias=True,
                        trainable=training,
                        kernel_initializer=self.dense_hparams.kernel_initializer,
                        bias_initializer=self.dense_hparams.bias_initializer)

                if logits.dtype != tf.float32:
                    logits = tf.cast(logits, tf.float32)
                    
                axis = 3 if self.model_hparams.compute_format=="NHWC" and use_final_conv else 1
                probs = layers.softmax(logits, name="softmax", axis=axis)

            return probs, logits

model_architectures = {
    'resnet50': {
        'layers': [3, 4, 6, 3],
        'widths': [64, 128, 256, 512],
        'expansions': 4,
    },

    'resnext101-32x4d': {
        'layers': [3, 4, 23, 3],
        'widths': [128, 256, 512, 1024],
        'expansions': 2,
        'cardinality': 32,
    },

    'se-resnext101-32x4d' : {
        'cardinality' : 32,
        'layers' : [3, 4, 23, 3],
        'widths' : [128, 256, 512, 1024],
        'expansions' : 2,
        'use_se': True,
        'se_ratio': 16,
    },

}
