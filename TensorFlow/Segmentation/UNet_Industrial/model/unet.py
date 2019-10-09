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
import horovod.tensorflow as hvd

from model import layers
from model import blocks

from utils import hvd_utils

from utils import losses
from utils import metrics

from utils import image_processing

from dllogger.logger import LOGGER

__all__ = ["UNet_v1"]


class UNet_v1(object):

    authorized_weight_init_methods = [
        "he_normal",
        "he_uniform",
        "glorot_normal",
        "glorot_uniform",
        "orthogonal",
    ]

    authorized_models_variants = [
        "original",
        "tinyUNet",
    ]

    def __init__(
        self,
        model_name,
        compute_format,
        input_format,
        n_output_channels,
        unet_variant,
        activation_fn,
        weight_init_method,
    ):

        if unet_variant == "original":  # Total Params: 36,950,273
            input_filters = 64
            unet_block_filters = [128, 256, 512]
            bottleneck_filters = 1024
            output_filters = 64

        elif unet_variant == "tinyUNet":  # Total Params: 1,824,945
            input_filters = 32
            unet_block_filters = [32, 64, 128]
            bottleneck_filters = 256
            output_filters = 32

        else:
            raise ValueError(
                "Unknown `UNet` variant: %s. Authorized: %s" % (unet_variant, UNet_v1.authorized_models_variants)
            )

        if activation_fn not in blocks.authorized_activation_fn:
            raise ValueError(
                "Unknown activation function: %s - Authorised: %s" % (activation_fn, blocks.authorized_activation_fn)
            )

        self.model_hparams = tf.contrib.training.HParams(
            compute_format=compute_format,
            input_format=input_format,
            input_filters=input_filters,
            unet_block_filters=unet_block_filters,
            bottleneck_filters=bottleneck_filters,
            output_filters=output_filters,
            n_output_channels=n_output_channels,
            model_name=model_name,
        )

        self.conv2d_hparams = tf.contrib.training.HParams(
            kernel_initializer=None, bias_initializer=tf.initializers.constant(0.0), activation_fn=activation_fn
        )

        if weight_init_method == "he_normal":
            self.conv2d_hparams.kernel_initializer = tf.initializers.variance_scaling(
                scale=2.0, distribution='truncated_normal', mode='fan_in'
            )

        elif weight_init_method == "he_uniform":
            self.conv2d_hparams.kernel_initializer = tf.initializers.variance_scaling(
                scale=2.0, distribution='uniform', mode='fan_in'
            )

        elif weight_init_method == "glorot_normal":
            self.conv2d_hparams.kernel_initializer = tf.initializers.variance_scaling(
                scale=1.0, distribution='truncated_normal', mode='fan_avg'
            )

        elif weight_init_method == "glorot_uniform":
            self.conv2d_hparams.kernel_initializer = tf.initializers.variance_scaling(
                scale=1.0, distribution='uniform', mode='fan_avg'
            )

        elif weight_init_method == "orthogonal":
            self.conv2d_hparams.kernel_initializer = tf.initializers.orthogonal(gain=1.0)

        else:
            raise ValueError(
                "Unknown weight init method: %s - Authorized: %s" %
                (weight_init_method, UNet_v1.authorized_weight_init_methods)
            )

    def __call__(self, features, labels, mode, params):

        if "debug_verbosity" not in params.keys():
            raise RuntimeError("Parameter `debug_verbosity` is missing...")

        if mode == tf.estimator.ModeKeys.TRAIN:

            if "rmsprop_decay" not in params.keys():
                raise RuntimeError("Parameter `rmsprop_decay` is missing...")

            if "rmsprop_momentum" not in params.keys():
                raise RuntimeError("Parameter `rmsprop_momentum` is missing...")

            if "learning_rate" not in params.keys():
                raise RuntimeError("Parameter `learning_rate` is missing...")

            if "learning_rate_decay_steps" not in params.keys():
                raise RuntimeError("Parameter `learning_rate` is missing...")

            if "learning_rate_decay_factor" not in params.keys():
                raise RuntimeError("Parameter `learning_rate` is missing...")

            if "weight_decay" not in params.keys():
                raise RuntimeError("Parameter `weight_decay` is missing...")

            if "loss_fn_name" not in params.keys():
                raise RuntimeError("Parameter `loss_fn_name` is missing...")

        if mode == tf.estimator.ModeKeys.PREDICT:
            y_pred, y_pred_logits = self.build_model(
                features, training=False, reuse=False, debug_verbosity=params["debug_verbosity"]
            )

            predictions = {'logits': y_pred}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        input_image, mask_image = features

        with tf.device("/gpu:0"):

            tf.identity(input_image, name="input_image_ref")
            tf.identity(mask_image, name="mask_image_ref")
            tf.identity(labels, name="labels_ref")

            y_pred, y_pred_logits = self.build_model(
                input_image,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                reuse=False,
                debug_verbosity=params["debug_verbosity"]
            )

            all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
            tf.identity(all_trainable_vars, name='trainable_parameters_count_ref')

            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metrics = dict()

            # ==================== Samples ==================== #

            image_uint8 = tf.cast((input_image + 1) * 127.5, dtype=tf.uint8)
            input_image_jpeg = tf.image.encode_jpeg(image_uint8[0], format='grayscale', quality=100)
            tf.identity(input_image_jpeg, name="input_image_jpeg_ref")

            for threshold in [None, 0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:
                binarize_img, binarize_img_jpeg = image_processing.binarize_output(y_pred[0], threshold=threshold)

                tf.identity(binarize_img_jpeg, name="output_sample_ths_%s_ref" % threshold)
                tf.summary.image('output_sample_ths_%s' % threshold, binarize_img, 10)

            # ==============+ Evaluation Metrics ==================== #

            with tf.name_scope("IoU_Metrics"):

                for threshold in [0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99]:

                    iou_score = metrics.iou_score(y_pred=y_pred, y_true=mask_image, threshold=threshold)

                    tf.identity(iou_score, name='iou_score_ths_%s_ref' % threshold)
                    tf.summary.scalar('iou_score_ths_%s' % threshold, iou_score)

                    if mode == tf.estimator.ModeKeys.EVAL:
                        eval_metrics["IoU_THS_%s" % threshold] = tf.metrics.mean(iou_score)

            labels = tf.cast(labels, tf.float32)
            labels_preds = tf.reduce_max(y_pred, axis=(1, 2, 3))

            with tf.variable_scope("Confusion_Matrix") as scope:

                tp, update_tp = tf.metrics.true_positives_at_thresholds(
                    labels=labels,
                    predictions=labels_preds,
                    thresholds=[0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99],
                )

                tn, update_tn = tf.metrics.true_negatives_at_thresholds(
                    labels=labels,
                    predictions=labels_preds,
                    thresholds=[0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99],
                )

                fp, update_fp = tf.metrics.false_positives_at_thresholds(
                    labels=labels,
                    predictions=labels_preds,
                    thresholds=[0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99],
                )

                fn, update_fn = tf.metrics.false_negatives_at_thresholds(
                    labels=labels,
                    predictions=labels_preds,
                    thresholds=[0.05, 0.125, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99],
                )

                if mode == tf.estimator.ModeKeys.TRAIN:
                    local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope.name)
                    confusion_matrix_reset_op = tf.initializers.variables(local_vars, name='reset_op')

                    with tf.control_dependencies([confusion_matrix_reset_op]):
                        with tf.control_dependencies([update_tp, update_tn, update_fp, update_fn]):
                            tp = tf.identity(tp)
                            tn = tf.identity(tn)
                            fp = tf.identity(fp)
                            fn = tf.identity(fn)

                else:
                    eval_metrics["Confusion_Matrix_TP"] = tp, update_tp
                    eval_metrics["Confusion_Matrix_TN"] = tn, update_tn
                    eval_metrics["Confusion_Matrix_FP"] = fp, update_fp
                    eval_metrics["Confusion_Matrix_FN"] = fn, update_fn

                tf.identity(tp, name='true_positives_ref')  # Confusion_Matrix/true_positives_ref:0
                tf.identity(tn, name='true_negatives_ref')  # Confusion_Matrix/true_negatives_ref:0
                tf.identity(fp, name='false_positives_ref')  # Confusion_Matrix/false_positives_ref:0
                tf.identity(fn, name='false_negatives_ref')  # Confusion_Matrix/false_negatives_ref:0

                tf.summary.scalar('true_positives', tp[3])  # For Ths = 0.5
                tf.summary.scalar('true_negatives', tn[3])  # For Ths = 0.5
                tf.summary.scalar('false_positives', fp[3])  # For Ths = 0.5
                tf.summary.scalar('false_negatives', fn[3])  # For Ths = 0.5

            binarized_mask, binarized_mask_jpeg = image_processing.binarize_output(mask_image[0], threshold=0.5)
            tf.identity(binarized_mask_jpeg, name="mask_sample_ref")
            tf.summary.image('sample_mask', binarized_mask, 10)

            ##########################

            mask_max_val = tf.reduce_max(mask_image)
            tf.identity(mask_max_val, name='mask_max_val_ref')

            mask_min_val = tf.reduce_min(mask_image)
            tf.identity(mask_min_val, name='mask_min_val_ref')

            mask_mean_val = tf.reduce_mean(mask_image)
            tf.identity(mask_mean_val, name='mask_mean_val_ref')

            mask_std_val = tf.math.reduce_std(mask_image)
            tf.identity(mask_std_val, name='mask_std_val_ref')

            ##########################

            output_max_val = tf.reduce_max(y_pred)
            tf.identity(output_max_val, name='output_max_val_ref')

            output_min_val = tf.reduce_min(y_pred)
            tf.identity(output_min_val, name='output_min_val_ref')

            output_mean_val = tf.reduce_mean(y_pred)
            tf.identity(output_mean_val, name='output_mean_val_ref')

            output_std_val = tf.math.reduce_std(y_pred)
            tf.identity(output_std_val, name='output_std_val_ref')

            with tf.variable_scope("losses"):

                # ==============+ Reconstruction Loss ==================== #

                if params["loss_fn_name"] == "x-entropy":
                    reconstruction_loss = losses.reconstruction_x_entropy(y_pred=y_pred, y_true=mask_image)

                elif params["loss_fn_name"] == "l2_loss":
                    reconstruction_loss = losses.reconstruction_l2loss(y_pred=y_pred, y_true=mask_image)

                elif params["loss_fn_name"] == "dice_sorensen":
                    reconstruction_loss = 1 - losses.dice_coe(y_pred=y_pred, y_true=mask_image, loss_type='sorensen')

                elif params["loss_fn_name"] == "dice_jaccard":
                    reconstruction_loss = 1 - losses.dice_coe(y_pred=y_pred, y_true=mask_image, loss_type='jaccard')

                elif params["loss_fn_name"] == "adaptive_loss":
                    reconstruction_loss = losses.adaptive_loss(
                        y_pred=y_pred,
                        y_pred_logits=y_pred_logits,
                        y_true=mask_image,
                        switch_at_threshold=0.3,
                        loss_type='sorensen'
                    )

                else:
                    raise ValueError("Unknown loss function received: %s" % params["loss_fn_name"])

                tf.identity(reconstruction_loss, name='reconstruction_loss_ref')
                tf.summary.scalar('reconstruction_loss', reconstruction_loss)

                if mode == tf.estimator.ModeKeys.TRAIN:

                    # ============== Regularization Loss ==================== #

                    l2_loss = losses.regularization_l2loss(weight_decay=params["weight_decay"])

                    tf.identity(l2_loss, name='l2_loss_ref')
                    tf.summary.scalar('l2_loss', l2_loss)

                    total_loss = tf.add(reconstruction_loss, l2_loss, name="total_loss")

                else:
                    total_loss = reconstruction_loss

                tf.identity(total_loss, name='total_loss_ref')
                tf.summary.scalar('total_loss', total_loss)

            if mode == tf.estimator.ModeKeys.TRAIN:

                with tf.variable_scope("optimizers"):

                    # Update Global Step
                    global_step = tf.train.get_or_create_global_step()
                    tf.identity(global_step, name="global_step_ref")

                    learning_rate = tf.train.exponential_decay(
                        learning_rate=params["learning_rate"],
                        decay_steps=params["learning_rate_decay_steps"],
                        decay_rate=params["learning_rate_decay_factor"],
                        global_step=global_step,
                        staircase=True
                    )

                    tf.identity(learning_rate, name="learning_rate_ref")
                    tf.summary.scalar('learning_rate_ref', learning_rate)

                    opt = tf.train.RMSPropOptimizer(
                        learning_rate=learning_rate,
                        use_locking=False,
                        centered=True,
                        decay=params["rmsprop_decay"],
                        momentum=params["rmsprop_momentum"],
                    )

                    if hvd_utils.is_using_hvd():
                        opt = hvd.DistributedOptimizer(opt, device_dense='/gpu:0')

                    if params["apply_manual_loss_scaling"]:

                        if not hvd_utils.is_using_hvd() or hvd.local_rank() == 0:
                            LOGGER.log("Applying manual Loss Scaling ...")

                        loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
                            init_loss_scale=2**32,  # 4,294,967,296
                            incr_every_n_steps=1000
                        )
                        opt = tf.contrib.mixed_precision.LossScaleOptimizer(opt, loss_scale_manager)

                    deterministic = True
                    gate_gradients = (tf.train.Optimizer.GATE_OP if deterministic else tf.train.Optimizer.GATE_NONE)

                    backprop_op = opt.minimize(total_loss, gate_gradients=gate_gradients, global_step=global_step)

                    train_op = tf.group(backprop_op, tf.get_collection(tf.GraphKeys.UPDATE_OPS))

                    return tf.estimator.EstimatorSpec(
                        mode,
                        loss=total_loss,
                        train_op=train_op,
                    )

            elif mode == tf.estimator.ModeKeys.EVAL:

                return tf.estimator.EstimatorSpec(
                    mode, loss=total_loss, eval_metric_ops=eval_metrics, predictions={"output": y_pred}
                )

            else:
                raise NotImplementedError('Unknown mode {}'.format(mode))

    def build_model(self, inputs, training=True, reuse=False, debug_verbosity=0):
        """
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/pdf/1505.04597
        """

        skip_connections = []

        with tf.variable_scope(self.model_hparams.model_name, reuse=reuse):

            with tf.variable_scope("input_reshape"):

                with tf.variable_scope("initial_zero_padding"):
                    inputs = tf.image.resize_image_with_crop_or_pad(inputs, target_height=512, target_width=512)

                if self.model_hparams.input_format == 'NHWC' and self.model_hparams.compute_format == 'NCHW':
                    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                    # This provides a large performance boost on GPU. See
                    # https://www.tensorflow.org/performance/performance_guide#data_formats

                    # Reshape inputs: NHWC => NCHW
                    net = tf.transpose(inputs, [0, 3, 1, 2])

                elif self.model_hparams.input_format == 'NCHW' and self.model_hparams.compute_format == 'NHWC':

                    # Reshape inputs: NCHW => NHWC
                    net = tf.transpose(inputs, [0, 2, 3, 1])

                else:
                    net = inputs

            # net, out = input_block(net, filters=64)

            net, out = blocks.input_unet_block(
                net,
                filters=self.model_hparams.input_filters,
                data_format=self.model_hparams.compute_format,
                is_training=training,
                conv2d_hparams=self.conv2d_hparams
            )

            skip_connections.append(out)

            for idx, filters in enumerate(self.model_hparams.unet_block_filters):
                # net, out = downsample_block(net, filters=filters, idx=idx)

                net, skip_connect = blocks.downsample_unet_block(
                    net,
                    filters=filters,
                    data_format=self.model_hparams.compute_format,
                    is_training=training,
                    conv2d_hparams=self.conv2d_hparams,
                    block_name="downsample_block_%d" % (idx + 1)
                )

                skip_connections.append(skip_connect)

            net = blocks.bottleneck_unet_block(
                net,
                filters=self.model_hparams.bottleneck_filters,
                data_format=self.model_hparams.compute_format,
                is_training=training,
                conv2d_hparams=self.conv2d_hparams,
            )

            for idx, filters in enumerate(reversed(self.model_hparams.unet_block_filters)):
                net = blocks.upsample_unet_block(
                    net,
                    residual_input=skip_connections.pop(),
                    filters=filters,
                    data_format=self.model_hparams.compute_format,
                    is_training=training,
                    conv2d_hparams=self.conv2d_hparams,
                    block_name='upsample_block_%d' % (idx + 1)
                )

            logits = blocks.output_unet_block(
                inputs=net,
                residual_input=skip_connections.pop(),
                filters=self.model_hparams.output_filters,
                n_output_channels=self.model_hparams.n_output_channels,
                data_format=self.model_hparams.compute_format,
                is_training=training,
                conv2d_hparams=self.conv2d_hparams,
                block_name='ouputs_block'
            )

            if self.model_hparams.compute_format == "NCHW":
                logits = tf.transpose(logits, [0, 2, 3, 1])

            outputs = layers.sigmoid(logits)

            return outputs, logits
