#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

"""Model definition for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

import itertools

import tensorflow as tf

from mask_rcnn import anchors

from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.models import resnet

from mask_rcnn.training import losses, learning_rates

from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_local_rank

from mask_rcnn.utils.meters import StandardMeter
from mask_rcnn.utils.metric_tracking import register_metric

from mask_rcnn.utils.lazy_imports import LazyImport
hvd = LazyImport("horovod.tensorflow")

MODELS = dict()


def create_optimizer(learning_rate, params):
    """Creates optimized based on the specified flags."""

    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=params['momentum'])

    if MPI_is_distributed():
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            name=None,
            device_dense='/gpu:0',
            device_sparse='',
            # compression=hvd.Compression.fp16,
            compression=hvd.Compression.none,
            sparse_as_dense=False
        )

    if params["use_amp"]:
        loss_scale = tf.train.experimental.DynamicLossScale(
            initial_loss_scale=(2 ** 15),
            increment_period=2000,
            multiplier=2.0
        )
        optimizer = tf.compat.v1.train.experimental.MixedPrecisionLossScaleOptimizer(optimizer, loss_scale=loss_scale)

    return optimizer


def compute_model_statistics(batch_size, is_training=True):
    """Compute number of parameters and FLOPS."""
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'

    from tensorflow.python.keras.backend import get_graph
    flops = tf.compat.v1.profiler.profile(get_graph(), options=options).total_float_ops
    flops_per_image = flops / batch_size

    logging.info('[%s Compute Statistics] %.1f GFLOPS/image' % (
        "Training" if is_training else "Inference",
        flops_per_image/1e9
    ))


def build_model_graph(features, labels, is_training, params):
    """Builds the forward model graph."""
    model_outputs = {}
    is_gpu_inference = not is_training and params['use_batched_nms']

    batch_size, image_height, image_width, _ = features['images'].get_shape().as_list()

    if 'source_ids' not in features:
        features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

    all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                  params['num_scales'], params['aspect_ratios'],
                                  params['anchor_scale'],
                                  (image_height, image_width))

    MODELS["backbone"] = resnet.Resnet_Model(
        "resnet50",
        data_format='channels_last',
        trainable=is_training,
        finetune_bn=params['finetune_bn']
    )

    backbone_feats = MODELS["backbone"](
        features['images'],
        training=is_training,
    )

    MODELS["FPN"] = fpn.FPNNetwork(params['min_level'], params['max_level'], trainable=is_training)
    fpn_feats = MODELS["FPN"](backbone_feats, training=is_training)

    model_outputs.update({'fpn_features': fpn_feats})

    def rpn_head_fn(features, min_level=2, max_level=6, num_anchors=3):
        """Region Proposal Network (RPN) for Mask-RCNN."""
        scores_outputs = dict()
        box_outputs = dict()

        MODELS["RPN_Heads"] = heads.RPN_Head_Model(name="rpn_head", num_anchors=num_anchors, trainable=is_training)

        for level in range(min_level, max_level + 1):
            scores_outputs[level], box_outputs[level] = MODELS["RPN_Heads"](features[level], training=is_training)

        return scores_outputs, box_outputs

    rpn_score_outputs, rpn_box_outputs = rpn_head_fn(
        features=fpn_feats,
        min_level=params['min_level'],
        max_level=params['max_level'],
        num_anchors=len(params['aspect_ratios'] * params['num_scales'])
    )

    if is_training:
        rpn_pre_nms_topn = params['train_rpn_pre_nms_topn']
        rpn_post_nms_topn = params['train_rpn_post_nms_topn']
        rpn_nms_threshold = params['train_rpn_nms_threshold']

    else:
        rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
        rpn_post_nms_topn = params['test_rpn_post_nms_topn']
        rpn_nms_threshold = params['test_rpn_nms_thresh']

    if params['use_custom_box_proposals_op']:
        rpn_box_scores, rpn_box_rois = roi_ops.custom_multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            all_anchors=all_anchors,
            image_info=features['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=params['rpn_min_size']
        )

    else:
        rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            all_anchors=all_anchors,
            image_info=features['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=params['rpn_min_size'],
            bbox_reg_weights=None,
            use_batched_nms=params['use_batched_nms']
        )


    rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)

    if is_training:
        rpn_box_rois = tf.stop_gradient(rpn_box_rois)
        rpn_box_scores = tf.stop_gradient(rpn_box_scores)  # TODO Jonathan: Unused => Shall keep ?

        # Sampling
        box_targets, class_targets, rpn_box_rois, proposal_to_label_map = training_ops.proposal_label_op(
            rpn_box_rois,
            labels['gt_boxes'],
            labels['gt_classes'],
            batch_size_per_im=params['batch_size_per_im'],
            fg_fraction=params['fg_fraction'],
            fg_thresh=params['fg_thresh'],
            bg_thresh_hi=params['bg_thresh_hi'],
            bg_thresh_lo=params['bg_thresh_lo']
        )

    # Performs multi-level RoIAlign.
    box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features=fpn_feats,
        boxes=rpn_box_rois,
        output_size=7,
        is_gpu_inference=is_gpu_inference
    )

    MODELS["Box_Head"] = heads.Box_Head_Model(
        num_classes=params['num_classes'],
        mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
        trainable=is_training
    )

    class_outputs, box_outputs, _ = MODELS["Box_Head"](inputs=box_roi_features)

    if not is_training:
        if params['use_batched_nms']:
            generate_detections_fn = postprocess_ops.generate_detections_gpu

        else:
            generate_detections_fn = postprocess_ops.generate_detections_tpu

        detections = generate_detections_fn(
            class_outputs=class_outputs,
            box_outputs=box_outputs,
            anchor_boxes=rpn_box_rois,
            image_info=features['image_info'],
            pre_nms_num_detections=params['test_rpn_post_nms_topn'],
            post_nms_num_detections=params['test_detections_per_image'],
            nms_threshold=params['test_nms'],
            bbox_reg_weights=params['bbox_reg_weights']
        )

        model_outputs.update({
            'num_detections': detections[0],
            'detection_boxes': detections[1],
            'detection_classes': detections[2],
            'detection_scores': detections[3],
        })

    else:  # is training
        encoded_box_targets = training_ops.encode_box_targets(
            boxes=rpn_box_rois,
            gt_boxes=box_targets,
            gt_labels=class_targets,
            bbox_reg_weights=params['bbox_reg_weights']
        )

        model_outputs.update({
            'rpn_score_outputs': rpn_score_outputs,
            'rpn_box_outputs': rpn_box_outputs,
            'class_outputs': class_outputs,
            'box_outputs': box_outputs,
            'class_targets': class_targets,
            'box_targets': encoded_box_targets,
            'box_rois': rpn_box_rois,
        })

    # Faster-RCNN mode.
    if not params['include_mask']:
        return model_outputs

    # Mask sampling
    if not is_training:
        selected_box_rois = model_outputs['detection_boxes']
        class_indices = model_outputs['detection_classes']

        # If using GPU for inference, delay the cast until when Gather ops show up
        # since GPU inference supports float point better.
        # TODO(laigd): revisit this when newer versions of GPU libraries is
        # released.
        if not params['use_batched_nms']:
            class_indices = tf.cast(class_indices, dtype=tf.int32)

    else:
        selected_class_targets, selected_box_targets, \
        selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
            class_targets=class_targets,
            box_targets=box_targets,
            boxes=rpn_box_rois,
            proposal_to_label_map=proposal_to_label_map,
            max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction'])
        )

        class_indices = tf.cast(selected_class_targets, dtype=tf.int32)

    mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features=fpn_feats,
        boxes=selected_box_rois,
        output_size=14,
        is_gpu_inference=is_gpu_inference
    )

    MODELS["Mask_Head"] = heads.Mask_Head_Model(
        class_indices,
        num_classes=params['num_classes'],
        mrcnn_resolution=params['mrcnn_resolution'],
        is_gpu_inference=is_gpu_inference,
        trainable=is_training,
        name="mask_head"
    )

    mask_outputs = MODELS["Mask_Head"](inputs=mask_roi_features)

    if MPI_local_rank() == 0:
        # Print #FLOPs in model.
        compute_model_statistics(batch_size, is_training=is_training)

    if is_training:
        mask_targets = training_ops.get_mask_targets(

            fg_boxes=selected_box_rois,
            fg_proposal_to_label_map=proposal_to_label_map,
            fg_box_targets=selected_box_targets,
            mask_gt_labels=labels['cropped_gt_masks'],
            output_size=params['mrcnn_resolution']
        )

        model_outputs.update({
            'mask_outputs': mask_outputs,
            'mask_targets': mask_targets,
            'selected_class_targets': selected_class_targets,
        })

    else:
        model_outputs.update({
            'detection_masks': tf.nn.sigmoid(mask_outputs),
        })

    return model_outputs


def _model_fn(features, labels, mode, params):
    """Model defination for the Mask-RCNN model based on ResNet.

    Args:
    features: the input image tensor and auxiliary information, such as
      `image_info` and `source_ids`. The image tensor has a shape of
      [batch_size, height, width, 3]. The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include score targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.
    """

    # Set up training loss and learning rate.
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if mode == tf.estimator.ModeKeys.PREDICT:

        if params['include_groundtruth_in_features'] and 'labels' in features:
            # In include groundtruth for eval.
            labels = features['labels']

        else:
            labels = None

        if 'features' in features:
            features = features['features']
            # Otherwise, it is in export mode, the features is past in directly.

    model_outputs = build_model_graph(features, labels, mode == tf.estimator.ModeKeys.TRAIN, params)

    model_outputs.update({
        'source_id': features['source_ids'],
        'image_info': features['image_info'],
    })

    if mode == tf.estimator.ModeKeys.PREDICT and 'orig_images' in features:
        model_outputs['orig_images'] = features['orig_images']

    # First check if it is in PREDICT mode or EVAL mode to fill out predictions.
    # Predictions are used during the eval step to generate metrics.
    if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        predictions = {}

        try:
            model_outputs['orig_images'] = features['orig_images']
        except KeyError:
            pass

        if labels and params['include_groundtruth_in_features']:
            # Labels can only be embedded in predictions. The prediction cannot output
            # dictionary as a value.
            predictions.update(labels)

        model_outputs.pop('fpn_features', None)
        predictions.update(model_outputs)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # If we are doing PREDICT, we can return here.
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # score_loss and box_loss are for logging. only total_loss is optimized.
    total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
        score_outputs=model_outputs['rpn_score_outputs'],
        box_outputs=model_outputs['rpn_box_outputs'],
        labels=labels,
        params=params
    )

    total_fast_rcnn_loss, fast_rcnn_class_loss, fast_rcnn_box_loss = losses.fast_rcnn_loss(
        class_outputs=model_outputs['class_outputs'],
        box_outputs=model_outputs['box_outputs'],
        class_targets=model_outputs['class_targets'],
        box_targets=model_outputs['box_targets'],
        params=params
    )

    # Only training has the mask loss.
    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/model_builder.py
    if mode == tf.estimator.ModeKeys.TRAIN and params['include_mask']:
        mask_loss = losses.mask_rcnn_loss(
            mask_outputs=model_outputs['mask_outputs'],
            mask_targets=model_outputs['mask_targets'],
            select_class_targets=model_outputs['selected_class_targets'],
            params=params
        )

    else:
        mask_loss = 0.

    trainable_variables = list(itertools.chain.from_iterable([model.trainable_variables for model in MODELS.values()]))

    l2_regularization_loss = params['l2_weight_decay'] * tf.add_n([
        tf.nn.l2_loss(v)
        for v in trainable_variables
        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
    ])

    total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + l2_regularization_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        # Predictions can only contain a dict of tensors, not a dict of dict of
        # tensors. These outputs are not used for eval purposes.
        del predictions['rpn_score_outputs']
        del predictions['rpn_box_outputs']

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=total_loss
        )

    if mode == tf.estimator.ModeKeys.TRAIN:

        learning_rate = learning_rates.step_learning_rate_with_linear_warmup(
            global_step=global_step,
            init_learning_rate=params['init_learning_rate'],
            warmup_learning_rate=params['warmup_learning_rate'],
            warmup_steps=params['warmup_steps'],
            learning_rate_levels=params['learning_rate_levels'],
            learning_rate_steps=params['learning_rate_steps']
        )

        optimizer = create_optimizer(learning_rate, params)

        grads_and_vars = optimizer.compute_gradients(total_loss, trainable_variables, colocate_gradients_with_ops=True)

        gradients, variables = zip(*grads_and_vars)
        grads_and_vars = []

        # Special treatment for biases (beta is named as bias in reference model)
        # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
        for grad, var in zip(gradients, variables):

            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad

            grads_and_vars.append((grad, var))

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    else:
        train_op = None
        learning_rate = None

    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group

    if not isinstance(replica_id, tf.Tensor) or tf.get_static_value(replica_id) == 0:

        register_metric(name="L2 loss", tensor=l2_regularization_loss, aggregator=StandardMeter())
        register_metric(name="Mask loss", tensor=mask_loss, aggregator=StandardMeter())
        register_metric(name="Total loss", tensor=total_loss, aggregator=StandardMeter())

        register_metric(name="RPN box loss", tensor=rpn_box_loss, aggregator=StandardMeter())
        register_metric(name="RPN score loss", tensor=rpn_score_loss, aggregator=StandardMeter())
        register_metric(name="RPN total loss", tensor=total_rpn_loss, aggregator=StandardMeter())

        register_metric(name="FastRCNN class loss", tensor=fast_rcnn_class_loss, aggregator=StandardMeter())
        register_metric(name="FastRCNN box loss", tensor=fast_rcnn_box_loss, aggregator=StandardMeter())
        register_metric(name="FastRCNN total loss", tensor=total_fast_rcnn_loss, aggregator=StandardMeter())

        register_metric(name="Learning rate", tensor=learning_rate, aggregator=StandardMeter())
        pass
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
    )


def mask_rcnn_model_fn(features, labels, mode, params):
    """Mask-RCNN model."""

    return _model_fn(
        features,
        labels,
        mode,
        params
    )
