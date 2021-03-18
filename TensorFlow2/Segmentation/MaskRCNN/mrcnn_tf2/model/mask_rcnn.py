import tensorflow as tf

from mrcnn_tf2.model import anchors
from mrcnn_tf2.model.losses import MaskRCNNLoss, FastRCNNLoss, RPNLoss
from mrcnn_tf2.model.models.fpn import FPNNetwork
from mrcnn_tf2.model.models.heads import RPNHead, BoxHead, MaskHead
from mrcnn_tf2.model.models.resnet50 import ResNet50
from mrcnn_tf2.ops import roi_ops, spatial_transform_ops, postprocess_ops, training_ops


class MaskRCNN(tf.keras.Model):

    def __init__(self, params, name='mrcnn', trainable=True, *args, **kwargs):
        super().__init__(name=name, trainable=trainable, *args, **kwargs)
        self._params = params

        self.backbone = ResNet50()

        self.fpn = FPNNetwork(
            min_level=self._params.min_level,
            max_level=self._params.max_level,
            trainable=trainable
        )

        self.rpn_head = RPNHead(
            name="rpn_head",
            num_anchors=len(self._params.aspect_ratios * self._params.num_scales),
            trainable=trainable
        )

        self.box_head = BoxHead(
            num_classes=self._params.num_classes,
            mlp_head_dim=self._params.fast_rcnn_mlp_head_dim,
            trainable=trainable
        )

        self.mask_head = MaskHead(
            num_classes=self._params.num_classes,
            mrcnn_resolution=self._params.mrcnn_resolution,
            trainable=trainable,
            name="mask_head"
        )

        self.mask_rcnn_loss = MaskRCNNLoss()

        self.fast_rcnn_loss = FastRCNNLoss(
            num_classes=self._params.num_classes
        )

        self.rpn_loss = RPNLoss(
            batch_size=self._params.train_batch_size,
            rpn_batch_size_per_im=self._params.rpn_batch_size_per_im,
            min_level=self._params.min_level,
            max_level=self._params.max_level
        )

    def call(self, inputs, training=None, mask=None):

        batch_size, image_height, image_width, _ = inputs['images'].get_shape().as_list()

        if 'source_ids' not in inputs:
            inputs['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)

        outputs = dict(inputs)

        all_anchors = anchors.Anchors(self._params.min_level, self._params.max_level,
                                      self._params.num_scales, self._params.aspect_ratios,
                                      self._params.anchor_scale,
                                      (image_height, image_width))

        backbone_feats = self.backbone(inputs['images'], training=training)

        fpn_feats = self.fpn(backbone_feats, training=training)

        outputs.update({'fpn_features': fpn_feats})

        def rpn_head_fn(features, min_level=2, max_level=6):
            """Region Proposal Network (RPN) for Mask-RCNN."""
            scores_outputs = dict()
            box_outputs = dict()

            for level in range(min_level, max_level + 1):
                scores_outputs[level], box_outputs[level] = self.rpn_head(features[level], training=training)

            return scores_outputs, box_outputs

        rpn_score_outputs, rpn_box_outputs = rpn_head_fn(
            features=fpn_feats,
            min_level=self._params.min_level,
            max_level=self._params.max_level
        )

        if training:
            rpn_pre_nms_topn = self._params.train_rpn_pre_nms_topn
            rpn_post_nms_topn = self._params.train_rpn_post_nms_topn
            rpn_nms_threshold = self._params.train_rpn_nms_threshold

        else:
            rpn_pre_nms_topn = self._params.test_rpn_pre_nms_topn
            rpn_post_nms_topn = self._params.test_rpn_post_nms_topn
            rpn_nms_threshold = self._params.test_rpn_nms_thresh

        rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
            scores_outputs=rpn_score_outputs,
            box_outputs=rpn_box_outputs,
            all_anchors=all_anchors,
            image_info=inputs['image_info'],
            rpn_pre_nms_topn=rpn_pre_nms_topn,
            rpn_post_nms_topn=rpn_post_nms_topn,
            rpn_nms_threshold=rpn_nms_threshold,
            rpn_min_size=self._params.rpn_min_size,
            bbox_reg_weights=None
        )

        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)

        if training:
            rpn_box_rois = tf.stop_gradient(rpn_box_rois)
            rpn_box_scores = tf.stop_gradient(rpn_box_scores)  # TODO Jonathan: Unused => Shall keep ?

            # Sampling
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = training_ops.proposal_label_op(
                rpn_box_rois,
                inputs['gt_boxes'],
                inputs['gt_classes'],
                batch_size_per_im=self._params.batch_size_per_im,
                fg_fraction=self._params.fg_fraction,
                fg_thresh=self._params.fg_thresh,
                bg_thresh_hi=self._params.bg_thresh_hi,
                bg_thresh_lo=self._params.bg_thresh_lo
            )

        # Performs multi-level RoIAlign.
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=rpn_box_rois,
            output_size=7,
            training=training
        )

        class_outputs, box_outputs, _ = self.box_head(inputs=box_roi_features)

        if not training:
            detections = postprocess_ops.generate_detections_gpu(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=inputs['image_info'],
                pre_nms_num_detections=self._params.test_rpn_post_nms_topn,
                post_nms_num_detections=self._params.test_detections_per_image,
                nms_threshold=self._params.test_nms,
                bbox_reg_weights=self._params.bbox_reg_weights
            )

            outputs.update({
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
                bbox_reg_weights=self._params.bbox_reg_weights
            )

            outputs.update({
                'rpn_score_outputs': rpn_score_outputs,
                'rpn_box_outputs': rpn_box_outputs,
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets,
                'box_rois': rpn_box_rois,
            })

        # Faster-RCNN mode.
        if not self._params.include_mask:
            return outputs

        # Mask sampling
        if not training:
            selected_box_rois = outputs['detection_boxes']
            class_indices = outputs['detection_classes']

        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(self._params.batch_size_per_im * self._params.fg_fraction)
            )

            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)

        mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=selected_box_rois,
            output_size=14,
            training=training
        )

        mask_outputs = self.mask_head(
            inputs=(mask_roi_features, class_indices),
            training=training
        )

        if training:
            mask_targets = training_ops.get_mask_targets(

                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=inputs['cropped_gt_masks'],
                output_size=self._params.mrcnn_resolution
            )

            outputs.update({
                'mask_outputs': mask_outputs,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })

        else:
            outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_outputs),
            })

        if training:
            self._add_losses(outputs)

        # filter out only the needed outputs
        model_outputs = [
            'source_ids', 'image_info',
            'num_detections', 'detection_boxes',
            'detection_classes', 'detection_scores',
            'detection_masks'
        ]
        return {
            name: tf.identity(tensor, name=name)
            for name, tensor in outputs.items()
            if name in model_outputs
        }

    def _add_losses(self, model_outputs):
        mask_rcnn_loss = self.mask_rcnn_loss(model_outputs)
        mask_rcnn_loss *= self._params.mrcnn_weight_loss_mask
        self.add_loss(mask_rcnn_loss)
        self.add_metric(mask_rcnn_loss, name='mask_rcnn_loss')

        fast_rcnn_class_loss, fast_rcnn_box_loss = self.fast_rcnn_loss(model_outputs)
        fast_rcnn_box_loss *= self._params.fast_rcnn_box_loss_weight
        self.add_loss(fast_rcnn_box_loss)
        self.add_metric(fast_rcnn_box_loss, name='fast_rcnn_box_loss')
        self.add_loss(fast_rcnn_class_loss)
        self.add_metric(fast_rcnn_class_loss, name='fast_rcnn_class_loss')

        rpn_score_loss, rpn_box_loss = self.rpn_loss(model_outputs)
        rpn_box_loss *= self._params.rpn_box_loss_weight
        self.add_loss(rpn_box_loss)
        self.add_metric(rpn_box_loss, name='rpn_box_loss')
        self.add_loss(rpn_score_loss)
        self.add_metric(rpn_score_loss, name='rpn_score_loss')

        l2_regularization_loss = tf.add_n([
            tf.nn.l2_loss(tf.cast(v, dtype=tf.float32))
            for v in self.trainable_variables
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ])
        l2_regularization_loss *= self._params.l2_weight_decay
        self.add_loss(l2_regularization_loss)
        self.add_metric(l2_regularization_loss, name='l2_regularization_loss')

    def get_config(self):
        pass
