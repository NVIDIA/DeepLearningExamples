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
"""Losses used for Mask-RCNN."""
import tensorflow as tf


class MaskRCNNLoss(tf.keras.layers.Layer):
    """
    Layer that computes the mask loss of Mask-RCNN.

    This layer implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`.
    """

    def __init__(self):
        super().__init__(trainable=False, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: dictionary with model outputs, which has to include:
                mask_outputs: a float tensor representing the prediction for each mask,
                    with a shape of [batch_size, num_masks, mask_height, mask_width].
                mask_targets: a float tensor representing the binary mask of ground truth
                    labels for each mask with a shape of [batch_size, num_masks, mask_height, mask_width].
                select_class_targets: a tensor with a shape of [batch_size, num_masks],
                    representing the foreground mask targets.
        Returns:
            mask_loss: a float tensor representing total mask loss.
        """
        mask_outputs = inputs['mask_outputs']
        mask_targets = inputs['mask_targets']
        select_class_targets = inputs['selected_class_targets']

        batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()

        weights = tf.tile(
            tf.reshape(tf.greater(select_class_targets, 0), [batch_size, num_masks, 1, 1]),
            [1, 1, mask_height, mask_width]
        )
        weights = tf.cast(weights, tf.float32)

        return _sigmoid_cross_entropy(
            multi_class_labels=mask_targets,
            logits=mask_outputs,
            weights=weights,
            sum_by_non_zeros_weights=True
        )


class FastRCNNLoss(tf.keras.layers.Layer):
    """
    Layer that computes the box and class loss (Fast-RCNN branch) of Mask-RCNN.

    This layer implements the classification and box regression loss of the
    Fast-RCNN branch in Mask-RCNN. As the `box_outputs` produces `num_classes`
    boxes for each RoI, the reference model expands `box_targets` to match the
    shape of `box_outputs` and selects only the target that the RoI has a maximum
    overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py)
    Instead, this function selects the `box_outputs` by the `class_targets` so
    that it doesn't expand `box_targets`.

    The loss computation has two parts: (1) classification loss is softmax on all
    RoIs. (2) box loss is smooth L1-loss on only positive samples of RoIs.
    Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py
    """

    def __init__(self, num_classes):
        super().__init__(trainable=False, dtype=tf.float32)
        self._num_classes = num_classes

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: dictionary with model outputs, which has to include:
                class_outputs: a float tensor representing the class prediction for each box
                    with a shape of [batch_size, num_boxes, num_classes].
                box_outputs: a float tensor representing the box prediction for each box
                    with a shape of [batch_size, num_boxes, num_classes * 4].
                class_targets: a float tensor representing the class label for each box
                    with a shape of [batch_size, num_boxes].
                box_targets: a float tensor representing the box label for each box
                    with a shape of [batch_size, num_boxes, 4].
        Returns:
            cls_loss: a float tensor representing total class loss.
            box_loss: a float tensor representing total box regression loss.
        """
        class_outputs = inputs['class_outputs']
        box_outputs = inputs['box_outputs']
        class_targets = inputs['class_targets']
        box_targets = inputs['box_targets']

        class_targets = tf.cast(class_targets, dtype=tf.int32)

        # Selects the box from `box_outputs` based on `class_targets`, with which
        # the box has the maximum overlap.
        batch_size, num_rois, _ = box_outputs.get_shape().as_list()
        box_outputs = tf.reshape(box_outputs, [batch_size, num_rois, self._num_classes, 4])

        box_indices = tf.reshape(
            class_targets +
            tf.tile(tf.expand_dims(tf.range(batch_size) * num_rois * self._num_classes, 1), [1, num_rois]) +
            tf.tile(tf.expand_dims(tf.range(num_rois) * self._num_classes, 0), [batch_size, 1]),
            [-1]
        )

        box_outputs = tf.matmul(
            tf.one_hot(
                box_indices,
                batch_size * num_rois * self._num_classes,
                dtype=box_outputs.dtype
            ),
            tf.reshape(box_outputs, [-1, 4])
        )

        box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])
        box_loss = _fast_rcnn_box_loss(
            box_outputs=box_outputs,
            box_targets=box_targets,
            class_targets=class_targets,
            normalizer=1.0
        )

        class_targets = tf.one_hot(class_targets, self._num_classes)
        class_loss = _fast_rcnn_class_loss(
            class_outputs=class_outputs,
            class_targets_one_hot=class_targets,
            normalizer=1.0
        )

        return class_loss, box_loss


class RPNLoss(tf.keras.layers.Layer):
    """
    Layer that computes total RPN detection loss.

    Computes total RPN detection loss including box and score from all levels.
    """

    def __init__(self, batch_size, rpn_batch_size_per_im, min_level, max_level):
        super().__init__(trainable=False, dtype=tf.float32)
        self._batch_size = batch_size
        self._rpn_batch_size_per_im = rpn_batch_size_per_im
        self._min_level = min_level
        self._max_level = max_level

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs: dictionary with model outputs, which has to include:
                score_outputs: an OrderDict with keys representing levels and values
                    representing scores in [batch_size, height, width, num_anchors].
                box_outputs: an OrderDict with keys representing levels and values
                    representing box regression targets in [batch_size, height, width, num_anchors * 4].
                score_targets_*: ground truth score targets
                box_targets_*: ground truth box targets
        Returns:
            rpn_score_loss: a float tensor representing total score loss.
            rpn_box_loss: a float tensor representing total box regression loss.
        """
        score_outputs = inputs['rpn_score_outputs']
        box_outputs = inputs['rpn_box_outputs']

        score_losses = []
        box_losses = []

        for level in range(int(self._min_level), int(self._max_level + 1)):

            score_targets_at_level = inputs['score_targets_%d' % level]
            box_targets_at_level = inputs['box_targets_%d' % level]

            score_losses.append(
                _rpn_score_loss(
                    score_outputs=score_outputs[level],
                    score_targets=score_targets_at_level,
                    normalizer=tf.cast(self._batch_size * self._rpn_batch_size_per_im, dtype=tf.float32)
                )
            )

            box_losses.append(_rpn_box_loss(
                box_outputs=box_outputs[level],
                box_targets=box_targets_at_level,
                normalizer=1.0
            ))

        # Sum per level losses to total loss.
        rpn_score_loss = tf.add_n(score_losses)
        rpn_box_loss = tf.add_n(box_losses)

        return rpn_score_loss, rpn_box_loss


def _huber_loss(y_true, y_pred, weights, delta):

    num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)

    huber_keras_loss = tf.keras.losses.Huber(
        delta=delta,
        reduction=tf.keras.losses.Reduction.SUM,
        name='huber_loss'
    )

    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    huber_loss = huber_keras_loss(
        y_true,
        y_pred,
        sample_weight=weights
    )

    assert huber_loss.dtype == tf.float32

    huber_loss = tf.math.divide_no_nan(huber_loss, num_non_zeros, name="huber_loss")

    assert huber_loss.dtype == tf.float32
    return huber_loss


def _sigmoid_cross_entropy(multi_class_labels, logits, weights, sum_by_non_zeros_weights=False):

    assert weights.dtype == tf.float32

    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=multi_class_labels,
        logits=logits,
        name="x-entropy"
    )

    assert sigmoid_cross_entropy.dtype == tf.float32

    sigmoid_cross_entropy = tf.math.multiply(sigmoid_cross_entropy, weights)
    sigmoid_cross_entropy = tf.math.reduce_sum(input_tensor=sigmoid_cross_entropy)

    assert sigmoid_cross_entropy.dtype == tf.float32

    if sum_by_non_zeros_weights:
        num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)
        sigmoid_cross_entropy = tf.math.divide_no_nan(
            sigmoid_cross_entropy,
            num_non_zeros,
            name="sum_by_non_zeros_weights"
        )

    assert sigmoid_cross_entropy.dtype == tf.float32
    return sigmoid_cross_entropy


def _softmax_cross_entropy(onehot_labels, logits):

    num_non_zeros = tf.math.count_nonzero(onehot_labels, dtype=tf.float32)

    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(onehot_labels),
        logits=logits
    )

    assert softmax_cross_entropy.dtype == tf.float32

    softmax_cross_entropy = tf.math.reduce_sum(input_tensor=softmax_cross_entropy)
    softmax_cross_entropy = tf.math.divide_no_nan(
        softmax_cross_entropy,
        num_non_zeros,
        name="softmax_cross_entropy"
    )

    assert softmax_cross_entropy.dtype == tf.float32
    return softmax_cross_entropy


def _rpn_score_loss(score_outputs, score_targets, normalizer=1.0):
    """Computes score loss."""

    with tf.name_scope('rpn_score_loss'):

        # score_targets has three values:
        # * (1) score_targets[i]=1, the anchor is a positive sample.
        # * (2) score_targets[i]=0, negative.
        # * (3) score_targets[i]=-1, the anchor is don't care (ignore).

        mask = tf.math.greater_equal(score_targets, 0)
        mask = tf.cast(mask, dtype=tf.float32)

        score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
        score_targets = tf.cast(score_targets, dtype=tf.float32)

        assert score_outputs.dtype == tf.float32
        assert score_targets.dtype == tf.float32

        score_loss = _sigmoid_cross_entropy(
            multi_class_labels=score_targets,
            logits=score_outputs,
            weights=mask,
            sum_by_non_zeros_weights=False
        )

        assert score_loss.dtype == tf.float32

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            score_loss /= normalizer

        assert score_loss.dtype == tf.float32

    return score_loss


def _rpn_box_loss(box_outputs, box_targets, normalizer=1.0, delta=1. / 9):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

    with tf.name_scope('rpn_box_loss'):
        mask = tf.not_equal(box_targets, 0.0)
        mask = tf.cast(mask, tf.float32)

        assert mask.dtype == tf.float32

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)

        assert box_loss.dtype == tf.float32

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer

        assert box_loss.dtype == tf.float32

    return box_loss


def _fast_rcnn_class_loss(class_outputs, class_targets_one_hot, normalizer=1.0):
    """Computes classification loss."""

    with tf.name_scope('fast_rcnn_class_loss'):
        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.

        class_loss = _softmax_cross_entropy(onehot_labels=class_targets_one_hot, logits=class_outputs)

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            class_loss /= normalizer

    return class_loss


def _fast_rcnn_box_loss(box_outputs, box_targets, class_targets, normalizer=1.0, delta=1.):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

    with tf.name_scope('fast_rcnn_box_loss'):
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

        # The loss is normalized by the sum of non-zero weights before additional
        # normalizer provided by the function caller.
        box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)

        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer

    return box_loss
