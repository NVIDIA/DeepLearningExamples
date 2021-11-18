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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Dict

from .focal_loss import FocalLoss
opt_focal_loss = FocalLoss.apply

def focal_loss(logits, targets, alpha: float, gamma: float, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """

    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


def new_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float, normalizer, label_smoothing: float = 0.1):
    """Compute the focal loss between `logits` and the golden `target` values.
    'New' is not the best descriptor, but this focal loss impl matches recent versions of
    the official Tensorflow impl of EfficientDet. It has support for label smoothing, however
    it is a bit slower, doesn't jit optimize well, and uses more memory.
    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: Divide loss by this value.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    # compute focal loss multipliers before label smoothing, such that it will not blow up the loss.
    pred_prob = logits.sigmoid()
    targets = targets.to(logits.dtype)
    onem_targets = 1. - targets
    p_t = (targets * pred_prob) + (onem_targets * (1. - pred_prob))
    alpha_factor = targets * alpha + onem_targets * (1. - alpha)
    modulating_factor = (1. - p_t) ** gamma

    # apply label smoothing for cross_entropy for each entry.
    if label_smoothing > 0.:
        targets = targets * (1. - label_smoothing) + .5 * label_smoothing
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # compute the final loss and return
    return (1 / normalizer) * alpha_factor * modulating_factor * ce


def huber_loss(
        input: torch.Tensor, target: torch.Tensor, delta: float = 1., weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def smooth_l1_loss(
        input: torch.Tensor, target: torch.Tensor, beta: float = 1. / 9, weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        err = torch.abs(input - target)
        loss = torch.where(err < beta, 0.5 * err.pow(2) / beta, err - 0.5 * beta)
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def _classification_loss(cls_outputs: torch.Tensor, cls_targets: torch.Tensor, num_positives: float, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.1):
    """Computes classification loss."""
    normalizer = num_positives
    classification_loss = new_focal_loss(cls_outputs, cls_targets, alpha, gamma, normalizer, label_smoothing)
    return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta: float = 0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_outputs, box_targets, weights=mask, delta=delta, size_average=False)
    box_loss /= normalizer
    return box_loss


def one_hot(x, num_classes: int):
    # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
    x_non_neg = (x >= 0).to(x.dtype)
    onehot = torch.zeros(x.shape + (num_classes,), device=x.device, dtype=x.dtype)
    onehot.scatter_(-1, (x * x_non_neg).unsqueeze(-1), 1)
    return onehot * x_non_neg.unsqueeze(-1)


class DetectionLoss(nn.Module):
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.label_smoothing = config.label_smoothing
        self.box_loss_weight = config.box_loss_weight
        self.fused_focal_loss = config.fused_focal_loss

    def forward(
            self, cls_outputs: List[torch.Tensor], box_outputs: List[torch.Tensor],
            targets: Dict[str, torch.Tensor], num_positives: torch.Tensor):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
            cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
                at each feature level (index)

            box_outputs: a List with values representing box regression targets in
                [batch_size, height, width, num_anchors * 4] at each feature level (index)

            cls_targets: groundtruth class targets.

            box_targets: groundtrusth box targets.

            num_positives: num positive grountruth anchors

        Returns:
            total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

            cls_loss: an integer tensor representing total class loss.

            box_loss: an integer tensor representing total box regression loss.
        """
        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = num_positives.sum() + 1.0
        levels = len(cls_outputs)

        cls_losses = []
        box_losses = []
        for l in range(levels):
            cls_key = 'cls_targets_%d' % (l+1)
            bbox_key = 'box_targets_%d' % (l+1)
            cls_targets_at_level = targets[cls_key].cuda()
            box_targets_at_level = targets[bbox_key].cuda()


            if self.fused_focal_loss:
                cls_output = cls_outputs[l].permute(0, 2, 3, 1).contiguous()
                bs, height, width, anchor = cls_targets_at_level.shape
                cls_output = cls_output.view(bs, height, width, anchor, -1)
                cls_loss = opt_focal_loss(cls_output,
                    cls_targets_at_level,
                    num_positives_sum,
                    self.num_classes, self.alpha, self.gamma, self.label_smoothing)
                cls_losses.append(cls_loss)
            else:
                # Onehot encoding for classification labels.
                cls_targets_at_level_oh = one_hot(cls_targets_at_level, self.num_classes)

                bs, height, width, _, _ = cls_targets_at_level_oh.shape
                cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
                cls_loss = _classification_loss(
                    cls_outputs[l].permute(0, 2, 3, 1),
                    cls_targets_at_level_oh,
                    num_positives_sum,
                    alpha=self.alpha, gamma=self.gamma, label_smoothing=self.label_smoothing)
                cls_loss = cls_loss.view(bs, height, width, -1, self.num_classes)
                cls_loss *= (cls_targets_at_level != -2).unsqueeze(-1).float()
                cls_losses.append(cls_loss.sum())

            box_losses.append(_box_loss(
                box_outputs[l].permute(0, 2, 3, 1),
                box_targets_at_level,
                num_positives_sum,
                delta=self.delta))

        # Sum per level losses to total loss.
        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss

