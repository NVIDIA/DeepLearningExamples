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
from monai.losses import FocalLoss


class DiceLoss(nn.Module):
    def __init__(self, include_background=False, smooth=1e-5, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.dims = (0, 2)
        self.eps = eps

    def forward(self, y_pred, y_true):
        num_classes, batch_size = y_pred.size(1), y_true.size(0)
        y_pred = y_pred.log_softmax(dim=1).exp()
        y_true, y_pred = y_true.view(batch_size, -1), y_pred.view(batch_size, num_classes, -1)
        y_true = F.one_hot(y_true.to(torch.int64), num_classes).permute(0, 2, 1)
        if not self.include_background:
            y_true, y_pred = y_true[:, 1:], y_pred[:, 1:]
        intersection = torch.sum(y_true * y_pred, dim=self.dims)
        cardinality = torch.sum(y_true + y_pred, dim=self.dims)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        mask = (y_true.sum(self.dims) > 0).to(dice_loss.dtype)
        dice_loss *= mask.to(dice_loss.dtype)
        dice_loss = dice_loss.sum() / mask.sum()
        return dice_loss


class Loss(nn.Module):
    def __init__(self, focal):
        super(Loss, self).__init__()
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal = FocalLoss(gamma=2.0)
        self.use_focal = focal

    def forward(self, y_pred, y_true):
        loss = self.dice(y_pred, y_true)
        if self.use_focal:
            loss += self.focal(y_pred, y_true)
        else:
            loss += self.cross_entropy(y_pred, y_true[:, 0].long())
        return loss
