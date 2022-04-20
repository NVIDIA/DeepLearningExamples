# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss


class Loss(nn.Module):
    def __init__(self, focal):
        super(Loss, self).__init__()
        if focal:
            self.loss_fn = DiceFocalLoss(
                include_background=False, softmax=True, to_onehot_y=True, batch=True, gamma=2.0
            )
        else:
            self.loss_fn = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, batch=True)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


class LossBraTS(nn.Module):
    def __init__(self, focal):
        super(LossBraTS, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False) if focal else nn.BCEWithLogitsLoss()

    def _loss(self, p, y):
        return self.dice(p, y) + self.ce(p, y.float())

    def forward(self, p, y):
        y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
        p_wt, p_tc, p_et = p[:, 0].unsqueeze(1), p[:, 1].unsqueeze(1), p[:, 2].unsqueeze(1)
        l_wt, l_tc, l_et = self._loss(p_wt, y_wt), self._loss(p_tc, y_tc), self._loss(p_et, y_et)
        return l_wt + l_tc + l_et
