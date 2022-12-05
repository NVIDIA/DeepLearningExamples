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

import torch
from torchmetrics import Metric


class Dice(Metric):
    full_state_update = False

    def __init__(self, n_class, brats):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.brats = brats
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((n_class,)), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, p, y, l):
        self.steps += 1
        self.dice += self.compute_stats_brats(p, y) if self.brats else self.compute_stats(p, y)
        self.loss += l

    def compute(self):
        return 100 * self.dice / self.steps, self.loss / self.steps

    def compute_stats_brats(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()
        y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
        y = torch.stack([y_wt, y_tc, y_et], dim=1)

        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                # no foreground class
                scores[i - 1] += 1 if (p_i != 1).all() else 0
                continue
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (2 * tp + fp + fn).to(torch.float)
            score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores

    def compute_stats(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = torch.argmax(p, dim=1)
        for i in range(1, self.n_class + 1):
            if (y != i).all():
                # no foreground class
                scores[i - 1] += 1 if (p != i).all() else 0
                continue
            tp, fn, fp = self.get_stats(p, y, i)
            denom = (2 * tp + fp + fn).to(torch.float)
            score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores

    @staticmethod
    def get_stats(p, y, c):
        tp = torch.logical_and(p == c, y == c).sum()
        fn = torch.logical_and(p != c, y == c).sum()
        fp = torch.logical_and(p == c, y != c).sum()
        return tp, fn, fp
