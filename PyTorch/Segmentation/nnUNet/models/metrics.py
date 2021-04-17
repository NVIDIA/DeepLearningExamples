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
from pytorch_lightning.metrics.functional import stat_scores
from pytorch_lightning.metrics.metric import Metric


class Dice(Metric):
    def __init__(self, nclass):
        super().__init__(dist_sync_on_step=True)
        self.add_state("n_updates", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((nclass,)), dist_reduce_fx="sum")

    def update(self, pred, target):
        self.n_updates += 1
        self.dice += self.compute_stats(pred, target)

    def compute(self):
        return 100 * self.dice / self.n_updates

    @staticmethod
    def compute_stats(pred, target):
        num_classes = pred.shape[1]
        scores = torch.zeros(num_classes - 1, device=pred.device, dtype=torch.float32)
        for i in range(1, num_classes):
            if (target != i).all():
                # no foreground class
                _, _pred = torch.max(pred, 1)
                scores[i - 1] += 1 if (_pred != i).all() else 0
                continue
            _tp, _fp, _tn, _fn, _ = stat_scores(pred=pred, target=target, class_index=i)
            denom = (2 * _tp + _fp + _fn).to(torch.float)
            score_cls = (2 * _tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0.0
            scores[i - 1] += score_cls
        return scores
