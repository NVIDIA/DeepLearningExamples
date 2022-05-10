# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pickle
import torch
from triton.deployment_toolkit.core import BaseMetricsCalculator
from timm.utils import accuracy, AverageMeter


class MetricsCalculator(BaseMetricsCalculator):
    def __init__(self):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    @property
    def metrics(self):
        return {'top1': self.top1.avg, 'top5': self.top5.avg}

    def update(
        self,
        ids,
        y_pred,
        x,
        y_real,
    ):
        output = torch.from_numpy(y_pred["OUTPUT__0"]).float()
        label = torch.from_numpy(y_real['OUTPUT__0'][:,0]).long()
        acc1, acc5 = accuracy(output.detach(), label, topk=(1, 5))
        self.top1.update(acc1.item(), output.shape[0])
        self.top5.update(acc5.item(), output.shape[0])
