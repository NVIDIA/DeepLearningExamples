# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import sys
import logging
from paddle import optimizer as optim


class Momentum:
    """
    Simple Momentum optimizer with velocity state.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        learning_rate(float|LRScheduler): The learning rate used to update parameters.
            Can be a float value or a paddle.optimizer.lr.LRScheduler.
    """

    def __init__(self, args, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.grad_clip = None
        self.multi_precision = args.amp

    def __call__(self):
        # model_list is None in static graph
        parameters = None
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            multi_precision=self.multi_precision,
            parameters=parameters)
        return opt


def build_optimizer(args, lr):
    """
    Build a raw optimizer with learning rate scheduler.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        lr(paddle.optimizer.lr.LRScheduler): A LRScheduler used for training.
    return:
        optim(paddle.optimizer): A normal optmizer.
    """
    optimizer_mod = sys.modules[__name__]
    opt = getattr(optimizer_mod, args.optimizer)(args, learning_rate=lr)()
    logging.info("build optimizer %s success..", opt)
    return opt
