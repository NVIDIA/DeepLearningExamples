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
import paddle


class Cosine:
    """
    Cosine learning rate decay.
    lr = eta_min + 0.5 * (learning_rate - eta_min) * (cos(epoch * (PI / epochs)) + 1)
    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        step_each_epoch(int): The number of steps in each epoch.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training.
            Default: -1, meaning initial learning rate.
    """

    def __init__(self, args, step_each_epoch, last_epoch=-1):
        super().__init__()
        if args.warmup_epochs >= args.epochs:
            args.warmup_epochs = args.epochs
        self.learning_rate = args.lr
        self.T_max = (args.epochs - args.warmup_epochs) * step_each_epoch
        self.eta_min = 0.0
        self.last_epoch = last_epoch
        self.warmup_steps = round(args.warmup_epochs * step_each_epoch)
        self.warmup_start_lr = args.warmup_start_lr

    def __call__(self):
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.
            last_epoch) if self.T_max > 0 else self.learning_rate
        if self.warmup_steps > 0:
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


def build_lr_scheduler(args, step_each_epoch):
    """
    Build a learning rate scheduler.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        step_each_epoch(int): The number of steps in each epoch.
    return:
        lr(paddle.optimizer.lr.LRScheduler): A learning rate scheduler.
    """
    # Turn last_epoch to last_step, since we update lr each step instead of each epoch.
    last_step = args.start_epoch * step_each_epoch - 1
    learning_rate_mod = sys.modules[__name__]
    lr = getattr(learning_rate_mod, args.lr_scheduler)(args, step_each_epoch,
                                                       last_step)
    if not isinstance(lr, paddle.optimizer.lr.LRScheduler):
        lr = lr()
    logging.info("build lr %s success..", lr)
    return lr
