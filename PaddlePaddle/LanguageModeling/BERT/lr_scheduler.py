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

import logging
import math
import paddle
from utils.utility import is_integer


class Poly:
    """
    Polynormial learning rate decay.
    lr = (learning_rate - end_lr) * (1 - min(step, decay_steps) / decay_steps) ^ power + end_lr
    If `power` is 1.0, it's also equivalent to linear learning rate decay.

    Args:
        learning_rate (float): The initial learning rate.
        num_steps(int): The total number of training steps.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0.
        power(float, optional): Power of polynomial. Default: 1.0.
        warmup(int|float, optional):
            If warmup is int, it indicates the number of warmup steps. Default: 0.
            If warmup is float, it indicates the proportion of warmup steps.
        warmup_start_lr(float, optional): Initial learning rate of warm up. Default: 0.0.
        last_step(int, optional): The step id of the last run. Can be set to resume training.
                                    Default: 0.
    """

    def __init__(self,
                 learning_rate,
                 num_steps,
                 end_lr=0.0,
                 power=1.0,
                 warmup=0,
                 warmup_start_lr=0.0,
                 last_step=0):
        super().__init__()
        self.end_lr = end_lr
        self.power = power
        self.learning_rate = learning_rate
        self.warmup_start_lr = warmup_start_lr
        self.last_step = last_step
        self.total_steps = num_steps
        self.warmup_steps = warmup if is_integer(warmup) else int(
            math.floor(warmup * self.total_steps))
        self.steps = self.total_steps - self.warmup_steps

        assert self.warmup_steps <= self.total_steps, "warmup steps can't be larger than total steps"

    def __call__(self):
        learning_rate = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.steps,
            end_lr=self.end_lr,
            power=self.power,
            last_epoch=self.
            last_step) if self.steps > 0 else self.learning_rate
        if self.warmup_steps > 0:
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_step)
        return learning_rate


def build_lr_scheduler(args):
    """
    Build a learning rate scheduler.

    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
    return:
        lr(paddle.optimizer.lr.LRScheduler): A learning rate scheduler.
    """

    lr = Poly(
        args.learning_rate,
        args.max_steps,
        warmup=args.warmup_proportion,
        last_step=args.last_step_of_checkpoint)
    if not isinstance(lr, paddle.optimizer.lr.LRScheduler):
        lr = lr()
    logging.info("build lr %s success..", lr)
    return lr
