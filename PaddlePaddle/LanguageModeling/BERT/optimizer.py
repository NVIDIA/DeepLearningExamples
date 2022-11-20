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
from paddle import optimizer as optim

_EXCLUDE_FROM_DECAY = ["b_0", "norm"]


class AdamW:
    """
    AdamW optimizer.
    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        learning_rate(float|LRScheduler, optional): The learning rate used to update parameters. Default: 0.001
            Can be a float value or a paddle.optimizer.lr.LRScheduler.
    """

    def __init__(self, args, learning_rate):
        self.learning_rate = learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.weight_decay = args.weight_decay
        self.multi_precision = args.amp

    def __call__(self):
        # not apply weight decay to all bias and layer_norm
        def apply_decay_func(name):
            return False if any(key in name
                                for key in _EXCLUDE_FROM_DECAY) else True

        # add grad clipping to prevent exploding gradients
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            apply_decay_param_fun=apply_decay_func,
            grad_clip=clip,
            multi_precision=self.multi_precision)
        return opt


class Lamb:
    """
    Lamb optimizer.
    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        learning_rate(float|LRScheduler, optional): The learning rate used to update parameters. Default: 0.001
            Can be a float value or a paddle.optimizer.lr.LRScheduler.
    """

    def __init__(self, args, learning_rate):
        self.learning_rate = learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.lamb_weight_decay = args.weight_decay
        self.multi_precision = args.amp

    def __call__(self):
        # not apply weight decay to all bias and layer_norm
        def exclude_from_decay_func(param):
            return True if any(key in param.name
                               for key in _EXCLUDE_FROM_DECAY) else False

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        opt = optim.Lamb(
            learning_rate=self.learning_rate,
            lamb_weight_decay=self.lamb_weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            exclude_from_weight_decay_fn=exclude_from_decay_func,
            grad_clip=clip)
        opt._multi_precision = True if self.multi_precision else False
        return opt


class DistributedFusedLamb:
    """
    DistributedFusedLamb optimizer.
    Args:
        args(Namespace): Arguments obtained from ArgumentParser.
        learning_rate(float|LRScheduler, optional): The learning rate used to update parameters. Default: 0.001
            Can be a float value or a paddle.optimizer.lr.LRScheduler.
    """

    def __init__(self, args, learning_rate):
        self.learning_rate = learning_rate
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.epsilon = args.epsilon
        self.lamb_weight_decay = args.weight_decay
        self.gradient_merge_steps = args.gradient_merge_steps

    def __call__(self):
        # not apply weight decay to all bias and layer_norm
        def exclude_from_decay_func(param):
            return True if any(key in param.name
                               for key in _EXCLUDE_FROM_DECAY) else False

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        opt = paddle.incubate.DistributedFusedLamb(
            learning_rate=self.learning_rate,
            lamb_weight_decay=self.lamb_weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            exclude_from_weight_decay_fn=exclude_from_decay_func,
            grad_clip=clip,
            clip_after_allreduce=True,
            is_grad_scaled_by_nranks=False,
            use_master_param_norm=True,
            gradient_accumulation_steps=self.gradient_merge_steps,
            use_master_acc_grad=True)
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
