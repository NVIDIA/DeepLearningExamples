# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import math

from common.fairseq.optim.adam import FairseqAdam
from common.fairseq.optim.fp16_optimizer import FP16Optimizer
from common.fairseq.optim.fused_adam import get_fused_adam_class
from common.utils import print_once


def lr_poly_policy(step, optimizer, lr, initial_lr_scale=0.0,
                   final_lr_scale=0.0, warmup_steps=1000, hold_steps=0,
                   num_steps=None, power=1.0):
    """Polynomial decay LR policy with an optional hold period."""
    assert step >= 1
    assert num_steps is not None
    assert power is not None

    start_lr = initial_lr_scale * lr
    end_lr = final_lr_scale * lr

    if step <= warmup_steps:
        new_lr = start_lr + (step) / warmup_steps * (lr - start_lr)
    elif step <= warmup_steps + hold_steps:
        new_lr = lr
    elif warmup_steps + hold_steps < step <= num_steps:
        remain = 1 - (step - warmup_steps) / (num_steps - warmup_steps)
        new_lr = (lr - end_lr) * remain ** power + end_lr
    else:
        new_lr = end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def lr_exp_policy(step, optimizer, initial_lr_scale, lr, final_lr_scale=0.0,
                  warmup_steps=1000, hold_steps=0, num_steps=float('inf'),
                  decay=None):
    """Exponential LR policy with an optional hold period.

    If `decay` factor is not supplied, it is calculated to reach `end_lr`
    on `num_steps` steps.

    Args:
        num_steps (int): Limits the number of decay steps.
        end_lr (float): The lowest possible LR.
        decay (float or None): Decay factor; if None, the it will be derived
            from `num_steps` and `end_lr`.
    """
    assert step >= 1

    start_lr = initial_lr_scale * lr
    end_lr = final_lr_scale * lr

    if decay is None:
        assert not math.isinf(num_steps) and end_lr > 0.0
        decay_steps = num_steps - warmup_steps - hold_steps
        decay = math.log(end_lr / lr) / decay_steps
    else:
        decay = math.log(decay)

    if step <= warmup_steps:
        new_lr = start_lr + (step) / warmup_steps * (lr - start_lr)
    elif step <= warmup_steps + hold_steps:
        new_lr = lr
    else:
        a = math.exp(decay * (min(step, num_steps) - warmup_steps - hold_steps))
        new_lr = max(a * lr, end_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_optimizer(model, args):

    kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.optimizer == 'adam' and (args.fp16 or args.bf16):

        print_once('WARNING: Using Fairseq FP16Optimizer')

        # based on fairseq.optim.FP16Optimizer.build_optimizer
        flatten = True  # not args.fp16_no_flatten_grads
        args.betas = args.adam_betas
        args.eps = args.adam_eps

        params = list(filter(lambda p: p.requires_grad, model.parameters()))

        fp32_params = FP16Optimizer.build_fp32_params(args, params,
                                                      flatten=flatten)

        # based on fairseq.optim.build_optimizer
        def build_optimizer(cfg, params, *extra_args, **extra_kwargs):
            if all(isinstance(p, dict) for p in params):
                params = [t for p in params for t in p.values()]
            params = list(filter(lambda p: p.requires_grad, params))
            return FairseqAdam(cfg, params, *extra_args, **extra_kwargs)

        if flatten:
            fp32_optimizer = build_optimizer(args, [fp32_params])
        else:
            fp32_optimizer = build_optimizer(args, fp32_params)

        if flatten and not fp32_optimizer.supports_flat_params:
            raise RuntimeError(
                f"chosen optimizer {fp32_optimizer.__class__.__name__} does "
                "not support flat params, please set --fp16-no-flatten-grads"
            )
        kwargs = {}
        optimizer = FP16Optimizer(args, params, fp32_optimizer, fp32_params,
                                  **kwargs)

    elif args.optimizer == 'adam' and not (args.fp16 or args.bf16):
        print_once('WARNING: Using FusedAdam instead of Adam')
        kw.update({'betas': args.adam_betas, 'eps': args.adam_eps})
        fused_adam_cls = get_fused_adam_class()
        optimizer = fused_adam_cls(model.parameters(), **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    return optimizer
