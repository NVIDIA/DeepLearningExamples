# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
#from fused_adam_local import FusedAdam
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
multi_tensor_l2norm = amp_C.multi_tensor_l2norm
lamb_compute_update = amp_C.multi_tensor_lamb_stage1_cuda
lamb_apply_update = amp_C.multi_tensor_lamb_stage2_cuda
scale = amp_C.multi_tensor_scale


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)
    
def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}


class BertLAMB(Optimizer):
    """Implements BERT version of LAMB algorithm.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: LAMBs b1. Default: 0.9
        b2: LAMBs b2. Default: 0.999
        e: LAMBs epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum global norm for the gradients. Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_poly',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertLAMB, self).__init__(params, defaults)
        self.step_count = 0
        self.b1 = b1
        self.b2 = b2
        self.epsilon = e
        self.max_global_grad_norm = max_grad_norm
        self.learning_rate = lr
        self.schedule = schedule
        self.warmup = warmup
        self.max_steps = t_total
        self.updates_created=False

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def apply_gradients(self, dummy_overflow_buf, lr_scheduled, per_param_decay, grad_list, param_list, momentum, velocity, update):
        # Compute global gradient norm
        global_grad_norm = multi_tensor_applier(
                        multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grad_list],
                        False)[0].item()

        # Compute per parameter norm
        param_norms = multi_tensor_applier(
                        multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [param_list],
                        True)[1]

        # Compute LAMB update
        multi_tensor_applier(
                        lamb_compute_update,
                        dummy_overflow_buf,
                        [grad_list, param_list, momentum, velocity, update],
                        torch.cuda.FloatTensor(per_param_decay),
                        self.step_count,
                        self.b1,
                        self.b2,
                        self.epsilon,
                        global_grad_norm,
                        self.max_global_grad_norm,
                        )

        # Computer per parameter update norm
        update_norms = multi_tensor_applier(
                        multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [update],
                        True)[1]

        # Apply LAMB update on parameters
        multi_tensor_applier(
                        lamb_apply_update,
                        dummy_overflow_buf,
                        [param_list, update],
                        param_norms,
                        update_norms,
                        lr_scheduled,
                        )

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        check = 1#torch.norm(all_grads, 2)

        grad_list = []
        param_list = []
        per_param_decay = []
        momentum = []
        velocity = []

        fp16_grad_list = []
        fp16_from_fp32_param_list = []
        fp32_param_list = []
        fp16_per_param_decay = []
        fp16_momentum = []
        fp16_velocity = []
        
        if not self.updates_created:
            self.update = []
            self.fp16_update = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Keep step here for compatibility with earlier resume from checkpoint
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['momentum'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # Exponential moving average of squared gradient values
                    state['velocity'] = torch.zeros_like(p.data, dtype=torch.float32)
                    # fp32 master weights
                if 'master_param' not in state.keys() and p.type() == 'torch.cuda.HalfTensor':
                    state['master_param'] = p.detach().clone().float()

                # ensure these 3 are float tensors
                if state['momentum'].type() != 'torch.cuda.FloatTensor':
                    state['momentum'] = state['momentum'].float()
                if state['velocity'].type() != 'torch.cuda.FloatTensor':
                    state['velocity'] = state['velocity'].float()
                if 'master_param' in state.keys() and state['master_param'].type() != 'torch.cuda.FloatTensor':
                    state['master_param'] = state['master_param'].float()

                # Append all params, gradients, decays, velocity, momentum and updates to a list
                if p.type() == 'torch.cuda.HalfTensor':
                    fp16_grad_list.append(grad)
                    fp32_param_list.append(state['master_param'])
                    fp16_from_fp32_param_list.append(p.data)
                    fp16_per_param_decay.append(group['weight_decay'])
                    fp16_momentum.append(state["momentum"])
                    fp16_velocity.append(state["velocity"])
                    if not self.updates_created:
                        #self.fp16_update.append(torch.empty_like(p.data, dtype=torch.float32))
                        # Use fp16 weights as temporary buffer for update term.
                        # This is safe because fp16 weights are overwritten after apply_gradients
                        self.fp16_update.append(p.data)
                else:
                    grad_list.append(grad)
                    param_list.append(p.data)
                    per_param_decay.append(group['weight_decay'])
                    momentum.append(state["momentum"])
                    velocity.append(state["velocity"])
                    if not self.updates_created:
                        self.update.append(torch.empty_like(p.data))
                state['step'] += 1
        self.updates_created=True
        update = self.update
        fp16_update = self.fp16_update

        self.step_count = state['step']
        # Calculate learning rate from input schedule
        # if self.max_steps != -1:
        schedule_fct = SCHEDULES[self.schedule]
        lr_scheduled = self.learning_rate * schedule_fct(self.step_count / self.max_steps, self.warmup)
        if torch.distributed.get_rank() == 0:
            print("Step {} LR {}".format(self.step_count, lr_scheduled))
        # else:
        #     lr_scheduled = self.learning_rate

        overflow_buf = torch.cuda.IntTensor([0])

        if len(grad_list) > 0:
            self.apply_gradients(overflow_buf, lr_scheduled, per_param_decay, grad_list, param_list, momentum, velocity, update)
        if len(fp16_grad_list) > 0:
            self.apply_gradients(overflow_buf, lr_scheduled, fp16_per_param_decay, fp16_grad_list, fp32_param_list, fp16_momentum, fp16_velocity, fp16_update)
            multi_tensor_applier(
                    scale,
                    overflow_buf,
                    [fp32_param_list, fp16_from_fp32_param_list],
                    1.)

        return loss

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

