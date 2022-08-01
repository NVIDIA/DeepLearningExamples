# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

"""
Train a network across multiple GPUs.
"""

import math
from collections import defaultdict
from itertools import chain

import torch
import torch.nn.functional as F
from torch.cuda import amp
from apex.parallel import DistributedDataParallel as DDP

from fairseq import distributed_utils, optim, utils
from fairseq.optim import lr_scheduler
from fairseq.meters import TimeMeter, AverageMeter
from fairseq.criterions import CRITERION_REGISTRY

import dllogger as DLLogger


class DDPTrainer():
    """Main class for data parallel training.

    This class supports data parallel training, where multiple workers each
    have a full model replica and gradients are accumulated synchronously via
    torch.distributed.all_reduce.
    """

    def __init__(self, args, model):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args

        self.model = model.cuda()
        self.criterion = CRITERION_REGISTRY[args.criterion](args).cuda()
        self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self.scaler = amp.GradScaler(enabled=self.args.amp, init_scale=2**15)

        if self.args.distributed_world_size > 1:
            self.model = DDP(model)

        self._buffered_stats = defaultdict(lambda: [])
        self._num_updates = 0
        self._optim_history = None
        self.throughput_meter = TimeMeter()
        self.avg_loss_meter = AverageMeter()

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            utils.save_state(
                filename, self.args, self.get_model(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename, load_optim=True):
        """Load all training state from a checkpoint file."""
        extra_state, optim_history, last_optim_state = \
            utils.load_model_state(filename, self.get_model())

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            #self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
            self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

            if load_optim:
                self._optim_history = optim_history
                # only reload optimizer and lr_scheduler if they match
                last_optim = self._optim_history[-1]
                if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                    self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                    if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                        self.optimizer.load_state_dict(last_optim_state)

                self._num_updates = last_optim['num_updates']

        return extra_state

    def train_step(self, sample, update_params=True, last_step=False):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model.train()
        if isinstance(self.model, DDP):
            if last_step:
                self.model.disable_allreduce()
            else:
                self.model.enable_allreduce()

        # forward and backward pass
        sample = self._prepare_sample(sample)
        loss, oom_fwd = self._forward(sample)

        # If this is a last batch forward pass is skipped on some workers
        # Batch with sample_size 0 is not accounted for in weighted loss
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
            'loss': utils.item(loss.data) if loss is not None else 0,
        }
        sample_size = sample['ntokens'] if sample is not None else 0
        oom_bwd = self._backward(loss)

        # buffer stats and logging outputs
        self._buffered_stats['sample_sizes'].append(sample_size)
        self._buffered_stats['logging_outputs'].append(logging_output)
        self._buffered_stats['ooms_fwd'].append(oom_fwd)
        self._buffered_stats['ooms_bwd'].append(oom_bwd)

        # update parameters
        if update_params and not last_step:
            # gather logging outputs from all replicas
            sample_sizes = self._buffered_stats['sample_sizes']
            logging_outputs = self._buffered_stats['logging_outputs']
            ooms_fwd = self._buffered_stats['ooms_fwd']
            ooms_bwd = self._buffered_stats['ooms_bwd']
            if self.args.distributed_world_size > 1:
                sample_sizes, logging_outputs, ooms_fwd, ooms_bwd = map(
                    lambda l: list(chain.from_iterable(l)),
                    zip(*distributed_utils.all_gather_list(
                        (sample_sizes, logging_outputs, ooms_fwd, ooms_bwd)
                    ))
                )
            ooms_fwd = sum(ooms_fwd)
            ooms_bwd = sum(ooms_bwd)
            ooms = ooms_fwd + ooms_bwd  # this is always <= distributed_world_size

            if ooms == self.args.distributed_world_size:
                print('| WARNING: OOM in all workers, skipping batch')
                self.zero_grad()
                return

            # aggregate stats and logging outputs
            grad_denom = sum(sample_sizes)
            for p in self.model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad /= grad_denom

            self._opt()

            # Handle logging
            ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
            self.throughput_meter.update(ntokens)
            info_log_data = {
                'tokens/s': self.throughput_meter.avg,
                'tokens': ntokens,
                'loss': sum(log.get('loss', 0) for log in logging_outputs) / ntokens / math.log(2)
            }
            self.avg_loss_meter.update(info_log_data['loss'])
            debug_log_data = {
                'batch_size': sum(log.get('nsentences', 0) for log in logging_outputs),
                'lr': self.get_lr(),
                'grad_denom': grad_denom,
                'updates': 1
            }

            DLLogger.log(step=self._num_updates, data=info_log_data, verbosity=0)
            DLLogger.log(step=self._num_updates, data=debug_log_data, verbosity=1)

            self.clear_buffered_stats()

    def _forward(self, sample):
        loss = None
        oom = 0
        try:
            if sample is not None:
                with amp.autocast(enabled=self.args.amp):
                    # calculate loss and sample size
                    logits, _ = self.model(**sample['net_input'])
                    target = sample['target']
                    probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                    loss = self.criterion(probs, target)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory in worker {}, skipping batch'.format(
                    self.args.distributed_rank), force=True)
                oom = 1
                loss = None
            else:
                raise e
        return loss, oom

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                self.scaler.scale(loss).backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory in worker {}, skipping batch'.format(
                        self.args.distributed_rank), force=True)
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def _opt(self):
        # take an optimization step
        self.scaler.step(self.optimizer.optimizer)
        self.scaler.update()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        self.lr_scheduler.step_update(self._num_updates)

    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        self.model.eval()
        # forward pass
        sample = self._prepare_sample(sample)
        with torch.no_grad():
            loss, oom_fwd = self._forward(sample)
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
        }
        loss = loss.item() if loss is not None else 0
        assert not oom_fwd, 'Ran out of memory during validation'

        # gather logging outputs from all GPUs
        if self.args.distributed_world_size > 1:
            losses, logging_outputs = zip(*distributed_utils.all_gather_list(
                (loss, logging_output)
            ))
        else:
            losses = [loss]
            logging_outputs = [logging_output]

        weight = sum(log.get('ntokens', 0) for log in logging_outputs)
        scaled_loss = sum(losses) / weight / math.log(2)

        return scaled_loss

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_throughput_meter(self):
        """Get the throughput meter"""
        return self.throughput_meter

    def get_model(self):
        """Get the model replica."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if not sample:
            return None
        return utils.move_to_cuda(sample)
