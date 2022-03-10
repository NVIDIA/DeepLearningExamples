# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from . import logger as log
from . import utils
from .logger import TrainingMetrics, ValidationMetrics
from .models.common import EMA


class Executor:
    def __init__(
        self,
        model: nn.Module,
        loss: Optional[nn.Module],
        cuda: bool = True,
        memory_format: torch.memory_format = torch.contiguous_format,
        amp: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        divide_loss: int = 1,
        ts_script: bool = False,
    ):
        assert not (amp and scaler is None), "Gradient Scaler is needed for AMP"

        def xform(m: nn.Module) -> nn.Module:
            if cuda:
                m = m.cuda()
            m.to(memory_format=memory_format)
            return m

        self.model = xform(model)
        if ts_script:
            self.model = torch.jit.script(self.model)
        self.ts_script = ts_script
        self.loss = xform(loss) if loss is not None else None
        self.amp = amp
        self.scaler = scaler
        self.is_distributed = False
        self.divide_loss = divide_loss
        self._fwd_bwd = None
        self._forward = None

    def distributed(self, gpu_id):
        self.is_distributed = True
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)
        torch.cuda.current_stream().wait_stream(s)

    def _fwd_bwd_fn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        with autocast(enabled=self.amp):
            loss = self.loss(self.model(input), target)
            loss /= self.divide_loss

        self.scaler.scale(loss).backward()
        return loss

    def _forward_fn(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), autocast(enabled=self.amp):
            output = self.model(input)
            loss = None if self.loss is None else self.loss(output, target)

        return output if loss is None else loss, output

    def optimize(self, fn):
        return fn

    @property
    def forward_backward(self):
        if self._fwd_bwd is None:
            if self.loss is None:
                raise NotImplementedError(
                    "Loss must not be None for forward+backward step"
                )
            self._fwd_bwd = self.optimize(self._fwd_bwd_fn)
        return self._fwd_bwd

    @property
    def forward(self):
        if self._forward is None:
            self._forward = self.optimize(self._forward_fn)
        return self._forward

    def train(self):
        self.model.train()
        if self.loss is not None:
            self.loss.train()

    def eval(self):
        self.model.eval()
        if self.loss is not None:
            self.loss.eval()


class Trainer:
    def __init__(
        self,
        executor: Executor,
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int,
        ema: Optional[float] = None,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.use_ema = False
        if ema is not None:
            self.ema_executor = deepcopy(self.executor)
            self.ema = EMA(ema, self.ema_executor.model)
            self.use_ema = True

        self.optimizer.zero_grad(set_to_none=True)
        self.steps_since_update = 0

    def train(self):
        self.executor.train()
        if self.use_ema:
            self.ema_executor.train()

    def eval(self):
        self.executor.eval()
        if self.use_ema:
            self.ema_executor.eval()

    def train_step(self, input, target, step=None):
        loss = self.executor.forward_backward(input, target)

        self.steps_since_update += 1

        if self.steps_since_update == self.grad_acc_steps:
            if self.executor.scaler is not None:
                self.executor.scaler.step(self.optimizer)
                self.executor.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps_since_update = 0

        torch.cuda.synchronize()

        if self.use_ema:
            self.ema(self.executor.model, step=step)

        return loss

    def validation_steps(self) -> Dict[str, Callable]:
        vsd: Dict[str, Callable] = {"val": self.executor.forward}
        if self.use_ema:
            vsd["val_ema"] = self.ema_executor.forward
        return vsd

    def state_dict(self) -> dict:
        res = {
            "state_dict": self.executor.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_ema:
            res["state_dict_ema"] = self.ema_executor.model.state_dict()

        return res


def train(
    train_step,
    train_loader,
    lr_scheduler,
    log_fn,
    timeout_handler,
    prof=-1,
    step=0,
):
    interrupted = False

    end = time.time()

    data_iter = enumerate(train_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr = lr_scheduler(i)
        data_time = time.time() - end

        loss = train_step(input, target, step=step + i)
        it_time = time.time() - end

        with torch.no_grad():
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.detach())
            else:
                reduced_loss = loss.detach()

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            lr=lr,
            loss=reduced_loss.item(),
        )

        end = time.time()
        if prof > 0 and (i + 1 >= prof):
            time.sleep(5)
            break
        if ((i + 1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break

    return interrupted


def validate(infer_fn, val_loader, log_fn, prof=-1, with_loss=True):
    top1 = log.AverageMeter()
    # switch to evaluate mode

    end = time.time()

    data_iter = enumerate(val_loader)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        if with_loss:
            loss, output = infer_fn(input, target)
        else:
            output = infer_fn(input)

        with torch.no_grad():
            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

            if torch.distributed.is_initialized():
                if with_loss:
                    reduced_loss = utils.reduce_tensor(loss.detach())
                prec1 = utils.reduce_tensor(prec1)
                prec5 = utils.reduce_tensor(prec5)
            else:
                if with_loss:
                    reduced_loss = loss.detach()

        prec1 = prec1.item()
        prec5 = prec5.item()
        infer_result = {
            "top1": (prec1, bs),
            "top5": (prec5, bs),
        }

        if with_loss:
            infer_result["loss"] = (reduced_loss.item(), bs)

        torch.cuda.synchronize()

        it_time = time.time() - end

        top1.record(prec1, bs)

        log_fn(
            compute_ips=utils.calc_ips(bs, it_time - data_time),
            total_ips=utils.calc_ips(bs, it_time),
            data_time=data_time,
            compute_time=it_time - data_time,
            **infer_result,
        )

        end = time.time()
        if (prof > 0) and (i + 1 >= prof):
            time.sleep(5)
            break

    return top1.get_val()


# Train loop {{{
def train_loop(
    trainer: Trainer,
    lr_scheduler,
    train_loader,
    train_loader_len,
    val_loader,
    logger,
    should_backup_checkpoint,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
):
    train_metrics = TrainingMetrics(logger)
    val_metrics = {
        k: ValidationMetrics(logger, k) for k in trainer.validation_steps().keys()
    }
    training_step = trainer.train_step

    prec1 = -1

    if early_stopping_patience > 0:
        epochs_since_improvement = 0
    backup_prefix = (
        checkpoint_filename[: -len("checkpoint.pth.tar")]
        if checkpoint_filename.endswith("checkpoint.pth.tar")
        else ""
    )

    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    with utils.TimeoutHandler() as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if logger is not None:
                logger.start_epoch()
            if not skip_training:
                if logger is not None:
                    data_iter = logger.iteration_generator_wrapper(
                        train_loader, mode="train"
                    )
                else:
                    data_iter = train_loader

                trainer.train()
                interrupted = train(
                    training_step,
                    data_iter,
                    lambda i: lr_scheduler(trainer.optimizer, i, epoch),
                    train_metrics.log,
                    timeout_handler,
                    prof=prof,
                    step=epoch * train_loader_len,
                )

            if not skip_validation:
                trainer.eval()
                for k, infer_fn in trainer.validation_steps().items():
                    if logger is not None:
                        data_iter = logger.iteration_generator_wrapper(
                            val_loader, mode="val"
                        )
                    else:
                        data_iter = val_loader

                    step_prec1, _ = validate(
                        infer_fn,
                        data_iter,
                        val_metrics[k].log,
                        prof=prof,
                    )

                    if k == "val":
                        prec1 = step_prec1

                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = False
                best_prec1 = 0

            if logger is not None:
                logger.end_epoch()

            if save_checkpoints and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                if should_backup_checkpoint(epoch):
                    backup_filename = "{}checkpoint-{}.pth.tar".format(
                        backup_prefix, epoch + 1
                    )
                else:
                    backup_filename = None
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "best_prec1": best_prec1,
                    **trainer.state_dict(),
                }
                utils.save_checkpoint(
                    checkpoint_state,
                    is_best,
                    checkpoint_dir=checkpoint_dir,
                    backup_filename=backup_filename,
                    filename=checkpoint_filename,
                )

            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            if interrupted:
                break


# }}}
