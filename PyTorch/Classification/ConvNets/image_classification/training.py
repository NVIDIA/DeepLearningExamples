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
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import models
from . import utils
import dllogger

from .optimizers import get_sgd_optimizer, get_rmsprop_optimizer
from .models.common import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast

ACC_METADATA = {"unit": "%", "format": ":.2f"}
IPS_METADATA = {"unit": "img/s", "format": ":.2f"}
TIME_METADATA = {"unit": "s", "format": ":.5f"}
LOSS_METADATA = {"format": ":.5f"}


class ModelAndLoss(nn.Module):
    def __init__(
        self,
        model,
        loss,
        cuda=True,
        memory_format=torch.contiguous_format,
    ):
        super(ModelAndLoss, self).__init__()

        if cuda:
            model = model.cuda().to(memory_format=memory_format)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self, gpu_id):
        self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(parameters, lr, args, state=None):
    if args.optimizer == 'sgd':
        optimizer = get_sgd_optimizer(parameters, lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay, nesterov=args.nesterov,
                                      bn_weight_decay=args.bn_weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = get_rmsprop_optimizer(parameters, lr, alpha=args.rmsprop_alpha, momentum=args.momentum,
                                          weight_decay=args.weight_decay,
                                          eps=args.rmsprop_eps,
                                          bn_weight_decay=args.bn_weight_decay)
    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric(
            "lr", log.LR_METER(), verbosity=dllogger.Verbosity.VERBOSE
        )

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric("lr", lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, end_lr = 0, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - end_lr))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(
    base_lr, warmup_length, epochs, final_multiplier=0.001, decay_factor=None, decay_step=1, logger=None
):
    """Exponential lr policy. Setting decay factor parameter overrides final_multiplier"""
    es = epochs - warmup_length

    if decay_factor is not None:
        epoch_decay = decay_factor
    else:
        epoch_decay = np.power(2, np.log2(final_multiplier) / math.floor(es/decay_step))

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** math.floor(e/decay_step))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def get_train_step(
    model_and_loss, optimizer, scaler, use_amp=False, batch_size_multiplier=1
):
    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)

        with autocast(enabled=use_amp):
            loss, output = model_and_loss(input_var, target_var)
            loss /= batch_size_multiplier
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

        scaler.scale(loss).backward()

        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss

    return _step


def train(
    train_loader,
    model_and_loss,
    optimizer,
    scaler,
    lr_scheduler,
    logger,
    epoch,
    steps_per_epoch,
    timeout_handler,
    ema=None,
    use_amp=False,
    prof=-1,
    batch_size_multiplier=1,
    register_metrics=True,
):
    interrupted = False
    if register_metrics and logger is not None:
        logger.register_metric(
            "train.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            "train.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            "train.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            "train.compute_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_train_step(
        model_and_loss,
        optimizer,
        scaler=scaler,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
    )

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter, mode='train')

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss = step(input, target, optimizer_step=optimizer_step)
        if ema is not None:
            ema(model_and_loss, epoch*steps_per_epoch+i)

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric("train.loss", loss.item(), bs)
            logger.log_metric("train.compute_ips", utils.calc_ips(bs, it_time - data_time))
            logger.log_metric("train.total_ips", utils.calc_ips(bs, it_time))
            logger.log_metric("train.data_time", data_time)
            logger.log_metric("train.compute_time", it_time - data_time)

        end = time.time()
        if prof > 0 and (i + 1 >= prof):
            time.sleep(5)
            break
        if ((i+1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break

    return interrupted


def get_val_step(model_and_loss, use_amp=False):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad(), autocast(enabled=use_amp):
            loss, output = model_and_loss(input_var, target_var)

            prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.data)
                prec1 = utils.reduce_tensor(prec1)
                prec5 = utils.reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(
    val_loader,
    model_and_loss,
    logger,
    epoch,
    use_amp=False,
    prof=-1,
    register_metrics=True,
    prefix="val",
):
    if register_metrics and logger is not None:
        logger.register_metric(
            f"{prefix}.top1",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            f"{prefix}.top5",
            log.ACC_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=ACC_METADATA,
        )
        logger.register_metric(
            f"{prefix}.loss",
            log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=LOSS_METADATA,
        )
        logger.register_metric(
            f"{prefix}.compute_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            f"{prefix}.total_ips",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT,
            metadata=IPS_METADATA,
        )
        logger.register_metric(
            f"{prefix}.data_time",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            f"{prefix}.compute_latency",
            log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            f"{prefix}.compute_latency_at100",
            log.LAT_100(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            f"{prefix}.compute_latency_at99",
            log.LAT_99(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )
        logger.register_metric(
            f"{prefix}.compute_latency_at95",
            log.LAT_95(),
            verbosity=dllogger.Verbosity.VERBOSE,
            metadata=TIME_METADATA,
        )

    step = get_val_step(model_and_loss, use_amp=use_amp)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, mode='val')

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(prec1.item(), bs)
        if logger is not None:
            logger.log_metric(f"{prefix}.top1", prec1.item(), bs)
            logger.log_metric(f"{prefix}.top5", prec5.item(), bs)
            logger.log_metric(f"{prefix}.loss", loss.item(), bs)
            logger.log_metric(f"{prefix}.compute_ips", utils.calc_ips(bs, it_time - data_time))
            logger.log_metric(f"{prefix}.total_ips", utils.calc_ips(bs, it_time))
            logger.log_metric(f"{prefix}.data_time", data_time)
            logger.log_metric(f"{prefix}.compute_latency", it_time - data_time)
            logger.log_metric(f"{prefix}.compute_latency_at95", it_time - data_time)
            logger.log_metric(f"{prefix}.compute_latency_at99", it_time - data_time)
            logger.log_metric(f"{prefix}.compute_latency_at100", it_time - data_time)

        end = time.time()
        if (prof > 0) and (i + 1 >= prof):
            time.sleep(5)
            break

    return top1.get_val()


# Train loop {{{


def train_loop(
    model_and_loss,
    optimizer,
    scaler,
    lr_scheduler,
    train_loader,
    val_loader,
    logger,
    should_backup_checkpoint,
    steps_per_epoch,
    ema=None,
    model_ema=None,
    use_amp=False,
    batch_size_multiplier=1,
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
    prec1 = -1
    use_ema = (model_ema is not None) and (ema is not None)

    if early_stopping_patience > 0:
        epochs_since_improvement = 0
    backup_prefix = checkpoint_filename[:-len("checkpoint.pth.tar")] if \
        checkpoint_filename.endswith("checkpoint.pth.tar") else ""
    
    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    with utils.TimeoutHandler() as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if logger is not None:
                logger.start_epoch()
            if not skip_training:
                interrupted = train(
                    train_loader,
                    model_and_loss,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    logger,
                    epoch,
                    steps_per_epoch,
                    timeout_handler,
                    ema=ema,
                    use_amp=use_amp,
                    prof=prof,
                    register_metrics=epoch == start_epoch,
                    batch_size_multiplier=batch_size_multiplier,
                )

            if not skip_validation:
                prec1, nimg = validate(
                    val_loader,
                    model_and_loss,
                    logger,
                    epoch,
                    use_amp=use_amp,
                    prof=prof,
                    register_metrics=epoch == start_epoch,
                )
                if use_ema:
                    model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                    prec1, nimg = validate(
                        val_loader,
                        model_ema,
                        logger,
                        epoch,
                        prof=prof,
                        register_metrics=epoch == start_epoch,
                        prefix='val_ema'
                    )

                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = True
                best_prec1 = 0

            if logger is not None:
                logger.end_epoch()

            if save_checkpoints and (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            ):
                if should_backup_checkpoint(epoch):
                    backup_filename = "{}checkpoint-{}.pth.tar".format(backup_prefix, epoch + 1)
                else:
                    backup_filename = None
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "state_dict": model_and_loss.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                }
                if use_ema:
                    checkpoint_state["state_dict_ema"] = ema.state_dict()

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
