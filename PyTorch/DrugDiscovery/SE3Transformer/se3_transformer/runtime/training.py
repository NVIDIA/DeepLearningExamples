# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import pathlib
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedLAMB
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from se3_transformer.data_loading import QM9DataModule
from se3_transformer.model import SE3TransformerPooled
from se3_transformer.model.fiber import Fiber
from se3_transformer.runtime import gpu_affinity
from se3_transformer.runtime.arguments import PARSER
from se3_transformer.runtime.callbacks import QM9MetricCallback, QM9LRSchedulerCallback, BaseCallback, \
    PerformanceCallback
from se3_transformer.runtime.inference import evaluate
from se3_transformer.runtime.loggers import LoggerCollection, DLLogger, WandbLogger, Logger
from se3_transformer.runtime.utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity


def save_state(model: nn.Module, optimizer: Optimizer, epoch: int, path: pathlib.Path, callbacks: List[BaseCallback]):
    """ Saves model, optimizer and epoch states to path (only once per node) """
    if get_local_rank() == 0:
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        checkpoint = {
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        for callback in callbacks:
            callback.on_checkpoint_save(checkpoint)

        torch.save(checkpoint, str(path))
        logging.info(f'Saved checkpoint to {str(path)}')


def load_state(model: nn.Module, optimizer: Optimizer, path: pathlib.Path, callbacks: List[BaseCallback]):
    """ Loads model, optimizer and epoch states from path """
    checkpoint = torch.load(str(path), map_location={'cuda:0': f'cuda:{get_local_rank()}'})
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for callback in callbacks:
        callback.on_checkpoint_load(checkpoint)

    logging.info(f'Loaded checkpoint from {str(path)}')
    return checkpoint['epoch']


def train_epoch(train_dataloader, loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, callbacks, args):
    losses = []
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit='batch',
                         desc=f'Epoch {epoch_idx}', disable=(args.silent or local_rank != 0)):
        *inputs, target = to_cuda(batch)

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(*inputs)
            loss = loss_fn(pred, target) / args.accumulate_grad_batches

        grad_scaler.scale(loss).backward()

        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        losses.append(loss.item())

    return np.mean(losses)


def train(model: nn.Module,
          loss_fn: _Loss,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          callbacks: List[BaseCallback],
          logger: Logger,
          args):
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), lr=args.learning_rate, betas=(args.momentum, 0.999),
                              weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    epoch_start = load_state(model, optimizer, args.load_ckpt_path, callbacks) if args.load_ckpt_path else 0

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)

    for epoch_idx in range(epoch_start, args.epochs):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        loss = train_epoch(train_dataloader, loss_fn, epoch_idx, grad_scaler, optimizer, local_rank, callbacks, args)
        if dist.is_initialized():
            loss = torch.tensor(loss, dtype=torch.float, device=device)
            torch.distributed.all_reduce(loss)
            loss = (loss / world_size).item()

        logging.info(f'Train loss: {loss}')
        logger.log_metrics({'train loss': loss}, epoch_idx)

        for callback in callbacks:
            callback.on_epoch_end()

        if not args.benchmark and args.save_ckpt_path is not None and args.ckpt_interval > 0 \
                and (epoch_idx + 1) % args.ckpt_interval == 0:
            save_state(model, optimizer, epoch_idx, args.save_ckpt_path, callbacks)

        if not args.benchmark and args.eval_interval > 0 and (epoch_idx + 1) % args.eval_interval == 0:
            evaluate(model, val_dataloader, callbacks, args)
            model.train()

            for callback in callbacks:
                callback.on_validation_end(epoch_idx)

    if args.save_ckpt_path is not None and not args.benchmark:
        save_state(model, optimizer, args.epochs, args.save_ckpt_path, callbacks)

    for callback in callbacks:
        callback.on_fit_end()


def print_parameters_count(model):
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable parameters: {num_params_trainable}')


if __name__ == '__main__':
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    args = PARSER.parse_args()

    logging.getLogger().setLevel(logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO)

    logging.info('====== SE(3)-Transformer ======')
    logging.info('|      Training procedure     |')
    logging.info('===============================')

    if args.seed is not None:
        logging.info(f'Using seed {args.seed}')
        seed_everything(args.seed)

    logger = LoggerCollection([
        DLLogger(save_dir=args.log_dir, filename=args.dllogger_name),
        WandbLogger(name=f'QM9({args.task})', save_dir=args.log_dir, project='se3-transformer')
    ])

    datamodule = QM9DataModule(**vars(args))
    model = SE3TransformerPooled(
        fiber_in=Fiber({0: datamodule.NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args)
    )
    loss_fn = nn.L1Loss()

    if args.benchmark:
        logging.info('Running benchmark mode')
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        callbacks = [PerformanceCallback(logger, args.batch_size * world_size)]
    else:
        callbacks = [QM9MetricCallback(logger, targets_std=datamodule.targets_std, prefix='validation'),
                     QM9LRSchedulerCallback(logger, epochs=args.epochs)]

    if is_distributed:
        gpu_affinity.set_affinity(gpu_id=get_local_rank(), nproc_per_node=torch.cuda.device_count())

    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    increase_l2_fetch_granularity()
    train(model,
          loss_fn,
          datamodule.train_dataloader(),
          datamodule.val_dataloader(),
          callbacks,
          logger,
          args)

    logging.info('Training finished successfully')
