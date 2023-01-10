# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


import argparse
import functools
import json
import logging
import os
import signal
from typing import Dict

from apex.contrib.clip_grad import clip_grad_norm_
from apex.optimizers import FusedAdam as Adam
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler

from moflow.config import CONFIGS, Config
from moflow.data.data_loader import NumpyTupleDataset
from moflow.data import transform
from moflow.model.model import MoFlow, MoFlowLoss
from moflow.model.utils import initialize
from moflow.runtime.logger import MetricsLogger, PerformanceLogger, setup_logging
from moflow.runtime.arguments import PARSER
from moflow.runtime.common import get_newest_checkpoint, load_state, save_state
from moflow.runtime.distributed_utils import (
    get_device, get_rank, get_world_size, init_distributed, reduce_tensor
)
from moflow.runtime.generate import infer
from moflow.utils import check_validity, convert_predictions_to_mols


torch._C._jit_set_autocast_mode(True)


def run_validation(model: MoFlow, config: Config, ln_var: float, args: argparse.Namespace,
                         is_distributed: bool, world_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    if is_distributed:
        model_callable = model.module
    else:
        model_callable = model
    result = infer(model_callable, config, device=device, ln_var=ln_var, batch_size=args.val_batch_size,
                   temp=args.temperature)
    mols = convert_predictions_to_mols(*result, correct_validity=args.correct_validity)
    validity_info = check_validity(mols)
    valid_ratio = torch.tensor(validity_info['valid_ratio'], dtype=torch.float32, device=device)
    unique_ratio = torch.tensor(validity_info['unique_ratio'], dtype=torch.float32, device=device)
    valid_value = reduce_tensor(valid_ratio, world_size).detach().cpu().numpy()
    unique_value = reduce_tensor(unique_ratio, world_size).detach().cpu().numpy()
    model.train()
    return {'valid': valid_value, 'unique': unique_value}


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.results_dir, exist_ok=True)

    # Device configuration
    device = get_device(args.local_rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()

    logger = setup_logging(args)
    if local_rank == 0:
        perf_logger = PerformanceLogger(logger, args.batch_size * world_size, args.warmup_steps)
        acc_logger = MetricsLogger(logger)

    if local_rank == 0:
        logging.info('Input args:')
        logging.info(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # Model configuration
    assert args.config_name in CONFIGS
    config = CONFIGS[args.config_name]
    data_file = config.dataset_config.dataset_file
    transform_fn = functools.partial(transform.transform_fn, config=config)
    valid_idx = transform.get_val_ids(config, args.data_dir)

    if local_rank == 0:
        logging.info('Config:')
        logging.info(str(config))
    model = MoFlow(config)
    model.to(device)
    loss_module = MoFlowLoss(config)
    loss_module.to(device)

    # Datasets:
    dataset = NumpyTupleDataset.load(
        os.path.join(args.data_dir, data_file),
        transform=transform_fn,
    )
    if len(valid_idx) == 0:
        raise ValueError('Empty validation set!')
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    train = torch.utils.data.Subset(dataset, train_idx)
    test = torch.utils.data.Subset(dataset, valid_idx)

    if world_size > 1:
        sampler = DistributedSampler(train, seed=args.seed, drop_last=False)
    else:
        sampler = None
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if local_rank == 0:
        logging.info(f'Using {world_size} GPUs')
        logging.info(f'Num training samples: {len(train)}')
        logging.info(f'Minibatch-size: {args.batch_size}')
        logging.info(f'Num Iter/Epoch: {len(train_dataloader)}')
        logging.info(f'Num epoch: {args.epochs}')

    if is_distributed:
        train_dataloader.sampler.set_epoch(-1)
    x, adj, *_ = next(iter(train_dataloader))
    x = x.to(device)
    adj = adj.to(device)
    with autocast(enabled=args.amp):
        initialize(model, (adj, x))

    model.to(memory_format=torch.channels_last)
    adj.to(memory_format=torch.channels_last)

    if args.jit:
        model.bond_model = torch.jit.script(model.bond_model)
        model.atom_model = torch.jit.script(model.atom_model)

    # make one pass in both directions to make sure that model works
    with torch.no_grad():
        _ = model(adj, x)
        _ = model.reverse(torch.randn(args.batch_size, config.z_dim, device=device))

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        loss_module = torch.nn.parallel.DistributedDataParallel(
            loss_module,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        model_callable = model.module
        loss_callable = loss_module.module
    else:
        model_callable = model
        loss_callable = loss_module

    # Loss and optimizer
    optimizer = Adam((*model.parameters(), *loss_module.parameters()), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    scaler = GradScaler()

    if args.save_epochs == -1:
        args.save_epochs = args.epochs
    if args.eval_epochs == -1:
        args.eval_epochs = args.epochs
    if args.steps == -1:
        args.steps = args.epochs * len(train_dataloader)

    snapshot_path = get_newest_checkpoint(args.results_dir)
    if snapshot_path is not None:
        snapshot_epoch, ln_var = load_state(snapshot_path, model_callable, optimizer=optimizer, device=device)
        loss_callable.ln_var = torch.nn.Parameter(torch.tensor(ln_var))
        first_epoch = snapshot_epoch + 1
        step = first_epoch * len(train_dataloader)
    else:
        first_epoch = 0
        step = 0

    if first_epoch >= args.epochs:
        logging.info(f'Model was already trained for {first_epoch} epochs')
        exit(0)

    for epoch in range(first_epoch, args.epochs):
        if local_rank == 0:
            acc_logger.reset()
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            if local_rank == 0:
                perf_logger.update()
            step += 1
            optimizer.zero_grad()
            x = batch[0].to(device)
            adj = batch[1].to(device=device,memory_format=torch.channels_last)

            # Forward, backward and optimize
            with_cuda_graph = (
                args.cuda_graph
                and step >= args.warmup_steps
                and x.size(0) == args.batch_size
            )
            with autocast(enabled=args.amp, cache_enabled=not with_cuda_graph):
                output = model(adj, x, with_cuda_graph=with_cuda_graph)
                nll_x, nll_adj = loss_module(*output)
                loss = nll_x + nll_adj

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

            # Print log info
            if (i + 1) % args.log_interval == 0:
                nll_x_value = reduce_tensor(nll_x, world_size).item()
                nll_adj_value = reduce_tensor(nll_adj, world_size).item()
                loss_value = nll_x_value + nll_adj_value

                if local_rank == 0:
                    acc_logger.update({
                        'loglik': loss_value,
                        'nll_x': nll_x_value,
                        'nll_adj': nll_adj_value
                    })

                    acc_logger.summarize(step=(epoch, i, i))
                    perf_logger.summarize(step=(epoch, i, i))

            if step >= args.steps:
                break

        if (epoch + 1) % args.eval_epochs == 0:
            with autocast(enabled=args.amp):
                metrics = run_validation(model, config, loss_callable.ln_var.item(), args, is_distributed, world_size, device)
            if local_rank == 0:
                acc_logger.update(metrics)

        # The same report for each epoch
        if local_rank == 0:
            acc_logger.summarize(step=(epoch,))
            perf_logger.summarize(step=(epoch,))

        # Save the model checkpoints
        if (epoch + 1) % args.save_epochs == 0:
            if local_rank == 0 or not is_distributed:
                save_state(args.results_dir, model_callable, optimizer, loss_callable.ln_var.item(), epoch, keep=5)

        if step >= args.steps:
            break

    if local_rank == 0:
        acc_logger.summarize(step=tuple())
        perf_logger.summarize(step=tuple())


if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    args = PARSER.parse_args()
    train(args)
