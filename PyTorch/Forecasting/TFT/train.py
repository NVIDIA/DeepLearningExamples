# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import os
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from apex import amp
from apex.optimizers import FusedAdam
#from torch.nn.parallel import DistributedDataParallel as DDP
from apex.parallel import DistributedDataParallel as DDP

import numpy as np

import dllogger

from modeling import TemporalFusionTransformer
from configuration import CONFIGS
from data_utils import TFTBinaryDataset, sample_data
from log_helper import setup_logger
from criterions import QuantileLoss
from inference import predict
from utils import PerformanceMeter
import gpu_affinity
from ema import ModelEma

def load_dataset(args, config):
    train_split = TFTBinaryDataset(os.path.join(args.data_path, 'train.bin'), config)
    train_split = sample_data(train_split, args.sample_data[0])
    if args.distributed_world_size > 1:
        data_sampler = DistributedSampler(train_split, args.distributed_world_size, args.distributed_rank, seed=args.seed + args.distributed_rank, drop_last=True)
    else:
        data_sampler = RandomSampler(train_split)
    train_loader = DataLoader(train_split, batch_size=args.batch_size, num_workers=4, sampler=data_sampler, pin_memory=True)

    valid_split = TFTBinaryDataset(os.path.join(args.data_path, 'valid.bin'), config)
    valid_split = sample_data(valid_split, args.sample_data[1])
    if args.distributed_world_size > 1:
        data_sampler = DistributedSampler(valid_split, args.distributed_world_size, args.distributed_rank, shuffle=False, drop_last=False)
    else:
        data_sampler = None
    valid_loader = DataLoader(valid_split, batch_size=args.batch_size, sampler=data_sampler, num_workers=4, pin_memory=True)

    test_split = TFTBinaryDataset(os.path.join(args.data_path, 'test.bin'), config)
    if args.distributed_world_size > 1:
        data_sampler = DistributedSampler(test_split, args.distributed_world_size, args.distributed_rank, shuffle=False, drop_last=False)
    else:
        data_sampler = None
    test_loader = DataLoader(test_split, batch_size=args.batch_size, sampler=data_sampler, num_workers=4, pin_memory=True)

    print_once(f'Train split length: {len(train_split)}')
    print_once(f'Valid split length: {len(valid_split)}')
    print_once(f'Test split length: {len(test_split)}')

    return train_loader, valid_loader, test_loader

def print_once(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def main(args):
    ### INIT DISTRIBUTED
    args.distributed_world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.distributed_world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        print_once(f'Distributed training with {args.distributed_world_size} GPUs')
        args.distributed_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        torch.cuda.synchronize()

    # Enable CuDNN autotuner
    nproc_per_node = torch.cuda.device_count()
    if args.affinity != 'disabled':
        affinity = gpu_affinity.set_affinity(
                args.local_rank,
                nproc_per_node,
                args.affinity
            )
        print(f'{args.local_rank}: thread affinity: {affinity}')

    torch.backends.cudnn.benchmark = True

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    setup_logger(args)

    config = CONFIGS[args.dataset]()
    if args.overwrite_config:
        config.__dict__.update(json.loads(args.overwrite_config))

    dllogger.log(step='HPARAMS', data={**vars(args), **vars(config)}, verbosity=1)

    model = TemporalFusionTransformer(config).cuda()
    if args.ema_decay:
        model_ema = ModelEma(model, decay=args.ema_decay)

    print_once('Model params: {}'.format(sum(p.numel() for p in model.parameters())))
    criterion = QuantileLoss(config).cuda()
    optimizer = FusedAdam(model.parameters(), lr=args.lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic")
    if args.distributed_world_size > 1:
        #model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        model = DDP(model)

    train_loader, valid_loader, test_loader = load_dataset(args, config)

    global_step = 0
    perf_meter = PerformanceMeter()

    for epoch in range(args.epochs):
        start = time.time()
        dllogger.log(step=global_step, data={'epoch': epoch}, verbosity=1)

        model.train() 
        for local_step, batch in enumerate(train_loader):
            perf_meter.reset_current_lap()
            batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
            predictions = model(batch)
            targets = batch['target'][:,config.encoder_length:,:]
            p_losses = criterion(predictions, targets)
            loss = p_losses.sum()

            if args.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if not args.grad_accumulation or (global_step+1) % args.grad_accumulation == 0:
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                if args.ema_decay:
                    model_ema.update(model)

            if args.distributed_world_size > 1:
                dist.all_reduce(p_losses)
                p_losses /= args.distributed_world_size
                loss = p_losses.sum()

            torch.cuda.synchronize()
            ips = perf_meter.update(args.batch_size * args.distributed_world_size,
                    exclude_from_total=local_step in [0, len(train_loader)-1])

            log_dict = {'P10':p_losses[0].item(), 'P50':p_losses[1].item(), 'P90':p_losses[2].item(), 'loss': loss.item(), 'items/s':ips}
            dllogger.log(step=global_step, data=log_dict, verbosity=1)
            global_step += 1

        validate(args, config, model_ema if args.ema_decay else model, criterion, valid_loader, global_step)

        if validate.early_stop_c >= args.early_stopping:
            print_once('Early stopping')
            break

    ### TEST PHASE ###
    state_dict = torch.load(os.path.join(args.results, 'checkpoint.pt'), map_location='cpu')
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict['model'])
    model.cuda().eval()

    tgt_scalers = pickle.load(open(os.path.join(args.data_path, 'tgt_scalers.bin'), 'rb'))
    cat_encodings = pickle.load(open(os.path.join(args.data_path,'cat_encodings.bin'), 'rb'))

    unscaled_predictions, unscaled_targets, _, _ = predict(args, config, model, test_loader, tgt_scalers, cat_encodings)
    losses = QuantileLoss(config)(unscaled_predictions, unscaled_targets)
    normalizer = unscaled_targets.abs().mean()
    quantiles = 2 * losses / normalizer

    if args.distributed_world_size > 1:
        quantiles = quantiles.cuda()
        dist.all_reduce(quantiles)
        quantiles /= args.distributed_world_size

    quantiles = {'test_p10': quantiles[0].item(), 'test_p50': quantiles[1].item(), 'test_p90': quantiles[2].item(), 'sum':sum(quantiles).item()}
    finish_log = {**quantiles, 'average_ips':perf_meter.avg, 'convergence_step':validate.conv_step}
    dllogger.log(step=(), data=finish_log, verbosity=1)

def validate(args, config, model, criterion, dataloader, global_step):
    if not hasattr(validate, 'best_valid_loss'):
        validate.best_valid_loss = float('inf')
    if not hasattr(validate, 'early_stop_c'):
        validate.early_stop_c = 0
    model.eval()

    losses = []
    validation_start = time.time()
    for batch in dataloader:
        with torch.no_grad():
            batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
            predictions = model(batch)
            targets = batch['target'][:,config.encoder_length:,:]
            p_losses = criterion(predictions, targets)
            bs = next(t for t in batch.values() if t is not None).shape[0]
            losses.append((p_losses, bs))

    validation_end = time.time()

    p_losses = sum([l[0]*l[1] for l in losses])/sum([l[1] for l in losses]) #takes into accunt that the last batch is not full
    if args.distributed_world_size > 1:
        dist.all_reduce(p_losses)
        p_losses = p_losses/args.distributed_world_size

    ips = len(dataloader.dataset) / (validation_end - validation_start)

    log_dict = {'P10':p_losses[0].item(), 'P50':p_losses[1].item(), 'P90':p_losses[2].item(), 'loss': p_losses.sum().item(), 'items/s':ips}

    if log_dict['loss'] < validate.best_valid_loss:
        validate.best_valid_loss = log_dict['loss']
        validate.early_stop_c = 0
        validate.conv_step = global_step
        if not dist.is_initialized() or dist.get_rank() == 0:
            state_dict = model.module.state_dict() if isinstance(model, (DDP, ModelEma)) else model.state_dict()
            ckpt = {'args':args, 'config':config, 'model':state_dict}
            torch.save(ckpt, os.path.join(args.results, 'checkpoint.pt'))
        if args.distributed_world_size > 1:
            dist.barrier()
    else:
        validate.early_stop_c += 1
        
    log_dict = {'val_'+k:v for k,v in log_dict.items()}
    dllogger.log(step=global_step, data=log_dict, verbosity=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=CONFIGS.keys(),
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Default number of training epochs')
    parser.add_argument('--sample_data', type=lambda x: int(float(x)), nargs=2, default=[-1, -1],
                        help="""Subsample the dataset. Specify number of training and valid examples.
                        Values can be provided in scientific notation. Floats will be truncated.""")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--grad_accumulation', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=1000,
                        help='Stop training if validation loss does not improve for more than this number of epochs.')
    parser.add_argument('--results', type=str, default='/results',
                        help='Directory in which results are stored')
    parser.add_argument('--log_file', type=str, default='dllogger.json',
                        help='Name of dllogger output file')
    parser.add_argument('--overwrite_config', type=str, default='',
                       help='JSON string used to overload config')
    parser.add_argument('--affinity', type=str,
                         default='socket_unique_interleaved',
                         choices=['socket', 'single', 'single_unique',
                                  'socket_unique_interleaved',
                                  'socket_unique_continuous',
                                  'disabled'],
                         help='type of CPU affinity')
    parser.add_argument("--ema_decay", type=float, default=0.0, help='Use exponential moving average')


    ARGS = parser.parse_args()
    main(ARGS)
