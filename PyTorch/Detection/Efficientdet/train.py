#!/usr/bin/env python
""" EfficientDet Training Script

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import time
import yaml
import os
from datetime import datetime
import ctypes
import numpy as np
import random
import copy

import torch
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as DDP

import dllogger

from effdet.factory import create_model
from effdet.evaluator import COCOEvaluator
from effdet.bench import unwrap_bench
from data import create_loader, CocoDetection
from utils.gpu_affinity import set_affinity
from utils.utils import *
from utils.optimizers import create_optimizer, clip_grad_norm_2
from utils.scheduler import create_scheduler
from utils.model_ema import ModelEma

torch.backends.cudnn.benchmark = True
_libcudart = ctypes.CDLL('libcudart.so')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='tf_efficientdet_d1', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias')
parser.set_defaults(redundant_bias=None)
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--pretrained-backbone-path', default='', type=str, metavar='PATH',
                    help='Start from pretrained backbone weights.')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume full model and optimizer state from checkpoint (default: False)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='0', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--input_size', type=int, default=None, metavar='PCT',
                    help='Image size (default: None) if this is not set default model image size is taken')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')

# Optimizer parameters
parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.0,
                    help='label smoothing (default: 0.0)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--dist-group-size', type=int, default=0,
                    help='Group size for sync-bn')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--eval-after', type=int, default=0, metavar='N',
                    help='Start evaluating after eval-after epochs')
parser.add_argument('--benchmark', action='store_true', default=False,
                    help='Turn this on when measuring performance')
parser.add_argument('--benchmark-steps', type=int, default=0, metavar='N',
                    help='Run training for this number of steps for performance measurement')
parser.add_argument('--dllogger-file', default='log.json', type=str, metavar='PATH',
                    help='File name of dllogger json file (default: log.json, current dir)')
parser.add_argument('--save-checkpoint-interval', type=int, default=10, metavar='N',
                    help='Save checkpoints after so many epochs')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--no-pin-mem', dest='pin_mem', action='store_false',
                    help='Disable pin CPU memory in DataLoader.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='map', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "map"')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
parser.add_argument("--memory-format", type=str, default="nchw", choices=["nchw", "nhwc"], 
                    help="memory layout, nchw or nhwc")
parser.add_argument("--fused-focal-loss", action='store_true',
                    help="Use fused focal loss for better performance.")

# Waymo
parser.add_argument('--waymo', action='store_true', default=False,
                    help='Train on Waymo dataset or COCO dataset. Default: False (COCO dataset)')
parser.add_argument('--num_classes', type=int, default=None, metavar='PCT',
                    help='Number of classes the model needs to be trained for (default: None)')
parser.add_argument('--remove-weights', nargs='*', default=[],
                    help='Remove these weights from the state dict before loading checkpoint (use case can be not loading heads)')
parser.add_argument('--freeze-layers', nargs='*', default=[],
                    help='Freeze these layers')
parser.add_argument('--waymo-train-annotation', default=None, type=str,
                    help='Absolute Path to waymo training annotation (default: "None")')
parser.add_argument('--waymo-val-annotation', default=None, type=str,
                    help='Absolute Path to waymo validation annotation (default: "None")')
parser.add_argument('--waymo-train', default=None, type=str,
                    help='Path to waymo training relative to waymo data (default: "None")')
parser.add_argument('--waymo-val', default=None, type=str,
                    help='Path to waymo validation relative to waymo data (default: "None")')



def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_outdirectory(path, *paths):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    return outdir

def main():
    setup_default_logging()  ## TODO(sugh) replace
    args, args_text = _parse_args()
    set_affinity(args.local_rank)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        torch.cuda.manual_seed_all(args.seed)
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    assert args.rank >= 0

    setup_dllogger(args.rank, filename=args.dllogger_file)
    dllogger.metadata('eval_batch_time', {'unit': 's'})
    dllogger.metadata('train_batch_time', {'unit': 's'})
    dllogger.metadata('eval_throughput', {'unit': 'images/s'})
    dllogger.metadata('train_throughout', {'unit': 'images/s'})
    dllogger.metadata('eval_loss', {'unit': None})
    dllogger.metadata('train_loss', {'unit': None})
    dllogger.metadata('map', {'unit': None})

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on 1 GPU.')

    if args.waymo:
        if (args.waymo_train is not None and args.waymo_val is None) or (args.waymo_train is None and args.waymo_val is not None):
            raise Exception("waymo_train or waymo_val is not set")

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )

    model = create_model(
        args.model,
        input_size=args.input_size,
        num_classes=args.num_classes,
        bench_task='train',
        pretrained=args.pretrained,
        pretrained_backbone_path=args.pretrained_backbone_path,
        redundant_bias=args.redundant_bias,
        checkpoint_path=args.initial_checkpoint,
        label_smoothing=args.smoothing,
        fused_focal_loss=args.fused_focal_loss,
        remove_params=args.remove_weights,
        freeze_layers=args.freeze_layers,
        strict_load=False
    )
    # FIXME decide which args to keep and overlay on config / pass to backbone
    #     num_classes=args.num_classes,
    input_size = model.config.image_size
    data_config = model.config
    print("Input size to be passed to dataloaders: {}".format(input_size))
    print("Image size used in model: {}".format(model.config.image_size))

    if args.rank == 0:
        dllogger.log(step='PARAMETER', data={'model_name':args.model, 'param_count': sum([m.numel() for m in model.parameters()])})
    model = model.cuda().to(memory_format=memory_format)

    # # optionally resume from a checkpoint

    if args.distributed:
        if args.sync_bn:
            try:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
    optimizer = create_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    resume_state = {}
    resume_epoch = None
    output_base = args.output if args.output else './output'
    resume_checkpoint_path = get_latest_checkpoint(os.path.join(output_base, 'train'))
    if args.resume and resume_checkpoint_path is not None:
        print("Trying to load checkpoint from {}".format(resume_checkpoint_path))
        resume_state, resume_epoch = resume_checkpoint(unwrap_bench(model), resume_checkpoint_path)
        if resume_epoch is not None:
            print("Resume training from {} epoch".format(resume_epoch))
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
        if args.amp and 'scaler' in resume_state:
            if args.local_rank == 0:
                logging.info('Restoring NVIDIA AMP state from checkpoint')
            scaler.load_state_dict(resume_state['scaler'])
    del resume_state

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        if args.resume and resume_checkpoint_path is not None:
            resume_path = resume_checkpoint_path
        else:
            resume_path = ''
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            resume=resume_path)

    if args.distributed:
        if args.local_rank == 0:
            logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
        model = DDP(model, device_ids=[args.device])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        dllogger.log(step="PARAMETER", data={'Scheduled_epochs': num_epochs}, verbosity=0)

    # Benchmark will always override every other setting.
    if args.benchmark:
        start_epoch = 0
        num_epochs = args.epochs

    if args.waymo:
        train_annotation_path = args.waymo_train_annotation
        train_image_dir = args.waymo_train
    else:
        train_anno_set = 'train2017'
        train_annotation_path = os.path.join(args.data, 'annotations', f'instances_{train_anno_set}.json')
        train_image_dir = train_anno_set
    dataset_train = CocoDetection(os.path.join(args.data, train_image_dir), train_annotation_path, data_config)

    loader_train = create_loader(
        dataset_train,
        input_size=input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        interpolation=args.train_interpolation,
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        memory_format=memory_format
    )

    loader_train_iter = iter(loader_train)
    steps_per_epoch = int(np.ceil( len(dataset_train) / (args.world_size * args.batch_size) ))

    if args.waymo:
        val_annotation_path = args.waymo_val_annotation
        val_image_dir = args.waymo_val
    else:
        val_anno_set = 'val2017'
        val_annotation_path = os.path.join(args.data, 'annotations', f'instances_{val_anno_set}.json')
        val_image_dir = val_anno_set
    dataset_eval = CocoDetection(os.path.join(args.data, val_image_dir), val_annotation_path, data_config)

    loader_eval = create_loader(
        dataset_eval,
        input_size=input_size,
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        memory_format=memory_format
    )

    evaluator = COCOEvaluator(dataset_eval.coco, distributed=args.distributed, waymo=args.waymo)

    eval_metric = args.eval_metric
    eval_metrics = None
    train_metrics = {}
    best_metric = -1
    is_best = False
    best_epoch = None
    saver = None
    output_dir = ''
    if args.rank == 0:
        output_base = args.output if args.output else './output'
        output_dir = get_outdirectory(output_base, 'train')
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(checkpoint_dir=output_dir)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, steps_per_epoch, model, loader_train_iter, optimizer, args,
                lr_scheduler=lr_scheduler, output_dir=output_dir, use_amp=args.amp, scaler=scaler, model_ema=model_ema)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            # the overhead of evaluating with coco style datasets is fairly high, so just ema or non, not both
            if model_ema is not None:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                
                if epoch >= args.eval_after:
                    eval_metrics = validate(model_ema.ema, loader_eval, args, evaluator, epoch, log_suffix=' (EMA)')
            else:
                eval_metrics = validate(model, loader_eval, args, evaluator, epoch)

            lr_scheduler.step(epoch + 1)

            if saver is not None and args.rank == 0 and epoch % args.save_checkpoint_interval == 0:
                if eval_metrics is not None:
                    # save proper checkpoint with eval metric
                    is_best = eval_metrics[eval_metric] > best_metric
                    best_metric = max(
                        eval_metrics[eval_metric],
                        best_metric
                    )
                    best_epoch = epoch
                else:
                    is_best = False
                    best_metric = 0
                saver.save_checkpoint(model, optimizer, epoch, model_ema=model_ema, metric=best_metric, is_best=is_best)


    except KeyboardInterrupt:
        dllogger.flush()
        torch.cuda.empty_cache()
    if best_metric > 0:
        train_metrics.update({'best_map': best_metric, 'best_epoch': best_epoch})
    if eval_metrics is not None:
        train_metrics.update(eval_metrics)
    dllogger.log(step=(), data=train_metrics, verbosity=0)


def train_epoch(
        epoch, steps_per_epoch, model, loader_iter, optimizer, args,
        lr_scheduler=None, output_dir='', use_amp=False, scaler=None, model_ema=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    throughput_m = AverageMeter()

    model.train()

    torch.cuda.synchronize()
    end = time.time()
    last_idx = steps_per_epoch - 1
    num_updates = epoch * steps_per_epoch
    for batch_idx in range(steps_per_epoch):
        input, target = next(loader_iter)
        last_batch = batch_idx == last_idx
        torch.cuda.synchronize()
        data_time_m.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(input, target)
            loss = output['loss']

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        for p in model.parameters():
            p.grad = None

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1
        if batch_idx == 10:
            batch_time_m.reset()
            throughput_m.reset()

        batch_time_m.update(time.time() - end)
        throughput_m.update(float(input.size(0) * args.world_size / batch_time_m.val))
        if last_batch or (batch_idx+1) % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.rank == 0:
                dllogger_data = {'train_batch_time': batch_time_m.avg, 
                'train_loss': losses_m.avg,
                'throughput': throughput_m.avg,
                'lr': lr,
                'train_data_time': data_time_m.avg}
                dllogger.log(step=(epoch, steps_per_epoch, batch_idx), data=dllogger_data, verbosity=0)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        torch.cuda.synchronize()
        end = time.time()
        if args.benchmark:
            if batch_idx >= args.benchmark_steps:
                break
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    metrics = {'train_loss': losses_m.avg, 'train_batch_time': batch_time_m.avg, 'train_throughout': throughput_m.avg}
    dllogger.log(step=(epoch,), data=metrics, verbosity=0)

    return metrics


def validate(model, loader, args, evaluator=None, epoch=0, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    throughput_m = AverageMeter()

    model.eval()

    torch.cuda.synchronize()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(input, target)
                loss = output['loss']

            if evaluator is not None:
                evaluator.add_predictions(output['detections'], target)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            throughput_m.update(float(input.size(0) * args.world_size / batch_time_m.val))
            end = time.time()
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                dllogger_data = {'eval_batch_time': batch_time_m.val, 'eval_loss': losses_m.val}
                dllogger.log(step=(epoch, last_idx, batch_idx), data=dllogger_data, verbosity=0)

    metrics = {'eval_batch_time': batch_time_m.avg, 'eval_throughput': throughput_m.avg, 'eval_loss': losses_m.avg}
    if evaluator is not None:
        metrics['map'] = evaluator.evaluate()
        if args.rank == 0:
            dllogger.log(step=(epoch,), data=metrics, verbosity=0)

    return metrics


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
