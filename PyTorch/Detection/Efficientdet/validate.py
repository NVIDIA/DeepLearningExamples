#!/usr/bin/env python
""" COCO validation script

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
import os
import json
import time
import logging
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import ctypes
import dllogger

from effdet.factory import create_model
from effdet.evaluator import COCOEvaluator
from utils.utils import setup_dllogger
from data import create_loader, CocoDetection
from utils.utils import AverageMeter, setup_default_logging

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
import numpy as np
import itertools

torch.backends.cudnn.benchmark = True
_libcudart = ctypes.CDLL('libcudart.so')


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--waymo', action='store_true', default=False,
                    help='Train on Waymo dataset or COCO dataset. Default: False (COCO dataset)')
parser.add_argument('--anno', default='val2017',
                    help='mscoco annotation set (one of val2017, train2017, test-dev2017)')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--input_size', type=int, default=None, metavar='PCT',
                    help='Image size (default: None) if this is not set default model image size is taken')
parser.add_argument('--num_classes', type=int, default=None, metavar='PCT',
                    help='Number of classes the model needs to be trained for (default: None)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='mean', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument("--memory-format", type=str, default="nchw", choices=["nchw", "nhwc"], 
                    help="memory layout, nchw or nhwc")
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--inference', dest='inference', action='store_true',
                    help='If true then inference else evaluation.')
parser.add_argument('--use-soft-nms', dest='use_soft_nms', action='store_true', default=False,
                    help='use softnms instead of default nms for eval')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
parser.add_argument('--dllogger-file', default='log.json', type=str, metavar='PATH',
                    help='File name of dllogger json file (default: log.json, current dir)')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--waymo-val', default=None, type=str,
                    help='Path to waymo validation images relative to data (default: "None")')
parser.add_argument('--waymo-val-annotation', default=None, type=str,
                    help='Absolute Path to waymo validation annotation (default: "None")')


def validate(args):
    setup_dllogger(0, filename=args.dllogger_file)
    dllogger.metadata('total_inference_time', {'unit': 's'})
    dllogger.metadata('inference_throughput', {'unit': 'images/s'})
    dllogger.metadata('inference_time', {'unit': 's'})
    dllogger.metadata('map', {'unit': None})
    dllogger.metadata('total_eval_time', {'unit': 's'})

    if args.checkpoint != '':
        args.pretrained = True
    args.prefetcher = not args.no_prefetcher
    if args.waymo:
        assert args.waymo_val is not None

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )
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

    # create model
    bench = create_model(
        args.model,
        input_size=args.input_size,
        num_classes=args.num_classes,
        bench_task='predict',
        pretrained=args.pretrained,
        redundant_bias=args.redundant_bias,
        checkpoint_path=args.checkpoint,
        checkpoint_ema=args.use_ema,
        soft_nms=args.use_soft_nms,
        strict_load=False
    )
    input_size = bench.config.image_size
    data_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda().to(memory_format=memory_format)

    if args.distributed > 1:
        raise ValueError("Evaluation is supported only on single GPU. args.num_gpu must be 1")
        bench = DDP(bench, device_ids=[args.device]) # torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    if args.waymo:
        annotation_path = args.waymo_val_annotation
        image_dir = args.waymo_val
    else:
        if 'test' in args.anno:
            annotation_path = os.path.join(args.data, 'annotations', f'image_info_{args.anno}.json')
            image_dir = 'test2017'
        else:
            annotation_path = os.path.join(args.data, 'annotations', f'instances_{args.anno}.json')
            image_dir = args.anno
    dataset = CocoDetection(os.path.join(args.data, image_dir), annotation_path, data_config)
    
    evaluator = COCOEvaluator(dataset.coco, distributed=args.distributed, waymo=args.waymo)

    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        fill_color=args.fill_color,
        num_workers=args.workers,
        distributed=args.distributed,
        pin_mem=args.pin_mem,
        memory_format=memory_format)

    img_ids = []
    results = []
    dllogger_metric = {}
    bench.eval()
    batch_time = AverageMeter()
    throughput = AverageMeter()
    torch.cuda.synchronize()
    end = time.time()
    total_time_start = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = bench(input, target['img_scale'], target['img_size'])
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            throughput.update(input.size(0) / batch_time.val)
            evaluator.add_predictions(output, target)
            torch.cuda.synchronize()

            # measure elapsed time
            if i == 9:
                batch_time.reset()
                throughput.reset()

            if args.rank == 0 and i % args.log_freq == 0:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                     )
                )
            end = time.time()

    torch.cuda.synchronize()
    dllogger_metric['total_inference_time'] = time.time() - total_time_start
    dllogger_metric['inference_throughput'] = throughput.avg
    dllogger_metric['inference_time'] = 1000 / throughput.avg
    total_time_start = time.time()
    mean_ap = 0.
    if not args.inference:
        if 'test' not in args.anno:
            mean_ap = evaluator.evaluate()
        else:
            evaluator.save_predictions(args.results)
        torch.cuda.synchronize()
        dllogger_metric['map'] = mean_ap
        dllogger_metric['total_eval_time'] = time.time() - total_time_start
    else:
        evaluator.save_predictions(args.results)

    if not args.distributed or args.rank == 0:
        dllogger.log(step=(), data=dllogger_metric, verbosity=0)

    return results


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()

