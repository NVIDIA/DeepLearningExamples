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
import math
import os
from datetime import datetime
import ctypes
import numpy as np

import torch
import torchvision.utils

from effdet.config import get_efficientdet_config
from data import create_loader, CocoDetection
from utils.utils import AverageMeter
from data.loader import IterationBasedBatchSampler

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
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--input-size', type=int, default=512, metavar='N',
                    help='input image size (default: 512)')
parser.add_argument('--prefetcher', action='store_true', default=True,
                    help='enable fast prefetcher')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)


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


def test_number_of_iters_and_elements():
    for batch_size in [4]:
        for drop_last in [False, True]:
            dataset = [i for i in range(10)]
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, batch_size, drop_last=drop_last
            )

            iter_sampler = IterationBasedBatchSampler(
                batch_sampler
            )
            iterator = iter(iter_sampler)
            print("Len of sampler {} ".format(len(iter_sampler)))
            print("=====================================================")
            print("Test batch size {} drop last {}".format(batch_size, drop_last))
            steps_per_epoch = int( np.ceil(len(dataset) / batch_size) )
            i = 0
            for epoch in range(3):
                for _ in range(steps_per_epoch):
                    batch = next(iterator)
                    start = (i % len(batch_sampler)) * batch_size
                    end = min(start + batch_size, len(dataset))
                    expected = [x for x in range(start, end)]
                    print("Epoch {} iteration {} batch {}".format(epoch, i, batch))
                    i += 1



def main():
    args, args_text = _parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    model_name = 'efficientdet_d0'
    data_config = get_efficientdet_config(model_name)

    train_anno_set = 'train2017'
    train_annotation_path = os.path.join(args.data, 'annotations', f'instances_{train_anno_set}.json')
    train_image_dir = train_anno_set
    dataset_train = CocoDetection(os.path.join(args.data, train_image_dir), train_annotation_path, data_config)
    print("Length of training dataset {}".format(len(dataset_train)))
    loader_train = create_loader(
        dataset_train,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        #re_prob=args.reprob,  # FIXME add back various augmentations
        #re_mode=args.remode,
        #re_count=args.recount,
        #re_split=args.resplit,
        #color_jitter=args.color_jitter,
        #auto_augment=args.aa,
        interpolation=args.train_interpolation,
        #mean=data_config['mean'],
        #std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        #collate_fn=collate_fn,
        pin_mem=args.pin_mem,
    )

    print("Iterations per epoch {}".format(math.ceil( len(dataset_train) / ( args.batch_size * args.world_size ))))
    data_time_m = AverageMeter()
    end = time.time()
    if args.local_rank == 0:
        print("Starting to test...")
    for batch_idx, (input, target) in enumerate(loader_train):
        data_time_m.update(time.time() - end)
        if args.local_rank == 0 and batch_idx % 20 == 0:
            print("batch time till {} is {}".format(batch_idx, data_time_m.avg))
        end = time.time()


if __name__ == "__main__":
    main()

#### USAGE ####
#
# NUM_PROC=8
# python -m torch.distributed.launch --nproc_per_node=$NUM_PROC data/dataloader_test.py /workspace/object_detection/datasets/coco -b 64 --workers 16
#