# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import os

import torch
from torch.utils.data import DataLoader

from src.utils import dboxes300_coco, COCODetection
from src.utils import SSDTransformer
from src.coco import COCO
#DALI import
from src.coco_pipeline import COCOPipeline, DALICOCOIterator

def get_train_loader(args, local_seed):
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    train_pipe = COCOPipeline(args.batch_size, args.local_rank, train_coco_root,
                    train_annotate, args.N_gpu, num_threads=args.num_workers,
                    output_fp16=args.amp, output_nhwc=False,
                    pad_output=False, seed=local_seed)
    train_pipe.build()
    test_run = train_pipe.schedule_run(), train_pipe.share_outputs(), train_pipe.release_outputs()
    train_loader = DALICOCOIterator(train_pipe, 118287 / args.N_gpu)
    return train_loader


def get_val_dataset(args):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco


def get_val_dataloader(dataset, args):
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        val_sampler = None

    val_dataloader = DataLoader(dataset,
                                batch_size=args.eval_batch_size,
                                shuffle=False,  # Note: distributed sampler is shuffled :(
                                sampler=val_sampler,
                                num_workers=args.num_workers)

    return val_dataloader

def get_coco_ground_truth(args):
    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file=val_annotate)
    return cocoGt
