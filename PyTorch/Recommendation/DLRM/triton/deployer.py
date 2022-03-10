#!/usr/bin/python

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
import json
import os
import sys

import torch
import numpy as np

from dlrm.data.datasets import SyntheticDataset
from dlrm.model.distributed import DistributedDlrm
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process
from triton import deployer_lib

sys.path.append('../')


def get_model_args(model_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--dump_perf_data", type=str, default=None)
    parser.add_argument("--model_checkpoint", type=str, default=None)

    parser.add_argument("--num_numerical_features", type=int, default=13)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--embedding_type", type=str, default="joint", choices=["joint", "multi_table"])
    parser.add_argument("--top_mlp_sizes", type=int, nargs="+",
                        default=[1024, 1024, 512, 256, 1])
    parser.add_argument("--bottom_mlp_sizes", type=int, nargs="+",
                        default=[512, 256, 128])
    parser.add_argument("--interaction_op", type=str, default="dot",
                        choices=["dot", "cat"])
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--dataset", type=str, required=True)

    return parser.parse_args(model_args)


def initialize_model(args, categorical_sizes, device_mapping):
    ''' return model, ready to trace '''
    device = "cuda:0" if not args.cpu else "cpu"
    model_config = {
        'top_mlp_sizes': args.top_mlp_sizes,
        'bottom_mlp_sizes': args.bottom_mlp_sizes,
        'embedding_dim': args.embedding_dim,
        'interaction_op': args.interaction_op,
        'categorical_feature_sizes': categorical_sizes,
        'num_numerical_features': args.num_numerical_features,
        'embedding_type': args.embedding_type,
        'hash_indices': False,
        'use_cpp_mlp': False,
        'fp16': args.fp16,
        'device': device,
    }

    model = DistributedDlrm.from_dict(model_config)
    model.to(device)

    if args.model_checkpoint:
        checkpoint_loader = make_distributed_checkpoint_loader(device_mapping=device_mapping, rank=0)
        checkpoint_loader.load_checkpoint(model, args.model_checkpoint)
        model.to(device)

    if args.fp16:
        model = model.half()

    return model


def get_dataloader(args, categorical_sizes):
    dataset_test = SyntheticDataset(num_entries=2000,
                                    batch_size=args.batch_size,
                                    numerical_features=args.num_numerical_features,
                                    categorical_feature_sizes=categorical_sizes,
                                    device="cpu" if args.cpu else "cuda:0")
    class RemoveOutput:
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, idx):
            value = self.dataset[idx]
            if args.fp16:
                value = (value[0].half(), value[1].long(), value[2])
            else:
                value = (value[0], value[1].long(), value[2])
            return value[:-1]

        def __len__(self):
            return len(self.dataset)

    test_loader = torch.utils.data.DataLoader(RemoveOutput(dataset_test),
                                              batch_size=None,
                                              num_workers=0,
                                              pin_memory=False)

    return test_loader


def main():
    # deploys and returns removed deployer arguments
    deployer, model_args = deployer_lib.create_deployer(sys.argv[1:],
                                                        get_model_args)

    with open(os.path.join(model_args.dataset, "model_size.json")) as f:
        categorical_sizes = list(json.load(f).values())
        categorical_sizes = [s + 1 for s in categorical_sizes]
        categorical_sizes = np.array(categorical_sizes)

    device_mapping = get_device_mapping(categorical_sizes, num_gpus=1)
    categorical_sizes = categorical_sizes[device_mapping['embedding'][0]].tolist()

    model = initialize_model(model_args, categorical_sizes, device_mapping)
    dataloader = get_dataloader(model_args, categorical_sizes)

    if model_args.dump_perf_data:
        input_0, input_1 = next(iter(dataloader))
        if model_args.fp16:
            input_0 = input_0.half()

        os.makedirs(model_args.dump_perf_data, exist_ok=True)
        input_0.detach().cpu().numpy()[0].tofile(os.path.join(model_args.dump_perf_data, "input__0"))
        input_1.detach().cpu().numpy()[0].tofile(os.path.join(model_args.dump_perf_data, "input__1"))

    deployer.deploy(dataloader, model)


if __name__=='__main__':
    main()
