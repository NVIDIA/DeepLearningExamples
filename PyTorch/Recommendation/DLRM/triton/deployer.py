#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved. 
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
import argparse
import deployer_lib
import json
# 
import sys
sys.path.append('../')

from dlrm.model import Dlrm
from dlrm.data.synthetic_dataset import SyntheticDataset

def get_model_args(model_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--dump_perf_data", type=str, default=None)
    parser.add_argument("--model_checkpoint", type=str, default=None)

    parser.add_argument("--num_numerical_features", type=int, default=13)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--top_mlp_sizes", type=int, nargs="+",
                        default=[1024, 1024, 512, 256, 1])
    parser.add_argument("--bottom_mlp_sizes", type=int, nargs="+",
                        default=[512, 256, 128])
    parser.add_argument("--interaction_op", type=str, default="dot",
                        choices=["dot", "cat"])
    parser.add_argument("--self_interaction", default=False, 
                        action="store_true")
    parser.add_argument("--hash_indices", default=False, 
                        action="store_true")
    parser.add_argument("--cpu", default=False, action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    
    return parser.parse_args(model_args)

def initialize_model(args, categorical_sizes):
    ''' return model, ready to trace '''
    base_device = "cuda:0" if not args.cpu else "cpu"
    model_config = {
        "top_mlp_sizes": args.top_mlp_sizes,
        "bottom_mlp_sizes": args.bottom_mlp_sizes,
        "embedding_dim": args.embedding_dim,
        "interaction_op": args.interaction_op,
        "self_interaction": args.self_interaction,
        "categorical_feature_sizes": categorical_sizes,
        "num_numerical_features": args.num_numerical_features,
        "hash_indices": args.hash_indices,
        "base_device": base_device
    }
        
    model = Dlrm.from_dict(model_config, sigmoid=True)
    model.to(base_device)

    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint,  
                                         map_location="cpu"))

    if args.fp16:
        model = model.half()

    return model

def get_dataloader(args, categorical_sizes):
    dataset_test = SyntheticDataset(num_entries=2000,
                                    batch_size=args.batch_size,
                                    dense_features=args.num_numerical_features,
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


if __name__=='__main__':
    deployer, model_args = deployer_lib.create_deployer(sys.argv[1:], 
            get_model_args) # deployer and returns removed deployer arguments
    with open(os.path.join(model_args.dataset, "model_size.json")) as f:
        categorical_sizes = list(json.load(f).values())

    model = initialize_model(model_args, categorical_sizes)
    dataloader = get_dataloader(model_args, categorical_sizes)

    if model_args.dump_perf_data:
        input_0, input_1 = next(iter(dataloader))
        if model_args.fp16:
            input_0 = input_0.half()

        os.makedirs(model_args.dump_perf_data, exist_ok=True)
        input_0.detach().cpu().numpy()[0].tofile(os.path.join(model_args.dump_perf_data, "input__0"))
        input_1.detach().cpu().numpy()[0].tofile(os.path.join(model_args.dump_perf_data, "input__1"))
        
    deployer.deploy(dataloader, model)
