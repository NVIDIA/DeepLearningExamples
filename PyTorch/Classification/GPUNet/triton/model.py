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

import torch
from timm.models.helpers import load_checkpoint
import os
import json

from models.gpunet_builder import GPUNet_Builder

def update_argparser(parser):
    parser.add_argument(
        "--config", type=str, required=True, help="Network to deploy")
    parser.add_argument(
        "--checkpoint", type=str, help="The checkpoint of the model. ")
    parser.add_argument("--precision", type=str, default="fp32", 
                        choices=["fp32", "fp16"], help="Inference precision")
    parser.add_argument(
        "--is-prunet", type=bool, required=True, help="Bool on whether network is a prunet")

def get_model(**model_args):
    dtype = model_args['precision']
    checkpoint = model_args['checkpoint']
    configPath = model_args['config']
    with open(configPath) as configFile:
        modelJSON = json.load(configFile)
        configFile.close()
    builder = GPUNet_Builder()
    model = builder.get_model(modelJSON)
    if dtype == 'fp16':
        dtype = torch.float16
    elif dtype == 'fp32':
        dtype = torch.float32
    else:
        raise NotImplementedError
    if model_args['is_prunet'] == "True":
        model.load_state_dict(torch.load(checkpoint))
    else:
        load_checkpoint(model, checkpoint, use_ema=True)

    model = model.to('cuda', dtype)
    model.eval()
    tensor_names = {"inputs": ["INPUT__0"],
                    "outputs": ["OUTPUT__0"]}

    return model, tensor_names

