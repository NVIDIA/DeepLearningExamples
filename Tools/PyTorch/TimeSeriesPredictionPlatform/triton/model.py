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

import os
import torch
import torch.nn as nn
import hydra

from typing import Dict, Tuple, Optional, List
from omegaconf import OmegaConf


def update_argparser(parser):
    parser.add_argument("--model-dir", type=str, help="Path to the model directory you would like to use (likely in outputs)", required=True)

class ModelWrapper(nn.Module):
    def __init__(self, model, test_func):
        super().__init__()
        self.model = model
        self.test_func = test_func
    def unwrap(self, t):
        if not torch.isnan(t).any():
            return t
        return None

    def forward(self, s_cat, s_cont, k_cat, k_cont, o_cat, o_cont, target, sample_weight, id, weight):
        wrapped_input = {}
        wrapped_input['s_cat'] = self.unwrap(s_cat)
        wrapped_input['s_cont'] = self.unwrap(s_cont)
        wrapped_input['k_cat'] = self.unwrap(k_cat)
        wrapped_input['k_cont'] = self.unwrap(k_cont)
        wrapped_input['o_cat'] = self.unwrap(o_cat)
        wrapped_input['o_cont'] = self.unwrap(o_cont)
        wrapped_input['sample_weight'] = self.unwrap(sample_weight)
        wrapped_input['target'] = target
        wrapped_input['id'] = id if id.numel() else None
        wrapped_input['weight'] = self.unwrap(weight)
        output = self.test_func(wrapped_input)
        return output

def get_model(**args):
    #get model config
    with open(os.path.join(args['model_dir'], ".hydra/config_merged.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    os.environ["TFT_SCRIPTING"] = "True"
    state_dict = torch.load(os.path.join(args['model_dir'], "best_checkpoint.zip"))['model_state_dict']
    model = hydra.utils.instantiate(config.model)
    test_method_name = 'predict' if hasattr(model, "predict") else '__call__'
    test_method = getattr(model, test_method_name)
    #load model
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    model = ModelWrapper(model, test_method).cuda()
    tensor_names = {
        "inputs": ['s_cat__0', 's_cont__1', 'k_cat__2', 'k_cont__3', 'o_cat__4', 'o_cont__5', 'target__6', 'sample_weight__7', 'id__8', 'weight__9'],
        "outputs": ["target__0"]
    }
    return model, tensor_names
