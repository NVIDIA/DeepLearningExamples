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
    def __init__(self, model, test_func, output_selector):
        super().__init__()
        self.model = model
        self.test_func = test_func
        self.output_selector = output_selector

    def forward(self, s_cat, s_cont, k_cat, k_cont, o_cat, o_cont, target, weight, sample_weight, id):
        wrapped_input = {}
        wrapped_input['s_cat'] = s_cat if len(s_cat.shape) != 10 else None
        wrapped_input['s_cont'] = s_cont if len(s_cont.shape) != 10 else None
        wrapped_input['k_cat'] = k_cat if len(k_cat.shape) != 10 else None
        wrapped_input['k_cont'] = k_cont if len(k_cont.shape) != 10 else None
        wrapped_input['o_cat'] = o_cat if len(o_cat.shape) != 10 else None
        wrapped_input['o_cont'] = o_cont if len(o_cont.shape) != 10 else None
        wrapped_input['weight'] = weight if len(weight.shape) != 10 else None
        wrapped_input['sample_weight'] = sample_weight if len(sample_weight.shape) != 10 else None
        wrapped_input['target'] = target
        wrapped_input['id'] = id if id.numel() else None
        output = self.test_func(wrapped_input)
        if self.output_selector >= 0:
            return output[..., self.output_selector : self.output_selector + 1]
        return output

def get_model(**args):
    #get model config
    with open(os.path.join(args['model_dir'], ".hydra/config_merged.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    os.environ["TFT_SCRIPTING"] = "True"
    state_dict = torch.load(os.path.join(args['model_dir'], "best_checkpoint.pth.tar"))['model_state_dict']
    if config.config.device.get("world_size", 1) > 1:
        model_params = list(state_dict.items())
        for k, v in model_params:
            if k[:7] == "module.":
                state_dict[k[7:]] = v
                del state_dict[k]
    config._target_ = config.config.model._target_
    model = hydra.utils.instantiate(config)
    test_method_name = config.config.model.get("test_method", "__call__")
    test_method = getattr(model, test_method_name)
    #load model
    preds_test_output_selector = config.config.model.get(
            "preds_test_output_selector", -1
        )
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    model = ModelWrapper(model, test_method, preds_test_output_selector).cuda()
    tensor_names = {
        "inputs": ['s_cat__0', 's_cont__1', 'k_cat__2', 'k_cont__3', 'o_cat__4', 'o_cont__5', 'target__6', 'weight__7', 'sample_weight__8', 'id__9'],
        "outputs": ["target__0"]
    }
    return model, tensor_names
