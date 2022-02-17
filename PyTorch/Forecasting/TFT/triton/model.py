# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import os
import torch
import torch.nn as nn


def update_argparser(parser):
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to be used", required=True)
    parser.add_argument("--precision", type=str, choices=['fp16', 'fp32'], required=True)

class TFTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, s_cat, s_cont, k_cat, k_cont, o_cat, o_cont, target, id):
        # wrapped_input = torch.jit.annotate(Dict[str, Optional[Tensor]], {})
        wrapped_input = {}
        input_names = ['s_cat', 's_cont', 'k_cat', 'k_cont', 'o_cat', 'o_cont', 'target', 'id']
        wrapped_input['s_cat'] = s_cat if s_cat.shape[1] != 1 else None
        wrapped_input['s_cont'] = s_cont if s_cont.shape[1] != 1 else None
        wrapped_input['k_cat'] = k_cat if k_cat.shape[1] != 1 else None
        wrapped_input['k_cont'] = k_cont if k_cont.shape[1] != 1 else None
        wrapped_input['o_cat'] = o_cat if o_cat.shape[1] != 1 else None
        wrapped_input['o_cont'] = o_cont if o_cont.shape[1] != 1 else None
        wrapped_input['target'] = target
        wrapped_input['id'] = id if id.numel() else None

        return self.model(wrapped_input)

def get_model(**args):
    #get model config
    os.environ["TFT_SCRIPTING"] = "True"
    from modeling import TemporalFusionTransformer

    state_dict = torch.load(os.path.join(args['checkpoint'], "checkpoint.pt"))
    config = state_dict['config']
    #create model
    model = TemporalFusionTransformer(config)
    #load model
    model.load_state_dict(state_dict['model'])
    model.eval()
    model.cuda()
    model = TFTWrapper(model).cuda()
    tensor_names = {
        "inputs": ['s_cat__0', 's_cont__1', 'k_cat__2', 'k_cont__3', 'o_cat__4', 'o_cont__5', 'target__6', 'id__7'],
        "outputs": ["target__0"]
    }
    return model, tensor_names