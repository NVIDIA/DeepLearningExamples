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
from argparse import Namespace
# 
import torch
from modeling import BertConfig, BertForQuestionAnswering


def update_argparser(parser):
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--checkpoint", type=str, default='', help="The checkpoint of the model.")
    parser.add_argument("--config-file", default=None, type=str, required=True, help="The BERT model config.")
    parser.add_argument("--fixed-batch-dim", default=False, action="store_true")
    parser.add_argument("--cpu", default=False, action="store_true")


def get_model_from_args(args):
    config = BertConfig.from_json_file(args.config_file)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    class BertForQuestionAnswering_int32_inputs(BertForQuestionAnswering):
        def forward(self, input_ids, segment_ids, attention_mask):
            input_ids, segment_ids, attention_mask = input_ids.long(), segment_ids.long(), attention_mask.long()
            return super().forward(input_ids, segment_ids, attention_mask)

    model = BertForQuestionAnswering_int32_inputs(config)

    model.enable_apex(False)
    if os.path.isfile(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        state_dict = state_dict["model"] if "model" in state_dict.keys() else state_dict
        model.load_state_dict(state_dict, strict=False)
    if args.precision == "fp16":
        model = model.half()
    device = "cuda:0" if not args.cpu else "cpu"
    model = model.to(device)
    model.eval()
    model.bermuda_batch_axis = 0 if not args.fixed_batch_dim else None
    return model


def get_model(**model_args):
    """return model, ready to be traced and tensor names"""

    args = Namespace(**model_args)
    model = get_model_from_args(args)
    tensor_names = {"inputs": ["input__0", "input__1", "input__2"], "outputs": ["output__0", "output__1"]}

    return model, tensor_names

