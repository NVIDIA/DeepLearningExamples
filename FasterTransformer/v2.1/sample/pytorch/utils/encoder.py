# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function
from typing import List

import sys
import torch

from transformers import BertConfig
from transformers.modeling_bert import BertEncoder


class EncoderWeights(object):
    def __init__(self, layer_num, hidden_dim, weights=None):
        self.layer_num = layer_num
        self.w = [[] for _ in range(layer_num)]
        if weights:
            if isinstance(weights, dict):
                for i in range(layer_num):
                    pre = 'bert.encoder.layer.' + str(i) + '.'
                    self.w[i].append(weights[pre + 'attention.self.query.weight'].transpose(-1, -2).contiguous())
                    self.w[i].append(weights[pre + 'attention.self.query.bias'])
                    self.w[i].append(weights[pre + 'attention.self.key.weight'].transpose(-1, -2).contiguous())
                    self.w[i].append(weights[pre + 'attention.self.key.bias'])
                    self.w[i].append(weights[pre + 'attention.self.value.weight'].transpose(-1, -2).contiguous())
                    self.w[i].append(weights[pre + 'attention.self.value.bias'])
                    self.w[i].append(weights[pre + 'attention.output.dense.weight'].transpose(-1, -2).contiguous())
                    self.w[i].append(weights[pre + 'attention.output.dense.bias'])
                    self.w[i].append(weights[pre + 'attention.output.LayerNorm.weight'])
                    self.w[i].append(weights[pre + 'attention.output.LayerNorm.bias'])
                    self.w[i].append(weights[pre + 'intermediate.dense.weight'].transpose(-1, -2).contiguous())
                    self.w[i].append(weights[pre + 'intermediate.dense.bias'])
                    self.w[i].append(weights[pre + 'output.dense.weight'].transpose(-1, -2).contiguous())
                    self.w[i].append(weights[pre + 'output.dense.bias'])
                    self.w[i].append(weights[pre + 'output.LayerNorm.weight'])
                    self.w[i].append(weights[pre + 'output.LayerNorm.bias'])
            else:
                for i in range(layer_num):
                    self.w[i].append(weights.layer[i].attention.self.query.weight.data.transpose(-1, -2).contiguous())
                    self.w[i].append(weights.layer[i].attention.self.query.bias.data)
                    self.w[i].append(weights.layer[i].attention.self.key.weight.data.transpose(-1, -2).contiguous())
                    self.w[i].append(weights.layer[i].attention.self.key.bias.data)
                    self.w[i].append(weights.layer[i].attention.self.value.weight.data.transpose(-1, -2).contiguous())
                    self.w[i].append(weights.layer[i].attention.self.value.bias.data)
                    self.w[i].append(weights.layer[i].attention.output.dense.weight.data.transpose(-1, -2).contiguous())
                    self.w[i].append(weights.layer[i].attention.output.dense.bias.data)
                    self.w[i].append(weights.layer[i].attention.output.LayerNorm.weight.data)
                    self.w[i].append(weights.layer[i].attention.output.LayerNorm.bias.data)
                    self.w[i].append(weights.layer[i].intermediate.dense.weight.data.transpose(-1, -2).contiguous())
                    self.w[i].append(weights.layer[i].intermediate.dense.bias.data)
                    self.w[i].append(weights.layer[i].output.dense.weight.data.transpose(-1, -2).contiguous())
                    self.w[i].append(weights.layer[i].output.dense.bias.data)
                    self.w[i].append(weights.layer[i].output.LayerNorm.weight.data)
                    self.w[i].append(weights.layer[i].output.LayerNorm.bias.data)
        else:
            for layer_weights in self.w:
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # q_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # q_bias
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # k_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # k_bias
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # v_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # v_bias
                layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # attr_output_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # attr_output_bias
                layer_weights.append(torch.zeros(hidden_dim))   # attr_output_layernorm_beta
                layer_weights.append(torch.zeros(hidden_dim))   # attr_output_layernorm_gamma
                layer_weights.append(torch.zeros(hidden_dim, 4 * hidden_dim))   # inter_kernel
                layer_weights.append(torch.zeros(4 * hidden_dim))   # inter_bias
                layer_weights.append(torch.zeros(4 * hidden_dim, hidden_dim))   # output_kernel
                layer_weights.append(torch.zeros(hidden_dim))   # output_bias
                layer_weights.append(torch.zeros(hidden_dim))   # output_layernorm_beta
                layer_weights.append(torch.zeros(hidden_dim))   # output_layernorm_gamma
                for i in range(len(layer_weights)):
                    torch.nn.init.uniform_(layer_weights[i], -1, 1)

    def to_cuda(self):
        for i in range(self.layer_num):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].cuda()

    def to_half(self):
        for i in range(self.layer_num):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].half()


class CustomEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights, path='./', use_ths=False, remove_padding=False):
        super().__init__()
        self.layer_num = layer_num
        self.encoders = []
        if use_ths:
            torch.classes.load_library(path)
            for i in range(layer_num):
                self.encoders.append(torch.classes.FasterTransformerEncoder(head_num, head_size, remove_padding, *weights.w[i]))
        else:
            sys.path.insert(0, path)
            from th_fastertransformer import FasterTransformerEncoder
            for i in range(layer_num):
                self.encoders.append(FasterTransformerEncoder(head_num, head_size, remove_padding, *weights.w[i]))

    def forward(self, hidden_states, attention_mask, sequence_lengths=torch.Tensor(0).to(torch.int).cuda()):
        for i in range(self.layer_num):
            hidden_states = self.encoders[i].forward(hidden_states, attention_mask, sequence_lengths)
        return (hidden_states,)


class CustomEncoder2(torch.nn.Module):
    w: List[List[torch.Tensor]]
    def __init__(self, layer_num, head_num, head_size, weights, path='./', remove_padding=False):
        super().__init__()
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_size = head_size
        self.remove_padding = remove_padding
        self.w = weights.w
        torch.ops.load_library(path)

    def forward(self, hidden_states, attention_mask, sequence_lengths=torch.Tensor(0).to(torch.int).cuda()):
        for i in range(self.layer_num):
            hidden_states = torch.ops.fastertransformer.encoder(self.head_num, self.head_size, self.remove_padding,
                                                                *self.w[i], hidden_states, attention_mask, sequence_lengths)
        return (hidden_states,)


class HuggingFaceEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights=None):
        super().__init__()
        hidden_dim = head_num * head_size
        conf = BertConfig(hidden_size=hidden_dim, intermediate_size=4*hidden_dim, num_attention_heads=head_num, num_hidden_layers=layer_num)
        self.encoder = BertEncoder(conf)
        if isinstance(weights, dict):
            w = {}
            for k, v in weights.items():
                if k.startswith('bert.encoder'):
                    w[k[13:]] = weights[k]
            self.encoder.load_state_dict(w)
        else:
            for i in range(layer_num):
                self.encoder.layer[i].attention.self.query.weight.data = weights.w[i][0].transpose(-1, -2).contiguous()
                self.encoder.layer[i].attention.self.query.bias.data = weights.w[i][1]
                self.encoder.layer[i].attention.self.key.weight.data = weights.w[i][2].transpose(-1, -2).contiguous()
                self.encoder.layer[i].attention.self.key.bias.data = weights.w[i][3]
                self.encoder.layer[i].attention.self.value.weight.data = weights.w[i][4].transpose(-1, -2).contiguous()
                self.encoder.layer[i].attention.self.value.bias.data = weights.w[i][5]
                self.encoder.layer[i].attention.output.dense.weight.data = weights.w[i][6].transpose(-1, -2).contiguous()
                self.encoder.layer[i].attention.output.dense.bias.data = weights.w[i][7]
                self.encoder.layer[i].attention.output.LayerNorm.weight.data = weights.w[i][8]
                self.encoder.layer[i].attention.output.LayerNorm.bias.data = weights.w[i][9]
                self.encoder.layer[i].intermediate.dense.weight.data = weights.w[i][10].transpose(-1, -2).contiguous()
                self.encoder.layer[i].intermediate.dense.bias.data = weights.w[i][11]
                self.encoder.layer[i].output.dense.weight.data = weights.w[i][12].transpose(-1, -2).contiguous()
                self.encoder.layer[i].output.dense.bias.data = weights.w[i][13]
                self.encoder.layer[i].output.LayerNorm.weight.data = weights.w[i][14]
                self.encoder.layer[i].output.LayerNorm.bias.data = weights.w[i][15]
        self.head_mask = [None] * layer_num

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        output = self.encoder(hidden_states, extended_attention_mask, self.head_mask)
        return output
