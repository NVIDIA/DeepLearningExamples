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

import sys
import torch

from onmt.decoders.transformer import TransformerDecoderLayer


class DecoderWeights(object):
    def __init__(self, layer_num, hidden_dim):
        self.layer_num = layer_num
        self.w = [[] for _ in range(layer_num)]
        for layer_weights in self.w:
            layer_weights.append(torch.zeros(hidden_dim))   # self_layernorm_gamma
            layer_weights.append(torch.zeros(hidden_dim))   # self_layernorm_beta
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_kernel_q
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_kernel_k
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_kernel_v
            layer_weights.append(torch.zeros(hidden_dim))   # self_bias_q
            layer_weights.append(torch.zeros(hidden_dim))   # self_bias_k
            layer_weights.append(torch.zeros(hidden_dim))   # self_bias_v
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # self_output_kernel
            layer_weights.append(torch.zeros(hidden_dim))   # self_output_bias
            layer_weights.append(torch.zeros(hidden_dim))   # cross_layernorm_gamma
            layer_weights.append(torch.zeros(hidden_dim))   # cross_layernorm_beta
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_q
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_k
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_kernel_v
            layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_q
            layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_k
            layer_weights.append(torch.zeros(hidden_dim))   # cross_bias_v
            layer_weights.append(torch.zeros(hidden_dim, hidden_dim))   # cross_output_kernel
            layer_weights.append(torch.zeros(hidden_dim))   # cross_output_bias
            layer_weights.append(torch.zeros(hidden_dim))   # ffn_layernorm_gamma
            layer_weights.append(torch.zeros(hidden_dim))   # ffn_layernorm_beta
            layer_weights.append(torch.zeros(hidden_dim, 4 * hidden_dim))   # inter_kernel
            layer_weights.append(torch.zeros(4 * hidden_dim))   # inter_bias
            layer_weights.append(torch.zeros(4 * hidden_dim, hidden_dim))   # output_kernel
            layer_weights.append(torch.zeros(hidden_dim))   # output_bias
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


def init_op_cache(layer_num, batch_size, beam_width, max_seq_len, hidden_dim, is_fp16):
    if is_fp16:
        self_cache = torch.zeros(layer_num, 2, 0, batch_size * beam_width, hidden_dim, dtype=torch.half).cuda()
        mem_cache = torch.zeros(layer_num, 2, batch_size * beam_width, max_seq_len, hidden_dim, dtype=torch.half).cuda()
    else:
        self_cache = torch.zeros(layer_num, 2, 0, batch_size * beam_width, hidden_dim).cuda()
        mem_cache = torch.zeros(layer_num, 2, batch_size * beam_width, max_seq_len, hidden_dim).cuda()
    return self_cache, mem_cache

def init_onmt_cache(layer_num, memory_bank):
    cache = {}
    for i in range(layer_num):
        layer_cache = {"memory_keys": None, "memory_values": None}
        layer_cache["self_keys"] = None
        layer_cache["self_values"] = None
        cache[i] = layer_cache
    return cache


class CustomDecoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights, is_fp16, path='./', use_ths=False):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.fp16 = is_fp16
        self.decoders = []
        if use_ths:
            torch.classes.load_library(path)
            for i in range(layer_num):
                self.decoders.append(torch.classes.FasterTransformerDecoder(head_num, head_size, *weights.w[i]))
        else:
            sys.path.insert(0, path)
            from th_fastertransformer import FasterTransformerDecoder
            for i in range(layer_num):
                self.decoders.append(FasterTransformerDecoder(head_num, head_size, *weights.w[i]))

    def forward(self, inputs, memory, memory_seq_lens, self_cache, mem_cache):
        if self.fp16:
            self_cache_tmp = torch.zeros(self.layer_num, 2, 1, self_cache.size(3), self.hidden_dim, dtype=torch.half).cuda()
        else:
            self_cache_tmp = torch.zeros(self.layer_num, 2, 1, self_cache.size(3), self.hidden_dim).cuda()
        self_cache = torch.cat([self_cache, self_cache_tmp], 2)
        output = inputs
        for i in range(self.layer_num):
            output = self.decoders[i].forward(output, memory, memory_seq_lens, self_cache[i], mem_cache[i])
        return output, self_cache, mem_cache


class ONMTDecoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.decoders = torch.nn.ModuleList()
        for i in range(layer_num):
            self.decoders.append(TransformerDecoderLayer(self.hidden_dim, head_num, 4 * self.hidden_dim, 0, 0))
        for i in range(layer_num):
            self.decoders[i].layer_norm_1.weight.data = weights.w[i][0]
            self.decoders[i].layer_norm_1.bias.data = weights.w[i][1]
            self.decoders[i].self_attn.linear_query.weight.data = weights.w[i][2].transpose(-1, -2).contiguous()
            self.decoders[i].self_attn.linear_keys.weight.data = weights.w[i][3].transpose(-1, -2).contiguous()
            self.decoders[i].self_attn.linear_values.weight.data = weights.w[i][4].transpose(-1, -2).contiguous()
            self.decoders[i].self_attn.linear_query.bias.data = weights.w[i][5]
            self.decoders[i].self_attn.linear_keys.bias.data = weights.w[i][6]
            self.decoders[i].self_attn.linear_values.bias.data = weights.w[i][7]
            self.decoders[i].self_attn.final_linear.weight.data = weights.w[i][8].transpose(-1, -2).contiguous()
            self.decoders[i].self_attn.final_linear.bias.data = weights.w[i][9]
            self.decoders[i].layer_norm_2.weight.data = weights.w[i][10]
            self.decoders[i].layer_norm_2.bias.data = weights.w[i][11]
            self.decoders[i].context_attn.linear_query.weight.data = weights.w[i][12].transpose(-1, -2).contiguous()
            self.decoders[i].context_attn.linear_keys.weight.data = weights.w[i][13].transpose(-1, -2).contiguous()
            self.decoders[i].context_attn.linear_values.weight.data = weights.w[i][14].transpose(-1, -2).contiguous()
            self.decoders[i].context_attn.linear_query.bias.data = weights.w[i][15]
            self.decoders[i].context_attn.linear_keys.bias.data = weights.w[i][16]
            self.decoders[i].context_attn.linear_values.bias.data = weights.w[i][17]
            self.decoders[i].context_attn.final_linear.weight.data = weights.w[i][18].transpose(-1, -2).contiguous()
            self.decoders[i].context_attn.final_linear.bias.data = weights.w[i][19]
            self.decoders[i].feed_forward.layer_norm.weight.data = weights.w[i][20]
            self.decoders[i].feed_forward.layer_norm.bias.data = weights.w[i][21]
            self.decoders[i].feed_forward.w_1.weight.data = weights.w[i][22].transpose(-1, -2).contiguous()
            self.decoders[i].feed_forward.w_1.bias.data = weights.w[i][23]
            self.decoders[i].feed_forward.w_2.weight.data = weights.w[i][24].transpose(-1, -2).contiguous()
            self.decoders[i].feed_forward.w_2.bias.data = weights.w[i][25]

    def forward(self, inputs, memory, src_pad_msk, cache, step):
        output = inputs
        for i in range(self.layer_num):
            output, _, _ = self.decoders[i](output, memory, src_pad_msk, None, cache[i], step)
        return output
