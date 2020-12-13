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

from transformers import BertConfig
from transformers.modeling_bert import BertEncoder
from .ckpt_quantization import checkpoint_quantization


class EncoderWeights(object):
    def __init__(self, layer_num, hidden_dim, weights=None):
        """weights need be a state_dict of bert model"""
        self.layer_num = layer_num
        self.int8 = False
        self.hidden_dim = hidden_dim
        self.weights = {}
        if weights is None:
            self._generated_weights = True
            for i in range(layer_num):
                pre = 'bert.encoder.layer.' + str(i) + '.'
                self.weights[pre + 'attention.self.query.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.query.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.self.key.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.key.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.self.value.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.self.value.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.dense.weight'] = torch.zeros(hidden_dim, hidden_dim)
                self.weights[pre + 'attention.output.dense.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.LayerNorm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'attention.output.LayerNorm.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'intermediate.dense.weight'] = torch.zeros(4 * hidden_dim, hidden_dim)
                self.weights[pre + 'intermediate.dense.bias'] = torch.zeros(4 * hidden_dim)
                self.weights[pre + 'output.dense.weight'] = torch.zeros(hidden_dim, 4 * hidden_dim)
                self.weights[pre + 'output.dense.bias'] = torch.zeros(hidden_dim)
                self.weights[pre + 'output.LayerNorm.weight'] = torch.zeros(hidden_dim)
                self.weights[pre + 'output.LayerNorm.bias'] = torch.zeros(hidden_dim)
            for k, v in self.weights.items():
                if not k.endswith('_amax'):
                    self.weights[k] = torch.nn.init.uniform_(v, -1, 1)
        else:
            self._generated_weights = False
            for k, v in weights.items():
                ks = k.split('.')
                if ks[-2] == 'LayerNorm':
                    if ks[-1] == 'gamma':
                        ks[-1] = 'weight'
                    elif ks[-1] == 'beta':
                        ks[-1] = 'bias'
                self.weights['.'.join(ks)] = v

    def listed_weights(self, layer_idx):
        ret = []
        pre = 'bert.encoder.layer.' + str(layer_idx) + '.'
        ret.append(self.weights[pre + 'attention.self.query.weight'])       # 0
        ret.append(self.weights[pre + 'attention.self.query.bias'])
        ret.append(self.weights[pre + 'attention.self.key.weight'])         # 2
        ret.append(self.weights[pre + 'attention.self.key.bias'])
        ret.append(self.weights[pre + 'attention.self.value.weight'])       # 4
        ret.append(self.weights[pre + 'attention.self.value.bias'])
        ret.append(self.weights[pre + 'attention.output.dense.weight'])     # 6
        ret.append(self.weights[pre + 'attention.output.dense.bias'])
        ret.append(self.weights[pre + 'attention.output.LayerNorm.weight'])
        ret.append(self.weights[pre + 'attention.output.LayerNorm.bias'])
        ret.append(self.weights[pre + 'intermediate.dense.weight'])         # 10
        ret.append(self.weights[pre + 'intermediate.dense.bias'])
        ret.append(self.weights[pre + 'output.dense.weight'])               # 12
        ret.append(self.weights[pre + 'output.dense.bias'])
        ret.append(self.weights[pre + 'output.LayerNorm.weight'])
        ret.append(self.weights[pre + 'output.LayerNorm.bias'])
        if not self.int8:
            ret[0] = ret[0].transpose(-1, -2).contiguous()
            ret[2] = ret[2].transpose(-1, -2).contiguous()
            ret[4] = ret[4].transpose(-1, -2).contiguous()
            ret[6] = ret[6].transpose(-1, -2).contiguous()
            ret[10] = ret[10].transpose(-1, -2).contiguous()
            ret[12] = ret[12].transpose(-1, -2).contiguous()
            ret.append(torch.tensor(0))
        else:
            ret.append(self.weights[pre + 'amaxList'])
        return ret

    def to_cuda(self):
        for k, v in self.weights.items():
            self.weights[k] = v.cuda()

    def to_half(self):
        if self.int8:
            raise RuntimeError("Cannot cast to half if the weights have been casted to int8.")
        for k, v in self.weights.items():
            self.weights[k] = v.half()

    def to_int8(self, is_per_channel, module_path='./', ths_path='./lib/libths_fastertransformer.so'):
        if self._generated_weights:
            if is_per_channel:
                amax_tensor_1 = torch.Tensor(self.hidden_dim).fill_(127.)
                amax_tensor_2 = torch.Tensor(self.hidden_dim * 4).fill_(127.)
            else:
                amax_tensor_1 = torch.tensor(127.)
                amax_tensor_2 = torch.tensor(127.)
            for i in range(self.layer_num):
                pre = 'bert.encoder.layer.' + str(i) + '.'
                self.weights[pre + 'attention.self.query._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.query._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.query._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.key._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.key._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.key._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.value._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.value._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.self.value._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_q_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_k_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_v_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.matmul_a_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.self.softmax_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.dense._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'attention.output.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.add_local_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'attention.output.add_residual_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'intermediate.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'intermediate.dense._weight_quantizer._amax'] = amax_tensor_2
                self.weights[pre + 'intermediate.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.dense._input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.dense._weight_quantizer._amax'] = amax_tensor_1
                self.weights[pre + 'output.dense._aftergemm_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.add_local_input_quantizer._amax'] = torch.tensor(127.)
                self.weights[pre + 'output.add_residual_input_quantizer._amax'] = torch.tensor(127.)
        if 'bert.encoder.layer.0.attention.self.query._input_quantizer._amax' not in self.weights:
            raise RuntimeError("There is no quantization node in the checkpoint, cannot be quantized to int8.")
        if self.int8:
            return
        self.int8 = True
        for k, v in self.weights.items():
            if k.endswith('bias') or k.endswith('LayerNorm.weight'):
                self.weights[k] = v.half()
            else:
                self.weights[k] = v.float().cpu()
        self.weights = checkpoint_quantization(self.weights, is_per_channel, module_path, ths_path, verbose=False)


class CustomEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights,
                 int8_mode=0, remove_padding=False, allow_gemm_test=False,
                 use_ths=False, path='./'):
        super().__init__()
        self.layer_num = layer_num
        self.remove_padding = remove_padding
        self.int8_mode = int8_mode
        self.encoders = []
        use_trt_kernel = True
        if use_ths:
            torch.classes.load_library(path)
            for i in range(layer_num):
                assert len(weights.listed_weights(i)) == 17
                try:
                    self.encoders.append(
                        torch.classes.FasterTransformer.Encoder(
                            *weights.listed_weights(i),
                            head_num, head_size, remove_padding, int8_mode, layer_num, i, allow_gemm_test, use_trt_kernel))
                except:
                    # legacy ths for 20.03 image
                    self.encoders.append(
                        torch.classes.FasterTransformerEncoder(
                            *weights.listed_weights(i),
                            head_num, head_size, remove_padding, int8_mode, layer_num, i, allow_gemm_test, use_trt_kernel))
            self.build_mask_remove_padding = torch.ops.fastertransformer.build_mask_remove_padding
            self.rebuild_padding = torch.ops.fastertransformer.rebuild_padding
        else:
            sys.path.insert(0, path)
            from th_fastertransformer import FasterTransformerEncoder, build_mask_remove_padding, rebuild_padding
            for i in range(layer_num):
                assert len(weights.listed_weights(i)) == 17
                self.encoders.append(
                    FasterTransformerEncoder(
                        *weights.listed_weights(i),
                        head_num, head_size, remove_padding, int8_mode, layer_num, i, allow_gemm_test, use_trt_kernel))
            self.build_mask_remove_padding = build_mask_remove_padding
            self.rebuild_padding = rebuild_padding

    def forward(self, hidden_states, attention_mask, sequence_lengths):
        if self.remove_padding:
            hidden_states, sequence_id_offset = self.build_mask_remove_padding(hidden_states, sequence_lengths)
            trt_seq_len = torch.cumsum(torch.cat([torch.tensor([0]).to(sequence_lengths).cuda(), sequence_lengths], dim=0), dim=0).to(torch.int32).cuda()
        else:
            sequence_id_offset = torch.tensor(0).to(torch.int32).cuda()
            batch = hidden_states.size(0)
            max_seq_len = hidden_states.size(1)
            padding_offset = torch.arange(0, batch*max_seq_len, max_seq_len).cuda()
            squence_offset_with_padding = sequence_lengths + padding_offset
            c = torch.cat([padding_offset, squence_offset_with_padding], dim=0)
            c_r = torch.reshape(c, [2, -1])
            t = torch.transpose(c_r, 0, 1)
            trt_seq_len = torch.reshape(t, [-1])
            trt_seq_len = torch.cat([trt_seq_len, torch.tensor([batch * max_seq_len]).to(trt_seq_len.dtype).cuda()], dim=0).to(torch.int32)
        for i in range(self.layer_num):
            hidden_states = self.encoders[i].forward(hidden_states, attention_mask, trt_seq_len, sequence_id_offset)
        if self.remove_padding:
            hidden_states = self.rebuild_padding(hidden_states, sequence_id_offset, attention_mask, 0)
        return (hidden_states,)


class HuggingFaceEncoder(torch.nn.Module):
    def __init__(self, layer_num, head_num, head_size, weights=None):
        super().__init__()
        hidden_dim = head_num * head_size
        conf = BertConfig(hidden_size=hidden_dim, intermediate_size=4*hidden_dim, num_attention_heads=head_num, num_hidden_layers=layer_num)
        self.encoder = BertEncoder(conf)
        w = {}
        for k, v in weights.weights.items():
            if k.startswith('bert.encoder') and not k.endswith('_amax'):
                w[k[13:]] = weights.weights[k]
        self.encoder.load_state_dict(w)
        self.head_mask = [None] * layer_num

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        output = self.encoder(hidden_states, extended_attention_mask, self.head_mask)
        return output
