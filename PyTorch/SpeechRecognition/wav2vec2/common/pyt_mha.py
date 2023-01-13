# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.fairseq.modules.multihead_attention import RotaryEmbedding


def mha_state_dict_to_fairseq(sd):
    """Concatenate q, k, v matrices and load as usual."""
    new_sd = {}
    qkv = defaultdict(dict)

    for key, val in sd.items():
        fields = key.split('.')
        if len(fields) < 2:
            continue
        prefix = '.'.join(fields[:-2] + [""])
        module, param = fields[-2:]

        if module in ['q_proj', 'k_proj', 'v_proj']:
            qkv[prefix][module + '.' + param] = val
        else:
            new_sd[key] = val

    for prefix, param_dict in qkv.items():
        # Stitch qkv params together
        assert len(param_dict) == 6
        new_sd[f"{prefix}qkv.weight"] = torch.cat(
            [param_dict[f"{k}_proj.weight"] for k in ["q", "k", "v"]], dim=0)
        new_sd[f"{prefix}qkv.bias"] = torch.cat(
            [param_dict[f"{k}_proj.bias"] for k in ["q", "k", "v"]], dim=0)

    return new_sd


class PytMultiheadAttention(nn.Module):
    """Drop-in replacement for Fairseq MHA.

    Calls torch.nn.functional with combined qkv.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        self_attention=True,
        rotary_embeddings=False,
    ):
        super().__init__()

        assert self_attention
        assert not rotary_embeddings, "Not yet supported"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rotary_embeddings = rotary_embeddings

        if self.rotary_embeddings:
            self.rotary_freq = RotaryEmbedding(embed_dim)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, 3 * num_heads * self.head_dim,
                             bias=bias)
        self.dropatt = nn.Dropout(dropout)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim,
                                  bias=bias)
        self.reset_parameters()

        def hook(state_dict, prefix, *args, **kwargs):
            this_keys = {k for k in state_dict.keys() if k.startswith(prefix)}
            new_sd = {k: v for k, v in state_dict.items() if k in this_keys}
            for k in this_keys:
                del state_dict[k]
            state_dict.update(mha_state_dict_to_fairseq(new_sd))

        self._register_load_state_dict_pre_hook(hook)

    def forward(self, query, key=None, value=None, key_padding_mask=None,
                attn_mask=None):

        return F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.qkv.weight,
            self.qkv.bias,
            None,
            None,
            False,
            self.dropatt.p,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            average_attn_weights=False,
        )

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """Split q, k, v matrices for bwd compatibility with Fairseq."""
        sd = super().state_dict(*args, destination, prefix, keep_vars)
        for key in list(sd.keys()):
            if not (key.endswith(".qkv.weight") or key.endswith(".qkv.bias")):
                continue
            *pref, qkv, param = key.split(".")
            pref = ".".join(pref)
            assert qkv == "qkv"
            q, k, v = torch.chunk(sd.pop(key), 3, dim=0)
            sd[f"{pref}.q_proj.{param}"] = q
            sd[f"{pref}.k_proj.{param}"] = k
            sd[f"{pref}.v_proj.{param}"] = v

        return sd

    def reset_parameters(self):
        # Init as in Fairseq with qkv_same_dim=True and separate qkv projs
        t = self.qkv.weight.size(0) // 3
        nn.init.xavier_uniform_(self.qkv.weight[0*t:1*t], gain=1 / (2 ** 0.5))
        nn.init.xavier_uniform_(self.qkv.weight[1*t:2*t], gain=1 / (2 ** 0.5))
        nn.init.xavier_uniform_(self.qkv.weight[2*t:3*t], gain=1 / (2 ** 0.5))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)


class Fp32Softmax(nn.Softmax):
    def forward(self, x):
        return F.softmax(x.float(), dim=self.dim).type_as(x)


class SlowMultiHeadAttention(nn.Module):
    """Drop-in replacement for Fairseq MHA."""
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True,
                 self_attention=True,
                 rotary_embeddings=None,
                 fp32_softmax=False,
        ):
        super().__init__()

        n_head = num_heads
        d_model = embed_dim
        d_head = embed_dim // n_head
        dropatt = dropout
        pre_lnorm = False
        assert self_attention
        assert rotary_embeddings is None, "Rotary embs not yet supported"

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv = nn.Linear(d_model, 3 * n_head * d_head, bias=bias)
        self.dropatt = nn.Dropout(dropatt)
        self.proj = nn.Linear(n_head * d_head, d_model, bias=bias)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.softmax = Fp32Softmax(dim=2) if fp32_softmax else nn.Softmax(dim=2)

    def state_dict(self):
        """Convert QKV to be compatible with Fairseq"""
        sd = super().state_dict()

        ret = {}
        for key, val in sd.items():
            fields = key.split('.')
            if len(fields) < 2:
                continue
            prefix = '.'.join(fields[:-2] + [""])
            module, param = fields[-2:]

            if module == 'qkv':
                q, k, v = torch.chunk(val, 3, dim=0)
                ret[f"{prefix}q_proj.{param}"] = q
                ret[f"{prefix}k_proj.{param}"] = k
                ret[f"{prefix}v_proj.{param}"] = v
            else:
                ret[key] = val
        return ret

    def load_state_dict(self, sd):

        from collections import defaultdict

        ret = {}
        qkv = defaultdict(dict)

        for key, val in sd.items():
            fields = key.split('.')
            if len(fields) < 2:
                continue
            prefix = '.'.join(fields[:-2] + [""])
            module, param = fields[-2:]

            if module in ['q_proj', 'k_proj', 'v_proj']:
                qkv[prefix][module + '.' + param] = val
            else:
                ret[key] = val

        for prefix, param_dict in qkv.items():
            # Stitch qkv params together
            assert len(param_dict) == 6
            ret[f"{prefix}qkv.weight"] = torch.cat(
                [param_dict[f"{k}_proj.weight"] for k in ["q", "k", "v"]],
                dim=0)
            ret[f"{prefix}qkv.bias"] = torch.cat(
                [param_dict[f"{k}_proj.bias"] for k in ["q", "k", "v"]],
                dim=0)

        super().load_state_dict(ret)

    def forward(self, inp, attn_mask=None):
        inp = inp.permute(1, 0, 2)  # (T, B, H) -> (B, T, H)

        if self.pre_lnorm:
            inp = self.layer_norm(inp)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv(inp), 3, dim=2)

        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = self.softmax(attn_score)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(
            inp.size(0), inp.size(1), n_head * d_head)

        output = self.proj(attn_vec)

        return output.permute(1, 0, 2)  # (B, T, H) -> (T, B, H)
