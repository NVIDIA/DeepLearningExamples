# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.proj_adaptive_softmax_jit import ProjectedAdaptiveLogSoftmax


class PositionalEmbedding(torch.jit.ScriptModule):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    @torch.jit.script_method
    def forward(self, pos_seq, bsz: Optional[int] = None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(torch.jit.ScriptModule):
    __constants__ = ['pre_lnorm']

    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    @torch.jit.script_method
    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(torch.jit.ScriptModule):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask.bool()
        else:
            return mask.flip(0).bool()

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    @torch.jit.script_method
    def _rel_shift(self, x, zero_triu: bool = False):
        zero_pad = torch.zeros((x.size(0), x.size(1), 1, x.size(3)),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)

        x_padded = x_padded.view(x.size(0), x.size(2) + 1, x.size(1), x.size(3))

        x = x_padded[:, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[None, :, :, None]

        return x

    @torch.jit.script_method
    def forward(self, w, r, attn_mask, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    __constants__ = ['pre_lnorm', 'n_head', 'd_head', 'scale']

    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    @torch.jit.script_method
    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask,
                mems: Optional[torch.Tensor] = None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head

        # AC = torch.einsum('ibnd,jbnd->bijn', (rw_head_q, w_head_k))    # bsz x qlen x klen x n_head
        rw_head_q = rw_head_q.view(qlen, bsz * self.n_head, self.d_head).permute(1, 0, 2)
        w_head_k = w_head_k.reshape(klen, bsz * self.n_head, self.d_head).permute(1, 2, 0)
        AC = torch.bmm(rw_head_q, w_head_k).view(bsz, self.n_head, qlen, klen).permute(0, 2, 3 ,1)

        rr_head_q = w_head_q + r_r_bias

        # BD = torch.einsum('ibnd,jnd->bijn', (rr_head_q, r_head_k))     # bsz x qlen x klen x n_head
        rr_head_q = rr_head_q.permute(2, 1, 0, 3).reshape(self.n_head, bsz * qlen, self.d_head)
        r_head_k = r_head_k.permute(1, 2, 0).view(self.n_head, self.d_head, klen)
        BD = torch.bmm(rr_head_q, r_head_k).permute(1, 2, 0).view(bsz, qlen, klen, self.n_head)

        BD = self._rel_shift(BD, False)

        # [bsz x qlen x klen x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [bsz x qlen x klen x n_head]
        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        # attn_vec = torch.einsum('bijn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_prob = attn_prob.permute(0, 3, 1 ,2).reshape(bsz * self.n_head, qlen, klen)
        w_head_v = w_head_v.permute(1, 2, 0, 3).reshape(bsz * self.n_head, klen, self.d_head)
        attn_vec = torch.bmm(attn_prob, w_head_v).permute(1, 0, 2).view(qlen, bsz, self.n_head, self.d_head)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.reshape(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
            output = output.type_as(w)

        return output


class RelPartialLearnableDecoderLayer(torch.jit.ScriptModule):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout,
                                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    @torch.jit.script_method
    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask,
                mems: Optional[torch.Tensor] = None
                ):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(torch.jit.ScriptModule):
    __constants__ = ['div_val', 'd_proj', 'd_embed', 'emb_scale']

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val != 1:
            raise RuntimeError('TorchScripted model supports only div_val == 1')
        if d_proj != d_embed:
            raise RuntimeError('TorchScripted model supports only d_proj == d_embed')
        self.emb_layers.append(nn.Embedding(n_token, d_embed))

    @torch.jit.script_method
    def forward(self, x):
        for emb_layer in self.emb_layers:
            x = emb_layer(x)

        x.mul_(self.emb_scale)

        return x


class MemTransformerLM(torch.jit.ScriptModule):
    __constants__ = ['same_length', 'mem_len', 'clamp_len', 'ext_len',
                     'n_layer', 'dtype']

    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, dtype, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.dtype = dtype

        self.attn_type = attn_type
        if attn_type != 0:
            raise RuntimeError('TorchScripted supports only attn_type == 0')

        self.layers = nn.ModuleList()
        # the default attention
        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
            )

        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                cutoffs, div_val=div_val)

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    @torch.jit.script_method
    def init_mems(self):
        mems = []
        for i in range(self.n_layer+1):
            empty = torch.empty(0, dtype=self.dtype, device=torch.device('cuda'))
            mems.append(empty)

        return mems

    def _update_mems(self, hids: List[torch.Tensor], mems: List[torch.Tensor],
                     qlen: int, mlen: int):
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        new_mems = []
        end_idx = mlen + max(0, qlen - 0 - self.ext_len)
        beg_idx = max(0, end_idx - self.mem_len)
        for i in range(len(hids)):

            cat = torch.cat([mems[i], hids[i]], dim=0)
            new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    @torch.jit.script_method
    def _forward(self, dec_inp, mems: List[torch.Tensor]):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0)
        klen = mlen + qlen
        if self.same_length:
            # all_ones = word_emb.new_ones(qlen, klen)
            all_ones = torch.ones((qlen, klen), device=torch.device('cuda'), dtype=torch.float32)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen) + torch.tril(all_ones, -mask_shift_len)).to(torch.bool)
        else:
            all_ones = torch.ones((qlen, klen), device=torch.device('cuda'), dtype=torch.float32)
            dec_attn_mask = torch.triu(
                all_ones, diagonal=1+mlen).to(torch.bool)

        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        i = 0
        for layer in self.layers:
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                             self.r_r_bias, dec_attn_mask=dec_attn_mask,
                             mems=mems_i)
            hids.append(core_out)
            i += 1

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    @torch.jit.script_method
    def forward(self, data, target, mems: Optional[List[torch.Tensor]]):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if mems is None:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
        loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
