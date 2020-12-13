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
import os
import math
import torch
import torch.nn as nn

from onmt.modules import Embeddings, AverageAttention
from onmt.decoders.decoder import DecoderBase
from onmt.decoders.transformer import TransformerDecoderLayer
from onmt.utils.misc import tile, sequence_mask


class DecodingWeights(object):
    def __init__(self, layer_num, hidden_dim, vocab_size, onmtcheckpoint=None, max_step_for_pe=2048):
        self.hidden_dim = hidden_dim
        self.max_step_for_pe = max_step_for_pe
        self.w = []
        if onmtcheckpoint:
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.layer_norm_1.weight'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.layer_norm_1.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_query.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_keys.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_values.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_query.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_keys.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.linear_values.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.final_linear.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.self_attn.final_linear.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.layer_norm_2.weight'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.layer_norm_2.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.linear_query.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.linear_keys.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.linear_values.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.linear_query.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.linear_keys.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.linear_values.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.final_linear.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.context_attn.final_linear.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.layer_norm.weight'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.layer_norm.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_1.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.weight'].transpose(-1, -2) for i in range(layer_num)],
                0).contiguous())
            self.w.append(torch.stack(
                [onmtcheckpoint['model']['decoder.transformer_layers.' + str(i) + '.feed_forward.w_2.bias'] for i in range(layer_num)],
                0).contiguous())
            self.w.append(onmtcheckpoint['model']['decoder.layer_norm.weight'])
            self.w.append(onmtcheckpoint['model']['decoder.layer_norm.bias'])
            self.w.append(onmtcheckpoint['model']['decoder.embeddings.make_embedding.emb_luts.0.weight'])
            self.w.append(self._get_position_encoding()) # pe_encoding
            self.w.append(onmtcheckpoint['generator']['0.weight'].transpose(-1, -2).contiguous())
            self.w.append(onmtcheckpoint['generator']['0.bias'])
        else:
            self.w.append(torch.zeros(layer_num, hidden_dim))   # self_layernorm_gamma
            self.w.append(torch.zeros(layer_num, hidden_dim))   # self_layernorm_beta
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # self_kernel_q
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # self_kernel_k
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # self_kernel_v
            self.w.append(torch.zeros(layer_num, hidden_dim))   # self_bias_q
            self.w.append(torch.zeros(layer_num, hidden_dim))   # self_bias_k
            self.w.append(torch.zeros(layer_num, hidden_dim))   # self_bias_v
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # self_output_kernel
            self.w.append(torch.zeros(layer_num, hidden_dim))   # self_output_bias
            self.w.append(torch.zeros(layer_num, hidden_dim))   # cross_layernorm_gamma
            self.w.append(torch.zeros(layer_num, hidden_dim))   # cross_layernorm_beta
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # cross_kernel_q
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # cross_kernel_k
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # cross_kernel_v
            self.w.append(torch.zeros(layer_num, hidden_dim))   # cross_bias_q
            self.w.append(torch.zeros(layer_num, hidden_dim))   # cross_bias_k
            self.w.append(torch.zeros(layer_num, hidden_dim))   # cross_bias_v
            self.w.append(torch.zeros(layer_num, hidden_dim, hidden_dim))   # cross_output_kernel
            self.w.append(torch.zeros(layer_num, hidden_dim))   # cross_output_bias
            self.w.append(torch.zeros(layer_num, hidden_dim))   # ffn_layernorm_gamma
            self.w.append(torch.zeros(layer_num, hidden_dim))   # ffn_layernorm_beta
            self.w.append(torch.zeros(layer_num, hidden_dim, 4 * hidden_dim))   # inter_kernel
            self.w.append(torch.zeros(layer_num, 4 * hidden_dim))   # inter_bias
            self.w.append(torch.zeros(layer_num, 4 * hidden_dim, hidden_dim))   # output_kernel
            self.w.append(torch.zeros(layer_num, hidden_dim))   # output_bias
            self.w.append(torch.zeros(hidden_dim))   # decoding_gamma
            self.w.append(torch.zeros(hidden_dim))   # decoding_beta
            self.w.append(torch.zeros(vocab_size, hidden_dim))   # embedding_table
            self.w.append(self._get_position_encoding()) # pe_encoding
            self.w.append(torch.zeros(hidden_dim, vocab_size))   # embedding_kernel
            self.w.append(torch.zeros(vocab_size))   # embedding_bias
            for i in range(len(self.w)):
                torch.nn.init.uniform_(self.w[i], -1, 1)

    def to_cuda(self):
        for i in range(len(self.w)):
            self.w[i] = self.w[i].cuda()

    def to_half(self):
        for i in range(len(self.w) - 1):    # embedding_bias is float32
            self.w[i] = self.w[i].half()

    def _get_position_encoding(self):
        pe = torch.zeros(self.max_step_for_pe, self.hidden_dim)
        position = torch.arange(0, self.max_step_for_pe).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.hidden_dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.hidden_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe


def gather_nd(params, indices):
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    params = params.reshape((-1, *tuple(torch.tensor(params.size()[ndim:]))))
    return params[idx]


def gather_tree(step_ids, parent_ids, max_sequence_lengths, end_token):
    beams = torch.empty_like(step_ids)
    beams.fill_(end_token)
    max_len = step_ids.size(0)
    batch_size = step_ids.size(1)
    beam_size = step_ids.size(-1)
    batch_beam = batch_size * beam_size
    for i in range(batch_beam):
        batch = i // beam_size
        beam = i % beam_size
        max_seq_len_b = min(max_len, max_sequence_lengths[batch])
        if max_seq_len_b <= 0:
            continue
        beams[max_seq_len_b - 1, batch, beam] = step_ids[max_seq_len_b - 1, batch, beam]
        parent = parent_ids[max_seq_len_b - 1, batch, beam]
        for level in range(max_seq_len_b - 2, -1, -1):
            if parent < 0 or parent > beam_size:
                raise ValueError("wrong parent id")
            beams[level, batch, beam] = step_ids[level, batch, parent]
            parent = parent_ids[level, batch, parent]
        finished = False
        for time in range(max_seq_len_b):
            if finished:
                beams[time, batch, beam] = end_token
            elif beams[time, batch, beam] == end_token:
                finished = True
    return beams


def finalize(beam_size, output_ids, parent_ids, out_seq_lens, end_id, max_seq_len=None, args=None):
    out_seq_lens = torch.reshape(out_seq_lens, (-1, beam_size))
    max_lens = torch.max(out_seq_lens, 1)[0]
    if max_seq_len:
        shape = (max_seq_len, -1, beam_size)
    else:
        shape = (torch.max(max_lens), -1, beam_size)
    output_ids = torch.reshape(output_ids, shape)
    parent_ids = torch.reshape(parent_ids, shape)
    if output_ids.is_cuda:
        if args.ths:
            torch.classes.load_library(args.ths_path)
            ids = torch.ops.fastertransformer.gather_tree(output_ids.to(torch.int32), parent_ids.to(torch.int32), max_lens.to(torch.int32), end_id)
        else:
            sys.path.insert(0, os.path.abspath(args.module_path))
            from th_fastertransformer import gather_tree as gather_tree_cuda
            ids = gather_tree_cuda(output_ids.to(torch.int32), parent_ids.to(torch.int32), max_lens.to(torch.int32), end_id)
    else:
        ids = gather_tree(output_ids, parent_ids, max_lens, end_id)
    ids = torch.einsum('ijk->jki', ids)    # batch_size, beam_size, max_seq_len
    lengths = torch.eq(ids, end_id)
    lengths = 1 - lengths.to(output_ids.dtype)
    lengths = torch.sum(lengths, -1)
    return ids, lengths


class FTDecoderLayer(nn.Module):
    def __init__(self, head_num, head_size, weights, args):
        super().__init__()
        self.args = args
        if args.ths:
            torch.classes.load_library(args.ths_path)
            self.dec_layer = torch.classes.FasterTransformerDecoder(head_num, head_size, *weights)
        else:
            sys.path.insert(0, os.path.abspath(args.module_path))
            from th_fastertransformer import FasterTransformerDecoder
            self.dec_layer = FasterTransformerDecoder(head_num, head_size, *weights)
    
    def forward(self, inputs, memory, memory_seq_lens, self_cache, mem_cache):
        if self.args.data_type == 'fp16':
            self_cache_tmp = torch.zeros(2, 1, self_cache.size(2), self_cache.size(3), dtype=torch.half).cuda()
        else:
            self_cache_tmp = torch.zeros(2, 1, self_cache.size(2), self_cache.size(3)).cuda()
        self_cache = torch.cat([self_cache, self_cache_tmp], 1)
        output = self.dec_layer.forward(inputs, memory, memory_seq_lens, self_cache, mem_cache)
        return output, self_cache, mem_cache


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    Args:
        num_layers (int): number of encoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads, args):
        super(TransformerDecoder, self).__init__()

        self.args = args
        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             full_context_alignment=full_context_alignment,
             alignment_heads=alignment_heads)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings, args):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            args=args)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v, batch_dim)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.args.model_type == 'ori' or self.args.model_type == 'torch_decoding':
            if self.state["cache"] is not None:
                _recursive_map(self.state["cache"])
        if self.args.model_type == 'decoder_ext' or self.args.model_type == 'torch_decoding_with_decoder_ext':
            if self.state["cache"] is not None:
                _recursive_map(self.state["cache"], 2)

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]

        if self.args.model_type == 'ori' or self.args.model_type == 'torch_decoding':
            src_max_len = self.state["src"].shape[0]
            src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
            tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

            with_align = kwargs.pop('with_align', False)
            attn_aligns = []

            for i, layer in enumerate(self.transformer_layers):
                layer_cache = self.state["cache"]["layer_{}".format(i)] \
                    if step is not None else None
                output, attn, attn_align = layer(
                    output,
                    src_memory_bank,
                    src_pad_mask,
                    tgt_pad_mask,
                    layer_cache=layer_cache,
                    step=step,
                    with_align=with_align)
                if attn_align is not None:
                    attn_aligns.append(attn_align)
        elif self.args.model_type == 'decoder_ext' or self.args.model_type == 'torch_decoding_with_decoder_ext':
            src_lens_ = src_lens.to(torch.int)
            for i, layer in enumerate(self.transformer_layers):
                layer_cache = self.state["cache"]["layer_{}".format(i)]
                output, self_cache_, mem_cache_ = layer(output, src_memory_bank, src_lens_, layer_cache['self'], layer_cache['mem'])
                layer_cache['self'] = self_cache_
                layer_cache['mem'] = mem_cache_

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attns = {}
        # attn = attn.transpose(0, 1).contiguous()

        # attns = {"std": attn}
        # if self._copy:
        #     attns["copy"] = attn
        # if with_align:
        #     attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
        #     # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        if self.args.model_type == 'ori' or self.args.model_type == 'torch_decoding':
            for i, layer in enumerate(self.transformer_layers):
                layer_cache = {"memory_keys": None, "memory_values": None}
                if isinstance(layer.self_attn, AverageAttention):
                    layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                        device=memory_bank.device)
                else:
                    layer_cache["self_keys"] = None
                    layer_cache["self_values"] = None
                self.state["cache"]["layer_{}".format(i)] = layer_cache
        elif self.args.model_type == 'decoder_ext' or self.args.model_type == 'torch_decoding_with_decoder_ext':
            max_seq_len = memory_bank.size(0)
            for i in range(len(self.transformer_layers)):
                layer_cache = {}
                if self.args.data_type == 'fp16':
                    layer_cache['self'] = torch.zeros(2, 0, batch_size, depth, dtype=torch.half).cuda()
                    layer_cache['mem'] = torch.zeros(1, 2, batch_size, max_seq_len, depth, dtype=torch.half).cuda()
                else:
                    layer_cache['self'] = torch.zeros(2, 0, batch_size, depth).cuda()
                    layer_cache['mem'] = torch.zeros(1, 2, batch_size, max_seq_len, depth).cuda()
                self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)


class CustomDecoding(nn.Module):
    def __init__(self, layer_num, head_num, head_size, vocab_size, start_id, end_id, weights, beam_search_diversity_rate=0.0, args=None):
        super().__init__()
        hidden_dim = head_num * head_size
        self.end_id = end_id
        self.args = args
        if args.ths:
            torch.classes.load_library(os.path.abspath(args.ths_path))
            self.decoding = torch.classes.FasterTransformerDecoding(head_num, head_size, hidden_dim, layer_num, vocab_size, start_id, end_id, beam_search_diversity_rate, *weights.w)
        else:
            sys.path.insert(0, os.path.abspath(args.module_path))
            from th_fastertransformer import FasterTransformerDecoding
            self.decoding = FasterTransformerDecoding(head_num, head_size, hidden_dim, layer_num, vocab_size, start_id, end_id, beam_search_diversity_rate, *weights.w)
    
    def forward(self, batch_size, beam_size, max_seq_len, memory, memory_seq_lens):
        extended_memory = tile(memory, beam_size)
        extended_memory_seq_lens = tile(memory_seq_lens, beam_size)
        output_ids, parent_ids, out_seq_lens = self.decoding.forward(batch_size, beam_size, max_seq_len, extended_memory, extended_memory_seq_lens)
        parent_ids = parent_ids % beam_size
        beams, lengths = finalize(beam_size, output_ids, parent_ids, out_seq_lens, self.end_id, max_seq_len, args=self.args)
        return beams, lengths


class ArgHelper(object):
    def __init__(self, model_type=None, data_type=None, module_path=None, ths=False, ths_path=None):
        self.model_type = model_type
        self.data_type = data_type
        self.module_path = module_path
        self.ths = ths
        self.ths_path = ths_path


class TorchDecoding(nn.Module):
    def __init__(self, layer_num, head_num, head_size, vocab_size, start_id, end_id, weights,
                 beam_search_diversity_rate=0.0, args=None):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_dim = head_num * head_size
        self.start_id = start_id
        self.end_id = end_id
        self.vocab_size = vocab_size
        self.diversity_rate = beam_search_diversity_rate
        self.args = args
        emb = Embeddings(self.hidden_dim, vocab_size, 1, position_encoding=True)
        self.decoder = TransformerDecoder(layer_num, self.hidden_dim, head_num, 4*self.hidden_dim,
                                          False, 'scaled-dot', 0, 0, emb, 0, False, False, -3, 0, args)
        self.generator = nn.Linear(self.hidden_dim, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.module_path = args.module_path
        if args.model_type == 'torch_decoding':
            for i in range(layer_num):
                self.decoder.transformer_layers[i].layer_norm_1.weight.data = weights.w[0][i]
                self.decoder.transformer_layers[i].layer_norm_1.bias.data = weights.w[1][i]
                self.decoder.transformer_layers[i].self_attn.linear_query.weight.data = weights.w[2][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].self_attn.linear_keys.weight.data = weights.w[3][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].self_attn.linear_values.weight.data = weights.w[4][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].self_attn.linear_query.bias.data = weights.w[5][i]
                self.decoder.transformer_layers[i].self_attn.linear_keys.bias.data = weights.w[6][i]
                self.decoder.transformer_layers[i].self_attn.linear_values.bias.data = weights.w[7][i]
                self.decoder.transformer_layers[i].self_attn.final_linear.weight.data = weights.w[8][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].self_attn.final_linear.bias.data = weights.w[9][i]
                self.decoder.transformer_layers[i].layer_norm_2.weight.data = weights.w[10][i]
                self.decoder.transformer_layers[i].layer_norm_2.bias.data = weights.w[11][i]
                self.decoder.transformer_layers[i].context_attn.linear_query.weight.data = weights.w[12][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].context_attn.linear_keys.weight.data = weights.w[13][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].context_attn.linear_values.weight.data = weights.w[14][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].context_attn.linear_query.bias.data = weights.w[15][i]
                self.decoder.transformer_layers[i].context_attn.linear_keys.bias.data = weights.w[16][i]
                self.decoder.transformer_layers[i].context_attn.linear_values.bias.data = weights.w[17][i]
                self.decoder.transformer_layers[i].context_attn.final_linear.weight.data = weights.w[18][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].context_attn.final_linear.bias.data = weights.w[19][i]
                self.decoder.transformer_layers[i].feed_forward.layer_norm.weight.data = weights.w[20][i]
                self.decoder.transformer_layers[i].feed_forward.layer_norm.bias.data = weights.w[21][i]
                self.decoder.transformer_layers[i].feed_forward.w_1.weight.data = weights.w[22][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].feed_forward.w_1.bias.data = weights.w[23][i]
                self.decoder.transformer_layers[i].feed_forward.w_2.weight.data = weights.w[24][i].transpose(-1, -2).contiguous()
                self.decoder.transformer_layers[i].feed_forward.w_2.bias.data = weights.w[25][i]
        elif args.model_type == 'torch_decoding_with_decoder_ext':
            w = []
            for i in range(layer_num):
                w.append([weights.w[j][i].clone().detach() for j in range(26)])
                for i in range(len(w[-1])):
                    w[-1][i] = w[-1][i].cuda()
                if args.data_type == 'fp16':
                    for i in range(len(w[-1])):
                        w[-1][i] = w[-1][i].half()
            decoder_layers = nn.ModuleList(
                [FTDecoderLayer(head_num, head_size, w[i], args) for i in range(layer_num)])
            self.decoder.transformer_layers = decoder_layers
        else:
            raise ValueError('wrong model_type')
        self.decoder.layer_norm.weight.data = weights.w[26]
        self.decoder.layer_norm.bias.data = weights.w[27]
        self.decoder.embeddings.make_embedding.emb_luts[0].weight.data = weights.w[28]
        self.generator.weight.data = weights.w[30].transpose(-1, -2).contiguous()
        self.generator.bias.data = weights.w[31]

    def forward(self, batch_size, beam_size, max_seq_len, memory, memory_seq_lens):
        extended_memory = tile(memory, beam_size)
        batchxbeam = extended_memory.size(0)
        extended_memory = extended_memory.transpose(0, 1).contiguous()

        extended_memory_seq_lens = tile(memory_seq_lens, beam_size)
        start_ids = extended_memory_seq_lens.new_full((batchxbeam,), self.start_id, dtype=torch.int64)

        initial_log_probs = extended_memory.new_full((beam_size,), -float("inf"), dtype=torch.float32)
        initial_log_probs[0] = 0.
        initial_log_probs = initial_log_probs.repeat(batch_size)
        sequence_lengths = extended_memory_seq_lens.new_full((batchxbeam,), 0)
        finished = extended_memory_seq_lens.new_full((batchxbeam,), 0, dtype=torch.bool)

        dtype_info = torch.finfo(extended_memory.dtype)
        eos_max_prob = extended_memory.new_full((batchxbeam, self.vocab_size), dtype_info.min)
        eos_max_prob[:, self.end_id] = dtype_info.max

        self.decoder.init_state(extended_memory, extended_memory, None)
        word_ids = start_ids
        cum_log_probs = initial_log_probs

        for step in range(max_seq_len):
            if not torch.bitwise_not(finished).any():
                break
            word_ids = word_ids.view(1, -1, 1)
            dec_out, dec_attn = self.decoder(word_ids, extended_memory, memory_lengths=extended_memory_seq_lens, step=step)
            logits = self.generator(dec_out.squeeze(0))
            logits = torch.where(finished.view(-1, 1), eos_max_prob, logits).to(torch.float32)
            log_probs = self.logsoftmax(logits.to(torch.float32))

            total_probs = log_probs + torch.unsqueeze(cum_log_probs, 1)
            total_probs = total_probs.view(-1, beam_size * self.vocab_size)

            # beamsearch
            # _, sample_ids = torch.topk(total_probs, beam_size)
            # sample_ids = sample_ids.view(-1)

            #diversesiblingsearch
            sibling_score = torch.arange(1, beam_size+1).to(total_probs.dtype).to(extended_memory.device) * self.diversity_rate # [beam_size]
            scores, ids = torch.topk(total_probs.view(-1, beam_size, self.vocab_size), beam_size) # [batch size, beam width, beam width]
            scores = scores + sibling_score # [batch size, beam width, beam width]
            scores = scores.view(-1, beam_size * beam_size)
            ids = ids + torch.unsqueeze(torch.unsqueeze(torch.arange(0, beam_size).to(extended_memory.device) * self.vocab_size, 0), -1)
            ids = ids.view(-1, beam_size * beam_size)
            _, final_ids = torch.topk(scores, beam_size) # [batch size, beam size]
            final_ids = final_ids.view(-1, 1)
            batch_index = torch.arange(0, batch_size).to(extended_memory.device).view(-1, 1).repeat(1, beam_size).view(-1, 1)
            index = torch.cat([batch_index, final_ids], 1)
            sample_ids = gather_nd(ids, index)

            word_ids = sample_ids % self.vocab_size  # [batch_size * beam_size]
            beam_ids = sample_ids // self.vocab_size  # [batch_size * beam_size]
            beam_indices = (torch.arange(batchxbeam).to(extended_memory.device) // beam_size) * beam_size + beam_ids

            sequence_lengths = torch.where(finished, sequence_lengths, sequence_lengths + 1)

            batch_pos = torch.arange(batchxbeam).to(extended_memory.device) // beam_size
            next_cum_log_probs = gather_nd(total_probs, torch.stack([batch_pos, sample_ids], -1))  # [batch_size * beam_size]
            finished = finished.index_select(0, beam_indices)
            sequence_lengths = sequence_lengths.index_select(0, beam_indices)

            self.decoder.map_state(lambda state, dim: state.index_select(dim, beam_indices))
            if step == 0:
                parent_ids = beam_ids.view(1, -1)
                output_ids = word_ids.view(1, -1)
            else:
                parent_ids = torch.cat((parent_ids, beam_ids.view(1, -1)))
                output_ids = torch.cat((output_ids, word_ids.view(1, -1)))
            cum_log_probs = torch.where(finished, cum_log_probs, next_cum_log_probs)
            finished = torch.bitwise_or(finished, torch.eq(word_ids, self.end_id))
        
        beams, lengths = finalize(beam_size, output_ids, parent_ids, sequence_lengths, self.end_id, args=self.args)
        return beams, lengths
