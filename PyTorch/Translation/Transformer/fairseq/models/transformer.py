# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict

from fairseq.modules import (
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, register_model,
    register_model_architecture,
)

from apex.normalization.fused_layer_norm import FusedLayerNorm

torch.set_printoptions(threshold=500000000, linewidth=1024)


@torch.jit.script
def jit_dropout_add(x, residual, prob, is_training):
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = F.dropout(x, p=prob, training=is_training)
    out = residual + out
    return out


@torch.jit.script
def jit_relu_dropout(x, prob, is_training):
    # type: (Tensor, float, bool) -> Tensor
    out = F.threshold(x, 0., 0.)
    out = F.dropout(out, p=prob, training=is_training)
    return out


@register_model('transformer')
class TransformerModel(nn.Module):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

    def __init__(self, encoder, decoder):
        super().__init__()
        self._is_generation_fast = False
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_model(cls, args):
        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        if args.share_all_embeddings:
            if args.src_vocab_size != args.tgt_vocab_size:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = Embedding(args.src_vocab_size, args.encoder_embed_dim, args.padding_idx)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = Embedding(args.src_vocab_size, args.encoder_embed_dim, args.padding_idx)
            decoder_embed_tokens = Embedding(args.tgt_vocab_size, args.decoder_embed_dim, args.padding_idx)

        encoder = TransformerEncoder(args, encoder_embed_tokens)
        decoder = TransformerDecoder(args, decoder_embed_tokens)

        return TransformerModel(encoder, decoder)

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_'):
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out, padding_mask = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, padding_mask)
        return decoder_out


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, args, embed_tokens, left_pad=True):
        super().__init__()
        self.dropout = args.dropout
        self.fuse_dropout_add = args.fuse_dropout_add
        self.fuse_relu_dropout = args.fuse_relu_dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = FusedLayerNorm(embed_dim) if args.fuse_layer_norm else nn.LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        # The tensor needs to copy transposed because
        # fused dropout is not capable of handing strided data
        if self.fuse_dropout_add:
            x = x.transpose(0, 1).contiguous()
        else:
            x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            _encoder_padding_mask = None
        else:
            _encoder_padding_mask = encoder_padding_mask

        # encoder layers
        for layer in self.layers:
            x = layer(x, _encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return x, encoder_padding_mask # x.shape == T x B x C, encoder_padding_mask.shape == B x T

    def reorder_encoder_out(self, encoder_out, encoder_padding_mask, new_order):
        if encoder_out is not None:
            encoder_out = encoder_out.index_select(1, new_order)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)
        return encoder_out, encoder_padding_mask


class TransformerDecoder(FairseqIncrementalDecoder):
    """Transformer decoder."""

    def __init__(self, args, embed_tokens, no_encoder_attn=False, left_pad=False):
        super().__init__()
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.fuse_dropout_add = args.fuse_dropout_add
        self.fuse_relu_dropout = args.fuse_relu_dropout

        embed_dim = embed_tokens.embedding_dim
        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(args.tgt_vocab_size, embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)
        else:
            self.embed_out = self.embed_tokens.weight
        self.normalize = args.decoder_normalize_before
        if self.normalize:
            self.layer_norm = FusedLayerNorm(embed_dim) if args.fuse_layer_norm else nn.LayerNorm(embed_dim)

    def forward(self,
                prev_output_tokens: Tensor,
                encoder_out: Tensor,
                encoder_padding_mask: Tensor,
                incremental_state: Optional[Dict[str, Dict[str, Tensor]]]=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        # The tensor needs to copy transposed because
        # fused dropout is not capable of handing strided data
        if self.fuse_dropout_add:
            x = x.transpose(0, 1).contiguous()
        else:
            x = x.transpose(0, 1)
        attn = None

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out,
                encoder_padding_mask if encoder_padding_mask.any() else None,
                incremental_state,
            )

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = F.linear(x, self.embed_out)

        return x, attn


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.fuse_dropout_add = args.fuse_dropout_add
        self.fuse_relu_dropout = args.fuse_relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.maybe_ln1 = MaybeLayerNorm(self.embed_dim, self.normalize_before, fuse=args.fuse_layer_norm)
        self.maybe_ln2 = MaybeLayerNorm(self.embed_dim, self.normalize_before, fuse=args.fuse_layer_norm)

    def forward(self, x: Tensor, encoder_padding_mask: Optional[Tensor]):
        residual = x

        x = self.maybe_ln1(x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x,
                              mask_future_timesteps=False,
                              key_padding_mask=encoder_padding_mask,
                              incremental_state=None,
                              need_weights=False,
                              static_kv=False)

        if self.fuse_dropout_add and self.training:
            x = jit_dropout_add(x, residual, self.dropout, self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        x = self.maybe_ln1(x, after=True)

        residual = x
        x = self.maybe_ln2(x, before=True)

        if self.fuse_relu_dropout:
            x = jit_relu_dropout(self.fc1(x), self.relu_dropout, self.training)
        else:
            x = F.threshold(self.fc1(x), 0.0, 0.0)
            x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)

        if self.fuse_dropout_add and self.training:
            x = jit_dropout_add(x, residual, self.dropout, self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        x = self.maybe_ln2(x, after=True)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        self.fuse_dropout_add = args.fuse_dropout_add
        self.fuse_relu_dropout = args.fuse_relu_dropout

        self.self_attn_layer_norm = MaybeLayerNorm(
            self.embed_dim, self.normalize_before, fuse=args.fuse_layer_norm)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = MaybeLayerNorm(
                self.embed_dim, self.normalize_before, fuse=args.fuse_layer_norm)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = MaybeLayerNorm(
            self.embed_dim, self.normalize_before, fuse=args.fuse_layer_norm)
        self.need_attn = True

    def forward(self,
                x: Tensor,
                encoder_out: Tensor,
                encoder_padding_mask: Optional[Tensor],
                incremental_state: Optional[Dict[str, Dict[str, Tensor]]]):
        residual = x
        x = self.self_attn_layer_norm(x, before=True)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            key_padding_mask=None,
            incremental_state=incremental_state,
            need_weights=False,
            static_kv=False
        )
        if self.fuse_dropout_add and self.training:
            x = jit_dropout_add(x, residual, self.dropout, self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        x = self.self_attn_layer_norm(x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.encoder_attn_layer_norm(x, before=True)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                mask_future_timesteps=False,
                need_weights=(not self.training and self.need_attn),
            )
            if self.fuse_dropout_add and self.training:
                x = jit_dropout_add(x, residual, self.dropout, self.training)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + x
            x = self.encoder_attn_layer_norm(x, after=True)

        residual = x
        x = self.final_layer_norm(x, before=True)
        if self.fuse_relu_dropout:
            x = jit_relu_dropout(self.fc1(x), self.relu_dropout, self.training)
        else:
            x = F.threshold(self.fc1(x), 0.0, 0.0)
            x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        if self.fuse_dropout_add and self.training:
            x = jit_dropout_add(x, residual, self.dropout, self.training)
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        x = self.final_layer_norm(x, after=True)
        return x, attn

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class MaybeLayerNorm(nn.Module):
    def __init__(self, embed_dim, normalize_before, fuse=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.normalize_before = normalize_before
        self.ln = FusedLayerNorm(embed_dim) if fuse else nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, before: bool = False, after: bool = False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.ln(x)
        else:
            return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)


@register_model_architecture('transformer', 'transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)
