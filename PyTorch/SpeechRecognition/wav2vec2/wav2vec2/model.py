# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

import contextlib
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common import fairseq_fake_modules
from common.fairseq import utils
from common.fairseq.data.data_utils import compute_mask_indices
from common.fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    Fp32MaskedGroupNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MaskedGroupNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from common.features import FilterbankFeatures
from common.helpers import load_wrapped_state
from common.pyt_mha import PytMultiheadAttention
from common.utils import print_once


class Fp32Conv1d(nn.Conv1d):
    """Casts to FP32. TorchScript ready, does not use inheritance.

    Details: https://github.com/pytorch/pytorch/issues/42885 .
    """

    def forward(self, x):
        return F.conv1d(
            x.float(), self.weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups
        ).to(dtype=x.dtype)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
    if isinstance(module, PytMultiheadAttention):
        normal_(module.qkv.weight.data)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class MaskedBlock(nn.Module):

    def __init__(self, *args):
        super().__init__()

        self.conv = args[0]
        self.drop = args[1]
        if len(args) == 4:
            self.norm = args[2]
            self.activation = args[3]
        else:
            self.norm = None
            self.activation = args[2]

        def hook(state_dict, prefix, *args, **kwargs):
            """Rename Blocks saved as nn.Sequential."""
            new_sd = {}
            for k, v in state_dict.items():
                if not k.startswith(prefix):
                    new_sd[k] = v
                else:
                    *pref, feat, conv, mod_num, layer_num, param = k.split(".")
                    assert feat == "feature_extractor" and conv == "conv_layers"
                    if layer_num == "0":
                        new_k = ".".join(pref + [feat, conv, mod_num, "conv", param])
                    elif layer_num == "2":
                        new_k = ".".join(pref + [feat, conv, mod_num, "norm", param])
                    else:
                        raise ValueError
                    print(f"Rename {k} --> {new_k}")
                    new_sd[new_k] = v
            state_dict.clear()
            state_dict.update(new_sd)

        self._register_load_state_dict_pre_hook(hook)

    def forward(self, x: Tensor, x_lens: Tensor):
        x = self.drop(self.conv(x))
        x_lens = (x_lens - self.conv.kernel_size[0]) / self.conv.stride[0] + 1
        x_lens = torch.floor(x_lens).long()

        if self.norm is not None:
            if isinstance(self.norm, nn.Sequential):
                # LayerNorm wraped with nn.Sequential
                raise ValueError("LayerNorm does not require masking")
            else:
                x = self.norm(x, x_lens)
        return self.activation(x), x_lens


class Wav2Vec2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_spectrogram_features = cfg.use_spectrogram_features

        if self.use_spectrogram_features:
            self.spec_feature_extractor = FilterbankFeatures(
                frame_stacking=cfg.spectrogram_feature_stacking,
                frame_subsampling=cfg.spectrogram_feature_subsampling,
                window_size=cfg.spectrogram_window_size,
                window_stride=cfg.spectrogram_window_stride,
                n_filt=cfg.spectrogram_n_filt).cuda()
            self.feature_extractr = None
            self.spec_feature_extractor.eval()
            self.embed = self.spec_feature_extractor.output_dim()
        else:
            feature_enc_layers = eval(cfg.conv_feature_layers)
            self.embed = feature_enc_layers[-1][0]
            self.spec_feature_extractor = None
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
                fp32_norms=cfg.fp32_conv_norms,
                masked=getattr(cfg, 'masked_feature_extractor', False),
            )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        self.fp32_cosine_sim = cfg.fp32_cosine_sim

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                    weight_proj_depth=cfg.quantizer_depth,
                    weight_proj_factor=cfg.quantizer_factor,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        self.conv_cfg_list = eval(self.cfg.conv_feature_layers)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=True,
                    mask_dropout=0.0,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = torch.arange(num, device=y.device).unsqueeze(-1)
                tszs = tszs.expand(-1, self.n_negatives).flatten()

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num),
                    device=y.device
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = torch.arange(num, device=y.device).unsqueeze(-1)
                tszs = tszs.expand(-1, self.cross_sample_negatives).flatten()

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                    device=y.device
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        if self.fp32_cosine_sim:
            logits = torch.cosine_similarity(x.float(), targets.float(),
                                             dim=-1).type_as(x)
        else:
            logits = torch.cosine_similarity(x, targets, dim=-1)

        logits = logits / self.logit_temp

        if neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                self._inftensor = float("-inf")
            logits[1:][neg_is_pos] = self._inftensor

        return logits

    def _conv_out_length(self, input_length: torch.Tensor, kernel_size: int, stride: int):
        return torch.floor((input_length - kernel_size) / stride + 1)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        for i in range(len(self.conv_cfg_list)):
            input_lengths = self._conv_out_length(
                input_lengths,
                self.conv_cfg_list[i][1],
                self.conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def infer(self, source: Tensor, padding_mask: Tensor):
        """Forward method for (masked) inference."""

        input_lengths = (1 - padding_mask.long()).sum(-1)
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)

        features, _ = self.feature_extractor.masked_forward(source, input_lengths)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        padding_mask = torch.zeros(
            features.shape[:2], dtype=features.dtype, device=features.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        padding_mask[
            (
                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,
            )
        ] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])) == 1

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)
        x, _ = self.encoder(x, padding_mask=padding_mask)
        return x, padding_mask

    def forward(
        self,
        source,
        padding_mask: Optional[Tensor] = None,
        mask=True,
        features_only=False,
        layer=-1,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        sub_batch_sizes=None,
        sub_batch_lens=None,
    ):
        masked_inference = self.feature_extractor.masked

        if self.spec_feature_extractor is not None:
            if padding_mask is not None and padding_mask.any():
                input_lengths = (1 - padding_mask.long()).sum(-1)
            else:
                input_lengths = (torch.zeros(source.size(0)) + source.size(1)).cuda()

            features, output_lengths = self.spec_feature_extractor(source, input_lengths)
            output_lengths = output_lengths.to(torch.long)

        else:
            if self.training and self.feature_grad_mult > 0:
                features = self.feature_extractor(source)
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    if masked_inference:
                        input_lengths = (1 - padding_mask.long()).sum(-1)
                        features, _ = self.feature_extractor.masked_forward(source, input_lengths)
                    else:
                        features = self.feature_extractor(source)

            if masked_inference or (padding_mask is not None and padding_mask.any()):
                input_lengths = (1 - padding_mask.long()).sum(-1)
                # apply conv formula to get real output_lengths
                output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            else:
                output_lengths = None

        features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if output_lengths is not None:

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])) == 1
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        split_accumulation = sub_batch_sizes is not None and sub_batch_sizes.size(0) > 1
        if split_accumulation:
            assert sub_batch_sizes is not None
            assert self.quantizer is not None
            assert not self.negatives_from_everywhere
            assert self.codebook_negatives == 0
            assert self.target_glu is None
            assert mask_indices is None
            assert mask_channel_indices is None
            assert mask

            split_sizes = sub_batch_sizes.tolist()
            sub_x, sub_y, sub_mask_indices, sub_negs = [], [], [], []

            for s, e in zip(np.cumsum(split_sizes) - split_sizes, np.cumsum(split_sizes)):
                x_, mask_indices_ = self.apply_mask(
                    features[s:e],
                    padding_mask[s:e] if padding_mask is not None else None,
                )
                sub_x.append(x_)
                sub_mask_indices.append(mask_indices_)
                y_ = unmasked_features[s:e][mask_indices_].view(
                     e-s, -1, unmasked_features.size(-1)
                )

                q_ = self.quantizer(y_, produce_targets=False)
                y_ = q_["x"]
                y_ = self.project_q(y_)

                negs_, _ = self.sample_negatives(
                    y_,
                    y_.size(1),
                    padding_count=padding_count,
                )
                sub_y.append(y_)
                sub_negs.append(negs_)

            x = torch.cat(sub_x, dim=0)
            mask_indices = torch.cat(sub_mask_indices, dim=0)

            x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

            if features_only:
                return {
                    "x": x,
                    "padding_mask": padding_mask,
                    "features": unmasked_features,
                    "layer_results": layer_results,
                }

            x = x[mask_indices]  # .view(x.size(0), -1, x.size(-1))
            x = self.final_proj(x)

            # At this point, x needs to be smartly reshaped / split into x_'s

            sub_x2 = []
            offset = 0
            for y_, mask_inds_, negs_ in zip(sub_y, sub_mask_indices, sub_negs):
                sz = mask_inds_.sum()
                x_ = x[offset:offset+sz].view(mask_inds_.size(0), -1, x.size(-1))
                x_ = self.compute_preds(x_, y_, negs_)
                sub_x2.append(x_)
                offset += sz

            x = torch.cat([x_.view(x_.size(0), 1, -1) for x_ in sub_x2], dim=2)

            result = {
                "x": x,
                "padding_mask": padding_mask,
                "features_pen": features_pen,
            }

            # TODO Reassemble q stats, currently using first chunk's stats
            q = q_

            if q["prob_perplexity"] is not None:
                result["prob_perplexity"] = q["prob_perplexity"]
                result["code_perplexity"] = q["code_perplexity"]
                result["num_vars"] = q["num_vars"]
                result["temp"] = q["temp"]

            return result

        # End split_accumulation ----------------------------------------------

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands = self.quantizer(unmasked_features, produce_targets=False)[
                    "x"
                ]
                negs, _ = self.sample_negatives(
                    neg_cands,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False, layer=-1):
        res = self.forward(
            source, padding_mask, mask=mask, features_only=True, layer=layer
        )
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(
        self,
        state_dict,
        strict=True,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        new_state_dict = state_dict

        return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

    def prepare_for_inference_(self, cfg):
        """Prepare model for inference."""
        kwargs = {}
        kwargs["beamable_mm_beam_size"] = (
            None
            if getattr(cfg.generation, "no_beamable_mm", False)
            else getattr(cfg.generation, "beam", 5)
        )
        kwargs["need_attn"] = getattr(cfg.generation, "print_alignment", False)
        if getattr(cfg.generation, "retain_dropout", False):
            kwargs["retain_dropout"] = cfg.generation.retain_dropout
            kwargs["retain_dropout_modules"] = cfg.generation.retain_dropout_modules
        self.make_generation_fast_(**kwargs)

    def make_generation_fast_(self, **kwargs):
        """
        Legacy entry point to optimize model for faster generation.
        Prefer prepare_for_inference_.
        """
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def train(mode=True):
            if mode:
                raise RuntimeError("cannot train after make_generation_fast")

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if (
                module != self
                and hasattr(module, "prepare_for_onnx_export_")
                and module not in seen
            ):
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    def remove_conv_wn(self):
        nn.utils.remove_weight_norm(self.encoder.pos_conv[0])

    def apply_conv_wn(self):
        nn.utils.weight_norm(self.encoder.pos_conv[0], name="weight", dim=2)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        fp32_norms: bool = True,
        masked: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}
        self.mode = mode
        self.masked = masked

        LayerNorm_ = Fp32LayerNorm if fp32_norms else nn.LayerNorm
        if masked and mode == "default":
            Block_ = MaskedBlock
            GroupNorm_ = Fp32MaskedGroupNorm if fp32_norms else MaskedGroupNorm
        else:
            Block_ = nn.Sequential
            GroupNorm_ = Fp32GroupNorm if fp32_norms else nn.GroupNorm

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            assert not (is_layer_norm and is_group_norm), (
                "layer norm and group norm are mutually exclusive")

            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            def make_norm():
                if is_group_norm:
                    l = GroupNorm_(dim, dim, affine=True)
                elif is_layer_norm:
                    l = nn.Sequential(TransposeLast(),
                                      LayerNorm_(dim, elementwise_affine=True),
                                      TransposeLast())
                return l

            has_norm = is_layer_norm or is_group_norm
            return Block_(make_conv(),
                          nn.Dropout(p=dropout),
                          *([make_norm()] if has_norm else []),
                          nn.GELU())
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)
        return x

    def masked_forward(self, x: Tensor, x_lens: Tensor):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x, x_lens = conv(x, x_lens)
        return x, x_lens


class Upsampler(nn.Module):
    def __init__(self, emb_dim, factor, mode="linear"):
        super().__init__()
        assert mode in ("linear", "naive")
        self.factor = factor
        if mode == "linear":
            self.linear = nn.Linear(emb_dim, emb_dim * factor)
        else:
            self.linear = None

    def forward(self, x):
        if self.linear is not None:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            x = self.linear(x)
            x = x.reshape(x.size(0), x.size(1) * self.factor, -1)
            x = x.transpose(0, 1)
        else:
            x = x.repeat_interleave(self.factor, dim=0)

        return x


class Downsampler(nn.Module):
    def __init__(self, emb_dim, factor, mode="linear"):
        super().__init__()
        assert mode in ("linear", "naive")
        self.factor = factor
        if mode == "linear":
            self.linear = nn.Linear(emb_dim * factor, emb_dim)
        else:
            self.linear = None

    def forward(self, x):
        if self.linear is not None:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            B, T, C = x.size()
            x = x.reshape(B, T // self.factor, C * self.factor)
            x = self.linear(x)
            x = x.transpose(0, 1)
        else:
            # T x B x C -> B x C x T
            x = x.permute(1, 2, 0)
            x = F.avg_pool1d(x, kernel_size=self.factor, stride=self.factor)
            x = x.permute(2, 0, 1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        PosConv = Fp32Conv1d if args.fp32_pos_conv else nn.Conv1d

        self.pos_conv = PosConv(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        def create_decoder_layers(n_layers):
            return nn.ModuleList([
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    rotary_embeddings=args.rotary_embeddings,
                    mha=args.mha,
                    fp32_transformer_layernorm=args.fp32_transformer_layernorm,
                    fp32_mha_softmax=args.fp32_mha_softmax,
                )
                for _ in range(n_layers)
            ])

        if args.hourglass_transformer:
            n_pre, (n_hourglass, self.shorten_factor), n_post = eval(
                args.hourglass_transformer)

            self.layers = create_decoder_layers(n_pre)
            self.hourglass_layers = create_decoder_layers(n_hourglass)
            self.post_layers = create_decoder_layers(n_post)

            assert args.hourglass_resample in ['linear', 'naive']
            # otherwise i want to resample before merging resutls
            assert not args.layer_norm_first

            kw = {'emb_dim': self.embedding_dim, 'factor': self.shorten_factor,
                  'mode': args.hourglass_resample}
            self.upsample_layer = Upsampler(**kw)
            self.downsample_layer = Downsampler(**kw)
        else:
            self.layers = create_decoder_layers(args.encoder_layers)
            self.hourglass_layers = None
            self.post_layers = None

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None,
                layer: int = -1):

        x, layer_results = self.extract_features(x, padding_mask, layer)

        if self.layer_norm_first and layer == -1:
            x = self.layer_norm(x)

        return x, layer_results

    def process_layers(self, x: Tensor, padding_mask: Optional[Tensor],
                       tgt_layer: int = -1):

        for i, layer in enumerate(self.layers):
            if not self.training or (torch.rand(1) > self.layerdrop):
                x, _ = layer(x, self_attn_padding_mask=padding_mask,
                             need_weights=False)
            if i == tgt_layer:
                return x
        return x

    def process_hourglass_layers(self, x: Tensor, padding_mask:
                                 Optional[Tensor], tgt_layer: int = -1):

        for i, layer in enumerate(self.hourglass_layers):
            if not self.training or (torch.rand(1) > self.layerdrop):
                x, _ = layer(x, self_attn_padding_mask=padding_mask,
                             need_weights=False)
            if i == tgt_layer:
                return x
        return x

    def process_post_layers(self, x: Tensor, padding_mask: Optional[Tensor],
                            tgt_layer: int = -1):

        if self.post_layers is None:
            return x
        else:
            for i, layer in enumerate(self.post_layers):
                if not self.training or (torch.rand(1) > self.layerdrop):
                    x, _ = layer(x, self_attn_padding_mask=padding_mask,
                                 need_weights=False)
                if i == tgt_layer:
                    return x
            return x

    def extract_features(self, x: Tensor, padding_mask: Optional[Tensor] = None,
                         tgt_layer: int = -1):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if self.hourglass_layers is not None:
            # we don't want to take outputs from inside of hourglass
            # as they are shortened and differnt
            n_layers_before_upsampling = (len(self.layers)  # pre layers
                                          + len(self.hourglass_layers))
            assert tgt_layer == -1 or tgt_layer >= n_layers_before_upsampling
            if tgt_layer is not None:
                tgt_layer = tgt_layer - n_layers_before_upsampling

            x = self.process_layers(x, padding_mask)
            res = x
            hourglass_pad_mask = padding_mask

            diff = ((self.shorten_factor - x.size(0) % self.shorten_factor)
                    % self.shorten_factor)

            if diff != 0:
                x = torch.cat([x, x.new_zeros(diff, x.size(1), x.size(2))])

            if hourglass_pad_mask is not None:
                if diff != 0:
                    hourglass_pad_mask = torch.cat([
                        hourglass_pad_mask,
                        x.new_ones(hourglass_pad_mask.size(0), diff)
                    ], dim=1)

                hourglass_pad_mask = (F.avg_pool1d(
                    hourglass_pad_mask.unsqueeze(0).float(),
                    self.shorten_factor,
                    self.shorten_factor
                ).int() > 0).squeeze(0)

            x = self.downsample_layer(x)
            x = self.process_hourglass_layers(x, hourglass_pad_mask)
            x = self.upsample_layer(x)

            if diff != 0:
                x = x[:-diff]

            x = x + res
            x = self.process_post_layers(x, padding_mask, tgt_layer)
        else:
            x = self.process_layers(x, padding_mask, tgt_layer)

        # T x B x C -> B x T x C
        return x.transpose(0, 1), []

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        rotary_embeddings: bool = False,
        mha: str = 'fairseq',
        fp32_transformer_layernorm: bool = False,
        fp32_mha_softmax: bool = False,
    ) -> None:

        assert not fp32_mha_softmax, "Support for FP32 MHA Softmax disabled"

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)

        MHA = {'fairseq': MultiheadAttention,
               'pyt': PytMultiheadAttention}[mha]

        self.self_attn = MHA(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            rotary_embeddings=rotary_embeddings
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        LN = Fp32LayerNorm if fp32_transformer_layernorm else LayerNorm

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LN(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LN(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class Wav2VecEncoder(nn.Module):
    def __init__(self, cfg, init_state_dict=None, output_size=None):
        super().__init__()

        self.apply_mask = cfg.apply_mask
        self.w2v_model = Wav2Vec2Model(cfg)

        if init_state_dict is not None:
            load_wrapped_state(self.w2v_model, init_state_dict)

        self.w2v_model.remove_pretraining_modules()

        d = cfg.encoder_embed_dim

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        tgt_d = None
        self.proj = None

        if output_size is not None:
            tgt_d = output_size
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            tgt_d = cfg.decoder_embed_dim

        if tgt_d is not None:
            self.proj = Linear(d, tgt_d)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates

    def extract_features(self, source, padding_mask, layer):

        assert not self.training

        with torch.no_grad():
            out = self.w2v_model.extract_features(
                source=source, padding_mask=padding_mask, mask=False,
                layer=layer)

        return out

    def infer(self, source: Tensor, padding_mask: Optional[Tensor],
              tbc: bool = True):
        assert padding_mask is not None

        x, padding_mask = self.w2v_model.infer(source, padding_mask)

        if tbc:
            # BTC -> TBC
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj is not None:
            x = self.proj(x)

        return x, padding_mask

    def forward(self, source: Tensor, padding_mask: Optional[Tensor],
                tbc: bool = True):

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            res = self.w2v_model.extract_features(
                source=source,
                padding_mask=padding_mask,
                mask=self.apply_mask and self.training
            )

        x = res["x"]
        padding_mask = res["padding_mask"]
        layer_results = res["layer_results"]

        if tbc:
            # BTC -> TBC
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj is not None:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask.transpose(0, 1)
            if padding_mask is not None
            else None,  # T x B
            "padding_mask": padding_mask,
            "layer_results": layer_results,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class Wav2VecCtc(nn.Module):
    def __init__(self, cfg, w2v_encoder):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @torch.jit.export
    def get_logits(self, logits: Tensor, padding_mask: Optional[Tensor], normalize: bool = False):

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise ValueError(f"invalid blank mode {self.blank_mode}")

        if padding_mask is not None and padding_mask.any():
            num_classes = logits.size(-1)
            masking_tensor = torch.full((num_classes,), float("-inf"),
                                        dtype=logits.dtype, device=logits.device)
            masking_tensor[0] = 0
            logits[padding_mask.T] = masking_tensor

        if normalize:
            logits = F.log_softmax(logits.float(), dim=-1)

        return logits

    @torch.jit.export
    def get_normalized_probs(self, logits: Tensor, padding_mask: Optional[Tensor], log_probs: bool):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(logits, padding_mask, normalize=False)

        if log_probs:
            return F.log_softmax(logits.float(), dim=-1)
        else:
            return F.softmax(logits.float(), dim=-1)

    def forward(self, source: Tensor, padding_mask: Optional[Tensor],
                tbc: bool = True):
        return self.w2v_encoder(source, padding_mask, tbc)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.w2v_encoder.set_num_updates(num_updates)
