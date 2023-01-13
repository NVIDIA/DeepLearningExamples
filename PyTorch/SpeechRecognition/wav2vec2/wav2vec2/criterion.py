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

import editdistance
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from common.fairseq import utils
from common.fairseq.data.data_utils import post_process
from common.utils import AttrDict


class Wav2vecCriterion(_Loss):
    def __init__(self, args):
        super().__init__(args)
        self.infonce = args.infonce
        self.loss_weights = args.loss_weights
        self.log_keys = [] if args.log_keys is None else args.log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"],
                           sub_batch_sizes=sample["sub_batch_sizes"],
                           sub_batch_lens=sample["sub_batch_lens"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)

        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        reduction = "sum" if reduce else "none"
        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )

        if 'sample_size' in sample:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        elif self.infonce:
            sample_size = target.numel()
        else:
            sample_size = target.long().sum().item()

        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)

            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]

            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)

            assert len(extra_losses) == len(self.loss_weights), \
                f"{len(extra_losses)}, {len(self.loss_weights)}"

            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        log_out = {
            "loss": loss.item() if reduce else loss.detach(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    log_out["logits"] = logits.cpu().numpy()

            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(
                            sample, net_output)
                    else:
                        original_target = target
                    log_out["target"] = original_target.cpu().numpy()

            elif lk in net_output:
                log_out[lk] = float(net_output[lk])

        if len(losses) > 1:
            for i, l in enumerate(losses):
                log_out[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max_ = logits.argmax(-1) == 0
                    min_ = logits.argmin(-1) == 0
                    both = max_ & min_
                    corr = max_.long().sum().item() - both.long().sum().item()
                    count = float(max_.numel())

                log_out["correct"] = corr
                log_out["count"] = count

        return loss, sample_size, log_out


class CTCCriterion(_Loss):
    def __init__(self, target_dictionary, blank_idx=0, pad_idx=1, eos_idx=2,
                 zero_infinity=True, sentence_avg=True, post_process='letter'):

        super().__init__()
        # keep all indexes for compatibility with fairseq
        self.blank_idx = blank_idx
        self.pad_idx = target_dictionary.pad()
        self.eos_idx = target_dictionary.eos()
        assert self.blank_idx != self.pad_idx != self.eos_idx

        self.target_dictionary = target_dictionary
        self.zero_infinity = zero_infinity
        self.sentence_avg = sentence_avg
        self.post_process = post_process

        # currently we don't support decoders (e.g., KenLM)
        self.w2l_decoder = None

    def forward(self, model, sample, reduce=True):
        net_out = model(**sample["net_input"])
        logp = model.get_normalized_probs(
            net_out["encoder_out"], net_out["padding_mask"], log_probs=True
        ).contiguous()

        T, B, _ = logp.size()

        if net_out["padding_mask"] is not None:
            lens = (~net_out["padding_mask"]).long().sum(-1)
        else:
            lens = logp.new_full((B,), T, dtype=torch.long)

        tgt = sample["target"]
        pad_mask = (tgt != self.pad_idx) & (tgt != self.eos_idx)
        tgt_flat = tgt.masked_select(pad_mask)
        tgt_lens = sample["target_lengths"]

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(logp, tgt_flat, lens, tgt_lens,
                              blank=self.blank_idx, reduction="sum",
                              zero_infinity=self.zero_infinity)
        log_out = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["id"].numel(),
            "sample_size": B if self.sentence_avg else sample["ntokens"]
        }

        if not model.training:
            log_out.update(self.calculate_wer(sample, logp, lens))

        return loss, log_out['sample_size'], log_out

    def calculate_wer(self, sample, logp, lens):
        with torch.no_grad():
            log = AttrDict({"wv_errs": 0, "w_errs": 0, "w_len": 0,
                            "c_errs": 0, "c_len": 0})

            logp_t = logp.transpose(0, 1).float().contiguous().cpu()
            tgt_labels = sample.get('target_label', sample['target'])

            head = lambda l: None if l is None or len(l) < 1 else l[0]

            for lp, L, tgt in zip(logp_t, lens, tgt_labels):
                lp = lp[:L].unsqueeze(0)

                if self.w2l_decoder is not None:
                    decoded = head(head(self.w2l_decoder.decode(lp)))
                else:
                    decoded = None

                mask = (tgt != self.pad_idx) & (tgt != self.eos_idx)
                tgt_units = self.target_dictionary.string(tgt[mask])
                tgt_units_arr = tgt[mask].tolist()

                toks = lp.argmax(dim=-1).unique_consecutive()
                pred_units_arr = toks[toks != self.blank_idx].tolist()

                log.c_errs += editdistance.eval(pred_units_arr, tgt_units_arr)
                log.c_len += len(tgt_units_arr)

                tgt_words = post_process(tgt_units, self.post_process).split()

                pred_units = self.target_dictionary.string(pred_units_arr)
                pred_words_raw = post_process(pred_units,
                                              self.post_process).split()

                if decoded is not None and "words" in decoded:
                    pred_words = decoded["words"]
                    log.w_errs += editdistance.eval(pred_words, tgt_words)
                    log.wv_errs += editdistance.eval(pred_words_raw, tgt_words)
                else:
                    dist = editdistance.eval(pred_words_raw, tgt_words)
                    log.w_errs += dist
                    log.wv_errs += dist

                log.w_len += len(tgt_words)

            return vars(log)
