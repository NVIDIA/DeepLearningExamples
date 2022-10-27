# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob,
                                    pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                    value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:query_lens[bid], :, :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(
                curr_logprob, target_seq, input_lengths=query_lens[bid:bid+1],
                target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
        cost = cost_total/attn_logprob.shape[0]
        return cost


class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention, eps=1e-12):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1],
                            min=eps)).sum()
        return -log_sum / hard_attention.sum()
