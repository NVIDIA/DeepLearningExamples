# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
import torch.nn.functional as F
from torch import nn

from common.utils import mask_from_lens


class FastPitchLoss(nn.Module):
    def __init__(self, dur_predictor_loss_scale=1.0,
                 pitch_predictor_loss_scale=1.0):
        super(FastPitchLoss, self).__init__()
        self.dur_predictor_loss_scale = dur_predictor_loss_scale
        self.pitch_predictor_loss_scale = pitch_predictor_loss_scale

    def forward(self, model_out, targets, is_training=True, meta_agg='mean'):
        mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred = model_out
        mel_tgt, dur_tgt, dur_lens, pitch_tgt = targets
        mel_tgt.requires_grad = False
        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        dur_mask = mask_from_lens(dur_lens, max_len=dur_tgt.size(1))
        log_dur_tgt = torch.log(dur_tgt.float() + 1)
        loss_fn = F.mse_loss
        dur_pred_loss = loss_fn(log_dur_pred, log_dur_tgt, reduction='none')
        dur_pred_loss = (dur_pred_loss * dur_mask).sum() / dur_mask.sum()

        ldiff = mel_tgt.size(1) - mel_out.size(1)
        mel_out = F.pad(mel_out, (0, 0, 0, ldiff, 0, 0), value=0.0)
        mel_mask = mel_tgt.ne(0).float()
        loss_fn = F.mse_loss
        mel_loss = loss_fn(mel_out, mel_tgt, reduction='none')
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()

        ldiff = pitch_tgt.size(1) - pitch_pred.size(1)
        pitch_pred = F.pad(pitch_pred, (0, ldiff, 0, 0), value=0.0)
        pitch_loss = F.mse_loss(pitch_tgt, pitch_pred, reduction='none')
        pitch_loss = (pitch_loss * dur_mask).sum() / dur_mask.sum()

        loss = mel_loss
        loss = (mel_loss + pitch_loss * self.pitch_predictor_loss_scale
                + dur_pred_loss * self.dur_predictor_loss_scale)

        meta = {
            'loss': loss.clone().detach(),
            'mel_loss': mel_loss.clone().detach(),
            'duration_predictor_loss': dur_pred_loss.clone().detach(),
            'pitch_loss': pitch_loss.clone().detach(),
            'dur_error': (torch.abs(dur_pred - dur_tgt).sum()
                          / dur_mask.sum()).detach(),
        }
        assert meta_agg in ('sum', 'mean')
        if meta_agg == 'sum':
            bsz = mel_out.size(0)
            meta = {k: v * bsz for k,v in meta.items()}
        return loss, meta
