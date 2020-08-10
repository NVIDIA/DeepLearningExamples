# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

from fastspeech.align_tacotron2 import get_tacotron2, get_duration
from fastspeech.trainer.trainer import Trainer
from fastspeech.utils.pytorch import to_device_async, to_cpu_numpy
from torch.nn import functional as F


class FastspeechTrainer(Trainer):

    def __init__(self, data_loader, model_name, model, optimizer_fn, final_steps, lr_scheduler_fn=None, step=0, ckpt_path=None, log_path=None,
                 n_epochs=None, save_steps=None, log_steps=10, device='cuda', use_amp='O0', nvprof_iter_start=None, nvprof_iter_end=None, pyprof_enabled=False, detect_anomaly=False, seed=None, pre_aligns=True):
        super(FastspeechTrainer, self).__init__(data_loader, model_name, model, optimizer_fn, final_steps, lr_scheduler_fn, step, ckpt_path,
                                                log_path, n_epochs, save_steps, log_steps, device, use_amp, nvprof_iter_start, nvprof_iter_end, pyprof_enabled, detect_anomaly, seed)
        self.pre_aligns = pre_aligns

        if not pre_aligns:
            self.tacotron2 = get_tacotron2(device, is_training=True)
            to_device_async(self.tacotron2, device)

    def loss(self, inputs, model):
        text = inputs["text_encoded"]
        text_pos = inputs["text_pos"]
        mel_tgt = inputs["mel"]

        text = to_device_async(text, self.device)
        text_pos = to_device_async(text_pos, self.device)
        mel_tgt = to_device_async(mel_tgt, self.device)

        if self.pre_aligns:
            dur_tgt = inputs["align"]  # preprocessed align
            dur_tgt = dur_tgt.float()
            dur_tgt = to_device_async(dur_tgt, self.device)
        else:
            text_len = inputs['text_len']
            mel_len = inputs['mel_len']
            dur_tgt = get_duration(
                text, text_len, mel_tgt, mel_len, self.tacotron2, self.device)

        # (B,H,T) => (B,T,H)
        mel_tgt = mel_tgt.transpose(1, 2)

        # Forward
        mel, mask, dur = model(
            text,
            text_pos,
            duration_target=dur_tgt,
            seq_output_len=mel_tgt.size(1))
        assert(mel.size(1) == mel_tgt.size(1))

        # Loss
        mel_loss = F.mse_loss(mel, mel_tgt, reduction='none')
        mel_mask = mel_tgt.ne(0).float()
        mel_loss *= mel_mask
        mel_loss = mel_loss.mean()

        dur_tgt = torch.log(dur_tgt + 1)
        dur_mask = text_pos.ne(0).float()
        dur_tgt *= dur_mask

        dur_pred_loss = F.mse_loss(dur, dur_tgt)

        loss = mel_loss + dur_pred_loss

        meta = {
            'mel_loss': to_cpu_numpy(mel_loss),
            'duration_predictor_loss': to_cpu_numpy(dur_pred_loss),
        }
        # meta = {}

        return loss, meta
