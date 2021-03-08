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

from fastspeech.inferencer.inferencer import Inferencer
from fastspeech.utils.logging import tprint
from fastspeech.utils.tensorboard import imshow_to_buf
from fastspeech.utils.pytorch import to_device_async, to_cpu_numpy
from torch.nn import functional as F


class FastSpeechInferencer(Inferencer):

    def __init__(self, model_name, model, data_loader, ckpt_path=None, ckpt_file=None, log_path=None, device='cuda', use_fp16=False, seed=None):
        super(FastSpeechInferencer, self).__init__(model_name, model, data_loader, ckpt_path, ckpt_file, log_path, device, use_fp16, seed)

    def infer(self, acts=None, seq_input_len=None, seq_output_len=None):
        inputs = next(self.data_loader_iter)

        text_encoded = inputs["text_encoded"]
        text_pos = inputs["text_pos"]

        if seq_input_len:
            text_encoded = F.pad(text_encoded, pad=(0, seq_input_len - text_encoded.size(1)))  # (b, t)
            text_pos = F.pad(text_pos, pad=(0, seq_input_len - text_pos.size(1)))  # (b, t)

        text_encoded = to_device_async(text_encoded, self.device)
        text_pos = to_device_async(text_pos, self.device)

        mel, mel_mask, _ = self.model(
            seq=text_encoded,
            pos=text_pos,
            seq_output_len=seq_output_len,
            use_fp16=self.use_fp16,
            acts=acts
        )

        # (B,T,H) => (B,H,T)
        mel = mel.transpose(1, 2)
        mel_mask = mel_mask.squeeze(2)

        outputs = dict()
        outputs['mel'] = mel
        outputs['mel_mask'] = mel_mask
        outputs['text'] = inputs["text_norm"]

        if "mel" in inputs:
            outputs['mel_tgt'] = inputs["mel"]

        if "wav" in inputs:
            outputs['wav_tgt'] = inputs["wav"]

        if "sr" in inputs:
            outputs['sr'] = inputs["sr"]

        return outputs

    def console_log(self, tag, output):
        # console logging
        msg = ""
        for key, value in sorted(output.items()):
            msg += ',\t{}: {}'.format(key, value)
        tprint(msg)

    # TODO generalize
    def tensorboard_log(self, tag, output_tensor):
        self.tbwriter.add_image('{}/{}'.format(tag, "mel"), imshow_to_buf(output_tensor['mel']), global_step=self.step)
        self.tbwriter.add_image('{}/{}'.format(tag, "mel_tgt"), imshow_to_buf(output_tensor['mel_tgt']), global_step=self.step)
        self.tbwriter.add_audio('{}/{}'.format(tag, "wav_tgt"), output_tensor['wav_tgt'], global_step=self.step, sample_rate=int(output_tensor['sr']))
        self.tbwriter.add_text('{}/{}'.format(tag, "text"), output_tensor['text'], global_step=self.step)