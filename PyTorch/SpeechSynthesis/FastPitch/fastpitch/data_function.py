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

import numpy as np

import torch

from common.utils import to_gpu
from tacotron2.data_function import TextMelLoader


class TextMelAliLoader(TextMelLoader):
    """
    """
    def __init__(self, *args):
        super(TextMelAliLoader, self).__init__(*args)
        if len(self.audiopaths_and_text[0]) != 4:
            raise ValueError('Expected four columns in audiopaths file')

    def __getitem__(self, index):
        # separate filename and text
        audiopath, durpath, pitchpath, text = self.audiopaths_and_text[index]
        len_text = len(text)
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        dur = torch.load(durpath)
        pitch = torch.load(pitchpath)
        return (text, mel, len_text, dur, pitch)


class TextMelAliCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self):
        self.n_frames_per_step = 1  # Taco2 bckwd compat

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        dur_padded = torch.zeros_like(text_padded, dtype=batch[0][3].dtype)
        dur_lens = torch.zeros(dur_padded.size(0), dtype=torch.int32)
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]][3]
            dur_padded[i, :dur.shape[0]] = dur
            dur_lens[i] = dur.shape[0]

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (self.n_frames_per_step - max_target_len
                               % self.n_frames_per_step)
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        pitch_padded = torch.zeros(dur_padded.size(0), dur_padded.size(1),
                                   dtype=batch[0][4].dtype)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :pitch.shape[0]] = pitch

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return (text_padded, input_lengths, mel_padded,
                output_lengths, len_x, dur_padded, dur_lens, pitch_padded)


def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, \
        output_lengths, len_x, dur_padded, dur_lens, pitch_padded = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    dur_padded = to_gpu(dur_padded).long()
    dur_lens = to_gpu(dur_lens).long()
    pitch_padded = to_gpu(pitch_padded).float()
    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         dur_padded, dur_lens, pitch_padded]
    y = [mel_padded, dur_padded, dur_lens, pitch_padded]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
