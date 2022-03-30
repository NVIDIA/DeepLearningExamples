# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  MIT License
#
#  Copyright (c) 2020 Jungil Kong
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# The following functions/classes were based on code from https://github.com/jik876/hifi-gan:
# mel_spectrogram, MelDataset

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from numpy import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common.audio_processing import dynamic_range_compression
from common.utils import load_filepaths_and_text, load_wav

MAX_WAV_VALUE = 32768.0

mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size,
                    fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    fmax_key = f'{fmax}_{y.device}'
    if fmax_key not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft-hop_size)/2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann_window[str(y.device)], center=center,
                      pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)

    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = dynamic_range_compression(spec)  # spectral normalize
    return spec


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True,
                 device=None, fmax_loss=None, fine_tuning=False,
                 base_mels_path=None, repeat=1, deterministic=False,
                 max_wav_value=MAX_WAV_VALUE):

        self.audio_files = training_files
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.max_wav_value = max_wav_value
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.repeat = repeat
        self.deterministic = deterministic
        self.rng = random.default_rng()

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('Dataset index out of range')
        rng = random.default_rng(index) if self.deterministic else self.rng
        index = index % len(self.audio_files)  # collapse **after** setting seed
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav(filename)
        audio = audio / self.max_wav_value
        if not self.fine_tuning:
            audio = normalize(audio) * 0.95
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = rng.integers(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)))

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size,
                                  self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path,
                os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel).float()

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = rng.integers(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    a = mel_start * self.hop_size
                    b = (mel_start + frames_per_seg) * self.hop_size
                    audio = audio[:, a:b]
                else:
                    mel = F.pad(mel, (0, frames_per_seg - mel.size(2)))
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)))

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size,
                                   self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files) * self.repeat


def get_data_loader(args, distributed_run, train=True, batch_size=None,
                    val_kwargs=None):

    filelists = args.training_files if train else args.validation_files
    files = load_filepaths_and_text(args.dataset_path, filelists)
    files = list(zip(*files))[0]

    dataset_kw = {
        'segment_size': args.segment_size,
        'n_fft': args.filter_length,
        'num_mels': args.num_mels,
        'hop_size': args.hop_length,
        'win_size': args.win_length,
        'sampling_rate': args.sampling_rate,
        'fmin': args.mel_fmin,
        'fmax': args.mel_fmax,
        'fmax_loss': args.mel_fmax_loss,
        'max_wav_value': args.max_wav_value,
        'fine_tuning': args.fine_tuning,
        'base_mels_path': args.input_mels_dir,
        'deterministic': not train
    }

    if train:
        dataset = MelDataset(files, **dataset_kw)
        sampler = DistributedSampler(dataset) if distributed_run else None
    else:
        dataset_kw.update(val_kwargs or {})
        dataset = MelDataset(files, **dataset_kw)
        sampler = (DistributedSampler(dataset, shuffle=False)
                   if distributed_run else None)

    loader = DataLoader(dataset,
                        # NOTE On DGX-1 and DGX A100 =1 is optimal
                        num_workers=args.num_workers if train else 1,
                        shuffle=(train and not distributed_run),
                        sampler=sampler,
                        batch_size=batch_size or args.batch_size,
                        pin_memory=True,
                        persistent_workers=True,
                        drop_last=train)
    return loader
