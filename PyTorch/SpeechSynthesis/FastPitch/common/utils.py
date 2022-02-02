# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import shutil
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

import torch
from scipy.io.wavfile import read


class BenchmarkStats:
    """ Tracks statistics used for benchmarking. """
    def __init__(self):
        self.num_frames = []
        self.losses = []
        self.mel_losses = []
        self.took = []

    def update(self, num_frames, losses, mel_losses, took):
        self.num_frames.append(num_frames)
        self.losses.append(losses)
        self.mel_losses.append(mel_losses)
        self.took.append(took)

    def get(self, n_epochs):
        frames_s = sum(self.num_frames[-n_epochs:]) / sum(self.took[-n_epochs:])
        return {'frames/s': frames_s,
                'loss': np.mean(self.losses[-n_epochs:]),
                'mel_loss': np.mean(self.mel_losses[-n_epochs:]),
                'took': np.mean(self.took[-n_epochs:]),
                'benchmark_epochs_num': n_epochs}

    def __len__(self):
        return len(self.losses)


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(fnames, dataset_path=None, has_speakers=False,
                            split="|"):
    def split_line(line, root=None):
        parts = line.strip().split(split)
        if has_speakers:
            paths, non_paths = parts[:-2], parts[-2:]
        else:
            paths, non_paths = parts[:-1], parts[-1:]
        if root:
            return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)
        else:
            return tuple(str(Path(p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            fpaths_and_text += [split_line(line, dataset_path) for line in f]
    return fpaths_and_text


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def prepare_tmp(path):
    if path is None:
        return
    p = Path(path)
    if p.is_dir():
        warnings.warn(f'{p} exists. Removing...')
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=False, exist_ok=False)
