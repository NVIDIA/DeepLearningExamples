# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import sys
from os.path import abspath, dirname
sys.path.append(abspath(dirname(__file__)+'/../'))

from fastpitch.data_function import TTSCollate, TTSDataset
from torch.utils.data import DataLoader
import numpy as np
import inspect
import torch
from typing import List
from common.text import cmudict

def get_dataloader_fn(batch_size: int = 8,
                      precision: str = "fp16",
                      heteronyms_path: str = 'cmudict/heteronyms',
                      cmudict_path: str = 'cmudict/cmudict-0.7b',
                      dataset_path: str = './LJSpeech_1.1',
                      filelist: str ="filelists/ljs_audio_pitch_text_test.txt",
                      text_cleaners: List = ['english_cleaners_v2'],
                      n_mel_channels: int = 80,
                      symbol_set: str ='english_basic',
                      p_arpabet: float = 1.0,
                      n_speakers: int = 1,
                      load_mel_from_disk: bool = False,
                      load_pitch_from_disk: bool = True,
                      pitch_mean: float = 214.72203,  # LJSpeech defaults
                      pitch_std: float = 65.72038,
                      max_wav_value: float = 32768.0,
                      sampling_rate: int = 22050,
                      filter_length: int = 1024,
                      hop_length: int = 256,
                      win_length: int = 1024,
                      mel_fmin: float = 0.0,
                      mel_fmax: float = 8000.0):

    if p_arpabet > 0.0:
        cmudict.initialize(cmudict_path, heteronyms_path)

    dataset = TTSDataset(dataset_path=dataset_path,
                         audiopaths_and_text=filelist,
                         text_cleaners=text_cleaners,
                         n_mel_channels=n_mel_channels,
                         symbol_set=symbol_set,
                         p_arpabet=p_arpabet,
                         n_speakers=n_speakers,
                         load_mel_from_disk=load_mel_from_disk,
                         load_pitch_from_disk=load_pitch_from_disk,
                         pitch_mean=pitch_mean,
                         pitch_std=pitch_std,
                         max_wav_value=max_wav_value,
                         sampling_rate=sampling_rate,
                         filter_length=filter_length,
                         hop_length=hop_length,
                         win_length=win_length,
                         mel_fmin=mel_fmin,
                         mel_fmax=mel_fmax)
    collate_fn = TTSCollate()
    dataloader = DataLoader(dataset, num_workers=8, shuffle=False,
                            sampler=None,
                            batch_size=batch_size, pin_memory=False,
                            collate_fn=collate_fn)

    def _get_dataloader():
        for idx, batch in enumerate(dataloader):

            text_padded, _, mel_padded, output_lengths, _, \
            pitch_padded, energy_padded, *_ = batch

            pitch_padded = pitch_padded.float()
            energy_padded = energy_padded.float()
            dur_padded = torch.zeros_like(pitch_padded)

            if precision == "fp16":
                pitch_padded = pitch_padded.half()
                dur_padded = dur_padded.half()
                mel_padded = mel_padded.half()
                energy_padded = energy_padded.half()

            ids = np.arange(idx*batch_size, idx*batch_size + batch_size)
            x = {"INPUT__0": text_padded.cpu().numpy()}
            y_real = {"OUTPUT__0": mel_padded.cpu().numpy(),
                      "OUTPUT__1": output_lengths.cpu().numpy(),
                      "OUTPUT__2": dur_padded.cpu().numpy(),
                      "OUTPUT__3": pitch_padded.cpu().numpy(),
                      "OUTPUT__4": energy_padded.cpu().numpy()}

            yield (ids, x, y_real)

    return _get_dataloader
