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

from fastpitch.data_function import TextMelAliCollate, TextMelAliLoader
from torch.utils.data import DataLoader
import numpy as np
import inspect
import torch
from typing import List

def get_dataloader_fn(text_cleaners: List = ['english_cleaners'],
                      n_mel_channels: int = 80,
                      n_speakers: int = 1,
                      symbol_set: str ='english_basic',
                      dataset_path: str = './LJSpeech_1.1',
                      filelist: str ="filelists/ljs_mel_dur_pitch_text_test_filelist.txt",
                      batch_size: int = 8,
                      precision: str = "fp16"):

    dataset = TextMelAliLoader(dataset_path=dataset_path,
                               audiopaths_and_text=filelist,
                               text_cleaners=text_cleaners,
                               n_mel_channels=n_mel_channels,
                               symbol_set=symbol_set,
                               n_speakers=n_speakers)
    collate_fn = TextMelAliCollate()
    dataloader = DataLoader(dataset, num_workers=8, shuffle=False,
                            sampler=None,
                            batch_size=batch_size, pin_memory=False,
                            collate_fn=collate_fn)

    def _get_dataloader():
        for idx, batch in enumerate(dataloader):
            text_padded, input_lengths, mel_padded, output_lengths, \
                len_x, dur_padded, dur_lens, pitch_padded, speaker = batch
            input_lengths = input_lengths.unsqueeze(-1)
            pitch_padded = pitch_padded.float()
            dur_padded = dur_padded.float()

            if precision == "fp16":
                pitch_padded = pitch_padded.half()
                dur_padded = dur_padded.half()
                mel_padded = mel_padded.half()

            ids = np.arange(idx*batch_size, idx*batch_size + batch_size)
            x = {"INPUT__0": text_padded.cpu().numpy(),
                 "INPUT__1": input_lengths.cpu().numpy()}
            y_real = {"OUTPUT__0": mel_padded.cpu().numpy(),
                      "OUTPUT__1": output_lengths.cpu().numpy(),
                      "OUTPUT__2": dur_padded.cpu().numpy(),
                      "OUTPUT__3": pitch_padded.cpu().numpy()}

            yield (ids, x, y_real)

    return _get_dataloader
