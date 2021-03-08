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

import csv

import pprint

import librosa
from torch.utils.data import Dataset
import pandas as pd
from fastspeech.text_norm import text_to_sequence
from fastspeech import audio
from fastspeech.utils.logging import tprint

import os
import pathlib

import fire
import numpy as np
from tqdm import tqdm

from fastspeech import hparam as hp

pp = pprint.PrettyPrinter(indent=4, width=1000)

class LJSpeechDataset(Dataset):

    def __init__(self, root_path, meta_file="metadata.csv",
                 sr=22050, n_fft=1024, win_len=1024, hop_len=256, n_mels=80, mel_fmin=0.0, mel_fmax=8000.0, exclude_mels=False, mels_path=None,
                 aligns_path=None, text_cleaner=['english_cleaners'], sort_by_length=False):
        self.root_path = root_path
        self.meta_file = meta_file
        self.text_cleaner = text_cleaner
        self.sr = sr
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.aligns_path = aligns_path
        self.mels_path = mels_path
        self.exclude_mels = exclude_mels
        self.sort_by_length = sort_by_length

        # Read metadata file.
        # - column: <name, transcription, normalized_transcription>
        self.metas = pd.read_csv(os.path.join(root_path, meta_file),
                                 sep="|",
                                 header=None,
                                 keep_default_na=False,
                                 quoting=csv.QUOTE_NONE,
                                 names=["name", "transcription", "normalized_transcription"],
                                 )
        if sort_by_length:
            self.metas.insert(3, 'length', self.metas['normalized_transcription'].str.len())
            self.metas.sort_values('length', ascending=True, inplace=True)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        name = self.metas.iloc[idx, 0]
        path = "{}/wavs/{}.wav".format(self.root_path, name)

        # Text normalization
        text = self.metas.iloc[idx, 1]
        text_norm = self.metas.iloc[idx, 2]
        text_encoded = np.array(text_to_sequence(text_norm, self.text_cleaner))
        text_pos = np.array([idx+1 for idx, _ in enumerate(text_encoded)])

        data = {
            "name": name,
            "text": text,
            "text_norm": text_norm,
            "text_encoded": text_encoded,
            "text_pos": text_pos,
            "text_len": text_encoded.shape[-1],
            "sr": self.sr
        }

        if not self.exclude_mels:
            wav, sr = librosa.load(path, sr=self.sr)  # wav is [-1.0, 1.0]
            if sr != self.sr:
                raise ValueError("{} SR doesn't match target {} SR".format(sr, self.sr))

            # Audio processing
            wav, _ = librosa.effects.trim(wav, frame_length=self.win_len, hop_length=self.hop_len)
            
            if self.mels_path:
                mel = np.load(os.path.join(self.mels_path, name + ".mel.npy"))
            else:
                mel = librosa.feature.melspectrogram(wav,
                                                    sr=sr,
                                                    n_fft=self.n_fft,
                                                    win_length=self.win_len,
                                                    hop_length=self.hop_len,
                                                    n_mels=self.n_mels,
                                                    fmin=self.mel_fmin,
                                                    fmax=self.mel_fmax,
                                                    power=1.0)
                mel = audio.dynamic_range_compression(mel)

            data_mel = {
                "wav": wav,
                "mel": mel,
                "mel_len": mel.shape[-1],
            }
            data.update(data_mel)

        if self.aligns_path:
            aligns = np.load(os.path.join(self.aligns_path, name + ".align.npy"))
            data['align'] = aligns

        return data


def preprocess_mel(hparam="base.yaml", **kwargs):
    """The script for preprocessing mel-spectrograms from the dataset.

    By default, this script assumes to load parameters in the default config file, fastspeech/hparams/base.yaml.

    Besides the flags, you can also set parameters in the config file via the command-line. For examples,
    --dataset_path=DATASET_PATH
        Path to dataset directory.
    --mels_path=MELS_PATH
        Path to output preprocessed mels directory.

    Refer to fastspeech/hparams/base.yaml to see more parameters.

    Args:
        hparam (str, optional): Path to default config file. Defaults to "base.yaml".
    """

    hp.set_hparam(hparam, kwargs)
    tprint("Hparams:\n{}".format(pp.pformat(hp)))
    
    pathlib.Path(hp.mels_path).mkdir(parents=True, exist_ok=True)

    dataset = LJSpeechDataset(hp.dataset_path, mels_path=None)

    for data in tqdm(dataset):
        name = data["name"]
        mel = data["mel"]

        save_path = os.path.join(hp.mels_path, name + ".mel.npy")

        if os.path.exists(save_path):
            continue

        # print(name, mel)
        np.save(save_path, mel)


if __name__ == '__main__':
    fire.Fire(preprocess_mel)