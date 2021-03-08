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

import pathlib

import fire
import torch
from tqdm import tqdm
from fastspeech.data_load import PadDataLoader
from fastspeech.dataset.ljspeech_dataset import LJSpeechDataset
import tacotron2.train
import tacotron2.hparams
from fastspeech import hparam as hp, DEFAULT_DEVICE
import os
import numpy as np

from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import to_device_async, to_cpu_numpy


def get_tacotron2(device, is_training=False):
    hparams = tacotron2.hparams.create_hparams()
    model = tacotron2.train.load_model(hparams)
    model.load_state_dict(torch.load(
        hp.tacotron2_path, map_location=torch.device(device))["state_dict"])
    if is_training:
        model.train()
    else:
        model.eval()
    return model


def get_duration(texts, text_lens, mels, mel_lens, tacotron2, device):
    texts = to_device_async(texts, device)
    text_lens = to_device_async(text_lens, device)
    mels = to_device_async(mels, device)
    mel_lens = to_device_async(mel_lens, device)

    _, _, _, aligns = tacotron2.forward(
        (texts, text_lens, mels, None, mel_lens))

    aligns = to_cpu_numpy(aligns)
    durs = torch.FloatTensor([compute_duration(align) for align in aligns])

    return durs


def compute_duration(align):
    """
    Warning. This code assumes the attention is monotonic.
    """
    d_mel, d_text = align.shape
    dur = np.array([0 for _ in range(d_text)])

    for i in range(d_mel):
        idx = np.argmax(align[i])
        dur[idx] += 1

    return dur


def preprocess_aligns(
        hparam="base.yaml",
        device=DEFAULT_DEVICE):
    """ The script for preprocessing alignments.

    By default, this script assumes to load parameters in the default config file, fastspeech/hparams/base.yaml.

    --dataset_path=DATASET_PATH
        Path to dataset directory.
    --tacotron2_path=TACOTRON2_PATH
        Path to tacotron2 checkpoint file.
    --aligns_path=ALIGNS_PATH
        Path to output preprocessed alignments directory.

    Refer to fastspeech/hparams/base.yaml to see more parameters.

    Args:
        hparam (str, optional): Path to default config file. Defaults to "base.yaml".
        device (str, optional): Device to use. Defaults to "cuda" if avaiable, or "cpu".
    """

    hp.set_hparam(hparam)

    pathlib.Path(hp.aligns_path).mkdir(parents=True, exist_ok=True)

    dataset = LJSpeechDataset(hp.dataset_path)
    dataloader = PadDataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=32, drop_last=True)

    tacotron2 = get_tacotron2(device, is_training=True)
    to_device_async(tacotron2, device)

    for batched in tqdm(dataloader):
        names = batched['name']
        texts = batched['text_encoded']
        text_lens = batched['text_len']
        mels = batched['mel']
        mel_lens = batched['mel_len']

        tprint("Processing {}.".format(', '.join(names)))
        durs = get_duration(texts, text_lens, mels,
                            mel_lens, tacotron2, device)

        for i, (name, dur) in enumerate(zip(names, durs)):
            save_path = os.path.join(hp.aligns_path, name + ".align.npy")

            if os.path.exists(save_path):
                continue

            np.save(save_path, dur)
            # assert sum(duration) == len(align)


if __name__ == '__main__':
    fire.Fire(preprocess_aligns)
