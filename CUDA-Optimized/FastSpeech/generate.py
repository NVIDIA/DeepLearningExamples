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

import os
import pathlib
import sys
import time

import fire
import librosa
import torch

from fastspeech.data_load import PadDataLoader
from fastspeech.dataset.text_dataset import TextDataset
from fastspeech.inferencer.fastspeech_inferencer import FastSpeechInferencer
from fastspeech.model.fastspeech import Fastspeech
from fastspeech import hparam as hp, DEFAULT_DEVICE
from fastspeech.utils.logging import tprint
from fastspeech.utils.time import TimeElapsed
from fastspeech.utils.pytorch import to_device_async, to_cpu_numpy
from fastspeech.infer import get_inferencer
from fastspeech.inferencer.waveglow_inferencer import WaveGlowInferencer

MAX_FILESIZE=128

# TODO test with different speeds
def generate(hparam='infer.yaml',
             text='test_sentences.txt',
             results_path='results',
             device=DEFAULT_DEVICE,
             **kwargs):
    """The script for generating waveforms from texts with a vocoder.

    By default, this script assumes to load parameters in the default config file, fastspeech/hparams/infer.yaml.

    Besides the flags, you can also set parameters in the config file via the command-line. For examples,
    --checkpoint_path=CHECKPOINT_PATH
        Path to checkpoint directory. The latest checkpoint will be loaded.
    --waveglow_path=WAVEGLOW_PATH
        Path to the WaveGlow checkpoint file.
    --waveglow_engine_path=WAVEGLOW_ENGINE_PATH
        Path to the WaveGlow engine file. It can be only used with --use_trt=True.
    --batch_size=BATCH_SIZE
        Batch size to use. Defaults to 1.

    Refer to fastspeech/hparams/infer.yaml to see more parameters.

    Args:
        hparam (str, optional): Path to default config file. Defaults to "infer.yaml".
        text (str, optional): a sample text or a text file path to generate its waveform. Defaults to 'test_sentences.txt'.
        results_path (str, optional): Path to output waveforms directory. Defaults to 'results'.
        device (str, optional): Device to use. Defaults to "cuda" if avaiable, or "cpu".
    """

    hp.set_hparam(hparam, kwargs)

    if os.path.isfile(text):
        f = open(text, 'r', encoding="utf-8")
        texts = f.read().splitlines()
    else:  # single string
        texts = [text]

    dataset = TextDataset(texts)
    data_loader = PadDataLoader(dataset,
                                batch_size=hp.batch_size,
                                num_workers=hp.n_workers,
                                shuffle=False,
                                drop_last=False)

    # text to mel
    model = Fastspeech(
        max_seq_len=hp.max_seq_len,
        d_model=hp.d_model,
        phoneme_side_n_layer=hp.phoneme_side_n_layer,
        phoneme_side_head=hp.phoneme_side_head,
        phoneme_side_conv1d_filter_size=hp.phoneme_side_conv1d_filter_size,
        phoneme_side_output_size=hp.phoneme_side_output_size,
        mel_side_n_layer=hp.mel_side_n_layer,
        mel_side_head=hp.mel_side_head,
        mel_side_conv1d_filter_size=hp.mel_side_conv1d_filter_size,
        mel_side_output_size=hp.mel_side_output_size,
        duration_predictor_filter_size=hp.duration_predictor_filter_size,
        duration_predictor_kernel_size=hp.duration_predictor_kernel_size,
        fft_conv1d_kernel=hp.fft_conv1d_kernel,
        fft_conv1d_padding=hp.fft_conv1d_padding,
        dropout=hp.dropout,
        n_mels=hp.num_mels,
        fused_layernorm=hp.fused_layernorm
    )

    fs_inferencer = get_inferencer(model, data_loader, device)

    # set up WaveGlow
    if hp.use_trt:
        from fastspeech.trt.waveglow_trt_inferencer import WaveGlowTRTInferencer
        wb_inferencer = WaveGlowTRTInferencer(
            ckpt_file=hp.waveglow_path, engine_file=hp.waveglow_engine_path, use_fp16=hp.use_fp16)
    else:
        wb_inferencer = WaveGlowInferencer(
            ckpt_file=hp.waveglow_path, device=device, use_fp16=hp.use_fp16)

    tprint("Generating {} sentences.. ".format(len(dataset)))

    with fs_inferencer, wb_inferencer:
        try:
            for i in range(len(data_loader)):
                tprint("------------- BATCH # {} -------------".format(i))

                with TimeElapsed(name="Inferece Time: E2E", format=":.6f"):
                    ## Text-to-Mel ##
                    with TimeElapsed(name="Inferece Time: FastSpeech", device=device, cuda_sync=True, format=":.6f"), torch.no_grad():
                        outputs = fs_inferencer.infer()

                    texts = outputs["text"]
                    mels = outputs["mel"]  # (b, n_mels, t)
                    mel_masks = outputs['mel_mask']  # (b, t)
                    # assert(mels.is_cuda)

                    # remove paddings
                    mel_lens = mel_masks.sum(axis=1)
                    max_len = mel_lens.max()
                    mels = mels[..., :max_len]
                    mel_masks = mel_masks[..., :max_len]

                    ## Vocoder ##
                    with TimeElapsed(name="Inferece Time: WaveGlow", device=device, cuda_sync=True, format=":.6f"), torch.no_grad():
                        wavs = wb_inferencer.infer(mels)
                        wavs = to_cpu_numpy(wavs)

                ## Write wavs ##
                pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)
                for i, (text, wav) in enumerate(zip(texts, wavs)):
                    tprint("TEXT #{}: \"{}\"".format(i, text))

                    # remove paddings in case of batch size > 1
                    wav_len = mel_lens[i] * hp.hop_len
                    wav = wav[:wav_len]

                    path = os.path.join(results_path, text[:MAX_FILESIZE] + ".wav")
                    librosa.output.write_wav(path, wav, hp.sr)

        except StopIteration:
            tprint("Generation has been done.")
        except KeyboardInterrupt:
            tprint("Generation has been canceled.")


if __name__ == '__main__':
    fire.Fire(generate)
