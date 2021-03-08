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

import pprint
import sys
import time

import fire
import torch
from tqdm import tqdm

from fastspeech import DEFAULT_DEVICE
from fastspeech import hparam as hp
from fastspeech.data_load import PadDataLoader
from fastspeech.dataset.ljspeech_dataset import LJSpeechDataset
from fastspeech.model.fastspeech import Fastspeech
from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import to_cpu_numpy, to_device_async
from fastspeech.infer import get_inferencer
from fastspeech.inferencer.waveglow_inferencer import WaveGlowInferencer
from contextlib import ExitStack
import numpy as np

try:
    from apex import amp
except ImportError:
    ImportError('Required to install apex.')

pp = pprint.PrettyPrinter(indent=4, width=1000)

WARMUP_ITERS = 3


def perf_inference(hparam="infer.yaml",
                   with_vocoder=False,
                   n_iters=None,
                   device=DEFAULT_DEVICE,
                   **kwargs):
    """The script for estimating inference performance.

    By default, this script assumes to load parameters in the default config file, fastspeech/hparams/infer.yaml.

    Besides the flags, you can also set parameters in the config file via the command-line. For examples,
    --dataset_path=DATASET_PATH
        Path to dataset directory.
    --checkpoint_path=CHECKPOINT_PATH
        Path to checkpoint directory. The latest checkpoint will be loaded.
    --batch_size=BATCH_SIZE
        Batch size to use. Defaults to 1.

    Refer to fastspeech/hparams/infer.yaml to see more parameters.

    Args:
        hparam (str, optional): Path to default config file. Defaults to "infer.yaml".
        with_vocoder (bool, optional): Whether or not to estimate with a vocoder. Defaults to False.
        n_iters (int, optional): Number of batches to estimate. Defaults to None (an epoch).
        device (str, optional): Device to use. Defaults to "cuda" if avaiable, or "cpu".

    """

    hp.set_hparam(hparam, kwargs)
    tprint("Hparams:\n{}".format(pp.pformat(hp)))
    tprint("Device count: {}".format(torch.cuda.device_count()))

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

    dataset = LJSpeechDataset(root_path=hp.dataset_path,
                            sr=hp.sr,
                            n_fft=hp.n_fft,
                            win_len=hp.win_len,
                            hop_len=hp.hop_len,
                            n_mels=hp.num_mels,
                            mel_fmin=hp.mel_fmin,
                            mel_fmax=hp.mel_fmax,
                            exclude_mels=True,
                            sort_by_length=True if hp.use_trt and hp.trt_multi_engine else False
                            )
    tprint("Dataset size: {}".format(len(dataset)))

    data_loader = PadDataLoader(dataset,
                                batch_size=hp.batch_size,
                                num_workers=hp.n_workers,
                                shuffle=False if hp.use_trt and hp.trt_multi_engine else True,
                                drop_last=True,
                                )

    fs_inferencer = get_inferencer(model, data_loader, device)

    if with_vocoder:
        if hp.use_trt:
            from fastspeech.trt.waveglow_trt_inferencer import WaveGlowTRTInferencer
            wb_inferencer = WaveGlowTRTInferencer(ckpt_file=hp.waveglow_path, engine_file=hp.waveglow_engine_path, use_fp16=hp.use_fp16)
        else:
            wb_inferencer = WaveGlowInferencer(ckpt_file=hp.waveglow_path, device=device, use_fp16=hp.use_fp16)

    with fs_inferencer, wb_inferencer if with_vocoder else ExitStack():

        tprint("Perf started. Batch size={}.".format(hp.batch_size))

        latencies = []
        throughputs = []

        n_iters = min(n_iters, len(data_loader)) if n_iters else len(data_loader)
        assert(n_iters > WARMUP_ITERS)
        for i in tqdm(range(n_iters)):
            start = time.time()

            outputs = fs_inferencer.infer()

            mels = outputs['mel']
            mel_masks = outputs['mel_mask']
            assert(mels.is_cuda)

            if with_vocoder:
                # remove padding
                max_len = mel_masks.sum(axis=1).max()
                mels = mels[..., :max_len]
                mel_masks = mel_masks[..., :max_len]

                with torch.no_grad():
                    wavs = wb_inferencer.infer(mels)
                wavs = to_cpu_numpy(wavs)
            else:
                # include time for DtoH copy
                to_cpu_numpy(mels)
                to_cpu_numpy(mel_masks)

            end = time.time()

            if i > WARMUP_ITERS-1:
                time_elapsed = end - start
                generated_samples = len(mel_masks.nonzero()) * hp.hop_len
                throughput = generated_samples / time_elapsed

                latencies.append(time_elapsed)
                throughputs.append(throughput)

        latencies.sort()

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        latency_90 = max(latencies[:int(len(latencies)*0.90)]) if n_iters > 1 else 0
        latency_95 = max(latencies[:int(len(latencies)*0.95)]) if n_iters > 1 else 0
        latency_99 = max(latencies[:int(len(latencies)*0.99)]) if n_iters > 1 else 0

        throughput = np.mean(throughputs)
        rtf = throughput / (hp.sr * hp.batch_size)

        tprint("Batch size\tPrecision\tAvg Latency(s)\tStd Latency(s)\tLatency 90%(s)\tLatency 95%(s)\tLatency 99%(s)\tThroughput(samples/s)\tAvg RTF\n\
        {}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{:.2f}".format(
            hp.batch_size,
            "FP16" if hp.use_fp16 else "FP32",
            avg_latency,
            std_latency,
            latency_90,
            latency_95,
            latency_99,
            int(throughput),
            rtf))


if __name__ == '__main__':
    fire.Fire(perf_inference)
