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

import fire
import torch

from fastspeech import DEFAULT_DEVICE
from fastspeech import hparam as hp
from fastspeech.data_load import PadDataLoader
from fastspeech.dataset.text_dataset import TextDataset
from fastspeech.inferencer.fastspeech_inferencer import FastSpeechInferencer
from fastspeech.model.fastspeech import Fastspeech
from fastspeech.trt.fastspeech_trt_inferencer import FastSpeechTRTInferencer
from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import to_cpu_numpy
from collections import OrderedDict
import sys
import numpy as np
from torch.nn import functional as F

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

pp = pprint.PrettyPrinter(indent=4, width=1000)
np.set_printoptions(threshold=sys.maxsize)

SAMPLE_TEXT = "the more you buy, the more you save."

def verify(hparam="trt.yaml",
           text=SAMPLE_TEXT,
           **kwargs):
    hp.set_hparam(hparam, kwargs)
    tprint("Hparams:\n{}".format(pp.pformat(hp)))
    tprint("Device count: {}".format(torch.cuda.device_count()))

    outs_trt, acts_trt = infer_trt(text)
    outs, acts = infer_pytorch(text)

    both, pytorch, trt = join_dict(acts, acts_trt)

    # print diff
    print("## Diff ##\n\n")
    for name, (act, act_trt) in both.items():
        act = act.float()
        act_trt = act_trt.float()
        diff = act.reshape(-1) - act_trt.reshape(-1)
        is_identical = diff.eq(0).all()
        errors = diff[diff.ne(0)]
        max_error = torch.max(torch.abs(errors)) if len(errors) > 0 else 0
        print("# {} #\n\n[PyTorch]\n{}\n\n[TRT]: \n{}\n\n[Diff]: \n{}\n\n[Errors]: \n{}\n- identical? {}\n- {} errors out of {}\n- max: {}\n\n".format(name,
                                                                                                                                                act, 
                                                                                                                                                act_trt, 
                                                                                                                                                diff, 
                                                                                                                                                errors, 
                                                                                                                                                is_identical, 
                                                                                                                                                len(errors), 
                                                                                                                                                len(diff),
                                                                                                                                                max_error,
                                                                                                                                                ))

    # print("## PyTorch ##\n\n")
    # for name, act in pytorch.items():
    #     print("[{}]\npytorch:\n{}\n\n".format(name, act))

    # print("## TRT ##\n\n")
    # for name, act in trt.items():
    #     print("[{}]\ttrt:\n{}\n\n".format(name, act_trt))

def join_dict(acts, acts_trt):
    both = dict()
    left = dict()
    right = dict()
    for k in acts:
        if k in acts_trt:
            both[k] = (acts[k], acts_trt[k])
        else:
            left[k] = acts[k]
    for k in acts_trt:
        if k not in acts:
            right[k] = acts_trt[k]
    return both, left, right


def infer_trt(text):
    # model
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

    # dataset
    dataset = TextDataset([text for _ in range(hp.batch_size)])
    data_loader = PadDataLoader(dataset,
                                batch_size=hp.batch_size,
                                num_workers=hp.n_workers,
                                drop_last=False)

    # inferencer
    inferencer = FastSpeechTRTInferencer('fastspeech',
                            model,
                            data_loader=data_loader,
                            ckpt_path=hp.checkpoint_path,
                            trt_max_ws_size=hp.trt_max_ws_size,
                            trt_file_path=hp.trt_file_path,
                            trt_force_build=hp.trt_force_build,
                            use_fp16=hp.use_fp16,
                            trt_max_input_seq_len=hp.trt_max_input_seq_len,
                            trt_max_output_seq_len=hp.trt_max_output_seq_len,
                            validate_accuracy=True,
                            )
    with inferencer:
        acts = dict()
        outs = inferencer.infer(acts=acts)

    return outs, acts


def infer_pytorch(text):
    # model
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

    # dataset
    dataset = TextDataset([text for _ in range(hp.batch_size)])
    data_loader = PadDataLoader(dataset,
                                batch_size=hp.batch_size,
                                num_workers=hp.n_workers,
                                drop_last=False)

    # inferencer
    with torch.no_grad():
        inferencer = FastSpeechInferencer('fastspeech',
                                model,
                                data_loader=data_loader,
                                ckpt_path=hp.checkpoint_path,
                                device='cuda',
                                use_fp16=hp.use_fp16,
                                )

        acts = dict()
        outs = inferencer.infer(acts=acts,
                                seq_input_len=hp.trt_max_input_seq_len,
                                seq_output_len=hp.trt_max_output_seq_len)

    return outs, acts


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    fire.Fire(verify)
