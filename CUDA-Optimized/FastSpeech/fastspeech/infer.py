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

import fire

from fastspeech import hparam as hp, DEFAULT_DEVICE

from fastspeech.dataset.ljspeech_dataset import LJSpeechDataset
from fastspeech.inferencer.fastspeech_inferencer import FastSpeechInferencer
from fastspeech.model.fastspeech import Fastspeech
from fastspeech.data_load import PadDataLoader
from fastspeech.utils.logging import tprint
import torch
import pprint
from fastspeech.utils.time import TimeElapsed

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

pp = pprint.PrettyPrinter(indent=4, width=1000)


def infer(hparam="infer.yaml",
          device=DEFAULT_DEVICE, 
          n_iters=1,
          **kwargs):
    """ The FastSpeech model inference script.

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
        device (str, optional): Device to use. Defaults to "cuda" if avaiable, or "cpu".
        n_iters (int, optional): Number of batches to infer. Defaults to 1.
    """

    hp.set_hparam(hparam, kwargs)
    tprint("Hparams:\n{}".format(pp.pformat(hp)))
    tprint("Device count: {}".format(torch.cuda.device_count()))

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

    dataset = LJSpeechDataset(root_path=hp.dataset_path,
                              meta_file=hp.meta_file,
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

    inferencer = get_inferencer(model, data_loader, device)

    try:
        n_iters = min(len(data_loader), n_iters) if n_iters else len(data_loader)
        tprint("Num of iters: {}".format(n_iters))
        with inferencer:
            for i in range(n_iters):
                    tprint("------------- INFERENCE : batch #{} -------------".format(i))
                    with TimeElapsed(name="Inference Time", cuda_sync=True):
                        out_batch = inferencer.infer()
                        # tprint("Output:\n{}".format(pp.pformat(out_batch)))
        tprint("Inference has been done.")
    except KeyboardInterrupt:
        tprint("Inference has been canceled.")


def get_inferencer(model, data_loader, device):
    if hp.use_trt:
        if hp.trt_multi_engine:
            from fastspeech.trt.fastspeech_trt_multi_engine_inferencer import FastSpeechTRTMultiEngineInferencer
            inferencer = FastSpeechTRTMultiEngineInferencer('fastspeech',
                                                            model,
                                                            data_loader=data_loader,
                                                            ckpt_path=hp.checkpoint_path,
                                                            trt_max_ws_size=hp.trt_max_ws_size,
                                                            trt_force_build=hp.trt_force_build,
                                                            use_fp16=hp.use_fp16,
                                                            trt_file_path_list=hp.trt_file_path_list,
                                                            trt_max_input_seq_len_list=hp.trt_max_input_seq_len_list,
                                                            trt_max_output_seq_len_list=hp.trt_max_output_seq_len_list,
                                                            )
        else:
            from fastspeech.trt.fastspeech_trt_inferencer import FastSpeechTRTInferencer
            inferencer = FastSpeechTRTInferencer('fastspeech',
                                                 model,
                                                 data_loader=data_loader,
                                                 ckpt_path=hp.checkpoint_path,
                                                 trt_max_ws_size=hp.trt_max_ws_size,
                                                 trt_file_path=hp.trt_file_path,
                                                 use_fp16=hp.use_fp16,
                                                 trt_force_build=hp.trt_force_build,
                                                 trt_max_input_seq_len=hp.trt_max_input_seq_len,
                                                 trt_max_output_seq_len=hp.trt_max_output_seq_len,
                                                 )
    else:
        inferencer = FastSpeechInferencer(
            'fastspeech',
            model,
            data_loader=data_loader,
            ckpt_path=hp.checkpoint_path,
            log_path=hp.log_path,
            device=device,
            use_fp16=hp.use_fp16)
    return inferencer


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    fire.Fire(infer)
