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
from torch.optim.lr_scheduler import LambdaLR

from fastspeech import DEFAULT_DEVICE
from fastspeech import hparam as hp
from fastspeech.data_load import PadDataLoader
from fastspeech.dataset.ljspeech_dataset import LJSpeechDataset
from fastspeech.model.fastspeech import Fastspeech
from fastspeech.trainer.fastspeech_trainer import FastspeechTrainer
from fastspeech.utils.logging import tprint

try:
    import apex
except ImportError:
    ImportError('Required to install apex.')

# import multiprocessing
# multiprocessing.set_start_method('spawn', True)

pp = pprint.PrettyPrinter(indent=4, width=1000)

def train(hparam="train.yaml",
          device=DEFAULT_DEVICE,
          **kwargs):
    """ The FastSpeech model training script.

    By default, this script assumes to load parameters in the default config file, fastspeech/hparams/train.yaml.

    Besides the flags, you can also set parameters in the config file via the command-line. For examples,
    --dataset_path=DATASET_PATH
        Path to dataset directory.
    --tacotron2_path=TACOTRON2_PATH
        Path to tacotron2 checkpoint file.
    --mels_path=MELS_PATH
        Path to preprocessed mels directory.
    --aligns_path=ALIGNS_PATH
        Path to preprocessed alignments directory.
    --log_path=LOG_PATH
        Path to log directory.
    --checkpoint_path=CHECKPOINT_PATH
        Path to checkpoint directory. The latest checkpoint will be loaded.
    --batch_size=BATCH_SIZE
        Batch size to use. Defaults to 16.

    Refer to fastspeech/hparams/train.yaml to see more parameters.

    Args:
        hparam (str, optional): Path to default config file. Defaults to "train.yaml".
        device (str, optional): Device to use. Defaults to "cuda" if avaiable, or "cpu".

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

    # dataset
    dataset = LJSpeechDataset(root_path=hp.dataset_path,
                              meta_file=hp.meta_file,
                              mels_path=hp.mels_path,
                              aligns_path=hp.aligns_path,
                              sr=hp.sr,
                              n_fft=hp.n_fft,
                              win_len=hp.win_len,
                              hop_len=hp.hop_len,
                              n_mels=hp.num_mels,
                              mel_fmin=hp.mel_fmin,
                              mel_fmax=hp.mel_fmax,
                              )
    tprint("Dataset size: {}".format(len(dataset)))

    # data loader
    data_loader = PadDataLoader(dataset,
                                batch_size=hp.batch_size,
                                num_workers=hp.n_workers,
                                drop_last=True,
                                )

    # optimizer
    def get_optimizer(model):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hp.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9)
        return optimizer

    def get_warmup_lr_scheduler(optimizer):
        d_model = hp.d_model
        warmup_steps = hp.warmup_steps
        lr = lambda step: d_model ** -0.5 * min((step + 1) ** -0.5,
                                                (step + 1) * warmup_steps ** -1.5) / hp.learning_rate
        scheduler = LambdaLR(optimizer, lr_lambda=[lr])
        return scheduler

    # trainer
    trainer = FastspeechTrainer(data_loader,
                                'fastspeech',
                                model,
                                optimizer_fn=get_optimizer,
                                final_steps=hp.final_steps,
                                log_steps=hp.log_step,
                                ckpt_path=hp.checkpoint_path,
                                save_steps=hp.save_step,
                                log_path=hp.log_path,
                                lr_scheduler_fn=get_warmup_lr_scheduler,
                                pre_aligns=True if hp.aligns_path else False,
                                device=device,
                                use_amp=hp.use_amp,
                                nvprof_iter_start=hp.nvprof_iter_start,
                                nvprof_iter_end=hp.nvprof_iter_end,
                                pyprof_enabled=hp.pyprof_enabled,
                                )
    trainer.train()


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    fire.Fire(train)
