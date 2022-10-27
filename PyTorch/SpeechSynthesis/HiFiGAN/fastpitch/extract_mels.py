# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from common.text import cmudict
from common.utils import init_distributed, prepare_tmp
from fastpitch.data_function import batch_to_gpu, TTSCollate, TTSDataset
from inference import CHECKPOINT_SPECIFIC_ARGS
from models import load_and_setup_model


def parse_args(parser):
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')

    general = parser.add_argument_group('general setup')
    general.add_argument('--checkpoint-path', type=str, required=True,
                         help='Checkpoint path to fastpitch model')
    general.add_argument('--resume', action='store_true',
                         help='Load last checkpoint from training')
    general.add_argument('--amp', action='store_true',
                         help='Enable AMP')
    general.add_argument('--cuda', action='store_true',
                         help='Run on GPU using CUDA')
    general.add_argument('--cudnn-benchmark', action='store_true',
                         help='Enable cudnn benchmark mode')
    general.add_argument('-bs', '--batch-size', type=int, required=True,
                         help='Batch size per GPU')

    data = parser.add_argument_group('dataset parameters')
    data.add_argument('--dataset-files', type=str, nargs='*', required=True,
                      help='Paths to dataset filelists.')
    data.add_argument('--text-cleaners', nargs='*',
                      default=['english_cleaners'], type=str,
                      help='Type of text cleaners for input text')
    data.add_argument('--symbol-set', type=str, default='english_basic',
                      help='Define symbol set for input text')
    data.add_argument('--p-arpabet', type=float, default=0.0,
                      help='Probability of using arpabets instead of graphemes '
                           'for each word; set 0 for pure grapheme training')
    data.add_argument('--heteronyms-path', type=str, default='data/cmudict/heteronyms',
                      help='Path to the list of heteronyms')
    data.add_argument('--cmudict-path', type=str, default='data/cmudict/cmudict-0.7b',
                      help='Path to the pronouncing dictionary')
    data.add_argument('--prepend-space-to-text', action='store_true',
                      help='Capture leading silence with a space token')
    data.add_argument('--append-space-to-text', action='store_true',
                      help='Capture trailing silence with a space token')

    cond = parser.add_argument_group('data for conditioning')
    cond.add_argument('--load-pitch-from-disk', action='store_true',
                      help='Use pitch cached on disk with prepare_dataset.py')
    cond.add_argument('--pitch-online-method', default='pyin', choices=['pyin'],
                      help='Calculate pitch on the fly during trainig')
    cond.add_argument('--pitch-online-dir', type=str, default=None,
                      help='A directory for storing pitch calculated on-line')

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                      help='Rank of the process for multiproc; do not set manually')
    dist.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                      help='Number of processes for multiproc; do not set manually')
    return parser


def main():
    parser = argparse.ArgumentParser(
        description='FastPitch spectrogram extraction', allow_abbrev=False)

    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    model, model_config, train_setup = load_and_setup_model(
        'FastPitch', parser, args.checkpoint_path, args.amp, unk_args=unk_args,
        device=torch.device('cuda' if args.cuda else 'cpu'))

    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    # use train_setup loaded from the checkpoint (sampling_rate, symbol_set, etc.)
    for k in CHECKPOINT_SPECIFIC_ARGS:

        if k in train_setup and getattr(args, k) != train_setup[k]:
            v = train_setup[k]
            print(f'Overwriting args.{k}={getattr(args, k)} with {v} '
                  f'from {args.checkpoint_path} checkpoint')
            setattr(args, k, v)

    if args.p_arpabet > 0.0:
        cmudict.initialize(args.cmudict_path, args.heteronyms_path)

    distributed_run = args.world_size > 1
    if distributed_run:
        init_distributed(args, args.world_size, args.local_rank)
        model = DDP(model, device_ids=[args.local_rank],
                    output_device=args.local_rank, find_unused_parameters=True)

    if args.local_rank == 0:
        Path(args.output).mkdir(exist_ok=True, parents=True)
        prepare_tmp(args.pitch_online_dir)

    args.n_speakers = model_config['n_speakers']
    args.n_mel_channels = model_config['n_mel_channels']
    trainset = TTSDataset(audiopaths_and_text=args.dataset_files,
                          load_mel_from_disk=False, **vars(args))

    dataset_loader = DataLoader(
        trainset, num_workers=16, shuffle=False, batch_size=args.batch_size,
        sampler=(DistributedSampler(trainset) if distributed_run else None),
        pin_memory=True, drop_last=False, collate_fn=TTSCollate())

    with torch.no_grad():
        for batch in tqdm(dataset_loader, 'Extracting mels'):
            x, y, num_frames = batch_to_gpu(batch)
            _, _, _, mel_lens, *_, audiopaths = x

            with torch.cuda.amp.autocast(enabled=args.amp):
                mel_out, *_ = model(x, use_gt_pitch=True)

            mel_out = mel_out.transpose(1, 2)
            assert mel_out.size(1) == args.n_mel_channels, mel_out.shape

            for apath, mel, len_ in zip(audiopaths, mel_out, mel_lens):
                np.save(Path(args.output, Path(apath).stem + '.npy'),
                        mel[:, :len_.item()].cpu().numpy())


if __name__ == '__main__':
    main()
