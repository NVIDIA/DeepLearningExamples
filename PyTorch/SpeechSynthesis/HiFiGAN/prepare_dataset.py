# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
from pathlib import Path

import torch
import tqdm
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from torch.utils.data import DataLoader

from fastpitch.data_function import TTSCollate, TTSDataset


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    parser.add_argument('--wav-text-filelists', required=True, nargs='+',
                        type=str, help='Files with audio paths and text')
    parser.add_argument('--extract-mels', action='store_true',
                        help='Calculate spectrograms from .wav files')
    parser.add_argument('--extract-pitch', action='store_true',
                        help='Extract pitch')
    parser.add_argument('--log-file', type=str, default='preproc_log.json',
                        help='Filename for logging')
    parser.add_argument('--n-speakers', type=int, default=1)
    # Mel extraction
    parser.add_argument('--max-wav-value', default=32768.0, type=float,
                        help='Maximum audiowave value')
    parser.add_argument('--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--filter-length', default=1024, type=int,
                        help='Filter length')
    parser.add_argument('--hop-length', default=256, type=int,
                        help='Hop (stride) length')
    parser.add_argument('--win-length', default=1024, type=int,
                        help='Window length')
    parser.add_argument('--mel-fmin', default=0.0, type=float,
                        help='Minimum mel frequency')
    parser.add_argument('--mel-fmax', default=8000.0, type=float,
                        help='Maximum mel frequency')
    parser.add_argument('--n-mel-channels', type=int, default=80)
    # Pitch extraction
    parser.add_argument('--f0-method', default='pyin', type=str,
                        choices=('pyin',), help='F0 estimation method')
    # Performance
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--n-workers', type=int, default=16)
    return parser


def main():
    parser = argparse.ArgumentParser(description='TTS Data Pre-processing')
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()
    if len(unk_args) > 0:
        raise ValueError(f'Invalid options {unk_args}')

    DLLogger.init(backends=[
        JSONStreamBackend(Verbosity.DEFAULT,
                          Path(args.dataset_path, args.log_file)),
        StdOutBackend(Verbosity.VERBOSE)])

    for k, v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k: v})
    DLLogger.flush()

    if args.extract_mels:
        Path(args.dataset_path, 'mels').mkdir(parents=False, exist_ok=True)

    if args.extract_pitch:
        Path(args.dataset_path, 'pitch').mkdir(parents=False, exist_ok=True)

    for filelist in args.wav_text_filelists:

        print(f'Processing {filelist}...')

        dataset = TTSDataset(
            args.dataset_path,
            filelist,
            text_cleaners=['english_cleaners_v2'],
            n_mel_channels=args.n_mel_channels,
            p_arpabet=0.0,
            n_speakers=args.n_speakers,
            load_mel_from_disk=False,
            load_pitch_from_disk=False,
            pitch_mean=None,
            pitch_std=None,
            max_wav_value=args.max_wav_value,
            sampling_rate=args.sampling_rate,
            filter_length=args.filter_length,
            hop_length=args.hop_length,
            win_length=args.win_length,
            mel_fmin=args.mel_fmin,
            mel_fmax=args.mel_fmax,
            betabinomial_online_dir=None,
            pitch_online_dir=None,
            pitch_online_method=args.f0_method if args.extract_pitch else None)

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.n_workers,
            collate_fn=TTSCollate(),
            pin_memory=False,
            drop_last=False)

        all_filenames = set()
        for i, batch in enumerate(tqdm.tqdm(data_loader)):

            _, input_lens, mels, mel_lens, _, pitch, _, _, attn_prior, fpaths = batch

            # Ensure filenames are unique
            for p in fpaths:
                fname = Path(p).name
                if fname in all_filenames:
                    raise ValueError(f'Filename is not unique: {fname}')
                all_filenames.add(fname)

            if args.extract_mels:
                for j, mel in enumerate(mels):
                    fname = Path(fpaths[j]).with_suffix('.pt').name
                    fpath = Path(args.dataset_path, 'mels', fname)
                    torch.save(mel[:, :mel_lens[j]], fpath)

            if args.extract_pitch:
                for j, p in enumerate(pitch):
                    fname = Path(fpaths[j]).with_suffix('.pt').name
                    fpath = Path(args.dataset_path, 'pitch', fname)
                    torch.save(p[:mel_lens[j]], fpath)


if __name__ == '__main__':
    main()
