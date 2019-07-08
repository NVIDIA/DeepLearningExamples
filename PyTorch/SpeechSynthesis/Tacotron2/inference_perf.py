# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import json
import time

from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model

from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger import tags
from dllogger.autologging import log_hardware, log_args

from apex import amp

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-m', '--model-name', type=str, default='', required=True,
                        help='Model to train')
    parser.add_argument('--input-text', type=str, default=None,
                        help='Path to tensor containing text (when running Tacotron2)')
    parser.add_argument('--input-mels', type=str, default=None,
                        help='Path to tensor containing mels (when running WaveGlow)')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--amp-run', action='store_true',
                        help='inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=1)
    parser.add_argument('--create-benchmark', action='store_true')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')

    return parser


def collate_text(batch):
    """Collate's training batch from normalized text and mel-spectrogram
    PARAMS
    ------
    batch: [text_normalized]
    """
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def main():
    """
    Launches inference benchmark.
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = args.log_file

    LOGGER.set_model_name("Tacotron2_PyT")
    LOGGER.set_backends([
        dllg.StdOutBackend(log_file=None,
                           logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1),
        dllg.JsonBackend(log_file,
                         logging_scope=dllg.TRAIN_ITER_SCOPE, iteration_interval=1)
    ])
    LOGGER.register_metric("items_per_sec",
                           metric_scope=dllg.TRAIN_ITER_SCOPE)

    log_hardware()
    log_args(args)

    # ### uncomment to generate new padded text
    # texts = []
    # f = open('qa/ljs_text_train_subset_2500.txt', 'r')
    # texts = f.readlines()
    # sequence = []
    # for i, text in enumerate(texts):
    #     sequence.append(torch.IntTensor(text_to_sequence(text, ['english_cleaners'])))

    # text_padded, input_lengths = collate_text(sequence)
    # text_padded = torch.autograd.Variable(text_padded).cuda().long()
    # torch.save(text_padded, "qa/text_padded.pt")
    # torch.save(input_lengths, "qa/input_lengths.pt")

    model = load_and_setup_model(args.model_name, parser, None,
                                 args.amp_run)

    dry_runs = 3
    num_iters = (16+dry_runs) if args.create_benchmark else (1+dry_runs)

    for i in range(num_iters):
        ## skipping the first inference which is slower
        if i >= dry_runs:
            LOGGER.iteration_start()

        if args.model_name == 'Tacotron2':
            text_padded = torch.load(args.input_text)
            text_padded = text_padded[:args.batch_size]
            text_padded = torch.autograd.Variable(text_padded).cuda().long()

            t0 = time.time()
            with torch.no_grad():
                _, mels, _, _ = model.infer(text_padded)
            t1 = time.time()
            inference_time= t1 - t0
            num_items = text_padded.size(0)*text_padded.size(1)

            # # ## uncomment to generate new padded mels
            # torch.save(mels, "qa/mel_padded.pt")

        if args.model_name == 'WaveGlow':
            mel_padded = torch.load(args.input_mels)
            mel_padded = torch.cat((mel_padded, mel_padded, mel_padded, mel_padded))
            mel_padded = mel_padded[:args.batch_size]
            mel_padded = mel_padded.cuda()

            if args.amp_run:
                mel_padded = mel_padded.half()

            t0 = time.time()
            with torch.no_grad():
                audios = model.infer(mel_padded)
                audios = audios.float()
            t1 = time.time()
            inference_time = t1 - t0
            num_items = audios.size(0)*audios.size(1)

        if i >= dry_runs:
            LOGGER.log(key="items_per_sec", value=(num_items/inference_time))
            LOGGER.iteration_stop()

    LOGGER.finish()

if __name__ == '__main__':
    main()
