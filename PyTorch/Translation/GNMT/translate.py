#!/usr/bin/env python

# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
import itertools
import sys
import warnings
from itertools import product

import torch

import seq2seq.utils as utils
from seq2seq.data.dataset import RawTextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.translator import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.inference import tables


def parse_args():
    """
    Parse commandline arguments.
    """
    def exclusive_group(group, name, default, help):
        destname = name.replace('-', '_')
        subgroup = group.add_mutually_exclusive_group(required=False)
        subgroup.add_argument(f'--{name}', dest=f'{destname}',
                              action='store_true',
                              help=f'{help} (use \'--no-{name}\' to disable)')
        subgroup.add_argument(f'--no-{name}', dest=f'{destname}',
                              action='store_false', help=argparse.SUPPRESS)
        subgroup.set_defaults(**{destname: default})

    parser = argparse.ArgumentParser(
        description='GNMT Translate',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('data setup')
    dataset.add_argument('-o', '--output', required=False,
                         help='full path to the output file \
                         if not specified, then the output will be printed')
    dataset.add_argument('-r', '--reference', default=None,
                         help='full path to the file with reference \
                         translations (for sacrebleu, raw text)')
    dataset.add_argument('-m', '--model', required=True,
                         help='full path to the model checkpoint file')

    source = dataset.add_mutually_exclusive_group(required=True)
    source.add_argument('-i', '--input', required=False,
                        help='full path to the input file (raw text)')
    source.add_argument('-t', '--input-text', nargs='+', required=False,
                        help='raw input text')

    exclusive_group(group=dataset, name='sort', default=False,
                    help='sorts dataset by sequence length')

    # parameters
    params = parser.add_argument_group('inference setup')
    params.add_argument('--batch-size', nargs='+', default=[128], type=int,
                        help='batch size per GPU')
    params.add_argument('--beam-size', nargs='+', default=[5], type=int,
                        help='beam size')
    params.add_argument('--max-seq-len', default=80, type=int,
                        help='maximum generated sequence length')
    params.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    params.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    params.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')
    # general setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', nargs='+', default=['fp16'],
                         choices=['fp16', 'fp32'], help='precision')

    exclusive_group(group=general, name='env', default=False,
                    help='print info about execution env')
    exclusive_group(group=general, name='bleu', default=True,
                    help='compares with reference translation and computes \
                    BLEU')
    exclusive_group(group=general, name='cuda', default=True,
                    help='enables cuda')
    exclusive_group(group=general, name='cudnn', default=True,
                    help='enables cudnn')

    batch_first_parser = general.add_mutually_exclusive_group(required=False)
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)

    general.add_argument('--print-freq', '-p', default=1, type=int,
                         help='print log every PRINT_FREQ batches')

    # benchmarking
    benchmark = parser.add_argument_group('benchmark setup')
    benchmark.add_argument('--target-perf', default=None, type=float,
                           help='target inference performance (in tokens \
                           per second)')
    benchmark.add_argument('--target-bleu', default=None, type=float,
                           help='target accuracy')

    benchmark.add_argument('--repeat', nargs='+', default=[1], type=float,
                           help='loops over the dataset REPEAT times, flag \
                           accepts multiple arguments, one for each specified \
                           batch size')
    benchmark.add_argument('--warmup', default=0, type=int,
                           help='warmup iterations for performance counters')
    benchmark.add_argument('--percentiles', nargs='+', type=int,
                           default=(50, 90, 95, 99, 100),
                           help='Percentiles for confidence intervals for \
                           throughput/latency benchmarks')
    exclusive_group(group=benchmark, name='tables', default=False,
                    help='print accuracy, throughput and latency results in \
                    tables')

    # distributed
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='global rank of the process, do not set!')
    distributed.add_argument('--local_rank', default=0, type=int,
                             help='local rank of the process, do not set!')

    args = parser.parse_args()

    if args.input_text:
        args.bleu = False

    if args.bleu and args.reference is None:
        parser.error('--bleu requires --reference')

    if 'fp16' in args.math and not args.cuda:
        parser.error('--math fp16 requires --cuda')

    if len(list(product(args.math, args.batch_size, args.beam_size))) > 1:
        args.target_bleu = None
        args.target_perf = None

    args.repeat = dict(itertools.zip_longest(args.batch_size,
                                             args.repeat,
                                             fillvalue=1))

    return args


def main():
    """
    Launches translation (inference).
    Inference is executed on a single GPU, implementation supports beam search
    with length normalization and coverage penalty.
    """
    args = parse_args()
    device = utils.set_device(args.cuda, args.local_rank)
    utils.init_distributed(args.cuda)
    args.rank = utils.get_rank()
    utils.setup_logging()

    if args.env:
        utils.log_env_info()

    logging.info(f'Run arguments: {args}')

    if not args.cuda and torch.cuda.is_available():
        warnings.warn('cuda is available but not enabled')
    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    # load checkpoint and deserialize to CPU (to save GPU memory)
    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    # build GNMT model
    tokenizer = Tokenizer()
    tokenizer.set_state(checkpoint['tokenizer'])
    model_config = checkpoint['model_config']
    model_config['batch_first'] = args.batch_first
    model_config['vocab_size'] = tokenizer.vocab_size
    model = GNMT(**model_config)
    model.load_state_dict(checkpoint['state_dict'])

    # construct the dataset
    if args.input:
        data = RawTextDataset(raw_datafile=args.input,
                              tokenizer=tokenizer,
                              sort=args.sort,
                              )
    elif args.input_text:
        data = RawTextDataset(raw_data=args.input_text,
                              tokenizer=tokenizer,
                              sort=args.sort,
                              )

    latency_table = tables.LatencyTable(args.percentiles)
    throughput_table = tables.ThroughputTable(args.percentiles)
    accuracy_table = tables.AccuracyTable('BLEU')

    dtype = {'fp32': torch.FloatTensor, 'fp16': torch.HalfTensor}

    for (math, batch_size, beam_size) in product(args.math, args.batch_size,
                                                 args.beam_size):
        logging.info(f'math: {math}, batch size: {batch_size}, '
                     f'beam size: {beam_size}')

        model.type(dtype[math])
        model = model.to(device)
        model.eval()

        # build the data loader
        loader = data.get_loader(
            batch_size=batch_size,
            batch_first=args.batch_first,
            pad=True,
            repeat=args.repeat[batch_size],
            num_workers=0,
            )

        # build the translator object
        translator = Translator(
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            beam_size=beam_size,
            max_seq_len=args.max_seq_len,
            len_norm_factor=args.len_norm_factor,
            len_norm_const=args.len_norm_const,
            cov_penalty_factor=args.cov_penalty_factor,
            print_freq=args.print_freq,
            )

        # execute the inference
        output, stats = translator.run(
            calc_bleu=args.bleu,
            eval_path=args.output,
            summary=True,
            warmup=args.warmup,
            reference_path=args.reference,
            )

        # print translated outputs
        if not args.output and args.rank == 0:
            logging.info(f'Translated output:')
            for out in output:
                print(out)

        key = (batch_size, beam_size)
        latency_table.add(key, {math: stats['runtimes']})
        throughput_table.add(key, {math: stats['throughputs']})
        accuracy_table.add(key, {math: stats['bleu']})

    if args.tables:
        accuracy_table.write('Inference accuracy', args.math)

        if 'fp16' in args.math and 'fp32' in args.math:
            relative = 'fp32'
        else:
            relative = None

        if 'fp32' in args.math:
            throughput_table.write('Inference throughput', 'fp32')
        if 'fp16' in args.math:
            throughput_table.write('Inference throughput', 'fp16',
                                   relative=relative)

        if 'fp32' in args.math:
            latency_table.write('Inference latency', 'fp32')
        if 'fp16' in args.math:
            latency_table.write('Inference latency', 'fp16',
                                relative=relative, reverse_speedup=True)

    passed = utils.benchmark(stats['bleu'], args.target_bleu,
                             stats['tokens_per_sec'], args.target_perf)
    return passed


if __name__ == '__main__':
    passed = main()
    if not passed:
        sys.exit(1)
