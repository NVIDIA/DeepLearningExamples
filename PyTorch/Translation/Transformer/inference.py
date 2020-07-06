#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
import numpy as np
import sys
import os
import time

import torch
from torch.serialization import default_restore_location

from fairseq import data, options, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import Dictionary
from fairseq.meters import StopwatchMeter
from fairseq.models.transformer import TransformerModel
import dllogger

from apply_bpe import BPE


Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

def _override_model_args(args, model_arg_overrides):
    # Uses model_arg_overrides {'arg_name': arg} to override model args
    for arg_name, arg_val in model_arg_overrides.items():
        setattr(args, arg_name, arg_val)
    return args

def load_ensemble_for_inference(filenames, model_arg_overrides=None):
    """Load an ensemble of models for inference.

    model_arg_overrides allows you to pass a dictionary model_arg_overrides --
    {'arg_name': arg} -- to override model args that were used during model
    training
    """
    # load model architectures and weights
    states = []
    for filename in filenames:
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        states.append(state)

    ensemble = []
    for state in states:
        args = state['args']
        
        if model_arg_overrides is not None:
            args = _override_model_args(args, model_arg_overrides)

        # build model for ensemble
        model = TransformerModel.build_model(args)
        model.load_state_dict(state['model'], strict=True)
        ensemble.append(model)

    src_dict = states[0]['extra_state']['src_dict']
    tgt_dict = states[0]['extra_state']['tgt_dict']

    return ensemble, args, src_dict, tgt_dict


def buffered_read(buffer_size):
    buffer = []
    for src_str in sys.stdin:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, src_dict, max_positions, bpe=None):
    tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, tokenize=tokenizer.tokenize_en, add_if_not_exist=False, bpe=bpe).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = data.EpochBatchIterator(
        dataset=data.LanguagePairDataset(tokens, lengths, src_dict),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']

def setup_logger(args):
    dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=1, filename=args.stat_file)])
    for k,v in vars(args).items():
        dllogger.log(step='PARAMETER', data= {k:v}, verbosity=0)
    container_setup_info = {
            'NVIDIA_PYTORCH_VERSION': os.environ.get('NVIDIA_PYTORCH_VERSION'),
            'PYTORCH_VERSION': os.environ.get('PYTORCH_VERSION'),
            'CUBLAS_VERSION' : os.environ.get('CUBLAS_VERSION'),
            'NCCL_VERSION' : os.environ.get('NCCL_VERSION'),
            'CUDA_DRIVER_VERSION' : os.environ.get('CUDA_DRIVER_VERSION'),
            'CUDNN_VERSION' : os.environ.get('CUDNN_VERSION'),
            'CUDA_VERSION' : os.environ.get('CUDA_VERSION')
            }
    dllogger.log(step='PARAMETER', data=container_setup_info, verbosity=0)
    dllogger.metadata('throughput', {'unit':'tokens/s', 'format':':/3f', 'GOAL':'MAXIMIZE', 'STAGE':'INFER'})

def main(args):
    if not args.no_dllogger:
        setup_logger(args)
    else:
        dllogger.init(backends=[])

    args.interactive = sys.stdin.isatty() # Just make the code more understendable
    if args.interactive:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1
    if args.buffer_size > 50000:
        print("WARNING: To prevent memory exhaustion buffer size is set to 50000", file=sys.stderr)
        args.buffer_size = 50000

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args, file=sys.stderr)

    use_cuda = torch.cuda.is_available() and not args.cpu

    processing_start = time.time()

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path), file=sys.stderr)
    model_paths = args.path.split(':')
    models, model_args, src_dict, tgt_dict = load_ensemble_for_inference(model_paths, model_arg_overrides=eval(args.model_overrides))
    if args.fp16:
        for model in models:
            model.half()

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(need_attn=args.print_alignment)

    # Initialize generator
    translator = SequenceGenerator(
        models,
        tgt_dict.get_metadata(),
        maxlen=args.max_target_positions,
        beam_size=args.beam, 
        stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), 
        len_penalty=args.lenpen,
        unk_penalty=args.unkpen,
        sampling=args.sampling, 
        sampling_topk=args.sampling_topk,
        minlen=args.min_len,
        sampling_temperature=args.sampling_temperature
    )

    if use_cuda:
        translator.cuda()

    # Load BPE codes file
    if args.bpe_codes:
        codes = open(args.bpe_codes, 'r')
        bpe = BPE(codes)
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    def make_result(src_str, hypos):
        result = Translation(
            src_str=src_str,
            hypos=[],
            pos_scores=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            hypo_str = tokenizer.Tokenizer.detokenize(hypo_str, 'de').strip()
            result.hypos.append((hypo['score'], hypo_str))
            result.pos_scores.append('P\t{}'.format(
                ' '.join(map(
                    lambda x: '{:.4f}'.format(x),
                    hypo['positional_scores'].tolist(),
                ))
            ))
            result.alignments.append(
                'A\t{}'.format(' '.join(map(lambda x: str(utils.item(x)), alignment)))
                if args.print_alignment else None
            )
        return result

    gen_timer = StopwatchMeter()

    def process_batch(batch):
        tokens = batch.tokens
        lengths = batch.lengths

        if use_cuda:
            tokens = tokens.cuda()
            lengths = lengths.cuda()

        translation_start = time.time()
        gen_timer.start()
        translations = translator.generate(
            tokens,
            lengths,
            maxlen=int(args.max_len_a * tokens.size(1) + args.max_len_b),
        )
        gen_timer.stop(sum(len(h[0]['tokens']) for h in translations))
        dllogger.log(step='infer', data={'latency': time.time() - translation_start})

        return [make_result(batch.srcs[i], t) for i, t in enumerate(translations)]

    if args.interactive:
        print('| Type the input sentence and press return:')
    for inputs in buffered_read(args.buffer_size):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs, args, src_dict, args.max_positions, bpe):
            indices.extend(batch_indices)
            results += process_batch(batch)

        for i in np.argsort(indices):
            result = results[i]
            print(result.src_str, file=sys.stderr)
            for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
                print(f'Score {hypo[0]}', file=sys.stderr)
                print(hypo[1])
                print(pos_scores, file=sys.stderr)
                if align is not None:
                    print(align, file=sys.stderr)

    log_dict = {
            'throughput': 1./gen_timer.avg,
            'latency_avg': sum(gen_timer.intervals)/len(gen_timer.intervals),
            'latency_p90': gen_timer.p(90),
            'latency_p95': gen_timer.p(95),
            'latency_p99': gen_timer.p(99),
            'total_infernece_time': gen_timer.sum,
            'total_run_time': time.time() - processing_start,
            }
    print('Translation time: {} s'.format(log_dict['total_infernece_time']), file=sys.stderr)
    print('Model throughput (beam {}): {} tokens/s'.format(args.beam, log_dict['throughput']), file=sys.stderr)
    print('Latency:\n\tAverage {:.3f}s\n\tp90 {:.3f}s\n\tp95 {:.3f}s\n\tp99 {:.3f}s'.format(
        log_dict['latency_avg'], log_dict['latency_p90'], log_dict['latency_p95'], log_dict['latency_p99'],
        file=sys.stderr))
    print('End to end time: {} s'.format(log_dict['total_run_time']), file=sys.stderr)
    dllogger.log(step=[], data=log_dict)


if __name__ == '__main__':
    parser = options.get_inference_parser()
    parser.add_argument('--no-dllogger', action='store_true')
    args = options.parse_args_and_arch(parser)
    main(args)
