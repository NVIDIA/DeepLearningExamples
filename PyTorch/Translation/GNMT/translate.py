#!/usr/bin/env python
import argparse
import logging
import os
import warnings
from ast import literal_eval
from itertools import product

import torch
import torch.distributed as dist

import seq2seq.utils as utils
from seq2seq.data.dataset import TextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.inference import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.utils import setup_logging


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
    dataset.add_argument('--dataset-dir', default='data/wmt16_de_en/',
                         help='path to directory with training/test data')
    dataset.add_argument('-i', '--input', required=True,
                         help='full path to the input file (tokenized)')
    dataset.add_argument('-o', '--output', required=True,
                         help='full path to the output file (tokenized)')
    dataset.add_argument('-r', '--reference', default=None,
                         help='full path to the file with reference \
                         translations (for sacrebleu)')
    dataset.add_argument('-m', '--model', required=True,
                         help='full path to the model checkpoint file')
    exclusive_group(group=dataset, name='sort', default=True,
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
                         choices=['fp16', 'fp32'], help='arithmetic type')

    exclusive_group(group=general, name='env', default=True,
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

    # distributed
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='global rank of the process, do not set!')
    distributed.add_argument('--local_rank', default=0, type=int,
                             help='local rank of the process, do not set!')

    args = parser.parse_args()

    if args.bleu and args.reference is None:
        parser.error('--bleu requires --reference')

    if 'fp16' in args.math and not args.cuda:
        parser.error('--math fp16 requires --cuda')

    return args


def main():
    """
    Launches translation (inference).
    Inference is executed on a single GPU, implementation supports beam search
    with length normalization and coverage penalty.
    """
    args = parse_args()
    utils.set_device(args.cuda, args.local_rank)
    utils.init_distributed(args.cuda)
    setup_logging()

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
    vocab_size = tokenizer.vocab_size
    model_config = checkpoint['model_config']
    model_config['batch_first'] = args.batch_first
    model = GNMT(vocab_size=vocab_size, **model_config)
    model.load_state_dict(checkpoint['state_dict'])

    for (math, batch_size, beam_size) in product(args.math, args.batch_size,
                                                 args.beam_size):
        logging.info(f'math: {math}, batch size: {batch_size}, '
                     f'beam size: {beam_size}')
        if math == 'fp32':
            dtype = torch.FloatTensor
        if math == 'fp16':
            dtype = torch.HalfTensor
        model.type(dtype)

        if args.cuda:
            model = model.cuda()
        model.eval()

        # construct the dataset
        test_data = TextDataset(src_fname=args.input,
                                tokenizer=tokenizer,
                                sort=args.sort)

        # build the data loader
        test_loader = test_data.get_loader(batch_size=batch_size,
                                           batch_first=args.batch_first,
                                           shuffle=False,
                                           pad=True,
                                           num_workers=0)

        # build the translator object
        translator = Translator(model=model,
                                tokenizer=tokenizer,
                                loader=test_loader,
                                beam_size=beam_size,
                                max_seq_len=args.max_seq_len,
                                len_norm_factor=args.len_norm_factor,
                                len_norm_const=args.len_norm_const,
                                cov_penalty_factor=args.cov_penalty_factor,
                                cuda=args.cuda,
                                print_freq=args.print_freq,
                                dataset_dir=args.dataset_dir)

        # execute the inference
        translator.run(calc_bleu=args.bleu, eval_path=args.output,
                       reference_path=args.reference, summary=True)


if __name__ == '__main__':
    main()
