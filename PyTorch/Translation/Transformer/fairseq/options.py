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

import argparse

import torch

from fairseq.criterions import CRITERION_REGISTRY
from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_CONFIG_REGISTRY
from fairseq.optim import OPTIMIZER_REGISTRY
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY
from fairseq.tasks import TASK_REGISTRY


def get_training_parser(default_task='translation'):
    parser = get_parser('Trainer', default_task)
    add_dataset_args(parser, train=True, gen=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    add_generation_args(parser)
    add_perf_args(parser)
    return parser


def get_generation_parser(interactive=False, default_task='translation'):
    parser = get_parser('Generation', default_task)
    add_dataset_args(parser, gen=True)
    add_generation_args(parser)
    add_perf_args(parser)
    if interactive:
        add_interactive_args(parser)
    return parser


def get_eval_lm_parser(default_task='language_modeling'):
    parser = get_parser('Evaluate Language Model', default_task)
    add_dataset_args(parser, gen=True)
    add_eval_lm_args(parser)
    return parser


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


def parse_args_and_arch(parser, input_args=None, parse_known=False):
    # The parser doesn't know about model/criterion/optimizer-specific args, so
    # we parse twice. First we parse the model/criterion/optimizer, then we
    # parse a second time after adding the *-specific arguments.
    # If input_args is given, we will parse those args instead of sys.argv.
    args, _ = parser.parse_known_args(input_args)


    # Add model-specific args to parser.
    if hasattr(args, 'arch'):
        model_specific_group = parser.add_argument_group(
            'Model-specific configuration',
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )
        ARCH_MODEL_REGISTRY[args.arch].add_args(model_specific_group)

    # Add *-specific args to parser.
    if hasattr(args, 'criterion'):
        CRITERION_REGISTRY[args.criterion].add_args(parser)
    if hasattr(args, 'optimizer'):
        OPTIMIZER_REGISTRY[args.optimizer].add_args(parser)
    if hasattr(args, 'lr_scheduler'):
        LR_SCHEDULER_REGISTRY[args.lr_scheduler].add_args(parser)
    if hasattr(args, 'task'):
        TASK_REGISTRY[args.task].add_args(parser)

    # Parse a second time.
    if parse_known:
        args, extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    # Post-process args.
    if hasattr(args, 'lr'):
        args.lr = eval_str_list(args.lr, type=float)
    if hasattr(args, 'update_freq'):
        args.update_freq = eval_str_list(args.update_freq, type=int)
    if hasattr(args, 'max_sentences_valid') and args.max_sentences_valid is None:
        args.max_sentences_valid = args.max_sentences

    # Apply architecture configuration.
    if hasattr(args, 'arch'):
        ARCH_CONFIG_REGISTRY[args.arch](args)

    if parse_known:
        return args, extra
    else:
        return args


def get_parser(desc, default_task='translation'):
    parser = argparse.ArgumentParser(
        description='Facebook AI Research Sequence-to-Sequence Toolkit -- ' + desc)
    parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='log progress every N batches (when progress bar is disabled)')
    parser.add_argument('--log-format', default=None, help='log format to use',
                        choices=['json', 'none', 'simple', 'tqdm'])
    parser.add_argument('--seed', default=1, type=int, metavar='N',
                        help='pseudo random number generator seed')
    parser.add_argument('--fp16', action='store_true', help='use FP16')
    parser.add_argument('--amp', action='store_true', help='use Automatic Mixed Precision')
    parser.add_argument('--amp-level', type=str, default="O1", help='choose apm\'s optimization level')

    parser.add_argument('--profile', type=int, default=None)
    # Task definitions can be found under fairseq/tasks/
    parser.add_argument(
        '--task', metavar='TASK', default=default_task, choices=TASK_REGISTRY.keys(),
        help='task: {} (default: {})'.format(', '.join(TASK_REGISTRY.keys()), default_task)
    )

    return parser


def add_dataset_args(parser, train=False, gen=False):
    group = parser.add_argument_group('Dataset and data loading')
    group.add_argument('--skip-invalid-size-inputs-valid-test', action='store_true',
                       help='ignore too long or too short lines in valid and test set')
    group.add_argument('--max-tokens', type=int, metavar='N',
                       help='maximum number of tokens in a batch')
    group.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    group.add_argument('--sentencepiece', action='store_true',
                        help='use when dataset uses sentencepiece encoding')
    if train:
        group.add_argument('--train-subset', default='train', metavar='SPLIT',
                           choices=['train', 'valid', 'test'],
                           help='data subset to use for training (train, valid, test)')
        group.add_argument('--valid-subset', default='valid', metavar='SPLIT',
                           help='comma separated list of data subsets to use for validation'
                                ' (train, valid, valid1, test, test1)')
        group.add_argument('--max-sentences-valid', type=int, metavar='N',
                           help='maximum number of sentences in a validation batch'
                                ' (defaults to --max-sentences)')
    if gen:
        group.add_argument('--gen-subset', default='test', metavar='SPLIT',
                           help='data subset to generate (train, valid, test)')
        group.add_argument('--num-shards', default=1, type=int, metavar='N',
                           help='shard generation over N shards')
        group.add_argument('--shard-id', default=0, type=int, metavar='ID',
                           help='id of the shard to generate (id < num_shards)')
    return group


def add_distributed_training_args(parser):
    group = parser.add_argument_group('Distributed training')
    group.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=torch.cuda.device_count(),
                       help='total number of GPUs across all nodes (default: all visible GPUs)')
    group.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current worker')
    group.add_argument('--local_rank', default=0, type=int,
                       help='rank of the current worker')
    group.add_argument('--distributed-backend', default='nccl', type=str,
                       help='distributed backend')
    group.add_argument('--distributed-init-method', default=None, type=str,
                       help='typically tcp://hostname:port that will be used to '
                            'establish initial connetion')
    group.add_argument('--distributed-port', default=-1, type=int,
                       help='port number (not required if using --distributed-init-method)')
    group.add_argument('--device-id', default=0, type=int,
                       help='which GPU to use (usually configured automatically)')
    return group


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                       help='force stop training at specified epoch')
    group.add_argument('--max-update', '--mu', default=0, type=int, metavar='N',
                       help='force stop training at specified update')
    group.add_argument('--target-bleu', default=0.0, type=float, metavar='TARGET',
                       help='force stop training after reaching target bleu')
    group.add_argument('--clip-norm', default=25, type=float, metavar='NORM',
                       help='clip threshold of gradients')
    group.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')
    group.add_argument('--update-freq', default='1', metavar='N',
                       help='update parameters every N_i batches, when in epoch i')

    # Optimizer definitions can be found under fairseq/optim/
    group.add_argument('--optimizer', default='nag', metavar='OPT',
                       choices=OPTIMIZER_REGISTRY.keys(),
                       help='optimizer: {} (default: nag)'.format(', '.join(OPTIMIZER_REGISTRY.keys())))
    group.add_argument('--lr', '--learning-rate', default='0.25', metavar='LR_1,LR_2,...,LR_N',
                       help='learning rate for the first N epochs; all epochs >N using LR_N'
                            ' (note: this may be interpreted differently depending on --lr-scheduler)')
    group.add_argument('--momentum', default=0.99, type=float, metavar='M',
                       help='momentum factor')
    group.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                       help='weight decay')

    # Learning rate schedulers can be found under fairseq/optim/lr_scheduler/
    group.add_argument('--lr-scheduler', default='reduce_lr_on_plateau',
                       help='learning rate scheduler: {} (default: reduce_lr_on_plateau)'.format(
                           ', '.join(LR_SCHEDULER_REGISTRY.keys())))
    group.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                       help='learning rate shrink factor for annealing, lr_new = (lr * lr_shrink)')
    group.add_argument('--min-lr', default=1e-5, type=float, metavar='LR',
                       help='minimum learning rate')
    group.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                       help='minimum loss scale (for FP16 training)')

    # Parallel backward + all-reduce optimization
    group.add_argument('--enable-parallel-backward-allred-opt', action='store_true',
                       help='enable all reduce of w-gradients in parallel with backward propagation (only for FP16 training)')
    group.add_argument('--parallel-backward-allred-opt-threshold', type=int, default=0, metavar='N',
                       help='min num of contiguous gradient elements before all-reduce is triggered')
    group.add_argument('--enable-parallel-backward-allred-opt-correctness-check', action='store_true',
                       help='compare w-gradient values obtained doing all-reduce in parallel vs. at the end')

    return group


def add_checkpoint_args(parser):
    group = parser.add_argument_group('Checkpointing')
    group.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                       help='path to save checkpoints')
    group.add_argument('--restore-file', default='checkpoint_last.pt',
                       help='filename in save-dir from which to load checkpoint')
    group.add_argument('--save-interval', type=int, default=1, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('--save-interval-updates', type=int, default=0, metavar='N',
                       help='save a checkpoint (and validate) every N updates')
    group.add_argument('--keep-interval-updates', type=int, default=-1, metavar='N',
                       help='keep last N checkpoints saved with --save-interval-updates')
    group.add_argument('--no-save', action='store_true',
                       help='don\'t save models or checkpoints')
    group.add_argument('--no-epoch-checkpoints', action='store_true',
                       help='only store last and best checkpoints')
    group.add_argument('--validate-interval', type=int, default=1, metavar='N',
                       help='validate every N epochs')
    return group


def add_common_eval_args(group):
    group.add_argument('--path', metavar='FILE',
                       help='path(s) to model file(s), colon separated')
    group.add_argument('--remove-bpe', nargs='?', const='@@ ', default=None,
                       help='remove BPE tokens before scoring')
    group.add_argument('--cpu', action='store_true', help='generate on CPU')
    group.add_argument('--quiet', action='store_true',
                       help='only print final scores')


def add_eval_lm_args(parser):
    group = parser.add_argument_group('LM Evaluation')
    add_common_eval_args(group)
    group.add_argument('--output-word-probs', action='store_true',
                       help='if set, outputs words and their predicted log probabilities to standard output')


def add_generation_args(parser):
    group = parser.add_argument_group('Generation')
    add_common_eval_args(group)
    group.add_argument('--beam', default=4, type=int, metavar='N',
                       help='beam size')
    group.add_argument('--nbest', default=1, type=int, metavar='N',
                       help='number of hypotheses to output')
    group.add_argument('--max-len-a', default=0, type=float, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--max-len-b', default=200, type=int, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--min-len', default=1, type=float, metavar='N',
                       help=('minimum generation length'))
    group.add_argument('--no-early-stop', action='store_true',
                       help=('continue searching even after finalizing k=beam '
                             'hypotheses; this is more correct, but increases '
                             'generation time by 50%%'))
    group.add_argument('--unnormalized', action='store_true',
                       help='compare unnormalized hypothesis scores')
    group.add_argument('--no-beamable-mm', action='store_true',
                       help='don\'t use BeamableMM in attention layers')
    group.add_argument('--lenpen', default=1, type=float,
                       help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    group.add_argument('--unkpen', default=0, type=float,
                       help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    group.add_argument('--replace-unk', nargs='?', const=True, default=None,
                       help='perform unknown replacement (optionally with alignment dictionary)')
    group.add_argument('--score-reference', action='store_true',
                       help='just score the reference translation')
    group.add_argument('--prefix-size', default=0, type=int, metavar='PS',
                       help='initialize generation by target prefix of given length')
    group.add_argument('--sampling', action='store_true',
                       help='sample hypotheses instead of using beam search')
    group.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
                       help='sample from top K likely next words instead of all words')
    group.add_argument('--sampling-temperature', default=1, type=float, metavar='N',
                       help='temperature for random sampling')
    group.add_argument('--print-alignment', action='store_true',
                       help='if set, uses attention feedback to compute and print alignment to source tokens')
    group.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                       help='a dictionary used to override model args at generation that were used during model training')
    group.add_argument('--online-eval', action='store_true',
                       help='score model at the end of epoch')
    group.add_argument('--ignore-case', action='store_true',
                       help='ignore case druing online eval')
    group.add_argument('--bpe-codes', default=None, type=str, metavar='CODES',
                        help='file with bpe codes')
    return group


def add_interactive_args(parser):
    group = parser.add_argument_group('Interactive')
    group.add_argument('--buffer-size', default=0, type=int, metavar='N',
                       help='read this many sentences into a buffer before processing them')


def add_model_args(parser):
    group = parser.add_argument_group('Model configuration')

    # Model definitions can be found under fairseq/models/
    #
    # The model architecture can be specified in several ways.
    # In increasing order of priority:
    # 1) model defaults (lowest priority)
    # 2) --arch argument
    # 3) --encoder/decoder-* arguments (highest priority)
    group.add_argument(
        '--arch', '-a', default='fconv', metavar='ARCH', required=True,
        choices=ARCH_MODEL_REGISTRY.keys(),
        help='model architecture: {} (default: fconv)'.format(
            ', '.join(ARCH_MODEL_REGISTRY.keys())),
    )

    # Criterion definitions can be found under fairseq/criterions/
    group.add_argument(
        '--criterion', default='cross_entropy', metavar='CRIT',
        choices=CRITERION_REGISTRY.keys(),
        help='training criterion: {} (default: cross_entropy)'.format(
            ', '.join(CRITERION_REGISTRY.keys())),
    )

    return group

def add_perf_args(parser):
    group = parser.add_argument_group('Performance')
    group.add_argument('--fuse-dropout-add', action='store_true',
                       help='Fuse dropout and residual adds.')
    group.add_argument('--fuse-relu-dropout', action='store_true',
                       help='Fuse Relu and Dropout.')
    return group
