# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
#
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

import os
import copy
import argparse
import distutils.util
import logging
import dllogger
from utils.task import Task
from utils.save_load import _PDOPT_SUFFIX, _PDPARAMS_SUFFIX, _PROGRESS_SUFFIX

_AUTO_LAST_EPOCH = 'auto'

_DEFAULT_BERT_CONFIG = {
    'bert-large-uncased': './bert_configs/bert-large-uncased.json',
    'bert-large-cased': './bert_configs/bert-large-cased.json',
    'bert-base-uncased': './bert_configs/bert-base-uncased.json',
    'bert-base-cased': './bert_configs/bert-base-cased.json'
}


def _get_full_path_of_ckpt(args):
    if args.from_checkpoint is None:
        args.last_step_of_checkpoint = 0
        return

    def _check_file_exist(path_with_prefix):
        pdopt_path = path_with_prefix + _PDOPT_SUFFIX
        pdparams_path = path_with_prefix + _PDPARAMS_SUFFIX
        progress_path = path_with_prefix + _PROGRESS_SUFFIX
        found = False
        if os.path.exists(pdopt_path) and os.path.exists(
                pdparams_path) and os.path.exists(progress_path):
            found = True
        return found, pdopt_path, pdparams_path, progress_path

    if not os.path.exists(args.from_checkpoint):
        logging.warning(
            f"Start training from scratch since no checkpoint is found.")
        args.from_checkpoint = None
        args.last_step_of_checkpoint = 0
        return

    target_from_checkpoint = os.path.join(args.from_checkpoint,
                                          args.model_prefix)
    if args.last_step_of_checkpoint is None:
        args.last_step_of_checkpoint = 0
    elif args.last_step_of_checkpoint == _AUTO_LAST_EPOCH:
        folders = os.listdir(args.from_checkpoint)
        args.last_step_of_checkpoint = 0
        for folder in folders:
            tmp_ckpt_path = os.path.join(args.from_checkpoint, folder,
                                         args.model_prefix)

            try:
                folder = int(folder)
            except ValueError:
                logging.warning(
                    f"Skip folder '{folder}' since its name is not integer-convertable."
                )
                continue

            if folder > args.last_step_of_checkpoint and \
                _check_file_exist(tmp_ckpt_path)[0]:
                args.last_step_of_checkpoint = folder
        step_with_prefix = os.path.join(str(args.last_step_of_checkpoint), args.model_prefix) \
                            if args.last_step_of_checkpoint > 0 else args.model_prefix
        target_from_checkpoint = os.path.join(args.from_checkpoint,
                                              step_with_prefix)
    else:
        try:
            args.last_step_of_checkpoint = int(args.last_step_of_checkpoint)
        except ValueError:
            raise ValueError(f"The value of --last-step-of-checkpoint should be None, {_AUTO_LAST_EPOCH}"  \
                            f" or integer >= 0, but receive {args.last_step_of_checkpoint}")

    args.from_checkpoint = target_from_checkpoint
    found, pdopt_path, pdparams_path, progress_path = _check_file_exist(
        args.from_checkpoint)
    if not found:
        args.from_checkpoint = None
        args.last_step_of_checkpoint = 0
        logging.warning(
            f"Cannot find {pdopt_path} and {pdparams_path} and {progress_path}, disable --from-checkpoint."
        )


def _get_full_path_of_pretrained_params(args, task=Task.pretrain):
    if args.from_pretrained_params is None and args.from_phase1_final_params is None:
        args.last_step_of_checkpoint = 0
        return
    if task == Task.pretrain and args.from_phase1_final_params is not None and args.last_step_of_checkpoint == 0:
        args.from_pretrained_params = args.from_phase1_final_params

    args.from_pretrained_params = os.path.join(args.from_pretrained_params,
                                               args.model_prefix)
    pdparams_path = args.from_pretrained_params + _PDPARAMS_SUFFIX
    if not os.path.exists(pdparams_path):
        args.from_pretrained_params = None
        logging.warning(
            f"Cannot find {pdparams_path}, disable --from-pretrained-params.")
    args.last_step_of_checkpoint = 0


def print_args(args):
    args_for_log = copy.deepcopy(args)
    dllogger.log(step='PARAMETER', data=vars(args_for_log))


def check_and_process_args(args, task=Task.pretrain):
    if task == Task.pretrain:
        assert not (args.from_checkpoint is not None and \
            args.from_pretrained_params is not None), \
           "--from-pretrained-params and --from-checkpoint should " \
           "not be set simultaneously."
        assert not (args.phase1 and args.phase2), \
            "--phase1 and --phase2 should not be set simultaneously in bert pretraining."
        if args.from_phase1_final_params is not None:
            assert args.phase2, "--from-phase1-final-params should only be used in phase2"

        # SQuAD finetuning does not support suspend-resume yet.(TODO)
        _get_full_path_of_ckpt(args)

    if args.bert_model == 'custom':
        assert args.config_file is not None, "--config-file must be specified if --bert-model=custom"
    elif args.config_file is None:
        args.config_file = _DEFAULT_BERT_CONFIG[args.bert_model]
        logging.info(
            f"According to the name of bert_model, the default config_file: {args.config_file} will be used."
        )
    if args.from_checkpoint is None:
        _get_full_path_of_pretrained_params(args, task)

    assert os.path.isfile(
        args.config_file), f"Cannot find config file in {args.config_file}"


def add_global_args(parser, task=Task.pretrain):
    group = parser.add_argument_group('Global')
    if task == Task.pretrain:
        group.add_argument(
            '--input-dir',
            type=str,
            default=None,
            required=True,
            help='The input data directory. Should be specified by users and contain .hdf5 files for the task.'
        )
        group.add_argument('--num-workers', default=4, type=int)
    if task == Task.squad:
        group.add_argument(
            '--train-file',
            type=str,
            default=None,
            help='SQuAD json for training. E.g., train-v1.1.json')
        group.add_argument(
            '--predict-file',
            type=str,
            default=None,
            help='SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json'
        )
        group.add_argument(
            "--eval-script",
            help="Script to evaluate squad predictions",
            default="evaluate.py",
            type=str)
        group.add_argument(
            '--epochs',
            type=int,
            default=3,
            help='The number of epochs for training.')

    group.add_argument(
        '--vocab-file',
        type=str,
        default=None,
        required=True,
        help="Vocabulary mapping/file BERT was pretrainined on")
    group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        required=True,
        help='The output directory where the model checkpoints will be written. Should be specified by users.'
    )
    group.add_argument(
        '--bert-model',
        type=str,
        default='bert-large-uncased',
        choices=('bert-base-uncased', 'bert-base-cased', 'bert-large-uncased',
                 'bert-large-cased', 'custom'),
        help='Specifies the type of BERT model to use. If it is set as custom, '
        'the path to the config file must be given by specifying --config-file')
    group.add_argument(
        '--config-file',
        type=str,
        default=None,
        help='The BERT model config. If set to None, `<--bert-model>.json` in folder `bert_configs` will be used.'
    )
    group.add_argument(
        '--max-steps',
        type=int,
        default=None,
        required=True if task == Task.pretrain else False,
        help='Total number of training steps to perform.')
    group.add_argument(
        '--log-freq', type=int, default=10, help='Frequency of logging loss.')
    group.add_argument(
        '--num-steps-per-checkpoint',
        type=int,
        default=100,
        help='Number of update steps until a model checkpoint is saved to disk.'
    )
    # Init model
    group.add_argument(
        '--from-pretrained-params',
        type=str,
        default=None,
        help='Path to pretrained parameters. If set to None, no pretrained params will be used.'
    )
    group.add_argument(
        '--from-checkpoint',
        type=str,
        default=None,
        help='A checkpoint path to resume training. If set to None, no checkpoint will be used. ' \
             'If not None, --from-pretrained-params will be ignored.')
    group.add_argument(
        '--last-step-of-checkpoint',
        type=str,
        default=None,
        help='The step id of the checkpoint given by --from-checkpoint. ' \
             'It should be None, auto, or integer > 0. If it is set as ' \
             'None, then training will start from the 1-th epoch. If it is set as ' \
             'auto, then it will search largest integer-convertable folder ' \
             ' --from-checkpoint, which contains required checkpoint. '
    )
    if task == Task.pretrain:
        group.add_argument(
            '--from-phase1-final-params',
            type=str,
            default=None,
            help='Path to final checkpoint of phase1, which will be used to ' \
   'initialize the parameter in the first step of phase2, and ' \
                 'ignored in the rest steps of phase2.'
        )
        group.add_argument(
            '--steps-this-run',
            type=int,
            default=None,
            help='If provided, only run this many steps before exiting.' \
        )
    group.add_argument(
        '--seed', type=int, default=42, help="random seed for initialization")
    group.add_argument(
        '--report-file',
        type=str,
        default='./report.json',
        help='A file in which to store JSON experiment report.')
    group.add_argument(
        '--model-prefix',
        type=str,
        default='bert_paddle',
        help='The prefix name of model files to save/load.')
    group.add_argument(
        '--show-config',
        type=distutils.util.strtobool,
        default=True,
        help='To show arguments.')
    group.add_argument(
        '--enable-cpu-affinity',
        type=distutils.util.strtobool,
        default=True,
        help='To enable in-built GPU-CPU affinity.')
    group.add_argument(
        '--benchmark', action='store_true', help='To enable benchmark mode.')
    group.add_argument(
        '--benchmark-steps',
        type=int,
        default=20,
        help='Steps for a benchmark run, only applied when --benchmark is set.')
    group.add_argument(
        '--benchmark-warmup-steps',
        type=int,
        default=20,
        help='Warmup steps for a benchmark run, only applied when --benchmark is set.'
    )
    return parser


def add_training_args(parser, task=Task.pretrain):
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--optimizer',
        default='Lamb',
        metavar="OPTIMIZER",
        choices=('Lamb', 'AdamW'),
        help='The name of optimizer. It should be one of {Lamb, AdamW}.')
    group.add_argument(
        '--gradient-merge-steps',
        type=int,
        default=1,
        help="Number of update steps to accumualte before performing a backward/update pass."
    )
    group.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='The initial learning rate.')
    group.add_argument(
        '--warmup-start-lr',
        type=float,
        default=0.0,
        help='The initial learning rate for warmup.')
    group.add_argument(
        '--warmup-proportion',
        type=float,
        default=0.01,
        help='Proportion of training to perform linear learning rate warmup for. '
        'For example, 0.1 = 10%% of training.')
    group.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help='The exponential decay rate for the 1st moment estimates.')
    group.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help='The exponential decay rate for the 2st moment estimates.')
    group.add_argument(
        '--epsilon',
        type=float,
        default=1e-6,
        help='A small float value for numerical stability.')
    group.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='The weight decay coefficient.')
    group.add_argument(
        '--max-seq-length',
        default=512,
        type=int,
        help='The maximum total input sequence length after WordPiece tokenization. \n'
        'Sequences longer than this will be truncated, and sequences shorter \n'
        'than this will be padded.')
    if task == Task.pretrain:
        group.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='The batch size for training')
        group.add_argument(
            '--phase1',
            action='store_true',
            help='The phase of BERT pretraining. It should not be set ' \
                'with --phase2 at the same time.'
        )
        group.add_argument(
            '--phase2',
            action='store_true',
            help='The phase of BERT pretraining. It should not be set ' \
                'with --phase1 at the same time.'
        )
        group.add_argument(
            '--max-predictions-per-seq',
            default=80,
            type=int,
            help='The maximum total of masked tokens in the input sequence')

    if task == Task.squad:
        group.add_argument(
            "--do-train", action='store_true', help="Whether to run training.")
        group.add_argument(
            "--do-predict",
            action='store_true',
            help="Whether to run eval on the dev set.")
        group.add_argument(
            "--do-eval",
            action='store_true',
            help="Whether to use evaluate accuracy of predictions")
        group.add_argument(
            "--train-batch-size",
            default=32,
            type=int,
            help="Total batch size for training.")
        group.add_argument(
            "--predict-batch-size",
            default=8,
            type=int,
            help="Total batch size for predictions.")
        group.add_argument(
            "--verbose-logging",
            action='store_true',
            help="If true, all of the warnings related to data processing will be printed. "
            "A number of warnings are expected for a normal SQuAD evaluation.")
        group.add_argument(
            "--doc-stride",
            default=128,
            type=int,
            help="When splitting up a long document into chunks, how much stride to take "
            "between chunks.")
        group.add_argument(
            "--max-query-length",
            default=64,
            type=int,
            help="The maximum number of tokens for the question. Questions longer than this "
            "will be truncated to this length.")
        group.add_argument(
            "--n-best-size",
            default=20,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json "
            "output file.")
        group.add_argument(
            "--max-answer-length",
            default=30,
            type=int,
            help="The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another.")
        group.add_argument(
            "--do-lower-case",
            action='store_true',
            help="Whether to lower case the input text. True for uncased models, False for cased models."
        )
        group.add_argument(
            '--version-2-with-negative',
            action='store_true',
            help='If true, the SQuAD examples contain some that do not have an answer.'
        )
        group.add_argument(
            '--null-score-diff-threshold',
            type=float,
            default=0.0,
            help="If null_score - best_non_null is greater than the threshold predict null."
        )
    return parser


def add_advance_args(parser):
    group = parser.add_argument_group('Advanced Training')
    group.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision training (AMP).')
    group.add_argument(
        '--scale-loss',
        type=float,
        default=1.0,
        help='The loss scalar for AMP training, only applied when --amp is set.'
    )
    group.add_argument(
        '--use-dynamic-loss-scaling',
        action='store_true',
        help='Enable dynamic loss scaling in AMP training, only applied when --amp is set.'
    )
    group.add_argument(
        '--use-pure-fp16',
        action='store_true',
        help='Enable pure FP16 training, only applied when --amp is set.')

    return parser


def parse_args(task=Task.pretrain):
    parser = argparse.ArgumentParser(
        description="PaddlePaddle BERT pretraining script"
        if task == Task.pretrain else "PaddlePaddle SQuAD finetuning script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = add_global_args(parser, task)
    parser = add_training_args(parser, task)
    parser = add_advance_args(parser)

    args = parser.parse_args()
    check_and_process_args(args, task)
    return args
