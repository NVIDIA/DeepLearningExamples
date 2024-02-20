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
import logging
import distutils.util
import dllogger
from utils.mode import RunScope
from utils.utility import get_num_trainers
from utils.save_load import _PDOPT_SUFFIX, _PDPARAMS_SUFFIX

_AUTO_LAST_EPOCH = 'auto'


def _get_full_path_of_ckpt(args):
    if args.from_checkpoint is None:
        args.last_epoch_of_checkpoint = -1
        return

    def _check_file_exist(path_with_prefix):
        pdopt_path = path_with_prefix + _PDOPT_SUFFIX
        pdparams_path = path_with_prefix + _PDPARAMS_SUFFIX
        found = False
        if os.path.exists(pdopt_path) and os.path.exists(pdparams_path):
            found = True
        return found, pdopt_path, pdparams_path

    target_from_checkpoint = os.path.join(args.from_checkpoint,
                                          args.model_prefix)
    if args.last_epoch_of_checkpoint is None:
        args.last_epoch_of_checkpoint = -1
    elif args.last_epoch_of_checkpoint == _AUTO_LAST_EPOCH:
        folders = os.listdir(args.from_checkpoint)
        args.last_epoch_of_checkpoint = -1
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

            if folder > args.last_epoch_of_checkpoint and \
               _check_file_exist(tmp_ckpt_path)[0]:
                args.last_epoch_of_checkpoint = folder
        epoch_with_prefix = os.path.join(str(args.last_epoch_of_checkpoint), args.model_prefix) \
                            if args.last_epoch_of_checkpoint > -1 else args.model_prefix
        target_from_checkpoint = os.path.join(args.from_checkpoint,
                                              epoch_with_prefix)
    else:
        try:
            args.last_epoch_of_checkpoint = int(args.last_epoch_of_checkpoint)
        except ValueError:
            raise ValueError(f"The value of --last-epoch-of-checkpoint should be None, {_AUTO_LAST_EPOCH}"  \
                            f" or integer >= 0, but receive {args.last_epoch_of_checkpoint}")

    args.from_checkpoint = target_from_checkpoint
    found, pdopt_path, pdparams_path = _check_file_exist(args.from_checkpoint)
    if not found:
        args.from_checkpoint = None
        args.last_epoch_of_checkpoint = -1
        logging.warning(
            f"Cannot find {pdopt_path} and {pdparams_path}, disable --from-checkpoint."
        )


def _get_full_path_of_pretrained_params(args):
    if args.from_pretrained_params is None:
        args.last_epoch_of_checkpoint = -1
        return

    args.from_pretrained_params = os.path.join(args.from_pretrained_params,
                                               args.model_prefix)
    pdparams_path = args.from_pretrained_params + _PDPARAMS_SUFFIX
    if not os.path.exists(pdparams_path):
        args.from_pretrained_params = None
        logging.warning(
            f"Cannot find {pdparams_path}, disable --from-pretrained-params.")
    args.last_epoch_of_checkpoint = -1


def print_args(args):
    args_for_log = copy.deepcopy(args)

    # Due to dllogger cannot serialize Enum into JSON.
    if hasattr(args_for_log, 'run_scope'):
        args_for_log.run_scope = args_for_log.run_scope.value

    dllogger.log(step='PARAMETER', data=vars(args_for_log))


def check_and_process_args(args):
    # Precess the scope of run
    run_scope = None
    for scope in RunScope:
        if args.run_scope == scope.value:
            run_scope = scope
            break
    assert run_scope is not None, \
           f"only support {[scope.value for scope in RunScope]} as run_scope"
    args.run_scope = run_scope

    # Precess image layout and channel
    args.image_channel = args.image_shape[0]
    if args.data_layout == "NHWC":
        args.image_shape = [
            args.image_shape[1], args.image_shape[2], args.image_shape[0]
        ]

    # Precess learning rate
    args.lr = get_num_trainers() * args.lr

    # Precess model loading
    assert not (args.from_checkpoint is not None and \
                args.from_pretrained_params is not None), \
           "--from-pretrained-params and --from-checkpoint should " \
           "not be set simultaneously."
    _get_full_path_of_pretrained_params(args)
    _get_full_path_of_ckpt(args)
    args.start_epoch = args.last_epoch_of_checkpoint + 1

    # Precess benchmark
    if args.benchmark:
        assert args.run_scope in [
            RunScope.TRAIN_ONLY, RunScope.EVAL_ONLY
        ], "If benchmark enabled, run_scope must be `train_only` or `eval_only`"

    # Only run one epoch when benchmark or eval_only.
    if args.benchmark or \
      (args.run_scope == RunScope.EVAL_ONLY):
        args.epochs = args.start_epoch + 1

    if args.run_scope == RunScope.EVAL_ONLY:
        args.eval_interval = 1


def add_general_args(parser):
    group = parser.add_argument_group('General')
    group.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoint/',
        help='A path to store trained models.')
    group.add_argument(
        '--inference-dir',
        type=str,
        default='./inference/',
        help='A path to store inference model once the training is finished.'
    )
    group.add_argument(
        '--run-scope',
        default='train_eval',
        choices=('train_eval', 'train_only', 'eval_only'),
        help='Running scope. It should be one of {train_eval, train_only, eval_only}.'
    )
    group.add_argument(
        '--epochs',
        type=int,
        default=90,
        help='The number of epochs for training.')
    group.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='The iteration interval to save checkpoints.')
    group.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='The iteration interval to test trained models on a given validation dataset. ' \
             'Ignored when --run-scope is train_only.'
    )
    group.add_argument(
        '--print-interval',
        type=int,
        default=10,
        help='The iteration interval to show training/evaluation message.')
    group.add_argument(
        '--report-file',
        type=str,
        default='./train.json',
        help='A file in which to store JSON experiment report.')
    group.add_argument(
        '--benchmark', action='store_true', help='To enable benchmark mode.')
    group.add_argument(
        '--benchmark-steps',
        type=int,
        default=100,
        help='Steps for benchmark run, only be applied when --benchmark is set.'
    )
    group.add_argument(
        '--benchmark-warmup-steps',
        type=int,
        default=100,
        help='Warmup steps for benchmark run, only be applied when --benchmark is set.'
    )
    group.add_argument(
        '--model-prefix',
        type=str,
        default="resnet_50_paddle",
        help='The prefix name of model files to save/load.')
    group.add_argument(
        '--from-pretrained-params',
        type=str,
        default=None,
        help='A folder path which contains pretrained parameters, that is a file in name' \
             ' --model-prefix + .pdparams. It should not be set with --from-checkpoint' \
             ' at the same time.'
    )
    group.add_argument(
        '--from-checkpoint',
        type=str,
        default=None,
        help='A checkpoint path to resume training. It should not be set ' \
             'with --from-pretrained-params at the same time. The path provided ' \
             'could be a folder contains < epoch_id/ckpt_files > or < ckpt_files >.'
    )
    group.add_argument(
        '--last-epoch-of-checkpoint',
        type=str,
        default=None,
        help='The epoch id of the checkpoint given by --from-checkpoint. ' \
             'It should be None, auto or integer >= 0. If it is set as ' \
             'None, then training will start from 0-th epoch. If it is set as ' \
             'auto, then it will search largest integer-convertable folder ' \
             ' --from-checkpoint, which contains required checkpoint. ' \
             'Default is None.'
    )
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
    return parser


def add_advance_args(parser):
    group = parser.add_argument_group('Advanced Training')
    # AMP
    group.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision training (AMP).')
    group.add_argument(
        '--scale-loss',
        type=float,
        default=1.0,
        help='The loss scalar for AMP training, only be applied when --amp is set.'
    )
    group.add_argument(
        '--use-dynamic-loss-scaling',
        action='store_true',
        help='Enable dynamic loss scaling in AMP training, only be applied when --amp is set.'
    )
    group.add_argument(
        '--use-pure-fp16',
        action='store_true',
        help='Enable pure FP16 training, only be applied when --amp is set.')
    group.add_argument(
        '--fuse-resunit',
        action='store_true',
        help='Enable CUDNNv8 ResUnit fusion, only be applied when --amp is set.')
    # ASP
    group.add_argument(
        '--asp',
        action='store_true',
        help='Enable automatic sparse training (ASP).')
    group.add_argument(
        '--prune-model',
        action='store_true',
        help='Prune model to 2:4 sparse pattern, only be applied when --asp is set.'
    )
    group.add_argument(
        '--mask-algo',
        default='mask_1d',
        choices=('mask_1d', 'mask_2d_greedy', 'mask_2d_best'),
        help='The algorithm to generate sparse masks. It should be one of ' \
             '{mask_1d, mask_2d_greedy, mask_2d_best}. This only be applied ' \
             'when --asp and --prune-model is set.'
    )
    # QAT
    group.add_argument(
        '--qat',
        action='store_true',
        help='Enable quantization aware training (QAT).')
    return parser


def add_dataset_args(parser):
    def float_list(x):
        return list(map(float, x.split(',')))

    def int_list(x):
        return list(map(int, x.split(',')))

    dataset_group = parser.add_argument_group('Dataset')
    dataset_group.add_argument(
        '--image-root',
        type=str,
        default='/imagenet',
        help='A root folder of train/val images. It should contain train and val folders, ' \
             'which store corresponding images.'
    )
    dataset_group.add_argument(
        '--image-shape',
        type=int_list,
        default=[4, 224, 224],
        help='The image shape. Its shape should be [channel, height, width].')

    # Data Loader
    dataset_group.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='The batch size for both training and evaluation.')
    dataset_group.add_argument(
        '--dali-random-seed',
        type=int,
        default=42,
        help='The random seed for DALI data loader.')
    dataset_group.add_argument(
        '--dali-num-threads',
        type=int,
        default=4,
        help='The number of threads applied to DALI data loader.')
    dataset_group.add_argument(
        '--dali-output-fp16',
        action='store_true',
        help='Output FP16 data from DALI data loader.')

    # Augmentation
    augmentation_group = parser.add_argument_group('Data Augmentation')
    augmentation_group.add_argument(
        '--crop-size',
        type=int,
        default=224,
        help='The size to crop input images.')
    augmentation_group.add_argument(
        '--rand-crop-scale',
        type=float_list,
        default=[0.08, 1.],
        help='Range from which to choose a random area fraction.')
    augmentation_group.add_argument(
        '--rand-crop-ratio',
        type=float_list,
        default=[3.0 / 4, 4.0 / 3],
        help='Range from which to choose a random aspect ratio (width/height).')
    augmentation_group.add_argument(
        '--normalize-scale',
        type=float,
        default=1.0 / 255.0,
        help='A scalar to normalize images.')
    augmentation_group.add_argument(
        '--normalize-mean',
        type=float_list,
        default=[0.485, 0.456, 0.406],
        help='The mean values to normalize RGB images.')
    augmentation_group.add_argument(
        '--normalize-std',
        type=float_list,
        default=[0.229, 0.224, 0.225],
        help='The std values to normalize RGB images.')
    augmentation_group.add_argument(
        '--resize-short',
        type=int,
        default=256,
        help='The length of the shorter dimension of the resized image.')
    return parser


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--model-arch-name',
        type=str,
        default='ResNet50',
        help='The model architecture name. It should be one of {ResNet50}.')
    group.add_argument(
        '--num-of-class',
        type=int,
        default=1000,
        help='The number classes of images.')
    group.add_argument(
        '--data-layout',
        default='NCHW',
        choices=('NCHW', 'NHWC'),
        help='Data format. It should be one of {NCHW, NHWC}.')
    group.add_argument(
        '--bn-weight-decay',
        action='store_true',
        help='Apply weight decay to BatchNorm shift and scale.')
    return parser


def add_training_args(parser):
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--label-smoothing',
        type=float,
        default=0.1,
        help='The ratio of label smoothing.')
    group.add_argument(
        '--optimizer',
        default='Momentum',
        metavar="OPTIMIZER",
        choices=('Momentum'),
        help='The name of optimizer. It should be one of {Momentum}.')
    group.add_argument(
        '--momentum',
        type=float,
        default=0.875,
        help='The momentum value of optimizer.')
    group.add_argument(
        '--weight-decay',
        type=float,
        default=3.0517578125e-05,
        help='The coefficient of weight decay.')
    group.add_argument(
        '--lr-scheduler',
        default='Cosine',
        metavar="LR_SCHEDULER",
        choices=('Cosine'),
        help='The name of learning rate scheduler. It should be one of {Cosine}.'
    )
    group.add_argument(
        '--lr', type=float, default=0.256, help='The initial learning rate.')
    group.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
        help='The number of epochs for learning rate warmup.')
    group.add_argument(
        '--warmup-start-lr',
        type=float,
        default=0.0,
        help='The initial learning rate for warmup.')
    return parser


def add_trt_args(parser):
    def int_list(x):
        return list(map(int, x.split(',')))

    group = parser.add_argument_group('Paddle-TRT')
    group.add_argument(
        '--device',
        type=int,
        default='0',
        help='The GPU device id for Paddle-TRT inference.'
    )
    group.add_argument(
        '--inference-dir',
        type=str,
        default='./inference',
        help='A path to load inference models.'
    )
    group.add_argument(
        '--data-layout',
        default='NCHW',
        choices=('NCHW', 'NHWC'),
        help='Data format. It should be one of {NCHW, NHWC}.')
    group.add_argument(
        '--precision',
        default='FP32',
        choices=('FP32', 'FP16', 'INT8'),
        help='The precision of TensorRT. It should be one of {FP32, FP16, INT8}.'
    )
    group.add_argument(
        '--workspace-size',
        type=int,
        default=(1 << 30),
        help='The memory workspace of TensorRT in MB.')
    group.add_argument(
        '--min-subgraph-size',
        type=int,
        default=3,
        help='The minimal subgraph size to enable PaddleTRT.')
    group.add_argument(
        '--use-static',
        type=distutils.util.strtobool,
        default=False,
        help='Fix TensorRT engine at first running.')
    group.add_argument(
        '--use-calib-mode',
        type=distutils.util.strtobool,
        default=False,
        help='Use the PTQ calibration of PaddleTRT int8.')
    group.add_argument(
        '--report-file',
        type=str,
        default='./inference.json',
        help='A file in which to store JSON inference report.')
    group.add_argument(
        '--use-synthetic',
        type=distutils.util.strtobool,
        default=False,
        help='Apply synthetic data for benchmark.')
    group.add_argument(
        '--benchmark-steps',
        type=int,
        default=100,
        help='Steps for benchmark run, only be applied when --benchmark is set.'
    )
    group.add_argument(
        '--benchmark-warmup-steps',
        type=int,
        default=100,
        help='Warmup steps for benchmark run, only be applied when --benchmark is set.'
    )
    group.add_argument(
        '--show-config',
        type=distutils.util.strtobool,
        default=True,
        help='To show arguments.')
    return parser


def parse_args(script='train'):
    assert script in ['train', 'inference']
    parser = argparse.ArgumentParser(
        description=f'PaddlePaddle RN50v1.5 {script} script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if script == 'train':
        parser = add_general_args(parser)
        parser = add_dataset_args(parser)
        parser = add_model_args(parser)
        parser = add_training_args(parser)
        parser = add_advance_args(parser)
        args = parser.parse_args()
        check_and_process_args(args)
    else:
        parser = add_trt_args(parser)
        parser = add_dataset_args(parser)
        args = parser.parse_args()
        # Precess image layout and channel
        args.image_channel = args.image_shape[0]
        if args.data_layout == "NHWC":
            args.image_shape = [
                args.image_shape[1], args.image_shape[2], args.image_shape[0]
            ]

    return args
