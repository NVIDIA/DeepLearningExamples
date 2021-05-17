# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
""" Command line argument parser """
import argparse


# ===================================================================
#  Parser setup
# ===================================================================
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


# noinspection PyTypeChecker
PARSER = argparse.ArgumentParser(
    usage='main.py MODE [arguments...]',
    description='NVIDIA implementation of MastRCNN for TensorFlow 2.x',
    formatter_class=lambda prog: CustomFormatter(prog, max_help_position=100),
    add_help=False
)

RUNTIME_GROUP = PARSER.add_argument_group('Runtime')
HYPER_GROUP = PARSER.add_argument_group('Hyperparameters')
LOGGING_GROUP = PARSER.add_argument_group('Logging')
UTILITY_GROUP = PARSER.add_argument_group('Utility')

# ===================================================================
#  Runtime arguments
# ===================================================================
RUNTIME_GROUP.add_argument(
    'mode',
    type=str,
    metavar='MODE',
    help=(
        'One of supported execution modes:'
        '\n\ttrain - run in training mode'
        '\n\teval - run evaluation on eval data split'
        '\n\tinfer - run inference on eval data split'
    ),
    choices=[
        'train', 'eval', 'infer'
    ]
)

RUNTIME_GROUP.add_argument(
    '--data_dir',
    type=str,
    default='/data',
    metavar='DIR',
    help='Input directory containing the dataset'
)

RUNTIME_GROUP.add_argument(
    '--model_dir',
    type=str,
    default='/results',
    metavar='DIR',
    help='Output directory for information related to the model'
)

RUNTIME_GROUP.add_argument(
    '--backbone_checkpoint',
    type=str,
    default='/weights/rn50_tf_amp_ckpt_v20.06.0/nvidia_rn50_tf_amp',
    metavar='FILE',
    help='Pretrained checkpoint for resnet'
)

RUNTIME_GROUP.add_argument(
    '--eval_file',
    type=str,
    default='/data/annotations/instances_val2017.json',
    metavar='FILE',
    help='Path to the validation json file'
)

RUNTIME_GROUP.add_argument(
    '--epochs',
    type=int,
    default=12,
    help='Number of training epochs'
)

RUNTIME_GROUP.add_argument(
    '--steps_per_epoch',
    type=int,
    help='Number of steps (batches) per epoch. Defaults to dataset size divided by batch size.'
)

RUNTIME_GROUP.add_argument(
    '--eval_samples',
    type=int,
    default=None,
    metavar='N',
    help='Number of evaluation samples'
)

# ===================================================================
#  Hyperparameters arguments
# ===================================================================
HYPER_GROUP.add_argument(
    '--train_batch_size',
    type=int,
    default=4,
    metavar='N',
    help='Batch size (per GPU) used during training'
)

HYPER_GROUP.add_argument(
    '--eval_batch_size',
    type=int,
    default=8,
    metavar='N',
    help='Batch size used during evaluation'
)

HYPER_GROUP.add_argument(
    '--seed',
    type=int,
    default=None,
    metavar='SEED',
    help='Set a constant seed for reproducibility'
)

HYPER_GROUP.add_argument(
    '--l2_weight_decay',
    type=float,
    default=1e-4,
    metavar='L2D',
    help='Weight of l2 regularization'
)

HYPER_GROUP.add_argument(
    '--init_learning_rate',
    type=float,
    default=0.0,
    metavar='LR',
    help='Initial learning rate'
)

HYPER_GROUP.add_argument(
    '--learning_rate_values',
    type=float,
    nargs='*',
    default=[1e-2, 1e-3, 1e-4],
    metavar='D',
    help='Learning rate decay levels that are then scaled by global batch size'
)

HYPER_GROUP.add_argument(
    '--learning_rate_boundaries',
    type=float,
    nargs='*',
    metavar='N',
    default=[0.3, 8.0, 10.0],
    help='Steps (in epochs) at which learning rate changes'
)

HYPER_GROUP.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Optimizer momentum'
)

HYPER_GROUP.add_argument(
    '--finetune_bn',
    action='store_true',
    help='Is batchnorm finetuned training mode'
)

HYPER_GROUP.add_argument(
    '--use_synthetic_data',
    action='store_true',
    help='Use synthetic input data, meant for testing only'
)

HYPER_GROUP.add_argument(
    '--xla',
    action='store_true',
    help='Enable XLA JIT Compiler'
)

HYPER_GROUP.add_argument(
    '--amp',
    action='store_true',
    help='Enable automatic mixed precision'
)

# ===================================================================
#  Logging arguments
# ===================================================================
LOGGING_GROUP.add_argument(
    '--log_file',
    type=str,
    default='mrcnn-dlll.json',
    metavar='FILE',
    help='Output file for DLLogger logs'
)

LOGGING_GROUP.add_argument(
    '--log_every',
    type=int,
    default=100,
    metavar='N',
    help='Log performance every N steps'
)

LOGGING_GROUP.add_argument(
    '--log_warmup_steps',
    type=int,
    default=100,
    metavar='N',
    help='Number of steps that will be ignored when collecting perf stats'
)

LOGGING_GROUP.add_argument(
    '--log_graph',
    action='store_true',
    help='Print details about TF graph'
)

LOGGING_GROUP.add_argument(
    '--log_tensorboard',
    type=str,
    metavar='PATH',
    help='When provided saves tensorboard logs to given dir'
)


# ===================================================================
#  Utility arguments
# ===================================================================
UTILITY_GROUP.add_argument(
    '-h', '--help',
    action='help',
    help='Show this help message and exit'
)

UTILITY_GROUP.add_argument(
    '-v', '--verbose',
    action='store_true',
    help='Displays debugging logs'
)

UTILITY_GROUP.add_argument(
    '--eagerly',
    action='store_true',
    help='Runs model in eager mode. Use for debugging only as it reduces performance.'
)
