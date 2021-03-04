# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse

# Default train dataset size
TRAIN_DATASET_SIZE = 59761827


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tensorflow2 WideAndDeep Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )

    locations = parser.add_argument_group('location of datasets')

    locations.add_argument('--train_data_pattern', type=str, default='/outbrain/tfrecords/train/part*', nargs='+',
                           help='Pattern of training file names. For example if training files are train_000.tfrecord, '
                                'train_001.tfrecord then --train_data_pattern is train_*')

    locations.add_argument('--eval_data_pattern', type=str, default='/outbrain/tfrecords/eval/part*', nargs='+',
                           help='Pattern of eval file names. For example if eval files are eval_000.tfrecord, '
                                'eval_001.tfrecord then --eval_data_pattern is eval_*')

    locations.add_argument('--transformed_metadata_path', type=str, default='/outbrain/tfrecords',
                           help='Path to transformed_metadata for feature specification reconstruction')

    locations.add_argument('--use_checkpoint', default=False, action='store_true',
                           help='Use checkpoint stored in model_dir path')

    locations.add_argument('--model_dir', type=str, default='/outbrain/checkpoints',
                           help='Destination where model checkpoint will be saved')

    locations.add_argument('--results_dir', type=str, default='/results',
                           help='Directory to store training results')

    locations.add_argument('--log_filename', type=str, default='log.json',
                           help='Name of the file to store dlloger output')

    training_params = parser.add_argument_group('training parameters')

    training_params.add_argument('--training_set_size', type=int, default=TRAIN_DATASET_SIZE,
                                 help='Number of samples in the training set')

    training_params.add_argument('--global_batch_size', type=int, default=131072,
                                 help='Total size of training batch')

    training_params.add_argument('--eval_batch_size', type=int, default=131072,
                                 help='Total size of evaluation batch')

    training_params.add_argument('--num_epochs', type=int, default=20,
                                 help='Number of training epochs')

    training_params.add_argument('--cpu', default=False, action='store_true',
                                 help='Run computations on the CPU')

    training_params.add_argument('--amp', default=False, action='store_true',
                                 help='Enable automatic mixed precision conversion')

    training_params.add_argument('--xla', default=False, action='store_true',
                                 help='Enable XLA conversion')

    training_params.add_argument('--linear_learning_rate', type=float, default=0.02,
                                 help='Learning rate for linear model')

    training_params.add_argument('--deep_learning_rate', type=float, default=0.00012,
                                 help='Learning rate for deep model')

    training_params.add_argument('--deep_warmup_epochs', type=float, default=6,
                                 help='Number of learning rate warmup epochs for deep model')

    model_construction = parser.add_argument_group('model construction')

    model_construction.add_argument('--deep_hidden_units', type=int, default=[1024, 1024, 1024, 1024, 1024], nargs="+",
                                    help='Hidden units per layer for deep model, separated by spaces')

    model_construction.add_argument('--deep_dropout', type=float, default=0.1,
                                    help='Dropout regularization for deep model')

    run_params = parser.add_argument_group('run mode parameters')

    run_params.add_argument('--evaluate', default=False, action='store_true',
                            help='Only perform an evaluation on the validation dataset, don\'t train')

    run_params.add_argument('--benchmark', action='store_true', default=False,
                            help='Run training or evaluation benchmark to collect performance metrics', )

    run_params.add_argument('--benchmark_warmup_steps', type=int, default=500,
                            help='Number of warmup steps before start of the benchmark')

    run_params.add_argument('--benchmark_steps', type=int, default=1000,
                            help='Number of steps for performance benchmark')

    run_params.add_argument('--affinity', type=str, default='socket_unique_interleaved',
                            choices=['socket', 'single', 'single_unique',
                                     'socket_unique_interleaved',
                                     'socket_unique_continuous',
                                     'disabled'],
                            help='Type of CPU affinity')

    return parser.parse_args()
