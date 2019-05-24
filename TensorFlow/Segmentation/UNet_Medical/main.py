# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""Entry point of the application.

This file serves as entry point to the training of UNet for segmentation of neuronal processes.

Example:
    Training can be adjusted by modifying the arguments specified below::

        $ python main.py --exec_mode train --model_dir /datasets ...

"""

import argparse
import os

import tensorflow as tf

from dllogger.logger import LOGGER

from utils.runner import Runner

PARSER = argparse.ArgumentParser(description="UNet-medical")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'train_and_predict', 'predict', 'benchmark'],
                    type=str,
                    default='train_and_predict',
                    help="""Which execution mode to run the model into"""
                    )

PARSER.add_argument('--model_dir',
                    type=str,
                    default='./results',
                    help="""Output directory for information related to the model"""
                    )

PARSER.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="""Input directory containing the dataset for training the model"""
                    )

PARSER.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="""Size of each minibatch per GPU""")

PARSER.add_argument('--max_steps',
                    type=int,
                    default=1000,
                    help="""Maximum number of steps (batches) used for training""")

PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help="""Random seed""")

PARSER.add_argument('--weight_decay',
                    type=float,
                    default=0.0005,
                    help="""Weight decay coefficient""")

PARSER.add_argument('--log_every',
                    type=int,
                    default=100,
                    help="""Log performance every n steps""")

PARSER.add_argument('--warmup_steps',
                    type=int,
                    default=200,
                    help="""Number of warmup steps""")

PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.01,
                    help="""Learning rate coefficient for SGD""")

PARSER.add_argument('--momentum',
                    type=float,
                    default=0.99,
                    help="""Momentum coefficient for SGD""")

PARSER.add_argument('--decay_steps',
                    type=float,
                    default=5000,
                    help="""Decay steps for inverse learning rate decay""")

PARSER.add_argument('--decay_rate',
                    type=float,
                    default=0.95,
                    help="""Decay rate for learning rate decay""")

PARSER.add_argument('--augment', dest='augment', action='store_true',
                    help="""Perform data augmentation during training""")
PARSER.add_argument('--no-augment', dest='augment', action='store_false')
PARSER.set_defaults(augment=False)

PARSER.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help="""Collect performance metrics during training""")
PARSER.add_argument('--no-benchmark', dest='benchmark', action='store_false')
PARSER.set_defaults(augment=False)

PARSER.add_argument('--use_amp', dest='use_amp', action='store_true',
                    help="""Train using TF-AMP""")
PARSER.set_defaults(use_amp=False)


def _cmd_params(flags):
    return {
        'model_dir': flags.model_dir,
        'batch_size': flags.batch_size,
        'data_dir': flags.data_dir,
        'max_steps': flags.max_steps,
        'weight_decay': flags.weight_decay,
        'dtype': tf.float32,
        'learning_rate': flags.learning_rate,
        'momentum': flags.momentum,
        'benchmark': flags.benchmark,
        'augment': flags.augment,
        'exec_mode': flags.exec_mode,
        'seed': flags.seed,
        'use_amp': flags.use_amp,
        'log_every': flags.log_every,
        'warmup_steps': flags.warmup_steps,
        'decay_steps': flags.decay_steps,
        'decay_rate': flags.decay_rate,
    }


def main(_):
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()

    params = _cmd_params(flags)

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_DISABLE_NVTX_RANGES'] = '1'

    if params['use_amp']:
        assert params['dtype'] == tf.float32, "TF-AMP requires FP32 precision"

        LOGGER.log("TF AMP is activated - Experimental Feature")
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    runner = Runner(params)

    if 'train' in params['exec_mode'] \
            or 'train_and predict' in params['exec_mode']:
        runner.train()
    if 'train_and predict' in params['exec_mode'] \
            or 'predict' in params['exec_mode']:
        runner.predict()
    if 'benchmark' in params['exec_mode']:
        runner.benchmark()


if __name__ == '__main__':
    tf.app.run()
