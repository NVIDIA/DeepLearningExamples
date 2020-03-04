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

"""Command line argument parsing"""
import argparse
from munch import Munch

PARSER = argparse.ArgumentParser(description="UNet-medical")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'train_and_predict', 'predict', 'evaluate', 'train_and_evaluate'],
                    type=str,
                    default='train_and_evaluate',
                    help="""Execution mode of running the model""")

PARSER.add_argument('--model_dir',
                    type=str,
                    default='./results',
                    help="""Output directory for information related to the model""")

PARSER.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="""Input directory containing the dataset for training the model""")

PARSER.add_argument('--log_dir',
                    type=str,
                    default=None,
                    help="""Output directory for training logs""")

PARSER.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="""Size of each minibatch per GPU""")

PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.0001,
                    help="""Learning rate coefficient for AdamOptimizer""")

PARSER.add_argument('--crossvalidation_idx',
                    type=int,
                    default=None,
                    help="""Chosen fold for cross-validation. Use None to disable cross-validation""")

PARSER.add_argument('--max_steps',
                    type=int,
                    default=1000,
                    help="""Maximum number of steps (batches) used for training""")

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

PARSER.add_argument('--seed',
                    type=int,
                    default=0,
                    help="""Random seed""")

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

PARSER.add_argument('--use_xla', dest='use_xla', action='store_true',
                    help="""Train using XLA""")
PARSER.set_defaults(use_amp=False)

PARSER.add_argument('--use_trt', dest='use_trt', action='store_true',
                    help="""Use TF-TRT""")
PARSER.set_defaults(use_trt=False)


def _cmd_params(flags):
    return Munch({
        'exec_mode': flags.exec_mode,
        'model_dir': flags.model_dir,
        'data_dir': flags.data_dir,
        'log_dir': flags.log_dir,
        'batch_size': flags.batch_size,
        'learning_rate': flags.learning_rate,
        'crossvalidation_idx': flags.crossvalidation_idx,
        'max_steps': flags.max_steps,
        'weight_decay': flags.weight_decay,
        'log_every': flags.log_every,
        'warmup_steps': flags.warmup_steps,
        'augment': flags.augment,
        'benchmark': flags.benchmark,
        'seed': flags.seed,
        'use_amp': flags.use_amp,
        'use_trt': flags.use_trt,
        'use_xla': flags.use_xla,
    })
