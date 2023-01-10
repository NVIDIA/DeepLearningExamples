# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import os

from moflow.config import CONFIGS
from moflow.runtime.logger import LOGGING_LEVELS


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--data_dir', type=str, default='/data', help='Location for the dataset.')
PARSER.add_argument('--config_name', type=str, default='zinc250k', choices=list(CONFIGS),
                    help='The config to choose. This parameter allows one to switch between different datasets '
                         'and their dedicated configurations of the neural network. By default, a pre-defined "zinc250k" config is used.')
PARSER.add_argument('--results_dir', type=str, default='/results', help='Directory where checkpoints are stored.')
PARSER.add_argument('--predictions_path', type=str, default='/results/predictions.smi',
                    help='Path to store generated molecules. If an empty string is provided, predictions will not be '
                         'saved (useful for benchmarking and debugging).')
PARSER.add_argument('--log_path', type=str, default=None,
                    help='Path for DLLogger log. This file will contain information about the speed and '
                         'accuracy of the model during training and inference. Note that if the file '
                         'already exists, new logs will be added at the end.')
PARSER.add_argument('--log_interval', type=int, default=20, help='Frequency for writing logs, expressed in steps.')
PARSER.add_argument('--warmup_steps', type=int, default=20,
                    help='Number of warmup steps. This value is used for benchmarking and for CUDA graph capture.')
PARSER.add_argument('--steps', type=int, default=-1,
                    help='Number of steps used for training/inference. This parameter allows finishing '
                         'training earlier than the specified number of epochs. If used with inference, '
                         'it allows generating  more molecules (by default only a single batch of molecules is generated).')
PARSER.add_argument('--save_epochs', type=int, default=5,
                    help='Frequency for saving checkpoints, expressed in epochs. If -1 is provided, checkpoints will not be saved.')
PARSER.add_argument('--eval_epochs', type=int, default=5,
                    help='Evaluation frequency, expressed in epochs. If -1 is provided, an evaluation will not be performed.')
PARSER.add_argument('--learning_rate', type=float, default=0.0005, help='Base learning rate.')
PARSER.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter for the optimizer.')
PARSER.add_argument('--beta2', type=float, default=0.99, help='beta2 parameter for the optimizer.')
PARSER.add_argument('--clip', type=float, default=1, help='Gradient clipping norm.')
PARSER.add_argument('--epochs', type=int, default=300,
                    help='Number of training epochs. Note that you can finish training mid-epoch by using "--steps" flag.')
PARSER.add_argument('--batch_size', type=int, default=512, help='Batch size per GPU.')
PARSER.add_argument('--num_workers', type=int, default=4, help='Number of workers in the data loader.')
PARSER.add_argument('--seed', type=int, default=1, help='Random seed used to initialize the distributed loaders.')
PARSER.add_argument('--local_rank', default=os.environ.get('LOCAL_RANK', 0), type=int,
                    help='rank of the GPU, used to launch distributed training. This argument is specified '
                         'automatically by `torchrun` and does not have to be provided by the user.')
PARSER.add_argument('--temperature', type=float, default=0.3, help='Temperature used for sampling.')
PARSER.add_argument('--val_batch_size', type=int, default=100, help='Number of molecules to generate during validation step.')
PARSER.add_argument('--allow_untrained', action='store_true',
                    help='Allow sampling molecules from an untrained network. Useful for performance benchmarking or debugging purposes.')
PARSER.add_argument('--correct_validity', action='store_true', help='Apply validity correction after the generation of the molecules.')
PARSER.add_argument('--amp', action='store_true', help='Use Automatic Mixed Precision.')
PARSER.add_argument('--cuda_graph', action='store_true', help='Capture GPU kernels with CUDA graphs. This option allows to speed up training.')
PARSER.add_argument('--jit', action='store_true', help='Compile the model with `torch.jit.script`. Can be used to speed up training or inference.')
PARSER.add_argument('--verbosity', type=int, default=1, choices=list(LOGGING_LEVELS),
                    help='Verbosity level. Specify the following values: 0, 1, 2, 3, where 0 means minimal '
                    'verbosity (errors only) and 3 - maximal (debugging).')
