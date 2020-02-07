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

import argparse

PARSER = argparse.ArgumentParser(description="VNet")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'predict', 'train_and_predict', 'train_and_evaluate'],
                    required=True,
                    type=str)

PARSER.add_argument('--data_normalization',
                    choices=['zscore'],
                    default='zscore',
                    type=str)

PARSER.add_argument('--activation',
                    choices=['relu'],
                    default='relu',
                    type=str)

PARSER.add_argument('--resize_interpolator',
                    choices=['linear'],
                    default='linear',
                    type=str)

PARSER.add_argument('--loss',
                    choices=['dice'],
                    default='dice',
                    type=str)

PARSER.add_argument('--normalization_layer',
                    choices=['batchnorm'],
                    default='batchnorm',
                    type=str)

PARSER.add_argument('--pooling',
                    choices=['conv_pool'],
                    default='conv_pool',
                    type=str)

PARSER.add_argument('--upsampling',
                    choices=['transposed_conv'],
                    default='transposed_conv',
                    type=str)

PARSER.add_argument('--seed',
                    default=0,
                    type=int)

PARSER.add_argument('--input_shape', nargs='+', type=int, default=[32, 32, 32])
PARSER.add_argument('--upscale_blocks', nargs='+', type=int, default=[3, 3])
PARSER.add_argument('--downscale_blocks', nargs='+', type=int, default=[3, 3, 3])

PARSER.add_argument('--convolution_size',
                    choices=[3, 5],
                    default=3,
                    type=int)

PARSER.add_argument('--batch_size',
                    required=True,
                    type=int)

PARSER.add_argument('--log_every',
                    default=10,
                    type=int)

PARSER.add_argument('--warmup_steps',
                    default=200,
                    type=int)

PARSER.add_argument('--train_epochs',
                    default=1,
                    type=int)

PARSER.add_argument('--optimizer',
                    choices=['rmsprop'],
                    default='rmsprop',
                    type=str)

PARSER.add_argument('--gradient_clipping',
                    choices=['global_norm'],
                    default='global_norm',
                    type=str)

PARSER.add_argument('--base_lr',
                    default=0.0001,
                    type=float)

PARSER.add_argument('--momentum',
                    default=0.0,
                    type=float)

PARSER.add_argument('--train_split',
                    default=1.0,
                    type=float)

PARSER.add_argument('--split_seed',
                    default=0,
                    type=int)

PARSER.add_argument('--model_dir',
                    required=True,
                    type=str)

PARSER.add_argument('--log_dir',
                    default=None,
                    type=str)

PARSER.add_argument('--data_dir',
                    required=True,
                    type=str)

PARSER.add_argument('--benchmark', dest='benchmark', action='store_true', default=False)
PARSER.add_argument('--use_amp', dest='use_amp', action='store_true', default=False)
PARSER.add_argument('--augment', dest='augment', action='store_true', default=False)
