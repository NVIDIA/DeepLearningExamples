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

"""Entry point of the application.

This file serves as entry point to the run of UNet for segmentation of neuronal processes.

Example:
    Training can be adjusted by modifying the arguments specified below::

        $ python main.py --exec_mode train --model_dir /dataset ...

"""

import horovod.tensorflow as hvd

from model.unet import Unet
from run import train, evaluate, predict
from utils.setup import get_logger, set_flags, prepare_model_dir
from utils.cmd_util import PARSER, parse_args
from utils.data_loader import Dataset


def main():
    """
    Starting point of the application
    """
    hvd.init()
    params = parse_args(PARSER.parse_args())
    set_flags(params)
    model_dir = prepare_model_dir(params)
    params.model_dir = model_dir
    logger = get_logger(params)

    model = Unet()

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold=params.crossvalidation_idx,
                      augment=params.augment,
                      gpu_id=hvd.rank(),
                      num_gpus=hvd.size(),
                      seed=params.seed)

    if 'train' in params.exec_mode:
        train(params, model, dataset, logger)

    if 'evaluate' in params.exec_mode:
        if hvd.rank() == 0:
            evaluate(params, model, dataset, logger)

    if 'predict' in params.exec_mode:
        if hvd.rank() == 0:
            predict(params, model, dataset, logger)


if __name__ == '__main__':
    main()
