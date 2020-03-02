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

This file serves as entry point to the run of UNet for segmentation of neuronal processes.

Example:
    Training can be adjusted by modifying the arguments specified below::

        $ python main.py --exec_mode train --model_dir /dataset ...

"""

import os

import horovod.tensorflow as hvd
import tensorflow as tf

from model.unet import Unet
from run import train, evaluate, predict, restore_checkpoint
from utils.cmd_util import PARSER, _cmd_params
from utils.data_loader import Dataset
from dllogger.logger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


def main():
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()
    params = _cmd_params(flags)

    backends = [StdOutBackend(Verbosity.VERBOSE)]
    if params.log_dir is not None:
        backends.append(JSONStreamBackend(Verbosity.VERBOSE, params.log_dir))
    logger = Logger(backends)

    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = 'data'

    os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    hvd.init()

    if params.use_xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if params.use_amp:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    # Build the  model
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
            model = restore_checkpoint(model, params.model_dir)
            evaluate(params, model, dataset, logger)

    if 'predict' in params.exec_mode:
        if hvd.rank() == 0:
            model = restore_checkpoint(model, params.model_dir)
            predict(params, model, dataset, logger)


if __name__ == '__main__':
    main()
