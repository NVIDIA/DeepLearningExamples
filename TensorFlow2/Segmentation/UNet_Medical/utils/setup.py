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

import os
import numpy as np
import tensorflow as tf

import dllogger as logger
import horovod.tensorflow as hvd
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend


def set_flags(params):
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '0'
    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

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


def prepare_model_dir(params):
    model_dir = os.path.join(params.model_dir, "model_checkpoint")
    model_dir = model_dir if (hvd.rank() == 0 and not params.benchmark) else None
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        if ('train' in params.exec_mode) and (not params.resume_training):
            os.system('rm -rf {}/*'.format(model_dir))

    return model_dir


def get_logger(params):
    backends = []
    if hvd.rank() == 0:
        backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            backends += [JSONStreamBackend(Verbosity.VERBOSE, params.log_dir)]
    logger.init(backends=backends)
    return logger
