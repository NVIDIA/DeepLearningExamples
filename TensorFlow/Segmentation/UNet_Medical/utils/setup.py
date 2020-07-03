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

import dllogger as logger
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend

from utils.model_fn import unet_fn


def set_flags():
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


def prepare_model_dir(params):
    model_dir = os.path.join(params.model_dir, "model_checkpoint")
    model_dir = model_dir if (hvd.rank() == 0 and not params.benchmark) else None
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        if ('train' in params.exec_mode) and (not params.resume_training):
            os.system('rm -rf {}/*'.format(model_dir))

    return model_dir


def build_estimator(params, model_dir):
    if params.use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    np.random.seed(params.seed)
    tf.compat.v1.random.set_random_seed(params.seed)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpu_options = tf.compat.v1.GPUOptions()
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    if params.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    run_config = tf.estimator.RunConfig(
        save_summary_steps=1,
        tf_random_seed=params.seed,
        session_config=config,
        save_checkpoints_steps=(params.max_steps // hvd.size()) if hvd.rank() == 0 else None,
        keep_checkpoint_max=1)

    estimator = tf.estimator.Estimator(
        model_fn=unet_fn,
        model_dir=model_dir,
        config=run_config,
        params=params)
    return estimator


def get_logger(params):
    backends = []
    if hvd.rank() == 0:
        backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            backends += [JSONStreamBackend(Verbosity.VERBOSE, params.log_dir)]
    logger.init(backends=backends)
    return logger
