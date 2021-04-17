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

import os
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd


def set_flags(params):
    # os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    # os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '0'
    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['HOROVOD_CACHE_CAPACITY'] = "0"
    os.environ['HOROVOD_CYCLE_TIME'] = "1.0"
    if params.intraop_threads:
        os.environ['TF_NUM_INTRAOP_THREADS'] = params.intraop_threads
    if params.interop_threads:
        os.environ['TF_NUM_INTEROP_THREADS'] = params.interop_threads

    if params.use_xla:
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_lazy_compilation=false --tf_xla_auto_jit=1 --tf_xla_async_io_level=1"
        os.environ['TF_EXTRA_PTXAS_OPTIONS'] = "-sw200428197=true"
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        assert tf.config.experimental.get_memory_growth(gpu)
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    if params.use_amp:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
