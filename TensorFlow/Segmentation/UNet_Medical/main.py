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

import os

import horovod.tensorflow as hvd
import math
import numpy as np
import tensorflow as tf
from PIL import Image

from utils.cmd_util import PARSER, _cmd_params
from utils.data_loader import Dataset
from utils.hooks.profiling_hook import ProfilingHook
from utils.hooks.training_hook import TrainingHook
from utils.model_fn import unet_fn
from dllogger.logger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


def main(_):
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()
    params = _cmd_params(flags)
    np.random.seed(params.seed)
    tf.compat.v1.random.set_random_seed(params.seed)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

    if params.use_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    hvd.init()

    # Build run config
    gpu_options = tf.compat.v1.GPUOptions()
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    if params.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    run_config = tf.estimator.RunConfig(
        save_summary_steps=1,
        tf_random_seed=None,
        session_config=config,
        save_checkpoints_steps=params.max_steps // hvd.size(),
        keep_checkpoint_max=1)

    # Build the estimator model
    estimator = tf.estimator.Estimator(
        model_fn=unet_fn,
        model_dir=params.model_dir,
        config=run_config,
        params=params)

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold=params.crossvalidation_idx,
                      augment=params.augment,
                      gpu_id=hvd.rank(),
                      num_gpus=hvd.size(),
                      seed=params.seed)

    if 'train' in params.exec_mode:
        max_steps = params.max_steps // (1 if params.benchmark else hvd.size())
        hooks = [hvd.BroadcastGlobalVariablesHook(0),
                 TrainingHook(logger,
                              max_steps=max_steps,
                              log_every=params.log_every)]

        if params.benchmark and hvd.rank() == 0:
            hooks.append(ProfilingHook(logger,
                                       batch_size=params.batch_size,
                                       log_every=params.log_every,
                                       warmup_steps=params.warmup_steps,
                                       mode='train'))

        estimator.train(
            input_fn=dataset.train_fn,
            steps=max_steps,
            hooks=hooks)

    if 'evaluate' in params.exec_mode:
        if hvd.rank() == 0:
            results = estimator.evaluate(input_fn=dataset.eval_fn, steps=dataset.eval_size)
            logger.log(step=(),
                       data={"eval_ce_loss": float(results["eval_ce_loss"]),
                             "eval_dice_loss": float(results["eval_dice_loss"]),
                             "eval_total_loss": float(results["eval_total_loss"]),
                             "eval_dice_score": float(results["eval_dice_score"])})

    if 'predict' in params.exec_mode:
        if hvd.rank() == 0:
            predict_steps = dataset.test_size
            hooks = None
            if params.benchmark:
                hooks = [ProfilingHook(logger,
                                       batch_size=params.batch_size,
                                       log_every=params.log_every,
                                       warmup_steps=params.warmup_steps,
                                       mode="test")]
                predict_steps = params.warmup_steps * 2 * params.batch_size

            predictions = estimator.predict(
                input_fn=lambda: dataset.test_fn(count=math.ceil(predict_steps / dataset.test_size)),
                hooks=hooks)
            binary_masks = [np.argmax(p['logits'], axis=-1).astype(np.uint8) * 255 for p in predictions]

            if not params.benchmark:
                multipage_tif = [Image.fromarray(mask).resize(size=(512, 512), resample=Image.BILINEAR)
                                 for mask in binary_masks]

                output_dir = os.path.join(params.model_dir, 'pred')

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                multipage_tif[0].save(os.path.join(output_dir, 'test-masks.tif'),
                                      compression="tiff_deflate",
                                      save_all=True,
                                      append_images=multipage_tif[1:])


if __name__ == '__main__':
    tf.compat.v1.app.run()
