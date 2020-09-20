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

from utils.setup import prepare_model_dir, get_logger, build_estimator, set_flags
from utils.cmd_util import PARSER, parse_args
from utils.data_loader import Dataset
from utils.hooks.profiling_hook import ProfilingHook
from utils.hooks.training_hook import TrainingHook


def main(_):
    """
    Starting point of the application
    """
    hvd.init()
    set_flags()
    params = parse_args(PARSER.parse_args())
    model_dir = prepare_model_dir(params)
    logger = get_logger(params)

    estimator = build_estimator(params, model_dir)

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
