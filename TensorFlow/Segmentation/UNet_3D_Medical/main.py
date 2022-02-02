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

""" Entry point of the application.

This file serves as entry point to the implementation of UNet3D for
medical image segmentation.

Example usage:
    $ python main.py --exec_mode train --data_dir ./data --batch_size 2
    --max_steps 1600 --amp

All arguments are listed under `python main.py -h`.
Full argument definition can be found in `arguments.py`.

"""
import os

import numpy as np
import horovod.tensorflow as hvd

from model.model_fn import unet_3d
from dataset.data_loader import Dataset, CLASSES
from runtime.hooks import get_hooks
from runtime.arguments import PARSER
from runtime.setup import build_estimator, set_flags, get_logger


def parse_evaluation_results(result, logger, step=()):
    """
    Parse DICE scores from the evaluation results

    :param result: Dictionary with metrics collected by the optimizer
    :param logger: Logger object
    :return:
    """
    data = {CLASSES[i]: float(result[CLASSES[i]]) for i in range(len(CLASSES))}
    data['mean_dice'] = sum([result[CLASSES[i]] for i in range(len(CLASSES))]) / len(CLASSES)
    data['whole_tumor'] = float(result['whole_tumor'])

    if hvd.rank() == 0:
        logger.log(step=step, data=data)

    return data


def main():
    """ Starting point of the application """
    hvd.init()
    set_flags()
    params = PARSER.parse_args()
    logger = get_logger(params)

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold_idx=params.fold,
                      n_folds=params.num_folds,
                      input_shape=params.input_shape,
                      params=params)

    estimator = build_estimator(params=params, model_fn=unet_3d)
    hooks = get_hooks(params, logger)

    if 'train' in params.exec_mode:
        max_steps = params.max_steps // (1 if params.benchmark else hvd.size())
        estimator.train(
            input_fn=dataset.train_fn,
            steps=max_steps,
            hooks=hooks)
    if 'evaluate' in params.exec_mode:
        result = estimator.evaluate(input_fn=dataset.eval_fn, steps=dataset.eval_size)
        _ = parse_evaluation_results(result, logger)
    if params.exec_mode == 'predict':
        if hvd.rank() == 0:
            predictions = estimator.predict(
                input_fn=dataset.test_fn, hooks=hooks)

            for idx, pred in enumerate(predictions):
                volume = pred['predictions']
                if not params.benchmark:
                    np.save(os.path.join(params.model_dir, "vol_{}.npy".format(idx)), volume)


if __name__ == '__main__':
    main()
