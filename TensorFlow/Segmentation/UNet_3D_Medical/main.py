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
import logging

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from dataset.data_loader import Dataset, CLASSES
from runtime.hooks import get_hooks, ProfilingHook, TrainingHook
from runtime.arguments import PARSER
from runtime.setup import prepare_model_dir, build_estimator, set_flags, get_logger


def parse_evaluation_results(result):
    data = {CLASSES[i]: result[CLASSES[i]] for i in range(len(CLASSES))}
    data['MeanDice'] = sum([result[CLASSES[i]] for i in range(len(CLASSES))]) / len(CLASSES)
    data['WholeTumor'] = result['WholeTumor']
    return data


def main():
    tf.get_logger().setLevel(logging.ERROR)
    hvd.init()
    params = PARSER.parse_args()
    model_dir = prepare_model_dir(params)
    logger = get_logger(params)

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold_idx=params.fold,
                      n_folds=params.num_folds,
                      params=params)

    estimator = build_estimator(params=params, model_dir=model_dir)

    max_steps = params.max_steps // (1 if params.benchmark else hvd.size())

    if 'train' in params.exec_mode:
        training_hooks = get_hooks(params, logger)
        estimator.train(
            input_fn=dataset.train_fn,
            steps=max_steps,
            hooks=training_hooks)

    if 'evaluate' in params.exec_mode:
        result = estimator.evaluate(input_fn=dataset.eval_fn, steps=dataset.eval_size)
        data = parse_evaluation_results(result)
        if hvd.rank() == 0:
            logger.log(step=(), data=data)

    if 'predict' == params.exec_mode:
        inference_hooks = get_hooks(params, logger)
        if hvd.rank() == 0:
            count = 1 if not params.benchmark else 2 * params.warmup_steps * params.batch_size // dataset.test_size
            predictions = estimator.predict(
                input_fn=lambda: dataset.test_fn(count=count,
                                                 drop_remainder=params.benchmark), hooks=inference_hooks)

            for idx, p in enumerate(predictions):
                volume = p['predictions']
                if not params.benchmark:
                    np.save(os.path.join(params.model_dir, "vol_{}.npy".format(idx)), volume)

    if 'debug_train' == params.exec_mode:
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        if hvd.rank() == 0:
            hooks += [TrainingHook(log_every=params.log_every,
                                   logger=logger,
                                   tensor_names=['total_loss_ref:0']),
                      ProfilingHook(warmup_steps=params.warmup_steps,
                                    global_batch_size=hvd.size() * params.batch_size,
                                    logger=logger,
                                    mode='train')]

        estimator.train(
            input_fn=dataset.synth_train_fn,
            steps=max_steps,
            hooks=hooks)

    if 'debug_predict' == params.exec_mode:
        if hvd.rank() == 0:
            hooks = [ProfilingHook(warmup_steps=params.warmup_steps,
                                   global_batch_size=params.batch_size,
                                   logger=logger,
                                   mode='inference')]
            count = 2 * params.warmup_steps
            predictions = estimator.predict(input_fn=lambda: dataset.synth_predict_fn(count=count),
                                            hooks=hooks)
            for p in predictions:
                _ = p['predictions']


if __name__ == '__main__':
    set_flags()
    main()
