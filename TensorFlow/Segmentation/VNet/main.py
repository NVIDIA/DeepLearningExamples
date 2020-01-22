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

# pylint: enable=line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import pickle
import shutil

import horovod.tensorflow as hvd
import tensorflow as tf

import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from hooks.profiling_hook import ProfilingHook
from hooks.train_hook import TrainHook
from utils.cmd_util import PARSER
from utils.data_loader import MSDDataset
from utils.model_fn import vnet_v2


def main(_):
    tf.get_logger().setLevel(logging.ERROR)

    hvd.init()

    FLAGS = PARSER.parse_args()

    backends = []

    if hvd.rank() == 0:
        backends += [StdOutBackend(Verbosity.DEFAULT)]

        if FLAGS.log_dir:
            backends += [JSONStreamBackend(Verbosity.DEFAULT, FLAGS.log_dir)]

    DLLogger.init(backends=backends)

    for key in vars(FLAGS):
        DLLogger.log(step="PARAMETER", data={str(key): vars(FLAGS)[key]})

    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_DISABLE_NVTX_RANGES'] = '1'

    dataset = MSDDataset(json_path=os.path.join(FLAGS.data_dir, 'dataset.json'),
                         dst_size=FLAGS.input_shape,
                         seed=FLAGS.seed,
                         interpolator=FLAGS.resize_interpolator,
                         data_normalization=FLAGS.data_normalization,
                         batch_size=FLAGS.batch_size,
                         train_split=FLAGS.train_split,
                         split_seed=FLAGS.split_seed)

    FLAGS.labels = dataset.labels

    gpu_options = tf.GPUOptions()
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    if FLAGS.use_amp:
        config.graph_options.rewrite_options.auto_mixed_precision = 1

    run_config = tf.estimator.RunConfig(
        save_summary_steps=None,
        save_checkpoints_steps=None if FLAGS.benchmark else dataset.train_steps * FLAGS.train_epochs,
        save_checkpoints_secs=None,
        tf_random_seed=None,
        session_config=config,
        keep_checkpoint_max=1)

    estimator = tf.estimator.Estimator(
        model_fn=vnet_v2,
        model_dir=FLAGS.model_dir if hvd.rank() == 0 else None,
        config=run_config,
        params=FLAGS)

    train_hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    if 'train' in FLAGS.exec_mode:
        steps = dataset.train_steps * FLAGS.train_epochs

        if FLAGS.benchmark:
            steps = FLAGS.warmup_steps * 2
            if hvd.rank() == 0:
                train_hooks += [ProfilingHook(FLAGS.warmup_steps, FLAGS.batch_size * hvd.size(), DLLogger)]
        else:
            if hvd.rank() == 0:
                train_hooks += [TrainHook(FLAGS.log_every, DLLogger)]

        estimator.train(
            input_fn=lambda: dataset.train_fn(FLAGS.augment),
            steps=steps,
            hooks=train_hooks)

    if 'evaluate' in FLAGS.exec_mode:
        if hvd.rank() == 0:
            if FLAGS.train_split >= 1.0:
                raise ValueError("Missing argument: --train_split < 1.0")

            result = estimator.evaluate(
                input_fn=dataset.eval_fn,
                steps=dataset.eval_steps,
                hooks=[])

            DLLogger.log(step=tuple(), data={'background_dice': str(result['background dice'])})
            DLLogger.log(step=tuple(), data={'anterior_dice': str(result['Anterior dice'])})
            DLLogger.log(step=tuple(), data={'posterior_dice': str(result['Posterior dice'])})

    if 'predict' in FLAGS.exec_mode:
        count = 1
        hooks = []
        if hvd.rank() == 0:
            if FLAGS.benchmark:
                count = math.ceil((FLAGS.warmup_steps * 2) / dataset.test_steps)
                hooks += [ProfilingHook(FLAGS.warmup_steps, FLAGS.batch_size * hvd.size(), DLLogger, training=False)]

            predictions = estimator.predict(input_fn=lambda: dataset.test_fn(count=count),
                                            hooks=hooks)

            pred = [p['prediction'] for p in predictions]

            predict_path = os.path.join(FLAGS.model_dir, 'predictions')
            if os.path.exists(predict_path):
                shutil.rmtree(predict_path)

            os.makedirs(predict_path)

            pickle.dump(pred, open(os.path.join(predict_path, 'predictions.pkl'), 'wb'))


if __name__ == '__main__':
    tf.compat.v1.app.run()

