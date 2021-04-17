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

import json
import logging
import os

import dllogger
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow_transform as tft
from data.outbrain.dataloader import train_input_fn, eval_input_fn
from data.outbrain.features import PREBATCH_SIZE
from trainer.utils.gpu_affinity import set_affinity


def init_cpu(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    init_logger(
        full=True,
        args=args,
        logger=logger
    )

    logger.warning('--gpu flag not set, running computation on CPU')


def init_gpu(args, logger):
    hvd.init()

    init_logger(
        full=hvd.rank() == 0,
        args=args,
        logger=logger
    )
    if args.affinity != 'disabled':
        gpu_id = hvd.local_rank()
        affinity = set_affinity(
            gpu_id=gpu_id,
            nproc_per_node=hvd.size(),
            mode=args.affinity
        )
        logger.warning(f'{gpu_id}: thread affinity: {affinity}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)


def init_logger(args, full, logger):
    if full:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(args.results_dir, args.log_filename)
        os.makedirs(args.results_dir, exist_ok=True)
        dllogger.init(backends=[
            dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                       filename=log_path),
            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
        logger.warning('command line arguments: {}'.format(json.dumps(vars(args))))
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        with open('{}/args.json'.format(args.results_dir), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        logger.setLevel(logging.ERROR)
        dllogger.init(backends=[])

    dllogger.log(data=vars(args), step='PARAMETER')


def create_config(args):
    assert not (args.cpu and args.amp), \
        'Automatic mixed precision conversion works only with GPU'
    assert not args.benchmark or args.benchmark_warmup_steps < args.benchmark_steps, \
        'Number of benchmark steps must be higher than warmup steps'
    logger = logging.getLogger('tensorflow')

    if args.cpu:
        init_cpu(args, logger)
    else:
        init_gpu(args, logger)

    num_gpus = 1 if args.cpu else hvd.size()
    gpu_id = 0 if args.cpu else hvd.rank()
    train_batch_size = args.global_batch_size // num_gpus
    eval_batch_size = args.eval_batch_size // num_gpus
    steps_per_epoch = args.training_set_size / args.global_batch_size

    feature_spec = tft.TFTransformOutput(
        args.transformed_metadata_path
    ).transformed_feature_spec()

    train_spec_input_fn = train_input_fn(
        num_gpus=num_gpus,
        id=gpu_id,
        filepath_pattern=args.train_data_pattern,
        feature_spec=feature_spec,
        records_batch_size=train_batch_size // PREBATCH_SIZE,
    )

    eval_spec_input_fn = eval_input_fn(
        num_gpus=num_gpus,
        id=gpu_id,
        repeat=None if args.benchmark else 1,
        filepath_pattern=args.eval_data_pattern,
        feature_spec=feature_spec,
        records_batch_size=eval_batch_size // PREBATCH_SIZE
    )

    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train_spec_input_fn,
        'eval_dataset': eval_spec_input_fn
    }

    return config
