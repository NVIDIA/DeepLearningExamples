# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import multiprocessing
import os

import dllogger
import horovod.tensorflow.keras as hvd
import tensorflow as tf

from data.feature_spec import FeatureSpec
from data.outbrain.dataloader import get_dataset
from data.outbrain.defaults import TEST_MAPPING, TRAIN_MAPPING, MULTIHOT_CHANNEL, ONEHOT_CHANNEL
from trainer.utils.gpu_affinity import set_affinity


def init_cpu(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    init_logger(full=True, args=args, logger=logger)

    logger.warning("--gpu flag not set, running computation on CPU")

    raise RuntimeError("CPU not supported with nvTabular dataloader")


def init_gpu(args, logger):
    hvd.init()

    init_logger(full=hvd.rank() == 0, args=args, logger=logger)
    if args.affinity != "disabled":
        gpu_id = hvd.local_rank()
        affinity = set_affinity(
            gpu_id=gpu_id, nproc_per_node=hvd.size(), mode=args.affinity
        )
        logger.warning(f"{gpu_id}: thread affinity: {affinity}")

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(
        max(2, (multiprocessing.cpu_count() // hvd.size()) - 2)
    )

    if args.amp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if args.xla:
        tf.config.optimizer.set_jit(True)

    # Max out L2 cache
    import ctypes

    _libcudart = ctypes.CDLL("libcudart.so")
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128



def init_logger(args, full, logger):
    if full:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(args.results_dir, args.log_filename)
        os.makedirs(args.results_dir, exist_ok=True)
        dllogger.init(
            backends=[
                dllogger.JSONStreamBackend(
                    verbosity=dllogger.Verbosity.VERBOSE, filename=log_path
                ),
                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE),
            ]
        )
        logger.warning("command line arguments: {}".format(json.dumps(vars(args))))
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        with open("{}/args.json".format(args.results_dir), "w") as f:
            json.dump(vars(args), f, indent=4)
    else:
        logger.setLevel(logging.ERROR)
        dllogger.init(backends=[])

    dllogger.log(data=vars(args), step="PARAMETER")
    dllogger.metadata("streaming_map_val", {"unit": None})
    dllogger.metadata("train_throughput", {"unit": "samples/s"})
    dllogger.metadata("validation_throughput", {"unit": "samples/s"})

def check_embedding_sizes(embedding_sizes: dict, feature_spec: FeatureSpec) -> None:
    onehot_features = feature_spec.get_names_by_channel(ONEHOT_CHANNEL)
    multihot_features = feature_spec.get_names_by_channel(MULTIHOT_CHANNEL)
    embedded_features = onehot_features + multihot_features
    for feature in embedded_features:
        assert feature in embedding_sizes
        assert isinstance(embedding_sizes[feature], int)

def create_config(args):
    assert not (
            args.cpu and args.amp
    ), "Automatic mixed precision conversion works only with GPU"
    assert (
            not args.benchmark or args.benchmark_warmup_steps < args.benchmark_steps
    ), "Number of benchmark steps must be higher than warmup steps"

    logger = logging.getLogger("tensorflow")

    if args.cpu:
        init_cpu(args, logger)
    else:
        init_gpu(args, logger)

    num_gpus = 1 if args.cpu else hvd.size()
    train_batch_size = args.global_batch_size // num_gpus
    eval_batch_size = args.eval_batch_size // num_gpus

    fspec_path = os.path.join(args.dataset_path, args.fspec_file)
    feature_spec = FeatureSpec.from_yaml(fspec_path)
    feature_spec.check_feature_spec(require_map_channel=args.map_calculation_enabled, world_size=hvd.size())
    train_dataset = get_dataset(
        feature_spec=feature_spec,
        mapping=TRAIN_MAPPING,
        batch_size=train_batch_size,
        shuffle=True,
        map_channel_enabled=False
    )

    eval_dataset = get_dataset(
        feature_spec=feature_spec,
        mapping=TEST_MAPPING,
        batch_size=eval_batch_size,
        shuffle=False,
        map_channel_enabled=args.map_calculation_enabled
    )

    with open(args.embedding_sizes_file) as opened:
        embedding_sizes = json.load(opened)
    check_embedding_sizes(embedding_sizes, feature_spec)
    config = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "feature_spec": feature_spec,
        "embedding_dimensions": embedding_sizes
    }

    return config
