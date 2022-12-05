# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import multiprocessing
import os
import pickle
import shutil
import sys
from functools import wraps
from pathlib import Path

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def hvd_init():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")


def set_tf_flags(args):
    os.environ["CUDA_CACHE_DISABLE"] = "0"
    os.environ["HOROVOD_GPU_ALLREDUCE"] = "NCCL"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = str(hvd.size())
    os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
    os.environ["TF_ADJUST_HUE_FUSED"] = "1"
    os.environ["TF_ADJUST_SATURATION_FUSED"] = "1"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_AUTOTUNE_THRESHOLD"] = "2"
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
    os.environ["TF_ENABLE_LAYOUT_NHWC"] = "1"
    os.environ["TF_CPP_VMODULE"] = "4"

    if args.xla:
        os.environ["TF_XLA_ENABLE_GPU_GRAPH_CAPTURE"] = "1"
        if args.amp:
            os.environ["XLA_FLAGS"] = "--xla_gpu_force_conv_nhwc"
        tf.config.optimizer.set_jit(True)

    if hvd.size() > 1:
        tf.config.threading.set_inter_op_parallelism_threads(max(2, (multiprocessing.cpu_count() // hvd.size()) - 2))
    else:
        tf.config.threading.set_inter_op_parallelism_threads(8)

    if args.amp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def is_main_process():
    return hvd.rank() == 0


def progress_bar(iterable, *args, quiet, **kwargs):
    if quiet or not is_main_process():
        return iterable
    return tqdm(iterable, *args, **kwargs)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if is_main_process():
            return fn(*args, **kwargs)

    return wrapped_fn


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_task_code(args):
    return f"{args.task}_{args.dim}d_tf2"


def get_config_file(args):
    task_code = get_task_code(args)
    path = os.path.join(args.data, "config.pkl")
    if not os.path.exists(path):
        path = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(path, "rb"))


def get_tta_flips(dim):
    if dim == 2:
        return [[1], [2], [1, 2]]
    return [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]


def make_empty_dir(path, force=False):
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            print(f"Output path {path} exists and is not a directory." "Please remove it and try again.")
            sys.exit(1)
        else:
            if not force:
                decision = input(f"Output path {path} exists. Continue and replace it? [Y/n]: ")
                if decision.strip().lower() not in ["", "y"]:
                    sys.exit(1)
            shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
