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

import ctypes
import os
import pickle
from subprocess import run

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print0(text):
    print(text)


def get_task_code(args):
    return f"{args.task}_{args.dim}d"


def get_config_file(args):
    if args.data != "/data":
        path = os.path.join(args.data, "config.pkl")
    else:
        task_code = get_task_code(args)
        path = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(path, "rb"))


def set_cuda_devices(args):
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


def verify_ckpt_path(args):
    if args.resume_training:
        resume_path_ckpt = os.path.join(
            args.ckpt_path if args.ckpt_path is not None else "", "checkpoints", "last.ckpt"
        )
        resume_path_results = os.path.join(args.results, "checkpoints", "last.ckpt")
        if os.path.exists(resume_path_ckpt):
            return resume_path_ckpt
        if os.path.exists(resume_path_results):
            return resume_path_results
        print("[Warning] Checkpoint not found. Starting training from scratch.")
        return None
    if args.ckpt_path is None or not os.path.isfile(args.ckpt_path):
        print(f"Provided checkpoint {args.ckpt_path} is not a file. Starting training from scratch.")
        return None
    return args.ckpt_path


def make_empty_dir(path):
    run(["rm", "-rf", path])
    os.makedirs(path)


def get_stats(pred, targ, class_idx):
    tp = np.logical_and(pred == class_idx, targ == class_idx).sum()
    fn = np.logical_and(pred != class_idx, targ == class_idx).sum()
    fp = np.logical_and(pred == class_idx, targ != class_idx).sum()
    return tp, fn, fp


def set_granularity():
    _libcudart = ctypes.CDLL("libcudart.so")
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128
