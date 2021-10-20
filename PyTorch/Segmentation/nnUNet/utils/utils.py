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

import glob
import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import call

import numpy as np
import torch
from dllogger import JSONStreamBackend, Logger, StdOutBackend, Verbosity
from sklearn.model_selection import KFold


def is_main_process():
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def set_cuda_devices(args):
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


def verify_ckpt_path(args):
    resume_path = os.path.join(args.results, "checkpoints", "last.ckpt")
    ckpt_path = resume_path if args.resume_training and os.path.exists(resume_path) else args.ckpt_path
    return ckpt_path


def get_task_code(args):
    return f"{args.task}_{args.dim}d"


def get_config_file(args):
    task_code = get_task_code(args)
    if args.data != "/data":
        path = os.path.join(args.data, "config.pkl")
    else:
        path = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(path, "rb"))


def get_dllogger(results):
    return Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(results, "logs.json")),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: f"Epoch: {step} "),
        ]
    )


def get_tta_flips(dim):
    if dim == 2:
        return [[2], [3], [2, 3]]
    return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]


def make_empty_dir(path):
    call(["rm", "-rf", path])
    os.makedirs(path)


def flip(data, axis):
    return torch.flip(data, dims=axis)


def positive_int(value):
    ivalue = int(value)
    assert ivalue > 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue


def float_0_1(value):
    ivalue = float(value)
    assert 0 <= ivalue <= 1, f"Argparse error. Expected float to be in range (0, 1), but got {value}"
    return ivalue


def get_unet_params(args):
    config = get_config_file(args)
    patch_size, spacings = config["patch_size"], config["spacings"]
    strides, kernels, sizes = [], [], patch_size[:]
    while True:
        spacing_ratio = [spacing / min(spacings) for spacing in spacings]
        stride = [2 if ratio <= 2 and size >= 2 * args.min_fmap else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
        if len(strides) == 6:
            break
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return config["in_channels"], config["n_class"], kernels, strides, patch_size


def log(logname, dice, results="/results"):
    dllogger = Logger(
        backends=[
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(results, logname)),
            StdOutBackend(Verbosity.VERBOSE, step_format=lambda step: ""),
        ]
    )
    metrics = {}
    metrics.update({"Mean dice": round(dice.mean().item(), 2)})
    metrics.update({f"L{j+1}": round(m.item(), 2) for j, m in enumerate(dice)})
    dllogger.log(step=(), data=metrics)
    dllogger.flush()


def layout_2d(img, lbl):
    batch_size, depth, channels, height, weight = img.shape
    img = torch.reshape(img, (batch_size * depth, channels, height, weight))
    if lbl is not None:
        lbl = torch.reshape(lbl, (batch_size * depth, 1, height, weight))
        return img, lbl
    return img


def get_split(data, idx):
    return list(np.array(data)[idx])


def load_data(path, files_pattern):
    return sorted(glob.glob(os.path.join(path, files_pattern)))


def get_path(args):
    if args.data != "/data":
        return args.data
    data_path = os.path.join(args.data, get_task_code(args))
    if args.exec_mode == "predict" and not args.benchmark:
        data_path = os.path.join(data_path, "test")
    return data_path


def get_test_fnames(args, data_path, meta=None):
    kfold = KFold(n_splits=args.nfolds, shuffle=True, random_state=12345)
    test_imgs = load_data(data_path, "*_x.npy")

    if args.exec_mode == "predict" and "val" in data_path:
        _, val_idx = list(kfold.split(test_imgs))[args.fold]
        test_imgs = sorted(get_split(test_imgs, val_idx))
        if meta is not None:
            meta = sorted(get_split(meta, val_idx))

    return test_imgs, meta


def get_main_args(strings=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg(
        "--exec_mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Execution mode to run the model",
    )
    arg("--data", type=str, default="/data", help="Path to data directory")
    arg("--results", type=str, default="/results", help="Path to results directory")
    arg("--logname", type=str, default=None, help="Name of dlloger output")
    arg("--task", type=str, help="Task number. MSD uses numbers 01-10")
    arg("--gpus", type=non_negative_int, default=1, help="Number of gpus")
    arg("--learning_rate", type=float, default=0.0008, help="Learning rate")
    arg("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    arg("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    arg("--tta", action="store_true", help="Enable test time augmentation")
    arg("--brats", action="store_true", help="Enable BraTS specific training and inference")
    arg("--deep_supervision", action="store_true", help="Enable deep supervision")
    arg("--more_chn", action="store_true", help="Create encoder with more channels")
    arg("--amp", action="store_true", help="Enable automatic mixed precision")
    arg("--benchmark", action="store_true", help="Run model benchmarking")
    arg("--focal", action="store_true", help="Use focal loss instead of cross entropy")
    arg("--sync_batchnorm", action="store_true", help="Enable synchronized batchnorm")
    arg("--save_ckpt", action="store_true", help="Enable saving checkpoint")
    arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    arg("--seed", type=non_negative_int, default=1, help="Random seed")
    arg("--skip_first_n_eval", type=non_negative_int, default=0, help="Skip the evaluation for the first n epochs.")
    arg("--ckpt_path", type=str, default=None, help="Path to checkpoint")
    arg("--fold", type=non_negative_int, default=0, help="Fold number")
    arg("--patience", type=positive_int, default=100, help="Early stopping patience")
    arg("--batch_size", type=positive_int, default=2, help="Batch size")
    arg("--val_batch_size", type=positive_int, default=4, help="Validation batch size")
    arg("--profile", action="store_true", help="Run dlprof profiling")
    arg("--momentum", type=float, default=0.99, help="Momentum factor")
    arg("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    arg("--save_preds", action="store_true", help="Enable prediction saving")
    arg("--dim", type=int, choices=[2, 3], default=3, help="UNet dimension")
    arg("--resume_training", action="store_true", help="Resume training from the last checkpoint")
    arg("--num_workers", type=non_negative_int, default=8, help="Number of subprocesses to use for data loading")
    arg("--epochs", type=non_negative_int, default=1000, help="Number of training epochs")
    arg("--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics")
    arg("--norm", type=str, choices=["instance", "batch", "group"], default="instance", help="Normalization layer")
    arg("--nvol", type=positive_int, default=1, help="Number of volumes which come into single batch size for 2D model")
    arg(
        "--data2d_dim",
        choices=[2, 3],
        type=int,
        default=3,
        help="Input data dimension for 2d model",
    )
    arg(
        "--oversampling",
        type=float_0_1,
        default=0.33,
        help="Probability of crop to have some region with positive label",
    )
    arg(
        "--overlap",
        type=float_0_1,
        default=0.5,
        help="Amount of overlap between scans during sliding window inference",
    )
    arg(
        "--affinity",
        type=str,
        default="socket_unique_interleaved",
        choices=[
            "socket",
            "single",
            "single_unique",
            "socket_unique_interleaved",
            "socket_unique_continuous",
            "disabled",
        ],
        help="type of CPU affinity",
    )
    arg(
        "--scheduler",
        action="store_true",
        help="Enable cosine rate scheduler with warmup",
    )
    arg(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Optimizer",
    )
    arg(
        "--min_fmap",
        type=non_negative_int,
        default=4,
        help="The minimal size that feature map can be reduced in bottleneck",
    )
    arg(
        "--blend",
        type=str,
        choices=["gaussian", "constant"],
        default="gaussian",
        help="How to blend output of overlapping windows",
    )
    arg(
        "--train_batches",
        type=non_negative_int,
        default=0,
        help="Limit number of batches for training (used for benchmarking mode only)",
    )
    arg(
        "--test_batches",
        type=non_negative_int,
        default=0,
        help="Limit number of batches for inference (used for benchmarking mode only)",
    )
    if strings is not None:
        arg(
            "strings",
            metavar="STRING",
            nargs="*",
            help="String for searching",
        )
        args = parser.parse_args(strings.split())
    else:
        args = parser.parse_args()
    return args
