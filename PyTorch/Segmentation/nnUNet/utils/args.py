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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace


def positive_int(value):
    ivalue = int(value)
    assert ivalue > 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected non-negative integer but got {value}"
    return ivalue


def float_0_1(value):
    fvalue = float(value)
    assert 0 <= fvalue <= 1, f"Argparse error. Expected float value to be in range (0, 1), but got {value}"
    return fvalue


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
    arg("--config", type=str, default=None, help="Config file with arguments")
    arg("--logname", type=str, default="logs.json", help="Name of dlloger output")
    arg("--task", type=str, default="01", help="Task number. MSD uses numbers 01-10")
    arg("--gpus", type=non_negative_int, default=1, help="Number of gpus")
    arg("--nodes", type=non_negative_int, default=1, help="Number of nodes")
    arg("--learning_rate", type=float, default=0.0008, help="Learning rate")
    arg("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    arg("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    arg("--tta", action="store_true", help="Enable test time augmentation")
    arg("--brats", action="store_true", help="Enable BraTS specific training and inference")
    arg("--deep_supervision", action="store_true", help="Enable deep supervision")
    arg("--invert_resampled_y", action="store_true", help="Resize predictions to match label size before resampling")
    arg("--amp", action="store_true", help="Enable automatic mixed precision")
    arg("--benchmark", action="store_true", help="Run model benchmarking")
    arg("--focal", action="store_true", help="Use focal loss instead of cross entropy")
    arg("--save_ckpt", action="store_true", help="Enable saving checkpoint")
    arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    arg("--seed", type=non_negative_int, default=None, help="Random seed")
    arg("--skip_first_n_eval", type=non_negative_int, default=0, help="Skip the evaluation for the first n epochs.")
    arg("--ckpt_path", type=str, default=None, help="Path for loading checkpoint")
    arg("--ckpt_store_dir", type=str, default="/results", help="Path for saving checkpoint")
    arg("--fold", type=non_negative_int, default=0, help="Fold number")
    arg("--patience", type=positive_int, default=100, help="Early stopping patience")
    arg("--batch_size", type=positive_int, default=2, help="Batch size")
    arg("--val_batch_size", type=positive_int, default=4, help="Validation batch size")
    arg("--momentum", type=float, default=0.99, help="Momentum factor")
    arg("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    arg("--save_preds", action="store_true", help="Enable prediction saving")
    arg("--dim", type=int, choices=[2, 3], default=3, help="UNet dimension")
    arg("--resume_training", action="store_true", help="Resume training from the last checkpoint")
    arg("--num_workers", type=non_negative_int, default=8, help="Number of subprocesses to use for data loading")
    arg("--epochs", type=non_negative_int, default=1000, help="Number of training epochs.")
    arg("--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics")
    arg("--nvol", type=positive_int, default=4, help="Number of volumes which come into single batch size for 2D model")
    arg("--depth", type=non_negative_int, default=5, help="The depth of the encoder")
    arg("--min_fmap", type=non_negative_int, default=4, help="Minimal dimension of feature map in the bottleneck")
    arg("--deep_supr_num", type=non_negative_int, default=2, help="Number of deep supervision heads")
    arg("--res_block", action="store_true", help="Enable residual blocks")
    arg("--filters", nargs="+", help="[Optional] Set U-Net filters", default=None, type=int)
    arg("--layout", type=str, default="NCDHW")
    arg("--brats22_model", action="store_true", help="Use BraTS22 model")
    arg(
        "--norm",
        type=str,
        choices=["instance", "instance_nvfuser", "batch", "group"],
        default="instance",
        help="Normalization layer",
    )
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
        default=0.4,
        help="Probability of crop to have some region with positive label",
    )
    arg(
        "--overlap",
        type=float_0_1,
        default=0.25,
        help="Amount of overlap between scans during sliding window inference",
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
        "--blend",
        type=str,
        choices=["gaussian", "constant"],
        default="constant",
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
        if args.config is not None:
            config = json.load(open(args.config, "r"))
            args = vars(args)
            args.update(config)
            args = Namespace(**args)

    with open(f"{args.results}/params.json", "w") as f:
        json.dump(vars(args), f)

    return args
