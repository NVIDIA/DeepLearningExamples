import os
import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import call

import torch


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
    config_file = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(config_file, "rb"))


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


def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--exec_mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Execution mode to run the model",
    )
    parser.add_argument("--data", type=str, default="/data", help="Path to data directory")
    parser.add_argument("--results", type=str, default="/results", help="Path to results directory")
    parser.add_argument("--logname", type=str, default=None, help="Name of dlloger output")
    parser.add_argument("--task", type=str, help="Task number. MSD uses numbers 01-10")
    parser.add_argument("--gpus", type=non_negative_int, default=1, help="Number of gpus")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    parser.add_argument("--negative_slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmarking")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable deep supervision")
    parser.add_argument("--sync_batchnorm", action="store_true", help="Enable synchronized batchnorm")
    parser.add_argument("--save_ckpt", action="store_true", help="Enable saving checkpoint")
    parser.add_argument("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--seed", type=non_negative_int, default=1, help="Random seed")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--fold", type=non_negative_int, default=0, help="Fold number")
    parser.add_argument("--patience", type=positive_int, default=100, help="Early stopping patience")
    parser.add_argument("--lr_patience", type=positive_int, default=70, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--batch_size", type=positive_int, default=2, help="Batch size")
    parser.add_argument("--val_batch_size", type=positive_int, default=4, help="Validation batch size")
    parser.add_argument("--steps", nargs="+", type=positive_int, required=False, help="Steps for multistep scheduler")
    parser.add_argument("--create_idx", action="store_true", help="Create index files for tfrecord")
    parser.add_argument("--profile", action="store_true", help="Run dlprof profiling")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momentum factor")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    parser.add_argument("--save_preds", action="store_true", help="Enable prediction saving")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3, help="UNet dimension")
    parser.add_argument("--resume_training", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--factor", type=float, default=0.3, help="Scheduler factor")
    parser.add_argument(
        "--num_workers", type=non_negative_int, default=8, help="Number of subprocesses to use for data loading"
    )
    parser.add_argument(
        "--min_epochs", type=non_negative_int, default=30, help="Force training for at least these many epochs"
    )
    parser.add_argument(
        "--max_epochs", type=non_negative_int, default=10000, help="Stop training after this number of epochs"
    )
    parser.add_argument(
        "--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics"
    )
    parser.add_argument(
        "--oversampling",
        type=float_0_1,
        default=0.33,
        help="Probability of crop to have some region with positive label",
    )
    parser.add_argument(
        "--norm", type=str, choices=["instance", "batch", "group"], default="instance", help="Normalization layer"
    )
    parser.add_argument(
        "--overlap",
        type=float_0_1,
        default=0.25,
        help="Amount of overlap between scans during sliding window inference",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "multistep", "cosine", "plateau"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="radam",
        choices=["sgd", "adam", "adamw", "radam", "fused_adam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--val_mode",
        type=str,
        choices=["gaussian", "constant"],
        default="gaussian",
        help="How to blend output of overlapping windows",
    )
    parser.add_argument(
        "--train_batches",
        type=non_negative_int,
        default=0,
        help="Limit number of batches for training (used for benchmarking mode only)",
    )
    parser.add_argument(
        "--test_batches",
        type=non_negative_int,
        default=0,
        help="Limit number of batches for inference (used for benchmarking mode only)",
    )
    args = parser.parse_args()
    return args
