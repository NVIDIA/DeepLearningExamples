import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a positive integer but got {value}")
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a non-negative integer but got {value}")
    return ivalue


def float_0_1(value):
    fvalue = float(value)
    if not (0 <= fvalue <= 1):
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a float from range (0, 1), but got {value}")
    return fvalue


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ArgParser(ArgumentParser):
    def arg(self, *args, **kwargs):
        return super().add_argument(*args, **kwargs)

    def flag(self, *args, **kwargs):
        return super().add_argument(*args, action="store_true", **kwargs)

    def boolean_flag(self, *args, **kwargs):
        return super().add_argument(*args, type=str2bool, nargs="?", const=True, metavar="BOOLEAN", **kwargs)


def get_main_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Runtime
    p.arg(
        "--exec-mode",
        "--exec_mode",
        type=str,
        choices=["train", "evaluate", "predict", "export"],
        default="train",
        help="Execution mode to run the model",
    )
    p.arg("--gpus", type=non_negative_int, default=1)
    p.arg("--data", type=Path, default=Path("/data"), help="Path to data directory")
    p.arg("--task", type=str, default="01", help="Task number, MSD uses numbers 01-10")
    p.arg("--dim", type=int, choices=[2, 3], default=3, help="UNet dimension")
    p.arg("--seed", type=non_negative_int, default=None, help="Random seed")
    p.flag("--benchmark", help="Run model benchmarking")
    p.boolean_flag("--tta", default=False, help="Enable test time augmentation")
    p.boolean_flag("--save-preds", "--save_preds", default=False, help="Save predictions")

    # Logging
    p.arg("--results", type=Path, default=Path("/results"), help="Path to results directory")
    p.arg("--logname", type=str, default="dllogger.json", help="DLLogger output filename")
    p.flag("--quiet", help="Minimalize stdout/stderr output")
    p.boolean_flag("--use-dllogger", "--use_dllogger", default=True, help="Use DLLogger logging")

    # Performance optimization
    p.boolean_flag("--amp", default=False, help="Enable automatic mixed precision")
    p.boolean_flag("--xla", default=False, help="Enable XLA compiling")

    # Training hyperparameters and loss fn customization
    p.arg("--batch-size", "--batch_size", type=positive_int, default=2, help="Batch size")
    p.arg("--learning-rate", "--learning_rate", type=float, default=0.0003, help="Learning rate")
    p.arg("--momentum", type=float, default=0.99, help="Momentum factor (SGD only)")
    p.arg(
        "--scheduler",
        type=str,
        default="cosine_annealing",
        choices=["none", "poly", "cosine", "cosine_annealing"],
        help="Learning rate scheduler",
    )
    p.arg("--end-learning-rate", type=float, default=0.00001, help="End learning rate for poly scheduler")
    p.arg(
        "--cosine-annealing-first-cycle-steps",
        type=positive_int,
        default=4096,
        help="Length of a cosine decay cycle in steps, only with 'cosine_annealing' scheduler",
    )
    p.arg(
        "--cosine-annealing-peak-decay", type=float_0_1, default=0.95, help="Multiplier reducing initial learning rate"
    )
    p.arg("--optimizer", type=str, default="adam", choices=["sgd", "adam", "radam"], help="Optimizer")
    p.boolean_flag("--deep-supervision", "--deep_supervision", default=False, help="Use deep supervision.")
    p.boolean_flag("--lookahead", default=False, help="Use Lookahead with the optimizer")
    p.arg("--weight-decay", "--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    p.boolean_flag(
        "--loss-batch-reduction",
        dest="reduce_batch",
        default=True,
        help="Reduce batch dimension first during loss calculation",
    )
    p.boolean_flag(
        "--loss-include-background",
        dest="include_background",
        default=False,
        help="Include background class to loss calculation",
    )

    # UNet architecture
    p.arg("--negative-slope", type=float, default=0.01, help="Negative slope for LeakyReLU")
    p.arg(
        "--norm",
        type=str,
        choices=["instance", "batch", "group", "none"],
        default="instance",
        help="Type of normalization layers",
    )

    # Checkpoints
    p.arg(
        "--ckpt-strategy",
        type=str,
        default="last_and_best",
        choices=["last_and_best", "last_only", "none"],
        help="Strategy how to save checkpoints",
    )
    p.arg("--ckpt-dir", type=Path, default=Path("/results/ckpt/"), help="Path to checkpoint directory")
    p.arg("--saved-model-dir", type=Path, help="Path to saved model directory (for evaluation and prediction)")
    p.flag("--resume-training", "--resume_training", help="Resume training from the last checkpoint")
    p.boolean_flag("--load_sm", default=False, help="Load exported savedmodel")
    p.boolean_flag("--validate", default=False, help="Validate exported savedmodel")

    # Data loading and processing
    p.arg(
        "--nvol",
        type=positive_int,
        default=2,
        help="Number of volumes which come into single batch size for 2D model",
    )
    p.arg(
        "--oversampling",
        type=float_0_1,
        default=0.33,
        help="Probability of crop to have some region with positive label",
    )
    p.arg(
        "--num-workers",
        type=non_negative_int,
        default=8,
        help="Number of subprocesses to use for data loading",
    )

    # Sliding window inference
    p.arg(
        "--overlap",
        type=float_0_1,
        default=0.25,
        help="Amount of overlap between scans during sliding window inference",
    )
    p.arg(
        "--blend",
        "--blend-mode",
        dest="blend_mode",
        type=str,
        choices=["gaussian", "constant"],
        default="constant",
        help="How to blend output of overlapping windows",
    )

    # Validation
    p.arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    p.arg("--fold", type=non_negative_int, default=0, help="Fold number")
    p.arg("--epochs", type=positive_int, default=1000, help="Number of epochs")
    p.arg("--skip-eval", type=non_negative_int, default=0, help="Skip evaluation for the first N epochs.")
    p.arg(
        "--steps-per-epoch",
        type=positive_int,
        help="Steps per epoch. By default ceil(training_dataset_size / batch_size / gpus)",
    )

    # Benchmarking
    p.arg(
        "--bench-steps",
        type=non_negative_int,
        default=200,
        help="Number of benchmarked steps in total",
    )
    p.arg(
        "--warmup-steps",
        type=non_negative_int,
        default=100,
        help="Number of warmup steps before collecting benchmarking statistics",
    )

    args = p.parse_args()
    return args
