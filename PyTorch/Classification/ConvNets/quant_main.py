# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import random
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from image_classification.training import *
from image_classification.utils import *
from image_classification.quantization import *
from image_classification.models import efficientnet_quant_b0, efficientnet_quant_b4

from main import prepare_for_training, add_parser_arguments as parse_training

import dllogger


def available_models():
    models = {
        m.name: m
        for m in [
            efficientnet_quant_b0,
            efficientnet_quant_b4,
        ]
    }
    return models


def parse_quantization(parser):
    model_names = available_models().keys()
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="efficientnet-quant-b0",
        choices=model_names,
        help="model architecture: "
        + " | ".join(model_names)
        + " (default: efficientnet-quant-b0)",
    )

    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="skip calibration before training, (default: false)",
    )


def parse_training_args(parser):
    from main import add_parser_arguments

    return add_parser_arguments(parser)


def main(args, model_args, model_arch):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    skip_calibration = args.skip_calibration or args.evaluate or args.resume is not None

    select_default_calib_method()

    (
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        start_epoch,
    ) = prepare_for_training(args, model_args, model_arch)

    print(f"RUNNING QUANTIZATION")

    if not skip_calibration:
        calibrate(trainer.model_and_loss.model, train_loader, logger, calib_iter=10)

    train_loop(
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger,
        should_backup_checkpoint(args),
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs)
        if args.run_epochs != -1
        else args.epochs,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints,
        checkpoint_dir=args.workspace,
        checkpoint_filename="quantized_" + args.checkpoint_filename,
    )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.end()
    print("Experiment ended")


if __name__ == "__main__":
    epilog = [
        "Based on the architecture picked by --arch flag, you may use the following options:\n"
    ]
    for model, ep in available_models().items():
        model_help = "\n".join(ep.parser().format_help().split("\n")[2:])
        epilog.append(model_help)
    parser = argparse.ArgumentParser(
        description="PyTorch ImageNet Training",
        epilog="\n".join(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parse_quantization(parser)
    parse_training(parser, skip_arch=True)

    args, rest = parser.parse_known_args()

    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)
    print(model_args)

    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args, model_arch)
