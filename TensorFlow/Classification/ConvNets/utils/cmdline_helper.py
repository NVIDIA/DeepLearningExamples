#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import argparse


class ArgumentParserUtil(object):
    def __init__(self, parser: argparse.ArgumentParser = None):
        self.parser = parser

    def build_data_parser_group(self):
        data_group = self.parser.add_argument_group("Dataset arguments")

        data_group.add_argument(
            "--data_dir",
            required=False,
            default=None,
            type=str,
            help="Path to dataset in TFRecord format. Files should be named 'train-*' and 'validation-*'.")

        data_group.add_argument("--data_idx_dir",
                                required=False,
                                default=None,
                                type=str,
                                help="Path to index files for DALI. Files should be named 'train-*' and 'validation-*'.")

        data_group.add_argument("--dali",
                                action="store_true",
                                default=False,
                                required=False,
                                help="Enable DALI data input.")

        data_group.add_argument("--synthetic_data_size",
                                required=False,
                                default=224,
                                type=int,
                                help="Dimension of image for synthetic dataset")

    def build_training_parser_group(self):
        train_group = self.parser.add_argument_group("Training arguments")

        train_group.add_argument("--lr_init",
                                 default=0.1,
                                 type=float,
                                 required=False,
                                 help="Initial value for the learning rate.")

        train_group.add_argument("--lr_warmup_epochs",
                                 default=5,
                                 type=int,
                                 required=False,
                                 help="Number of warmup epochs for learning rate schedule.")

        train_group.add_argument("--weight_decay",
                                 default=1e-4,
                                 type=float,
                                 required=False,
                                 help="Weight Decay scale factor.")

        train_group.add_argument("--weight_init",
                                 default="fan_out",
                                 choices=["fan_in", "fan_out"],
                                 type=str,
                                 required=False,
                                 help="Model weight initialization method.")

        train_group.add_argument("--momentum",
                                 default=0.9,
                                 type=float,
                                 required=False,
                                 help="SGD momentum value for the Momentum optimizer.")

        train_group.add_argument("--label_smoothing",
                                 type=float,
                                 default=0.0,
                                 required=False,
                                 help="The value of label smoothing.")

        train_group.add_argument("--mixup",
                                 type=float,
                                 default=0.0,
                                 required=False,
                                 help="The alpha parameter for mixup (if 0 then mixup is not applied).")

        train_group.add_argument("--cosine_lr",
                                 "--use_cosine",
                                 "--use_cosine_lr"
                                 "--cosine",
                                 action="store_true",
                                 default=False,
                                 required=False,
                                 help="Use cosine learning rate schedule.")

    def build_generic_optimization_parser_group(self):
        goptim_group = self.parser.add_argument_group("Generic optimization arguments")

        goptim_group.add_argument("--xla",
                                  "--use_xla",
                                  action="store_true",
                                  default=False,
                                  required=False,
                                  help="Enable XLA (Accelerated Linear Algebra) computation for improved performance.")
        goptim_group.add_argument("--data_format",
                                  choices=['NHWC', 'NCHW'],
                                  type=str,
                                  default='NHWC',
                                  required=False,
                                  help="Data format used to do calculations")

        goptim_group.add_argument("--amp",
                                  "--use_tf_amp",
                                  action="store_true",
                                  dest="amp",
                                  default=False,
                                  required=False,
                                  help="Enable Automatic Mixed Precision to speedup computation using tensor cores.")

        goptim_group.add_argument("--cpu",
                                  action="store_true",
                                  dest="cpu",
                                  default=False,
                                  required=False,
                                  help="Run model on CPU instead of GPU")

        amp_group = self.parser.add_argument_group("Automatic Mixed Precision arguments")
        amp_group.add_argument("--static_loss_scale",
                               "--loss_scale",
                               default=-1,
                               required=False,
                               help="Use static loss scaling in FP32 AMP.")
        amp_group.add_argument("--use_static_loss_scaling", required=False, action="store_true", help=argparse.SUPPRESS)


def parse_cmdline(available_arch):

    p = argparse.ArgumentParser(description="JoC-RN50v1.5-TF")

    p.add_argument('--arch',
                   choices=available_arch,
                   type=str,
                   default='resnet50',
                   required=False,
                   help="""Architecture of model to run""")

    p.add_argument('--mode',
                   choices=[
                       'train', 'train_and_evaluate', 'evaluate', 'predict', 'training_benchmark', 'inference_benchmark'
                   ],
                   type=str,
                   default='train_and_evaluate',
                   required=False,
                   help="""The execution mode of the script.""")

    p.add_argument('--export_dir',
                   required=False,
                   default=None,
                   type=str,
                   help="Directory in which to write exported SavedModel.")

    p.add_argument('--to_predict',
                   required=False,
                   default=None,
                   type=str,
                   help="Path to file or directory of files to run prediction on.")

    p.add_argument('--batch_size', type=int, required=True, help="""Size of each minibatch per GPU.""")

    p.add_argument('--num_iter', type=int, required=False, default=1, help="""Number of iterations to run.""")
    p.add_argument('--run_iter',
                   type=int,
                   required=False,
                   default=-1,
                   help="""Number of training iterations to run on single run.""")

    p.add_argument('--iter_unit',
                   choices=['epoch', 'batch'],
                   type=str,
                   required=False,
                   default='epoch',
                   help="""Unit of iterations.""")

    p.add_argument(
        '--warmup_steps',
        default=50,
        type=int,
        required=False,
        help="""Number of steps considered as warmup and not taken into account for performance measurements.""")

    p.add_argument('--model_dir',
                   type=str,
                   required=False,
                   default=None,
                   help="""Directory in which to write model. If undefined, results dir will be used.""")

    p.add_argument('--results_dir',
                   type=str,
                   required=False,
                   default='.',
                   help="""Directory in which to write training logs, summaries and checkpoints.""")

    p.add_argument('--log_filename',
                   type=str,
                   required=False,
                   default='log.json',
                   help="Name of the JSON file to which write the training log")

    p.add_argument('--display_every',
                   default=10,
                   type=int,
                   required=False,
                   help="""How often (in batches) to print out running information.""")

    p.add_argument('--seed', type=int, default=None, help="""Random seed.""")

    p.add_argument('--gpu_memory_fraction',
                   type=float,
                   default=0.7,
                   help="""Limit memory fraction used by training script for DALI""")

    p.add_argument('--gpu_id',
                   type=int,
                   default=0,
                   help="""Specify ID of the target GPU on multi-device platform. Effective only for single-GPU mode.""")

    p.add_argument('--finetune_checkpoint',
                   required=False,
                   default=None,
                   type=str,
                   help="Path to pre-trained checkpoint which will be used for fine-tuning")

    p.add_argument("--use_final_conv",
                   default=False,
                   required=False,
                   action="store_true",
                   help="Use convolution operator instead of MLP as last layer.")

    p.add_argument('--quant_delay',
                   type=int,
                   default=0,
                   required=False,
                   help="Number of steps to be run before quantization starts to happen")

    p.add_argument("--quantize",
                   default=False,
                   required=False,
                   action="store_true",
                   help="Quantize weights and activations during training. (Defaults to Assymmetric quantization)")

    p.add_argument("--use_qdq",
                   default=False,
                   required=False,
                   action="store_true",
                   help="Use QDQV3 op instead of FakeQuantWithMinMaxVars op for quantization. QDQv3 does only scaling")

    p.add_argument("--symmetric",
                   default=False,
                   required=False,
                   action="store_true",
                   help="Quantize weights and activations during training using symmetric quantization.")

    parser_util = ArgumentParserUtil(p)
    parser_util.build_data_parser_group()
    parser_util.build_training_parser_group()
    parser_util.build_generic_optimization_parser_group()

    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS
