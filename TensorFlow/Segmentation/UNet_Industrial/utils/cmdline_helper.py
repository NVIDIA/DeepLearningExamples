#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#
# ==============================================================================

import argparse

from datasets import known_datasets
from model.unet import UNet_v1
from model.blocks.activation_blck import authorized_activation_fn


def _add_bool_argument(parser, name=None, default=False, required=False, help=None):

    if not isinstance(default, bool):
        raise ValueError()

    feature_parser = parser.add_mutually_exclusive_group(required=required)

    feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    feature_parser.set_defaults(name=default)


def parse_cmdline():

    p = argparse.ArgumentParser(description="JoC-UNet_v1-TF")

    p.add_argument(
        '--unet_variant',
        default="tinyUNet",
        choices=UNet_v1.authorized_models_variants,
        type=str,
        required=False,
        help="""Which model size is used. This parameter control directly the size and the number of parameters"""
    )

    p.add_argument(
        '--activation_fn',
        choices=authorized_activation_fn,
        type=str,
        default="relu",
        required=False,
        help="""Which activation function is used after the convolution layers"""
    )

    p.add_argument(
        '--exec_mode',
        choices=['train', 'train_and_evaluate', 'evaluate', 'training_benchmark', 'inference_benchmark'],
        type=str,
        required=True,
        help="""Which execution mode to run the model into"""
    )

    p.add_argument(
        '--iter_unit',
        choices=['epoch', 'batch'],
        type=str,
        required=True,
        help="""Will the model be run for X batches or X epochs ?"""
    )

    p.add_argument('--num_iter', type=int, required=True, help="""Number of iterations to run.""")

    p.add_argument('--batch_size', type=int, required=True, help="""Size of each minibatch per GPU.""")

    p.add_argument(
        '--warmup_step',
        default=200,
        type=int,
        required=False,
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    p.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help="""Directory in which to write training logs, summaries and checkpoints."""
    )

    p.add_argument(
        '--log_dir',
        type=str,
        required=False,
        default="dlloger_out.json",
        help="""Directory in which to write logs."""
    )

    _add_bool_argument(
        parser=p,
        name="save_eval_results_to_json",
        default=False,
        required=False,
        help="Whether to save evaluation results in JSON format."
    )

    p.add_argument('--data_dir', required=False, default=None, type=str, help="Path to dataset directory")

    p.add_argument(
        '--dataset_name',
        choices=list(known_datasets.keys()),
        type=str,
        required=True,
        help="""Name of the dataset used in this run (only DAGM2007 is supported atm.)"""
    )

    p.add_argument(
        '--dataset_classID',
        default=None,
        type=int,
        required=False,
        help="""ClassID to consider to train or evaluate the network (used for DAGM)."""
    )

    p.add_argument(
        '--data_format',
        choices=['NHWC', 'NCHW'],
        type=str,
        default="NCHW",
        required=False,
        help="""Which Tensor format is used for computation inside the mode"""
    )

    _add_bool_argument(
        parser=p,
        name="amp",
        default=False,
        required=False,
        help="Enable Automatic Mixed Precision to speedup FP32 computation using tensor cores"
    )

    _add_bool_argument(
        parser=p, name="xla", default=False, required=False, help="Enable Tensorflow XLA to maximise performance."
    )

    p.add_argument(
        '--weight_init_method',
        choices=UNet_v1.authorized_weight_init_methods,
        default="he_normal",
        type=str,
        required=False,
        help="""Which initialisation method is used to randomly intialize the model during training"""
    )

    p.add_argument('--learning_rate', default=1e-4, type=float, required=False, help="""Learning rate value.""")

    p.add_argument(
        '--learning_rate_decay_factor',
        default=0.8,
        type=float,
        required=False,
        help="""Decay factor to decrease the learning rate."""
    )

    p.add_argument(
        '--learning_rate_decay_steps',
        default=500,
        type=int,
        required=False,
        help="""Decay factor to decrease the learning rate."""
    )

    p.add_argument('--rmsprop_decay', default=0.9, type=float, required=False, help="""RMSProp - Decay value.""")

    p.add_argument('--rmsprop_momentum', default=0.8, type=float, required=False, help="""RMSProp - Momentum value.""")

    p.add_argument('--weight_decay', default=1e-5, type=float, required=False, help="""Weight Decay scale factor""")

    _add_bool_argument(
        parser=p, name="use_auto_loss_scaling", default=False, required=False, help="Use AutoLossScaling with TF-AMP"
    )

    p.add_argument(
        '--loss_fn_name',
        type=str,
        default="adaptive_loss",
        required=False,
        help="""Loss function Name to use to train the network"""
    )

    _add_bool_argument(
        parser=p, name="augment_data", default=True, required=False, help="Choose whether to use data augmentation"
    )

    p.add_argument(
        '--display_every',
        type=int,
        default=50,
        required=False,
        help="""How often (in batches) to print out debug information."""
    )

    p.add_argument(
        '--debug_verbosity',
        choices=[0, 1, 2],
        default=0,
        type=int,
        required=False,
        help="""Verbosity Level: 0 minimum, 1 with layer creation debug info, 2 with layer + var creation debug info."""
    )

    p.add_argument('--seed', type=int, default=None, help="""Random seed.""")

    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS
