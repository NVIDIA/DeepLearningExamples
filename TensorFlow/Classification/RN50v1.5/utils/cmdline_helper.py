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


def _add_bool_argument(parser, name=None, default=False, required=False, help=None):

    if not isinstance(default, bool):
        raise ValueError()

    feature_parser = parser.add_mutually_exclusive_group(required=required)

    feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    feature_parser.set_defaults(name=default)


def parse_cmdline():

    p = argparse.ArgumentParser(description="JoC-RN50v1.5-TF")

    p.add_argument(
        '--mode',
        choices=['train', 'train_and_evaluate', 'evaluate', 'predict', 'training_benchmark', 'inference_benchmark'],
        type=str,
        default='train_and_evaluate',
        required=False,
        help="""The execution mode of the script."""
    )

    p.add_argument(
        '--data_dir',
        required=False,
        default=None,
        type=str,
        help="Path to dataset in TFRecord format. Files should be named 'train-*' and 'validation-*'."
    )

    p.add_argument(
        '--data_idx_dir',
        required=False,
        default=None,
        type=str,
        help="Path to index files for DALI. Files should be named 'train-*' and 'validation-*'."
    )
    
    p.add_argument(
        '--export_dir',
        required=False,
        default=None,
        type=str,
        help="Directory in which to write exported SavedModel."
    )

    p.add_argument(        
        '--to_predict',
        required=False,
        default=None,
        type=str,
        help="Path to file or directory of files to run prediction on."
    )
    
    p.add_argument(
        '--batch_size', 
        type=int, 
        required=False, 
        help="""Size of each minibatch per GPU."""
    )

    p.add_argument(
        '--num_iter',
        type=int, 
        required=False, 
        default=1,
        help="""Number of iterations to run."""
    )

    p.add_argument(
        '--iter_unit',
        choices=['epoch', 'batch'],
        type=str,
        required=False,    
        default='epoch',
        help="""Unit of iterations."""
    )

    p.add_argument(
        '--warmup_steps',
        default=50,
        type=int,
        required=False,
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    # Tensor format used for the computation.
    p.add_argument(
        '--data_format',
        choices=['NHWC', 'NCHW'],
        type=str,
        default='NCHW',
        required=False,
        help=argparse.SUPPRESS 
    )
    
    p.add_argument(
        '--model_dir',
        type=str,
        required=False,
        default=None,
        help="""Directory in which to write model. If undefined, results dir will be used."""
    )


    p.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help="""Directory in which to write training logs, summaries and checkpoints."""
    )

    p.add_argument(
        '--display_every', 
        default=10,
        type=int, 
        required=False, 
        help="""How often (in batches) to print out running information."""
    )

    p.add_argument(
        '--lr_init',
        default=0.1,
        type=float,
        required=False,
        help="""Initial value for the learning rate."""
    )
    
    p.add_argument(
        '--lr_warmup_epochs',
        default=5,
        type=int,
        required=False,
        help="""Number of warmup epochs for learning rate schedule."""
    )

    p.add_argument(
        '--weight_decay', 
        default=1e-4,
        type=float, 
        required=False, 
        help="""Weight Decay scale factor."""
    )

    p.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        required=False,
        help="""SGD momentum value for the Momentum optimizer."""
    )
    
    #Select fp32 or non-AMP fp16 precision arithmetic.
    p.add_argument(
        '--precision',
        choices=['fp32', 'fp16'],
        type=str,
        default='fp32',
        required=False,
        help=argparse.SUPPRESS 
    )

    p.add_argument(
        '--loss_scale',
        type=float,
        default=256.0,
        required=False,
        help="""Loss scale for FP16 Training and Fast Math FP32."""
    )
    
    p.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        required=False,
        help="""The value of label smoothing."""
    )
    
    p.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        required=False,
        help="""The alpha parameter for mixup (if 0 then mixup is not applied)."""
    )
    
    _add_bool_argument(
        parser=p,
        name="use_static_loss_scaling",
        default=False,
        required=False,
        help="Use static loss scaling in FP16 or FP32 AMP."
    )

    _add_bool_argument(
        parser=p,
        name="use_xla",
        default=False,
        required=False,
        help="Enable XLA (Accelerated Linear Algebra) computation for improved performance."
    )

    _add_bool_argument(
        parser=p,
        name="use_dali",
        default=False,
        required=False,
        help="Enable DALI data input."
    )

    _add_bool_argument(
        parser=p,
        name="use_tf_amp",
        default=False,
        required=False,
        help="Enable Automatic Mixed Precision to speedup FP32 computation using tensor cores."
    )
    
    _add_bool_argument(
        parser=p,
        name="use_cosine_lr",
        default=False,
        required=False,
        help="Use cosine learning rate schedule."
    )
    
    p.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help="""Random seed."""
    )
    
    p.add_argument(
        '--gpu_memory_fraction',
        type=float,
        default=0.7,
        help="""Limit memory fraction used by training script for DALI"""
    )
    
    p.add_argument(
        '--gpu_id',
        type=int,
        default=0,        
        help="""Specify ID of the target GPU on multi-device platform. Effective only for single-GPU mode."""
    )
    
    
    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS
