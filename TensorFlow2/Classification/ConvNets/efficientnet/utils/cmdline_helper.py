#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import argparse
import yaml


def _add_bool_argument(parser, name=None, default=False, required=False, help=None):

    if not isinstance(default, bool):
        raise ValueError()

    feature_parser = parser.add_mutually_exclusive_group(required=required)

    feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    feature_parser.set_defaults(name=default)


def parse_cmdline():

    p = argparse.ArgumentParser(description="JoC-RN50v1.5-TF")

    # ====== Define the common flags across models. ======
    p.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help=('The directory where the model and training/evaluation summaries'
                'are stored.'))

    p.add_argument(
        '--config_file',
        type=str,
        default=None,
        help=('A YAML file which specifies overrides. Note that this file can be '
                'used as an override template to override the default parameters '
                'specified in Python. If the same parameter is specified in both '
                '`--config_file` and `--params_override`, the one in '
                '`--params_override` will be used finally.'))

    p.add_argument(
        '--params_override',
        type=str,
        default=None,
        help=('a YAML/JSON string or a YAML file which specifies additional '
                'overrides over the default parameters and those specified in '
                '`--config_file`. Note that this is supposed to be used only to '
                'override the model parameters, but not the parameters like TPU '
                'specific flags. One canonical use case of `--config_file` and '
                '`--params_override` is users first define a template config file '
                'using `--config_file`, then use `--params_override` to adjust the '
                'minimal set of tuning parameters, for example setting up different'
                ' `train_batch_size`. '
                'The final override order of parameters: default_model_params --> '
                'params from config_file --> params in params_override.'
                'See also the help message of `--config_file`.'))

    p.add_argument(
        '--save_checkpoint_freq',
        type=int,
        default=1,
        help='Number of epochs to save checkpoint.')

    p.add_argument(
      '--data_dir',
      type=str,
      default='.',
      required=True,
      help='The location of the input data. Files should be named `train-*` and `validation-*`.')
    
    p.add_argument(
        '--mode',
        type=str,
        default='train_and_eval',
        required=False,
        help='Mode to run: `train`, `eval`, `train_and_eval` or `export`.')
    
    p.add_argument(
        '--arch',
        type=str,
        default='efficientnet-b0',
        required=False,
        help='The type of the model, e.g. EfficientNet, etc.')
    
    p.add_argument(
        '--dataset',
        type=str,
        default='ImageNet',
        required=False,
        help='The name of the dataset, e.g. ImageNet, etc.')
    
    p.add_argument(
        '--log_steps',
        type=int,
        default=100,
        help='The interval of steps between logging of batch level stats.')

    p.add_argument(
        '--time_history',
        action='store_true',
        default=True,
        help='Logging the time for training steps.')
    
    p.add_argument(
        '--use_xla',
        action='store_true',
        default=False,
        help='Set to True to enable XLA')
    
    p.add_argument(
        '--use_amp',
        action='store_true',
        default=False,
        help='Set to True to enable AMP')

    p.add_argument(
        '--intraop_threads',
        type=str,
        default='',
        help='intra thread should match the number of CPU cores')

    p.add_argument(
        '--interop_threads',
        type=str,
        default='',
        help='inter thread should match the number of CPU sockets')
    
    p.add_argument(
        '--export_dir', required=False, default=None, type=str, help="Directory in which to write exported SavedModel."
    )

    p.add_argument(
        '--results_dir',
        type=str,
        required=False,
        default='.',
        help="Directory in which to write training logs, summaries and checkpoints."
    )

    p.add_argument(
        '--inference_checkpoint',
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint to do inference on."
    )

    p.add_argument(
        '--to_predict',
        type=str,
        required=False,
        default=None,
        help="Path to image to do inference on."
    )


    p.add_argument(
        '--log_filename',
        type=str,
        required=False,
        default='log.json',
        help="Name of the JSON file to which write the training log"
    )

    p.add_argument(
        '--display_every',
        default=10,
        type=int,
        required=False,
        help="How often (in batches) to print out running information."
    )

    #model_params:
    p.add_argument(
        '--num_classes', type=int, default=1000, required=False, help="Number of classes to train on.")

    p.add_argument(
        '--batch_norm', type=str, default='default', required=False, help="Type of Batch norm used.")

    p.add_argument(
        '--activation', type=str, default='swish', required=False, help="Type of activation to be used.")

    #optimizer:
    p.add_argument(
        '--optimizer', type=str, default='rmsprop', required=False, help="Optimizer to be used.")

    p.add_argument(
        '--momentum', type=float, default=0.9, required=False, help="The value of Momentum.")

    p.add_argument(
        '--epsilon', type=float, default=0.001, required=False, help="The value of Epsilon for optimizer.")

    p.add_argument(
        '--decay', type=float, default=0.9, required=False, help="The value of decay.")

    p.add_argument(
        '--moving_average_decay', type=float, default=0.0, required=False, help="The value of moving average.")

    p.add_argument(
        '--lookahead', action='store_true', default=False, required=False, help="Lookahead.")

    p.add_argument(
        '--nesterov', action='store_true', default=False, required=False, help="nesterov bool.")

    p.add_argument(
        '--beta_1', type=float, default=0.0, required=False, help="beta1 for Adam/AdamW.")

    p.add_argument(
        '--beta_2', type=float, default=0.0, required=False, help="beta2 for Adam/AdamW..")
    
    #loss:
    p.add_argument(
        '--label_smoothing', type=float, default=0.1, required=False, help="The value of label smoothing.")
    p.add_argument(
        '--mixup_alpha', type=float, default=0.0, required=False, help="Mix up alpha")

    # Training specific params
    p.add_argument(
        '--max_epochs',
        default=300,
        type=int,
        required=False,
        help="Number of steps of training."
    )

    p.add_argument(
        '--num_epochs_between_eval', 
        type=int, 
        default=1, 
        required=False, 
        help="Eval after how many steps of training.")

    p.add_argument(
        '--steps_per_epoch',
        default=None,
        type=int,
        required=False,
        help="Number of steps of training."
    )
    # LR Params
    p.add_argument(
        '--warmup_epochs',
        default=5,
        type=int,
        required=False,
        help="Number of steps considered as warmup and not taken into account for performance measurements."
    )

    p.add_argument(
        '--lr_init', default=0.008, type=float, required=False, help="Initial value for the learning rate."
    )

    p.add_argument(
        '--lr_decay', type=str, default='exponential', required=False, help="Type of LR Decay.")

    p.add_argument('--lr_decay_rate', default=0.97, type=float, required=False, help="LR Decay rate.")

    p.add_argument('--lr_decay_epochs', default=2.4, type=float, required=False, help="LR Decay epoch.")

    p.add_argument(
        '--lr_warmup_epochs',
        default=5,
        type=int,
        required=False,
        help="Number of warmup epochs for learning rate schedule."
    )

    p.add_argument('--weight_decay', default=5e-6, type=float, required=False, help="Weight Decay scale factor.")

    p.add_argument(
        '--weight_init',
        default='fan_out',
        choices=['fan_in', 'fan_out'],
        type=str,
        required=False,
        help="Model weight initialization method."
    )

    p.add_argument(
        '--train_num_examples', type=int, default=1281167, required=False, help="Training number of examples.")

    p.add_argument(
    '--train_batch_size', type=int, default=32, required=False, help="Training batch size per GPU.")

    p.add_argument(
    '--augmenter_name', type=str, default='autoaugment', required=False, help="Type of Augmentation during preprocessing only during training.")

    #Rand-augment params
    p.add_argument(
        '--num_layers', type=int, default=None, required=False, help="Rand Augmentation parameter.")
    p.add_argument(
        '--magnitude', type=float, default=None, required=False, help="Rand Augmentation parameter.")
    p.add_argument(
        '--cutout_const', type=float, default=None, required=False, help="Rand/Auto Augmentation parameter.")
    p.add_argument(
        '--translate_const', type=float, default=None, required=False, help="Rand/Auto Augmentation parameter.")
    #Auto-augment params
    p.add_argument(
        '--autoaugmentation_name', type=str, default=None, required=False, help="Auto-Augmentation parameter.")
    #evaluation:

    # Tensor format used for the computation.
    p.add_argument(
        '--data_format', choices=['NHWC', 'NCHW'], type=str, default='NCHW', required=False, help=argparse.SUPPRESS
    )

    # validation_dataset:
    p.add_argument(
    '--eval_num_examples', type=int, default=50000, required=False, help="Evaluation number of examples")
    p.add_argument(
    '--eval_batch_size', type=int, default=32, required=False, help="Evaluation batch size per GPU.")
    p.add_argument(
    '--predict_batch_size', type=int, default=32, required=False, help="Predict batch size per GPU.")
    p.add_argument(
    '--skip_eval', action='store_true', default=False, required=False, help="Skip eval during training.")

    p.add_argument(
        '--resume_checkpoint', action='store_true', default=False, required=False, help="Resume from a checkpoint in the model_dir.")

    p.add_argument('--use_dali', action='store_true', default=False,
                        help='Use dali for data loading and preprocessing of train dataset.')

    p.add_argument('--use_dali_eval', action='store_true', default=False,
                        help='Use dali for data loading and preprocessing of eval dataset.')

    p.add_argument(
        '--index_file', type=str, default='', required=False,
        help="Path to index file required for dali.")

    p.add_argument('--benchmark', action='store_true', default=False, required=False, help="Benchmarking or not")
    # Callbacks options
    p.add_argument(
    '--enable_checkpoint_and_export', action='store_true', default=True, required=False, help="Evaluation number of examples")
    p.add_argument(
    '--enable_tensorboard', action='store_true', default=False, required=False, help="Enable Tensorboard logging.")
    p.add_argument(
    '--write_model_weights', action='store_true', default=False, required=False, help="whether to write model weights to visualize as image in TensorBoard..")

    p.add_argument('--seed', type=int, default=None, required=False, help="Random seed.")

    p.add_argument('--dtype', type=str, default='float32', required=False, help="Only permitted `float32`,`bfloat16`,`float16`,`fp32`,`bf16`")

    p.add_argument('--run_eagerly', action='store_true', default=False, required=False, help="Random seed.")
    


    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS