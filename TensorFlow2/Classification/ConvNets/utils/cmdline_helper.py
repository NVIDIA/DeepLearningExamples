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

    p = argparse.ArgumentParser(description="JoC-Efficientnet-v2-TF, full list of general hyperparameters." 
                                "Model specific hyperparameters must be provided via a config file (see --cfg)"
                                "User config can be provided in a separate config file like config/efficientnet_v2/s_cfg.py"
                                "Use True/False or 1/0 for boolean flags.")

    p.add_argument(
        '--cfg',
        type=str,
        default=None,
        required=True,
        help=('Path to the config file that contains the hyperparameters of your model.'
              'The path must be relative to the current dir. The user can override model hyperparameters'
              'in the command line (see --mparams).'))

    p.add_argument(
        '--mparams',
        type=str,
        default=None,
        required=False,
        help=('A comma seperated list of key=val where the key is a model hyperparameter and the val is the new value.'
              'This flag becomes handy when you want to override the model hyperparameters placed in the model config (see --cfg).'))
    
    #######################runtime-related hparams##########################

    p.add_argument( '--log_steps', type=int, default=100, help='The interval of steps between logging of batch level stats.')

    p.add_argument( '--mode', type=str, default='train_and_eval', required=False, help='Mode to run: `train`, `eval`, `train_and_eval`, `training_benchmark` or `predict`.')

    p.add_argument( '--time_history', type=str, default=1, help='Enable a callback to log the time for training steps.')

    p.add_argument( '--use_xla', action='store_true', default=False, help='Have this flag to enable xla')

    p.add_argument( '--use_amp', action='store_true', default=False, help='Have this flag to enable training with automated mixed precision (AMP)')

    p.add_argument( '--intraop_threads', type=str, default=None, help='intra thread should match the number of CPU cores')

    p.add_argument( '--interop_threads', type=str, default=None, help='inter thread should match the number of CPU sockets')

    p.add_argument( '--model_dir', type=str, default='/results/', help=('The directory where the model and training/evaluation summaries'
            'are stored. When resuming from a previous checkpoint,'
            'all necessary files should be placed in this directory '))

    p.add_argument('--log_dir', type=str, default=None,
                   help=('The directory where the model and training/evaluation summaries'
                         'are stored. '))
    
    p.add_argument( '--log_filename', type=str, default='log.json', help="Name of the JSON file to which write the training log")

    p.add_argument('--seed', type=int, default=None, required=False, help="Random seed.")

    # Tensor format used for the computation.
    p.add_argument('--data_format', choices=['channels_first', 'channels_last'], type=str, default='channels_first', required=False, help=argparse.SUPPRESS)

    p.add_argument('--run_eagerly', type=str, default=0, help="Set this flag 1/0 to run/disable eager execution mode.")

    p.add_argument('--memory_limit', type=int, default=None, help="Set the maximum GPU memory (MB) that can be allocated by tensorflow. Sometimes tensorflow"
                   "allocates more GPU memory than it actually needs, which results in OOM or halt without stopping. Setting this to be "
                    "slightly less than full GPU memory will help prevent this. For example, on A100 80G gpu, this value can be set to 81000")

    p.add_argument('--weights_format', type=str, default='ckpt', required=False, help="Whether to read pretrained weights from a ckpt or SavedModel format")

    #######################train-related hparams##########################

    # Tensor format used for the computation.
    p.add_argument('--train_img_size', default=224, type=int, required=False, help="Image size used for training dataset.")

    p.add_argument( '--max_epochs',  default=300, type=int, required=False, help="Number of epochs of training.")

    p.add_argument( '--steps_per_epoch', default=None, type=int,  required=False,  help="Manually set training steps that will be executed in each epoch, leave blank to iter over the whole training dataset every epoch." )

    p.add_argument('--train_batch_size', type=int, default=32, required=False, help="Training batch size per GPU.")

    p.add_argument('--train_use_dali', action='store_true', default=False, help='Have this flag to use dali for data loading and preprocessing of dataset, attention, dali does not support having auto-augmentation or  image mixup.')

    ##optimizer##
    p.add_argument(
        '--optimizer', type=str, default='rmsprop', required=False, help="Optimizer to be used.")

    p.add_argument(
        '--momentum', type=float, default=0.9, required=False, help="The value of Momentum used when optimizer name is `rmsprop` or `momentum`.")

    p.add_argument(
        '--beta_1', type=float, default=0.0, required=False, help="beta1 for Adam/AdamW.")

    p.add_argument(
        '--beta_2', type=float, default=0.0, required=False, help="beta2 for Adam/AdamW..")

    p.add_argument(
        '--nesterov', action='store_true', default=False, required=False, help="nesterov bool for momentum SGD.")

    p.add_argument(
        '--opt_epsilon', type=float, default=0.001, required=False, help="The value of Epsilon for optimizer, required for `adamw`, `adam` and `rmsprop`.")

    p.add_argument(
        '--decay', type=float, default=0.9, required=False, help="The value of decay for `rmsprop`.")

    p.add_argument(
        '--weight_decay', default=5e-6, type=float, required=False, help="Weight Decay scale factor, for adamw or can be used in layers as L2 reg.")

    p.add_argument(
        '--label_smoothing', type=float, default=0.1, required=False, help="The value of label smoothing.")

    p.add_argument(
        '--moving_average_decay', type=float, default=0.0, required=False, help="Empirically it has been found that using the moving average of the trained parameters"
        "of a deep network is better than using its trained parameters directly. This optimizer"
        "allows you to compute this moving average and swap the variables at save time so that"
        "any code outside of the training loop will use by default the average values instead"
        "of the original ones.")

    p.add_argument(
        '--lookahead', action='store_true', default=False, required=False, help="Having this flag to enable lookahead, the optimizer iteratively updates two sets of weights: the search directions for weights"
        "are chosen by the inner optimizer, while the `slow weights` are updated each k steps"
        "based on the directions of the `fast weights` and the two sets of weights are "
        "synchronized. This method improves the learning stability and lowers the variance of"
        "its inner optimizer.")

    p.add_argument(
        '--intratrain_eval_using_ema', action='store_true', default=True, required=False, help="Model evaluation during training can be done using the original weights,"
        "or using EMA weights. The latter takes place if moving_average_decay > 0 and intratrain_eval_using_ema is requested")

    p.add_argument(
        '--grad_accum_steps', type=int, default=1, required=False, help="Use multiple steps to simulate a large batch size")

    p.add_argument(
        '--grad_clip_norm', type=float, default=0, required=False,
        help="grad clipping is used in the custom train_step, which is called when grad_accum_steps > 1. Any non-zero value activates grad clipping")

    p.add_argument(
        '--hvd_fp16_compression', action='store_true', default=True, required=False, help="Optimize grad reducing across all workers")

    p.add_argument(
        '--export_SavedModel', action='store_true', default=False, required=False, help='Have this flag to export the trained model into SavedModel format after training is complete. When `moving_average_decay` > 0'
        'it will store the set of weights with better accuracy between the original and EMA weights. This flag also has the effect of exporting the model as SavedModel at the end of evaluation.')


    ##lr schedule##

    p.add_argument('--lr_init', default=0.008, type=float, required=False, help="Initial value for the learning rate without scaling, the final learning rate is scaled by ."
            "lr_init * global_batch_size / 128.")

    p.add_argument('--lr_decay', choices=['exponential', 'piecewise_constant_with_warmup', 'cosine', 'linearcosine'], type=str, default='exponential', required=False, help="Choose from the supported decay types")

    p.add_argument('--lr_decay_rate', default=0.97, type=float, required=False, help="LR Decay rate for exponential decay.")

    p.add_argument('--lr_decay_epochs', default=2.4, type=float, required=False, help="LR Decay epoch for exponential decay.")

    p.add_argument('--lr_warmup_epochs', default=5, type=int, required=False, help="Number of warmup epochs for learning rate schedule.")

    p.add_argument('--metrics', default=['accuracy', 'top_5'], nargs='+', action='extend', required=False, help="Metrics used to evaluate the model")

    p.add_argument('--resume_checkpoint', action='store_true', default=True, required=False, help="Resume from a checkpoint in the model_dir.")

    p.add_argument('--save_checkpoint_freq', type=int, default=5, required=False,  help='Number of epochs to save checkpoint.')

    ##progressive training##
    p.add_argument('--n_stages', type=int, default=1, required=False, help='Number of stages for progressive training in efficientnet_v2.')

    p.add_argument('--base_img_size', type=int, default=128, required=False, help='Used to determine image size for stage 1 in progressive training. Image size will then be scaled linearly until it reaches train_img_size in the last stage of training.')##Nima

    p.add_argument('--base_mixup', type=float, default=0, required=False, help='Mixup alpha for stage 1 in progressive training. Will then be scaled linearly until it reaches mixup_alpha in the last stage of training.')##Nima

    p.add_argument('--base_cutmix', type=float, default=0, required=False, help='Cutmix alpha for stage 1 in progressive training.  Will then be scaled linearly until it reaches cutmix_alpha in the last stage of training.')##Nima

    p.add_argument('--base_randaug_mag', type=float, default=5, required=False, help='Strength of random augmentation for stage 1 in progressive training.  Will then be scaled linearly until it reaches raug_magnitude in the last stage of training.')##Nima

    ##callbacks##
    p.add_argument('--enable_checkpoint_saving', action='store_true', default=True, required=False, help="saves model checkpoints during trining at desired intervals.")

    p.add_argument('--enable_tensorboard', action='store_true', default=False, required=False, help=argparse.SUPPRESS)

    p.add_argument('--tb_write_model_weights', action='store_true', default=False, required=False, help=argparse.SUPPRESS)

    #######################eval-related hparams##########################

    p.add_argument('--skip_eval', action='store_true', default=False, required=False, help="Skip eval at the end of training.")
    
    p.add_argument('--n_repeat_eval',  type=int, default=1, required=False, help="Number of time to repeat evaluation. Useful to check variations in throughputs.") 

    p.add_argument('--num_epochs_between_eval', type=int, default=1, required=False,  help="Eval after how many epochs of training.")

    p.add_argument('--eval_use_dali', action='store_true', default=False, help='Use dali for data loading and preprocessing of eval dataset.')

    p.add_argument('--eval_batch_size', type=int, default=100, required=False, help="Evaluation batch size per GPU.")

    p.add_argument('--eval_img_size', default=224, type=int, required=False, help="Image size used for validation dataset.")


    #######################predict mode related hparams##########################

    p.add_argument('--predict_img_dir', type=str, required=False, default='/infer_data', help="Path to image to do inference on.")

    p.add_argument('--predict_ckpt', type=str, required=False,  default=None, help="Path to checkpoint to do inference on.")

    p.add_argument('--predict_img_size', default=224, type=int, required=False,help="Image size used for inference.")

    p.add_argument('--predict_batch_size', type=int, default=32, required=False, help="Predict batch size per GPU.")

    p.add_argument('--benchmark', action='store_true', default=False, required=False, help="Benchmarking or not. Available in the predict mode.")

    ####################### data related hparams##########################

    p.add_argument('--dataset', type=str, default='ImageNet', required=False, help='The name of the dataset, e.g. ImageNet, etc.')

    p.add_argument('--augmenter_name', type=str, default='autoaugment', required=False, help="Type of Augmentation during preprocessing only during training.")

    ##Rand-augment params##
    p.add_argument('--raug_num_layers', type=int, default=None, required=False, help="Rand Augmentation parameter.")

    p.add_argument('--raug_magnitude', type=float, default=None, required=False, help="Rand Augmentation parameter.")

    p.add_argument('--cutout_const', type=float, default=None, required=False, help="Rand/Auto Augmentation parameter.")

    p.add_argument('--mixup_alpha', type=float, default=0., required=False, help="Mix up alpha")

    p.add_argument('--cutmix_alpha', type=float, default=0., required=False, help="Cut mix alpha")

    p.add_argument('--defer_img_mixing', action='store_true', default=False, required=False, help="Have this flag to perform image mixing in the compute graph")

    p.add_argument('--translate_const', type=float, default=None, required=False, help="Rand/Auto Augmentation parameter.")

    p.add_argument('--disable_map_parallelization', action='store_true', default=False, required=False, help="Have this flag to disable map parallelization of tf.Dataset. While this flag will hurt the throughput of multi-GPU/node sessions, it can prevent OOM errors during 1-GPU training sessions.")###Must add to scripts

    ##Auto-augment params
    p.add_argument('--autoaugmentation_name', type=str, default=None, required=False, help="Auto-Augmentation parameter.")

    ##Dali usage
    p.add_argument('--index_file', type=str, default=None, required=False, help="Path to index file required for dali.")

    # dataset and split
    p.add_argument('--data_dir', type=str, default='/data/', required=False, help='The location of the input data. Files should be named `train-*` and `validation-*`.')

    p.add_argument('--num_classes', type=int, default=1000, required=False, help="Number of classes to train on.")

    p.add_argument('--train_num_examples', type=int, default=1281167, required=False, help="Training number of examples.")

    p.add_argument('--eval_num_examples', type=int, default=50000, required=False, help="Evaluation number of examples")

    p.add_argument('--mean_subtract_in_dpipe', action='store_true', default=False, required=False, help="Whether to perform mean image subtraction in the data pipeline (dpipe) or not. If set to False, you can implement this in the compute graph.")##Nima

    p.add_argument('--standardize_in_dpipe', action='store_true', default=False, required=False, help="Whether to perform image standardization in the data pipeline (dpipe) or not.  If set to False, you can implement this in the compute graph.")##Nima


    FLAGS, unknown_args = p.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS
