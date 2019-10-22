# !/usr/bin/env python
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

"""
Usage:

    python export_saved_model.py \
        --activation_fn='relu' \
        --batch_size=16 \
        --data_format='NCHW' \
        --input_dtype="fp32" \
        --export_dir="exported_models" \
        --model_checkpoint_path="path/to/checkpoint/model.ckpt-2500" \
        --unet_variant='tinyUNet' \
        --use_xla \
        --use_tf_amp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import pprint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from dllogger.logger import LOGGER

from model.unet import UNet_v1
from model.blocks.activation_blck import authorized_activation_fn

from utils.cmdline_helper import _add_bool_argument


def get_export_flags():
    parser = argparse.ArgumentParser(description="JoC-UNet_v1-TF-ExportFlags")

    parser.add_argument('--export_dir', default=None, required=True, type=str, help='The export directory.')
    parser.add_argument('--model_checkpoint_path', default=None, required=True, help='Checkpoint path.')

    parser.add_argument(
        '--data_format',
        choices=['NHWC', 'NCHW'],
        type=str,
        default="NCHW",
        required=False,
        help="""Which Tensor format is used for computation inside the mode"""
    )

    parser.add_argument(
        '--input_dtype',
        choices=['fp32', 'fp16'],
        type=str,
        default="fp32",
        required=False,
        help="""Tensorflow dtype of the input tensor"""
    )

    parser.add_argument(
        '--unet_variant',
        default="tinyUNet",
        choices=UNet_v1.authorized_models_variants,
        type=str,
        required=False,
        help="""Which model size is used. This parameter control directly the size and the number of parameters"""
    )

    parser.add_argument(
        '--activation_fn',
        choices=authorized_activation_fn,
        type=str,
        default="relu",
        required=False,
        help="""Which activation function is used after the convolution layers"""
    )

    _add_bool_argument(
        parser=parser,
        name="use_tf_amp",
        default=False,
        required=False,
        help="Enable Automatic Mixed Precision Computation to maximise performance."
    )

    _add_bool_argument(
        parser=parser,
        name="use_xla",
        default=False,
        required=False,
        help="Enable Tensorflow XLA to maximise performance."
    )

    parser.add_argument('--batch_size', default=16, type=int, help='Evaluation batch size.')

    FLAGS, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    return FLAGS


def export_model(RUNNING_CONFIG):

    if RUNNING_CONFIG.use_tf_amp:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    model = UNet_v1(
        model_name="UNet_v1",
        input_format="NHWC",
        compute_format=RUNNING_CONFIG.data_format,
        n_output_channels=1,
        unet_variant=RUNNING_CONFIG.unet_variant,
        weight_init_method="he_normal",
        activation_fn=RUNNING_CONFIG.activation_fn
    )

    config_proto = tf.ConfigProto()

    config_proto.allow_soft_placement = True
    config_proto.log_device_placement = False

    config_proto.gpu_options.allow_growth = True

    if RUNNING_CONFIG.use_xla:  # Only working on single GPU
        LOGGER.log("XLA is activated - Experimental Feature")
        config_proto.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    config_proto.gpu_options.force_gpu_compatible = True  # Force pinned memory

    run_config = tf.estimator.RunConfig(
        model_dir=None,
        tf_random_seed=None,
        save_summary_steps=1e9,  # disabled
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        session_config=config_proto,
        keep_checkpoint_max=None,
        keep_checkpoint_every_n_hours=1e9,  # disabled
        log_step_count_steps=1e9,
        train_distribute=None,
        device_fn=None,
        protocol=None,
        eval_distribute=None,
        experimental_distribute=None
    )

    estimator = tf.estimator.Estimator(
        model_fn=model,
        model_dir=RUNNING_CONFIG.model_checkpoint_path,
        config=run_config,
        params={'debug_verbosity': 0}
    )

    LOGGER.log('[*] Exporting the model ...')

    input_type = tf.float32 if RUNNING_CONFIG.input_dtype else tf.float16

    def get_serving_input_receiver_fn():

        input_shape = [RUNNING_CONFIG.batch_size, 512, 512, 1]

        def serving_input_receiver_fn():
            features = tf.placeholder(dtype=input_type, shape=input_shape, name='input_tensor')

            return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors=features)

        return serving_input_receiver_fn

    export_path = estimator.export_saved_model(
        export_dir_base=RUNNING_CONFIG.export_dir,
        serving_input_receiver_fn=get_serving_input_receiver_fn(),
        checkpoint_path=RUNNING_CONFIG.model_checkpoint_path
    )

    LOGGER.log('[*] Done! path: `%s`' % export_path.decode())


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.disable_eager_execution()

    flags = get_export_flags()

    for endpattern in [".index", ".meta"]:
        file_to_check = flags.model_checkpoint_path + endpattern
        if not os.path.isfile(file_to_check):
            raise FileNotFoundError("The checkpoint file `%s` does not exist" % file_to_check)

    print(" ========================= Export Flags =========================\n")
    pprint.pprint(dict(flags._get_kwargs()))
    print("\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    export_model(flags)
