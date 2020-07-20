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

import os

import warnings

warnings.simplefilter("ignore")

import tensorflow as tf

import horovod.tensorflow as hvd
from utils import hvd_utils

from runtime import Runner

from utils.cmdline_helper import parse_cmdline
from utils.logging import init_dllogger

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_EXTRA_PTXAS_OPTIONS"] = "-sw200428197=true" # TODO: NINJA WAR

    FLAGS = parse_cmdline()
    
    init_dllogger(FLAGS.log_dir)

    RUNNING_CONFIG = tf.contrib.training.HParams(
        exec_mode=FLAGS.exec_mode,
        save_eval_results_to_json=FLAGS.save_eval_results_to_json,

        # ======= Directory HParams ======= #
        log_dir=os.path.join(FLAGS.results_dir, "logs"),
        model_dir=os.path.join(FLAGS.results_dir, "checkpoints"),
        summaries_dir=os.path.join(FLAGS.results_dir, "summaries"),
        sample_dir=os.path.join(FLAGS.results_dir, "samples"),
        data_dir=FLAGS.data_dir,
        dataset_name=FLAGS.dataset_name,
        dataset_hparams=dict(),

        # ========= Model HParams ========= #
        unet_variant=FLAGS.unet_variant,
        activation_fn=FLAGS.activation_fn,
        input_format='NHWC',
        compute_format=FLAGS.data_format,
        input_shape=(512, 512),
        mask_shape=(512, 512),
        n_channels=1,
        input_normalization_method="zero_one",

        # ======== Runtime HParams ======== #
        amp=FLAGS.amp,
        xla=FLAGS.xla,

        # ======= Training HParams ======== #
        iter_unit=FLAGS.iter_unit,
        num_iter=FLAGS.num_iter,
        warmup_steps=FLAGS.warmup_step,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        learning_rate_decay_steps=FLAGS.learning_rate_decay_steps,
        rmsprop_decay=FLAGS.rmsprop_decay,
        rmsprop_momentum=FLAGS.rmsprop_momentum,
        weight_decay=FLAGS.weight_decay,
        use_auto_loss_scaling=FLAGS.use_auto_loss_scaling,
        loss_fn_name=FLAGS.loss_fn_name,
        augment_data=FLAGS.augment_data,
        weight_init_method=FLAGS.weight_init_method,

        # ======== Debug Flags ======== #
        # 0: No debug
        # 1: Layer Creation Debug Info
        # 2: Layer + Var Creation Debug Info
        debug_verbosity=FLAGS.debug_verbosity,
        log_every_n_steps=FLAGS.display_every,
        seed=FLAGS.seed,
    )

    # ===================================

    if RUNNING_CONFIG.dataset_name == "DAGM2007":
        RUNNING_CONFIG.dataset_hparams["class_id"] = FLAGS.dataset_classID

    runner = Runner(
        input_format=RUNNING_CONFIG.input_format,
        compute_format=RUNNING_CONFIG.compute_format,
        n_channels=RUNNING_CONFIG.n_channels,
        model_variant=RUNNING_CONFIG.unet_variant,
        activation_fn=RUNNING_CONFIG.activation_fn,
        input_shape=RUNNING_CONFIG.input_shape,
        mask_shape=RUNNING_CONFIG.mask_shape,
        input_normalization_method=RUNNING_CONFIG.input_normalization_method,

        # Training HParams
        augment_data=RUNNING_CONFIG.augment_data,
        loss_fn_name=RUNNING_CONFIG.loss_fn_name,
        weight_init_method=RUNNING_CONFIG.weight_init_method,

        #  Runtime HParams
        amp=RUNNING_CONFIG.amp,
        xla=RUNNING_CONFIG.xla,

        # Directory Params
        log_dir=RUNNING_CONFIG.log_dir,
        model_dir=RUNNING_CONFIG.model_dir,
        sample_dir=RUNNING_CONFIG.sample_dir,
        data_dir=RUNNING_CONFIG.data_dir,
        dataset_name=RUNNING_CONFIG.dataset_name,
        dataset_hparams=RUNNING_CONFIG.dataset_hparams,

        # Debug Params
        debug_verbosity=RUNNING_CONFIG.debug_verbosity,
        log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
        seed=RUNNING_CONFIG.seed
    )

    if RUNNING_CONFIG.exec_mode in ["train", "train_and_evaluate", "training_benchmark"]:
        runner.train(
            iter_unit=RUNNING_CONFIG.iter_unit,
            num_iter=RUNNING_CONFIG.num_iter,
            batch_size=RUNNING_CONFIG.batch_size,
            warmup_steps=RUNNING_CONFIG.warmup_steps,
            weight_decay=RUNNING_CONFIG.weight_decay,
            learning_rate=RUNNING_CONFIG.learning_rate,
            learning_rate_decay_factor=RUNNING_CONFIG.learning_rate_decay_factor,
            learning_rate_decay_steps=RUNNING_CONFIG.learning_rate_decay_steps,
            rmsprop_decay=RUNNING_CONFIG.rmsprop_decay,
            rmsprop_momentum=RUNNING_CONFIG.rmsprop_momentum,
            use_auto_loss_scaling=FLAGS.use_auto_loss_scaling,
            augment_data=RUNNING_CONFIG.augment_data,
            is_benchmark=RUNNING_CONFIG.exec_mode == 'training_benchmark'
        )

    if RUNNING_CONFIG.exec_mode in ["train_and_evaluate", 'evaluate', 'inference_benchmark'] and hvd.rank() == 0:
        runner.evaluate(
            iter_unit=RUNNING_CONFIG.iter_unit if RUNNING_CONFIG.exec_mode != "train_and_evaluate" else "epoch",
            num_iter=RUNNING_CONFIG.num_iter if RUNNING_CONFIG.exec_mode != "train_and_evaluate" else 1,
            warmup_steps=RUNNING_CONFIG.warmup_steps,
            batch_size=RUNNING_CONFIG.batch_size,
            is_benchmark=RUNNING_CONFIG.exec_mode == 'inference_benchmark',
            save_eval_results_to_json=RUNNING_CONFIG.save_eval_results_to_json
        )
