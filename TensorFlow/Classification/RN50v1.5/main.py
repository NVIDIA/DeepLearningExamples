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

import os

import warnings
warnings.simplefilter("ignore")

import tensorflow as tf

import horovod.tensorflow as hvd
from utils import hvd_utils

from runtime import Runner

from utils.cmdline_helper import parse_cmdline

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    FLAGS = parse_cmdline()

    RUNNING_CONFIG = tf.contrib.training.HParams(
        mode=FLAGS.mode,
        
        # ======= Directory HParams ======= #
        log_dir=FLAGS.results_dir,
        model_dir=FLAGS.model_dir if FLAGS.model_dir is not None else FLAGS.results_dir,
        summaries_dir=FLAGS.results_dir,
        data_dir=FLAGS.data_dir,
        data_idx_dir=FLAGS.data_idx_dir,
        export_dir=FLAGS.export_dir,
        
        # ========= Model HParams ========= #
        n_classes=1001,
        input_format='NHWC',
        compute_format=FLAGS.data_format,
        dtype=tf.float32 if FLAGS.precision == "fp32" else tf.float16,
        height=224,
        width=224,
        n_channels=3,
        
        # ======= Training HParams ======== #
        iter_unit=FLAGS.iter_unit,
        num_iter=FLAGS.num_iter,
        warmup_steps=FLAGS.warmup_steps,
        batch_size=FLAGS.batch_size,
        log_every_n_steps=FLAGS.display_every,
        lr_init=FLAGS.lr_init,
        lr_warmup_epochs=FLAGS.lr_warmup_epochs,
        weight_decay=FLAGS.weight_decay,
        momentum=FLAGS.momentum,
        loss_scale=FLAGS.loss_scale,
        label_smoothing=FLAGS.label_smoothing,
        mixup=FLAGS.mixup,
        use_cosine_lr=FLAGS.use_cosine_lr,
        use_static_loss_scaling=FLAGS.use_static_loss_scaling,
        distort_colors=False,
        
        # ======= Optimization HParams ======== #
        use_xla=FLAGS.use_xla,
        use_tf_amp=FLAGS.use_tf_amp,
        use_dali=FLAGS.use_dali,
        gpu_memory_fraction=FLAGS.gpu_memory_fraction,
        gpu_id=FLAGS.gpu_id,
        
        seed=FLAGS.seed,
    )

    # ===================================

    runner = Runner(
        # ========= Model HParams ========= #
        n_classes=RUNNING_CONFIG.n_classes,
        input_format=RUNNING_CONFIG.input_format,
        compute_format=RUNNING_CONFIG.compute_format,
        dtype=RUNNING_CONFIG.dtype,
        n_channels=RUNNING_CONFIG.n_channels,
        height=RUNNING_CONFIG.height,
        width=RUNNING_CONFIG.width,
        distort_colors=RUNNING_CONFIG.distort_colors,
        log_dir=RUNNING_CONFIG.log_dir,
        model_dir=RUNNING_CONFIG.model_dir,
        data_dir=RUNNING_CONFIG.data_dir,
        data_idx_dir=RUNNING_CONFIG.data_idx_dir,
        

        # ======= Optimization HParams ======== #
        use_xla=RUNNING_CONFIG.use_xla,
        use_tf_amp=RUNNING_CONFIG.use_tf_amp,
        use_dali=RUNNING_CONFIG.use_dali,
        gpu_memory_fraction=RUNNING_CONFIG.gpu_memory_fraction,
        gpu_id=RUNNING_CONFIG.gpu_id,
        seed=RUNNING_CONFIG.seed
    )

    if RUNNING_CONFIG.mode in ["train", "train_and_evaluate", "training_benchmark"]:

        runner.train(
            iter_unit=RUNNING_CONFIG.iter_unit,
            num_iter=RUNNING_CONFIG.num_iter,
            batch_size=RUNNING_CONFIG.batch_size,
            warmup_steps=RUNNING_CONFIG.warmup_steps,
            log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
            weight_decay=RUNNING_CONFIG.weight_decay,
            lr_init=RUNNING_CONFIG.lr_init,
            lr_warmup_epochs=RUNNING_CONFIG.lr_warmup_epochs,
            momentum=RUNNING_CONFIG.momentum,
            loss_scale=RUNNING_CONFIG.loss_scale,       
            label_smoothing=RUNNING_CONFIG.label_smoothing,
            mixup=RUNNING_CONFIG.mixup,
            use_static_loss_scaling=RUNNING_CONFIG.use_static_loss_scaling,
            use_cosine_lr=RUNNING_CONFIG.use_cosine_lr,
            is_benchmark=RUNNING_CONFIG.mode == 'training_benchmark',
            
        )

    if RUNNING_CONFIG.mode in ["train_and_evaluate", 'evaluate', 'inference_benchmark']:

        if RUNNING_CONFIG.mode == 'inference_benchmark' and hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")

        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:

            runner.evaluate(
                iter_unit=RUNNING_CONFIG.iter_unit if RUNNING_CONFIG.mode != "train_and_evaluate" else "epoch",
                num_iter=RUNNING_CONFIG.num_iter if RUNNING_CONFIG.mode != "train_and_evaluate" else 1,
                warmup_steps=RUNNING_CONFIG.warmup_steps,
                batch_size=RUNNING_CONFIG.batch_size,
                log_every_n_steps=RUNNING_CONFIG.log_every_n_steps,
                is_benchmark=RUNNING_CONFIG.mode == 'inference_benchmark',
                export_dir=RUNNING_CONFIG.export_dir
            )
            
    if RUNNING_CONFIG.mode == 'predict':
        if FLAGS.to_predict is None:
            raise ValueError("No data to predict on.")
        
        if not os.path.isfile(FLAGS.to_predict):
            raise ValueError("Only prediction on single images is supported!")
        
        if hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")
            
        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
              runner.predict(FLAGS.to_predict)
