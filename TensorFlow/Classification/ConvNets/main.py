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
import dllogger

from utils import hvd_utils
from runtime import Runner
from model.resnet import model_architectures

from utils.cmdline_helper import parse_cmdline

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    FLAGS = parse_cmdline(model_architectures.keys())
    hvd.init()

    if hvd.rank() == 0:
        log_path = os.path.join(FLAGS.results_dir, FLAGS.log_filename)
        os.makedirs(FLAGS.results_dir, exist_ok=True)

        dllogger.init(
            backends=[
                dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=log_path),
                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)
            ]
        )
    else:
        dllogger.init(backends=[])
    dllogger.log(data=vars(FLAGS), step='PARAMETER')

    runner = Runner(
        # ========= Model HParams ========= #
        n_classes=1001,
        architecture=FLAGS.arch,
        input_format='NHWC',
        compute_format=FLAGS.data_format,
        dtype=tf.float32 if FLAGS.precision == 'fp32' else tf.float16,
        n_channels=3,
        height=224,
        width=224,
        distort_colors=False,
        log_dir=FLAGS.results_dir,
        model_dir=FLAGS.model_dir if FLAGS.model_dir is not None else FLAGS.results_dir,
        data_dir=FLAGS.data_dir,
        data_idx_dir=FLAGS.data_idx_dir,
        weight_init=FLAGS.weight_init,
        use_xla=FLAGS.use_xla,
        use_tf_amp=FLAGS.use_tf_amp,
        use_dali=FLAGS.use_dali,
        gpu_memory_fraction=FLAGS.gpu_memory_fraction,
        gpu_id=FLAGS.gpu_id,
        seed=FLAGS.seed
    )

    if FLAGS.mode in ["train", "train_and_evaluate", "training_benchmark"]:
        runner.train(
            iter_unit=FLAGS.iter_unit,
            num_iter=FLAGS.num_iter,
            run_iter=FLAGS.run_iter,
            batch_size=FLAGS.batch_size,
            warmup_steps=FLAGS.warmup_steps,
            log_every_n_steps=FLAGS.display_every,
            weight_decay=FLAGS.weight_decay,
            lr_init=FLAGS.lr_init,
            lr_warmup_epochs=FLAGS.lr_warmup_epochs,
            momentum=FLAGS.momentum,
            loss_scale=FLAGS.loss_scale,
            label_smoothing=FLAGS.label_smoothing,
            mixup=FLAGS.mixup,
            use_static_loss_scaling=FLAGS.use_static_loss_scaling,
            use_cosine_lr=FLAGS.use_cosine_lr,
            is_benchmark=FLAGS.mode == 'training_benchmark',
            use_final_conv=FLAGS.use_final_conv,
            quantize=FLAGS.quantize,
            symmetric=FLAGS.symmetric,
            quant_delay = FLAGS.quant_delay,
            use_qdq = FLAGS.use_qdq,
            finetune_checkpoint = FLAGS.finetune_checkpoint,
        )

    if FLAGS.mode in ["train_and_evaluate", 'evaluate', 'inference_benchmark']:

        if FLAGS.mode == 'inference_benchmark' and hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")

        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:

            runner.evaluate(
                iter_unit=FLAGS.iter_unit if FLAGS.mode != "train_and_evaluate" else "epoch",
                num_iter=FLAGS.num_iter if FLAGS.mode != "train_and_evaluate" else 1,
                warmup_steps=FLAGS.warmup_steps,
                batch_size=FLAGS.batch_size,
                log_every_n_steps=FLAGS.display_every,
                is_benchmark=FLAGS.mode == 'inference_benchmark',
                export_dir=FLAGS.export_dir,
                quantize=FLAGS.quantize,
                symmetric=FLAGS.symmetric,
                use_final_conv=FLAGS.use_final_conv,
                use_qdq=FLAGS.use_qdq
            )

    if FLAGS.mode == 'predict':
        if FLAGS.to_predict is None:
            raise ValueError("No data to predict on.")

        if not os.path.isfile(FLAGS.to_predict):
            raise ValueError("Only prediction on single images is supported!")

        if hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")

        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            runner.predict(FLAGS.to_predict, quantize=FLAGS.quantize, symmetric=FLAGS.symmetric, use_qdq=FLAGS.use_qdq, use_final_conv=FLAGS.use_final_conv)
