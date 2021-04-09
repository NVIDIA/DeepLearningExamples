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

import os

import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
import horovod.tensorflow as hvd
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger
from utils import hvd_utils

from utils.setup import set_flags
from runtime import Runner
from utils.cmdline_helper import parse_cmdline

if __name__ == "__main__":

    hvd.init()
    FLAGS = parse_cmdline()
    set_flags(FLAGS)

    backends = []
    if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
        # Prepare Model Dir
        log_path = os.path.join(FLAGS.model_dir, FLAGS.log_filename)
        os.makedirs(FLAGS.model_dir, exist_ok=True)
        # Setup dlLogger
        backends+=[
            JSONStreamBackend(verbosity=Verbosity.VERBOSE, filename=log_path),
            StdOutBackend(verbosity=Verbosity.DEFAULT)
        ]
    DLLogger.init(backends=backends)
    DLLogger.log(data=vars(FLAGS), step='PARAMETER')

    runner = Runner(FLAGS, DLLogger)

    if FLAGS.mode in ["train", "train_and_eval", "training_benchmark"]:
        runner.train()
        
    if FLAGS.mode in ['eval', 'evaluate', 'inference_benchmark']:
        if FLAGS.mode == 'inference_benchmark' and hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")
        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            runner.evaluate()
            
    if FLAGS.mode == 'predict':
        if FLAGS.to_predict is None:
            raise ValueError("No data to predict on.")

        if not os.path.isdir(FLAGS.to_predict):
            raise ValueError("Provide directory with images to infer!")

        if hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")

        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            runner.predict(FLAGS.to_predict, FLAGS.inference_checkpoint)