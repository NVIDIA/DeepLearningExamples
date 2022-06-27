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


from copy import deepcopy
from importlib import import_module
from config.defaults import base_config
from config.defaults import Config

from utils.setup import set_flags
from runtime import Runner
from utils.cmdline_helper import parse_cmdline


def get_module_path(sys_path):
    """[summary]
    converts the path to a py module to a format suitable for the import_module function.
    Ex: config/model/hparams.py -> config.model.hparams
    Args:
        sys_path (string): module path in sys format

    Returns:
        string: new format
    """
    no_ext = sys_path.split('.')[0]
    return no_ext.replace('/','.')


if __name__== "__main__":
    
    # get command line args 
    FLAGS = parse_cmdline()
    config = Config(FLAGS.__dict__)
    
    # get model hyperparameters from the user-provided model config
    model_config = import_module(get_module_path(FLAGS.cfg))
    model_config = Config(model_config.config)
    
    #override model hyper parameters by those provided by the user via cmd
    model_config.override(FLAGS.mparams)
    config.mparams = model_config
    
    # make sure number of classes in the model config is consistent with data loader config
    config.num_classes = config.mparams.num_classes 
    
    #========== horovod initialization
    hvd.init()
    
    #========== set up env variables, tf flags, and seeds
    set_flags(config)

    #========== set up the loggers and log dir
    backends = []
    if not hvd_utils.is_using_hvd() or hvd.rank() == 0:
        # Prepare Model Dir
        os.makedirs(config.model_dir, exist_ok=True)
        
        # Setup dlLogger
        backends+=[
            JSONStreamBackend(verbosity=Verbosity.VERBOSE, filename=config.log_filename),
            StdOutBackend(verbosity=Verbosity.DEFAULT)
        ]
    DLLogger.init(backends=backends)
    DLLogger.log(data=vars(config), step='PARAMETER')
    DLLogger.metadata('avg_exp_per_second_training', {'unit': 'samples/s'})
    DLLogger.metadata('avg_exp_per_second_training_per_GPU', {'unit': 'samples/s'})
    DLLogger.metadata('avg_exp_per_second_eval', {'unit': 'samples/s'})
    DLLogger.metadata('avg_exp_per_second_eval_per_GPU', {'unit': 'samples/s'})
    DLLogger.metadata('latency_pct', {'unit': 'ms'})
    DLLogger.metadata('latency_90pct', {'unit': 'ms'})
    DLLogger.metadata('latency_95pct', {'unit': 'ms'})
    DLLogger.metadata('latency_99pct', {'unit': 'ms'})
    DLLogger.metadata('eval_loss', {'unit': None})
    DLLogger.metadata('eval_accuracy_top_1', {'unit': None})
    DLLogger.metadata('eval_accuracy_top_5', {'unit': None})
    DLLogger.metadata('training_loss', {'unit': None})
    DLLogger.metadata('training_accuracy_top_1', {'unit': None})
    DLLogger.metadata('training_accuracy_top_5', {'unit': None})

    #========== initialize the runner
    runner = Runner(config, DLLogger)

    #========== determine the operation mode of the runner (tr,eval,predict)
    if config.mode in ["train", "train_and_eval", "training_benchmark"]:
        runner.train()
    if config.mode in ['eval', 'evaluate', 'inference_benchmark']:
        if config.mode == 'inference_benchmark' and hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")
        elif hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU evaluation is implemented.")
        else:
            runner.evaluate() 
    if config.mode == 'predict':
        if config.predict_img_dir is None:
            raise ValueError("No data to predict on.")

        if not os.path.isdir(config.predict_img_dir):
            raise ValueError("Provide directory with images to infer!")

        if hvd_utils.is_using_hvd():
            raise NotImplementedError("Only single GPU inference is implemented.")

        elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
            runner.predict(config.predict_img_dir, config.predict_ckpt)


