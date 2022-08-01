# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pandas as pd

import dllogger
from dllogger import JSONStreamBackend, Logger, StdOutBackend
from .backends import AggregatorBackend, TensorBoardBackend, AverageMeter
from omegaconf import OmegaConf

from distributed_utils import is_main_process

def jsonlog_2_df(path, keys):
    with open(path, 'r') as f:
        log = [json.loads(l[4:]) for l in f.readlines()]
        log = [l for l in log if l['type'] == 'LOG' and isinstance(l['step'], (int, list))]
        assert log[-1]['step'] == [], "Logfile is corrupted"
        log[-1]['step']=log[-2]['step'] # Every log ends with step == []
        log = [
                {
                    **{k:v for k,v in l.items() if not isinstance(v, dict)},
                    **(l['data'] if 'data' in l else {}),
                    'timestamp':float(l['timestamp'])*1000
                }
                for l in log
              ]
        log = [{k:v for k,v in l.items() if k in keys} for l in log]
        df = pd.DataFrame(log)
        df = df.groupby('step').mean()
        return df

def empty_step_format(step):
    return ""


def empty_prefix_format(timestamp):
    return ""


def no_string_metric_format(metric, metadata, value):
    unit = metadata["unit"] if "unit" in metadata.keys() else ""
    format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    if metric == "String":
        return "{} {}".format(format.format(value) if value is not None else value, unit)
    return "{} : {} {}".format(metric, format.format(value) if value is not None else value, unit)


def setup_logger(config, resume_training=False):
    log_filename = config.get("log_filename", "log.json")
    if is_main_process():
        backends = [
            TensorBoardBackend(verbosity=dllogger.Verbosity.VERBOSE),
            JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=log_filename, append=True),
            AggregatorBackend(verbosity=dllogger.Verbosity.VERBOSE, agg_dict={"loss": AverageMeter}),
            StdOutBackend(
                verbosity=dllogger.Verbosity.DEFAULT,
                step_format=empty_step_format,
                metric_format=no_string_metric_format,
                prefix_format=empty_prefix_format,
                ),
        ]

        logger = Logger(backends=backends)
    else:
        logger = Logger(backends=[])

    container_setup_info = get_framework_env_vars()
    logger.log(step="PARAMETER", data=container_setup_info, verbosity=dllogger.Verbosity.VERBOSE)

    if not resume_training:
        logger.metadata("loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        logger.metadata("val_loss", {"unit": "nat", "GOAL": "MINIMIZE", "STAGE": "VAL"})
    return logger

def restart_logger(config, logger):
    """An utility function to nealty close every backend holding resources"""
    for b in logger.backends:
        if hasattr(b, 'close'):
            b.close()
    return setup_logger(config, resume_training=True)

def log_parameters(logger, config):
    model_config = flatten_config(config.model)
    trainer_config = flatten_config(config.trainer)
    additional_fields = {'seed': config.seed}
    logger.log(step="PARAMETER", data={**model_config, **trainer_config, **additional_fields}, verbosity=dllogger.Verbosity.VERBOSE)

def flatten_config(config):
    config = OmegaConf.to_container(config, resolve=True)
    if '_target_' in config:
        del config['_target_']
    if 'config' in config:
        c = config['config']
        config = {**c, **config}
        del config['config']
    config = pd.json_normalize(config, sep='.')
    config = config.to_dict(orient='records')[0]
    return config

def get_framework_env_vars():
    return {
        "NVIDIA_PYTORCH_VERSION": os.environ.get("NVIDIA_PYTORCH_VERSION"),
        "PYTORCH_VERSION": os.environ.get("PYTORCH_VERSION"),
        "CUBLAS_VERSION": os.environ.get("CUBLAS_VERSION"),
        "NCCL_VERSION": os.environ.get("NCCL_VERSION"),
        "CUDA_DRIVER_VERSION": os.environ.get("CUDA_DRIVER_VERSION"),
        "CUDNN_VERSION": os.environ.get("CUDNN_VERSION"),
        "CUDA_VERSION": os.environ.get("CUDA_VERSION"),
        "NVIDIA_PIPELINE_ID": os.environ.get("NVIDIA_PIPELINE_ID"),
        "NVIDIA_BUILD_ID": os.environ.get("NVIDIA_BUILD_ID"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }
