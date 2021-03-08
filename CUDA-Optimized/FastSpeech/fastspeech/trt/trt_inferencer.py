# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import abc
import ctypes
import glob
import os
import pathlib
import sys
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt import Dims, ElementWiseOperation, MatrixOperation, Weights

from fastspeech.text_norm.symbols import symbols
from fastspeech.trt import TRT_LOGGER
from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import remove_module_in_state_dict, to_cpu_numpy


class TRTInferencer(object):

    def __init__(self, model_name, model, data_loader, ckpt_path=None, ckpt_file=None, trt_max_ws_size=1, trt_file_path=None, trt_force_build=False, use_fp16=False):
        self.model_name = model_name
        self.model = model
        self.data_loader = data_loader
        self.ckpt_path = ckpt_path
        self.ckpt_file = ckpt_file
        self.trt_max_ws_size = trt_max_ws_size
        self.trt_file_path = trt_file_path
        self.trt_force_build = trt_force_build
        self.use_fp16 = use_fp16

        self.batch_size = data_loader.batch_size

        self.plugins = dict()

        self.data_loader_iter = iter(self.data_loader)

        # checkpoint path
        if self.ckpt_path:
            self.ckpt_path = os.path.join(self.ckpt_path, self.model_name)
            pathlib.Path(self.ckpt_path).mkdir(parents=True, exist_ok=True)

            # load checkpoint
            self.load(ckpt_file)

        self.engine = self.build_engine()

    def __enter__(self):
        self.context = self.engine.create_execution_context()

    def __exit__(self, exception_type, exception_value, traceback):
        self.context.__del__()
        self.engine.__del__()

    def load(self, ckpt_file):
        # load latest checkpoint file if not defined.
        if not ckpt_file:
            files_exist = glob.glob(os.path.join(self.ckpt_path, '*'))
            if files_exist:
                ckpt_file = max(files_exist, key=os.path.getctime)

        if ckpt_file:
            state_dict = torch.load(ckpt_file, map_location='cpu')

            self.step = state_dict['step']
            self.model.load_state_dict(
                remove_module_in_state_dict(state_dict['model']))

            tprint('[Load] Checkpoint \'{}\'. Step={}'.format(
                ckpt_file, self.step))
        else:
            tprint('No checkpoints in {}. Load skipped.'.format(self.ckpt_path))

    def load_plugin(self, path):
        ctypes.cdll.LoadLibrary(path)

    def get_plugin_creator(self, plugin_name):
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
        plugin_creator = None
        for c in plugin_creator_list:
            if c.name == plugin_name:
                plugin_creator = c
        return plugin_creator

    def get_plugin(self, name):
        return self.plugins[name]

    @abc.abstractmethod
    def create_plugins(self):
        return NotImplemented

    @abc.abstractmethod
    def build_engine(self):
        return NotImplemented

    @abc.abstractmethod
    def infer(self):
        return NotImplemented
