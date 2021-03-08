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

import ctypes
import glob
import os
import pathlib
import sys
from collections import OrderedDict

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt import Dims, ElementWiseOperation, MatrixOperation, Weights

import fastspeech.trt.common as common
from fastspeech.trt import TRT_LOGGER
from fastspeech.trt.fastspeech_trt_inferencer import FastSpeechTRTInferencer
from fastspeech.trt.trt_inferencer import TRTInferencer
from fastspeech.utils.logging import tprint
from fastspeech.utils.nvtx import Nvtx
from fastspeech.utils.pytorch import (remove_module_in_state_dict,
                                      to_cpu_numpy, to_gpu_async)


class FastSpeechTRTMultiEngineInferencer(FastSpeechTRTInferencer):
    
    def __init__(self, model_name, model, data_loader, ckpt_path=None, ckpt_file=None,
                 trt_max_ws_size=1, trt_force_build=False, use_fp16=False,
                 trt_file_path_list=[], trt_max_input_seq_len_list=[], trt_max_output_seq_len_list=[]):

        self.trt_file_path_list = trt_file_path_list
        self.trt_max_input_seq_len_list = trt_max_input_seq_len_list
        self.trt_max_output_seq_len_list = trt_max_output_seq_len_list

        # sort by trt_max_input_seq_len in ascending order.
        self.max_seq_lens_and_file_path_list = sorted(zip(self.trt_max_input_seq_len_list,
                                                          self.trt_max_output_seq_len_list,
                                                          self.trt_file_path_list))
        self.engine = None
        self.context = None

        super(FastSpeechTRTMultiEngineInferencer, self).__init__(model_name, model, data_loader, ckpt_path, ckpt_file,
                                                                 trt_max_ws_size, None, trt_force_build, use_fp16, 
                                                                 None, None, False)

    def __enter__(self):
        for engine in self.engine_list:
            self.context_list.append(engine.create_execution_context())

    def __exit__(self, exception_type, exception_value, traceback):
        for engine, context in zip(self.engine_list, self.context_list):
            context.__del__()
            engine.__del__()

    def build_engine(self):
        # load engines and create contexts
        self.engine_list = []
        self.context_list = []
        for i, (trt_max_input_seq_len, trt_max_output_seq_len, trt_file_path) in enumerate(self.max_seq_lens_and_file_path_list):
            if trt_file_path and os.path.isfile(trt_file_path) and not self.trt_force_build:
                with open(trt_file_path, 'rb') as f:
                    engine_str = f.read()
                with trt.Runtime(TRT_LOGGER) as runtime:
                    engine = runtime.deserialize_cuda_engine(engine_str)
                tprint('TRT Engine Loaded from {} successfully.'.format(trt_file_path))
            else:
                self.trt_max_input_seq_len = trt_max_input_seq_len
                self.trt_max_output_seq_len = trt_max_output_seq_len
                self.trt_file_path = trt_file_path

                tprint('Building a TRT Engine..')
                engine = self.do_build_engine()
                tprint('TRT Engine Built.')

                with open(self.trt_file_path, 'wb') as f:
                    f.write(engine.serialize())
                tprint('TRT Engine Saved in {}.'.format(self.trt_file_path))

            self.engine_list.append(engine)

    def set_engine_and_context(self, length):
        for i, (trt_max_input_seq_len, trt_max_output_seq_len, trt_file_path) in enumerate(self.max_seq_lens_and_file_path_list):
            if length <= trt_max_input_seq_len:
                self.engine = self.engine_list[i]
                self.context = self.context_list[i]
                self.trt_max_input_seq_len = trt_max_input_seq_len
                self.trt_max_output_seq_len = trt_max_output_seq_len
                self.trt_file_path = trt_file_path
                break
        else:
            self.engine = self.engine_list[-1]
            self.context = self.context_list[-1]
            self.trt_max_input_seq_len = trt_max_input_seq_len
            self.trt_max_output_seq_len = trt_max_output_seq_len
            self.trt_file_path = trt_file_path
        tprint('TRT Engine {} is selected.'.format(self.trt_file_path))

    def infer(self, acts=None):
        inputs = next(self.data_loader_iter)

        text_encoded = inputs["text_encoded"]  # (b, t)
        text_pos = inputs["text_pos"]  # (b, t)

        self.set_engine_and_context(length=text_encoded.size(1))

        text_encoded = F.pad(text_encoded, pad=(0, self.trt_max_input_seq_len - text_encoded.size(1)))  # (b, t)
        text_pos = F.pad(text_pos, pad=(0, self.trt_max_input_seq_len - text_pos.size(1)))  # (b, t)

        text_mask = text_pos.ne(0)  # padded is False

        # TODO: process word emb in TRT if the API allows.
        with torch.no_grad():
            text_encoded = self.model.word_emb(text_encoded)
        
        # create input/output buffers
        input_buffers = common.create_inputs_from_torch(self.engine, [text_encoded, text_mask])
        output_buffers = common.create_outputs_from_torch(self.engine)

        # bindings
        bindings = [int(data.data_ptr()) for data in (input_buffers + output_buffers)]

        # execute
        # self.context.profiler = trt.Profiler()
        stream = cuda.Stream()
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # self.context.execute(batch_size=self.batch_size, bindings=bindings)
        stream.synchronize()

        outputs = dict()
        outputs['mel'] = output_buffers[-2]
        outputs['mel_mask'] = output_buffers[-1]
        outputs['text'] = inputs["text_norm"]

        # activations for verifying accuracy.
        if acts is not None:
            act_names = common.trt_output_names(self.engine)
            n_acts = len(output_buffers) - 2  # exclude outputs(mel and mel_mask)
            for i in range(n_acts):
                acts[act_names[i]] = output_buffers[i]

        return outputs