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

import argparse
import os
from itertools import chain

import numpy as np
import torch

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

##
# Common
##

def GiB(val):
    return val * 1 << 30


def input_binding_indices(engine):
    return [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]


def output_binding_indices(engine):
    return [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]


def trt_input_names(engine):
    return [engine.get_binding_name(i) for i in input_binding_indices(engine)]


def trt_output_names(engine):
    return [engine.get_binding_name(i) for i in output_binding_indices(engine)]


def set_input_shapes(engine, context, inputs):
    def is_dimension_dynamic(dim):
        return dim is None or dim <= 0

    def is_shape_dynamic(shape):
        return any([is_dimension_dynamic(dim) for dim in shape])

    for idx, tensor in enumerate(inputs):
        if engine.is_shape_binding(idx) and is_shape_dynamic(context.get_shape(idx)):
            context.set_shape_input(idx, tensor)
        elif is_shape_dynamic(engine.get_binding_shape(idx)):
            context.set_binding_shape(idx, tensor.shape)
    
    return context


##
# Pytorch Compatibility
##

# Modified from https://github.com/NVIDIA-AI-IOT/jetbot/blob/cf3e264ae6/jetbot/tensorrt_model.py

def torch_dtype_to_trt(dtype):
    if dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def create_inputs_from_torch(engine, inputs_torch):
    input_ids = input_binding_indices(engine)
    for i, idx in enumerate(input_ids):
        inputs_torch[i] = inputs_torch[i].to(torch_device_from_trt(engine.get_location(idx)))
        inputs_torch[i] = inputs_torch[i].type(torch_dtype_from_trt(engine.get_binding_dtype(idx)))        
    return inputs_torch


def create_outputs_from_torch(engine, outputs_shapes=None):
    output_ids = output_binding_indices(engine)
    outputs = [None] * len(output_ids)
    for i, idx in enumerate(output_ids):
        dtype = torch_dtype_from_trt(engine.get_binding_dtype(idx))
        shape = outputs_shapes[i] if outputs_shapes and outputs_shapes[i] else tuple(engine.get_binding_shape(idx))
        device = torch_device_from_trt(engine.get_location(idx))
        output = torch.empty(size=shape, dtype=dtype, device=device)
        outputs[i] = output
    return outputs
