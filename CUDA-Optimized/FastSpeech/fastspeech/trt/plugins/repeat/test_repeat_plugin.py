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

from __future__ import print_function
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import ctypes
import os
import time
import sys

logger = trt.Logger(trt.Logger.INFO)

PLUGIN_PATH = '/home/dahn/git/fastspeech/fastspeech/trt/plugins/repeat/RepeatPlugin.so'
ctypes.cdll.LoadLibrary(PLUGIN_PATH)

def get_plugin_creator(plugin_name):
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def build_engine(shape, shape2):
    plugin_creator = get_plugin_creator('RepeatPlugin')
    if plugin_creator == None:
        print('Plugin not found. Exiting')
        exit()

    builder = trt.Builder(logger)
    builder.max_batch_size = 1024
    builder.max_workspace_size = 1 << 20
    builder.fp16_mode = use_fp16
    network = builder.create_network()
    
    tensor = network.add_input('input1', trt.DataType.FLOAT, shape)
    tensor2 = network.add_input('input2', trt.DataType.FLOAT, shape2)    
    tensor = network.add_plugin_v2(
        [tensor, tensor2], 
        plugin_creator.create_plugin('RepeatPlugin', trt.PluginFieldCollection([
                trt.PluginField('maxOutputLength', np.array([MAX_OUTPUT_LENGTH], dtype=np.int32), trt.PluginFieldType.INT32)
        ]))
    ).get_output(0)

    network.mark_output(tensor)

    return builder.build_cuda_engine(network)
    
def run_trt(input1, input2):
    batch_size = input1.shape[0]

    engine = build_engine(input1.shape[1:], input2.shape[1:])

    context = engine.create_execution_context()
    
    d_input1 = cuda.mem_alloc(input1.nbytes)
    d_input2 = cuda.mem_alloc(input2.nbytes)

    output = np.zeros(shape=(batch_size, MAX_OUTPUT_LENGTH, input1.shape[2]), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    cuda.memcpy_htod(d_input1, input1)
    cuda.memcpy_htod(d_input2, input2)

    bindings = [int(d_input1), int(d_input2), int(d_output)]    

    start = time.time()
    context.execute(batch_size, bindings)
    end = time.time()
    time_elapsed = end - start
    print("time elapsed: {:06f}".format(time_elapsed))

    cuda.memcpy_dtoh(output, d_output)
    
    return output

use_fp16 = len(sys.argv) > 1 and sys.argv[1].isdigit() and int(sys.argv[1]) == 1
print('Use FP16:', use_fp16)

##
# accuray test
##

MAX_OUTPUT_LENGTH=8

inputs = np.array([
    [[1, 2], [4, 5], [7, 8]],
    [[3, 4], [5, 6], [8, 9]]
], np.float32)

masks = np.ones((2,3,1), np.float32)
repeats = np.array([
    [[0, 2, 10]],
    [[1, 2, 1]]
], np.float32)
output = run_trt(inputs, repeats)
print(output)
print(output.shape)
print(type(output))

output_mask = run_trt(masks, repeats)
print(output_mask)
print(output_mask.shape)
print(type(output_mask))


##
# latency test
##

# MAX_OUTPUT_LENGTH=1024
# inputs = np.full((16, 256, 384), 2, np.float32)
# masks = np.ones((16, 256, 384), np.float32)
# repeats = np.full((16, 256), 4, np.float32)

# output = run_trt(inputs, repeats)
# output_mask = run_trt(masks, repeats)