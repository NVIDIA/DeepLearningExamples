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
import sys
import time

logger = trt.Logger(trt.Logger.INFO)

PLUGIN_PATH = '/home/dahn/git/fastspeech/fastspeech/trt/plugins/add_pos_enc/AddPosEncPlugin.so'
ctypes.cdll.LoadLibrary(PLUGIN_PATH)

def get_plugin_creator(plugin_name):
    trt.init_libnvinfer_plugins(logger, '')
    plugin_creator_list = trt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

def build_engine(shape):
    plugin_creator = get_plugin_creator('AddPosEncPlugin')
    if plugin_creator == None:
        print('Plugin not found. Exiting')
        exit()

    builder = trt.Builder(logger)
    builder.max_batch_size = 1024
    builder.max_workspace_size = 1 << 20
    builder.fp16_mode = use_fp16
    network = builder.create_network()
    
    tensor = network.add_input('data', trt.DataType.FLOAT, shape)
    tensor = network.add_plugin_v2(
        [tensor], 
        plugin_creator.create_plugin('AddPosEncPlugin', trt.PluginFieldCollection())
    ).get_output(0)

    network.mark_output(tensor)
    return builder.build_cuda_engine(network)
    
def run_trt(data):
    engine = build_engine(data.shape[1:])

    context = engine.create_execution_context()
    
    d_data = cuda.mem_alloc(data.nbytes)

    output = np.zeros_like(data, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    
    cuda.memcpy_htod(d_data, data)

    bindings = [int(d_data), int(d_output)]    

    start = time.time()
    context.execute(data.shape[0], bindings)
    end = time.time()
    time_elapsed = end - start
    print("time elapsed: {:06f}".format(time_elapsed))

    cuda.memcpy_dtoh(output, d_output)
    
    return output

use_fp16 = len(sys.argv) > 1 and sys.argv[1].isdigit() and int(sys.argv[1]) == 1
print('Use FP16:', use_fp16)

output = run_trt(np.zeros((16, 128, 384), np.float32))

print(output)
print(output.shape)