# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
'''Contains helper functions for TRT components of JASPER inference
'''
import pycuda.driver as cuda
import tensorrt as trt

# Simple class: more explicit than dealing with 2-tuple
class HostDeviceMem(object):
    '''Type for managing host and device buffers

    A simple class which is more explicit that dealing with a 2-tuple.
    '''
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def build_engine_from_parser(model_path, batch_size, is_fp16=True, is_verbose=False, max_workspace_size=4*1024*1024*1024):
    '''Builds TRT engine from an ONNX file
    Note that network output 1 is unmarked so that the engine will not use
    vestigial length calculations associated with masked_fill
    '''
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if is_verbose else trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder:
        builder.max_batch_size = batch_size
        builder.fp16_mode = is_fp16
        builder.max_workspace_size = max_workspace_size
        with builder.create_network() as network:
            with trt.OnnxParser(network, TRT_LOGGER) as parser:
                with open(model_path, 'rb') as model:
                    parser.parse(model.read())
                
                return builder.build_cuda_engine(network)

def deserialize_engine(engine_path, is_verbose):
    '''Deserializes TRT engine at engine_path
    '''
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if is_verbose else trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers_with_existing_inputs(engine, inp, batch_size=1):
    '''
    allocate_buffers() (see TRT python samples) but uses an existing inputs on device

    inp:  List of pointers to device memory. Pointers are in the same order as
          would be produced by allocate_buffers(). That is, inputs are in the
          order defined by iterating through `engine`
    '''

    # Add input to bindings
    bindings = []
    outputs = []
    stream = cuda.Stream()
    inp_idx = 0

    for binding in engine:
        if engine.binding_is_input(binding):
            bindings.append(inp[inp_idx])
            inp_idx += 1
        else:
            # Unchanged from do_inference()
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes*2)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return outputs, bindings, stream
