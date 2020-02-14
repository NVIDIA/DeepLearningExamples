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
import onnxruntime as ort
import numpy as np

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

def build_engine_from_parser(args):
    '''Builds TRT engine from an ONNX file
    Note that network output 1 is unmarked so that the engine will not use
    vestigial length calculations associated with masked_fill
    '''
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if args.verbose else trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = 64

    if args.trt_fp16:
        builder.fp16_mode = True
        print("Optimizing for FP16")
        config_flags = 1 << int(trt.BuilderFlag.FP16) # | 1 << int(trt.BuilderFlag.STRICT_TYPES)
        max_size = 4*1024*1024*1024
        max_len = args.max_seq_len
    else:
        config_flags = 0
        max_size = 4*1024*1024*1024
        max_len = args.max_seq_len
    if args.max_workspace_size > 0:
        builder.max_workspace_size = args.max_workspace_size
    else:
        builder.max_workspace_size = max_size
        
    config = builder.create_builder_config()
    config.flags = config_flags
    
    if not args.static_shape:
        profile = builder.create_optimization_profile()
        if args.transpose:
            profile.set_shape("FEATURES", min=(1,192,64), opt=(args.engine_batch_size,256,64), max=(builder.max_batch_size, max_len, 64))
        else:
            profile.set_shape("FEATURES", min=(1,64,192), opt=(args.engine_batch_size,64,256), max=(builder.max_batch_size, 64, max_len))        
        config.add_optimization_profile(profile)    
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(args.onnx_path, 'rb') as model:
            parsed = parser.parse(model.read())
            print ("Parsing returned ", parsed, "dynamic_shape= " , not args.static_shape, "\n")
            return builder.build_engine(network, config=config)

def deserialize_engine(engine_path, is_verbose):
    '''Deserializes TRT engine at engine_path
    '''
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if is_verbose else trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers_with_existing_inputs(context, inp):
    '''
    allocate_buffers() (see TRT python samples) but uses an existing inputs on device

    inp:  List of pointers to device memory. Pointers are in the same order as
          would be produced by allocate_buffers(). That is, inputs are in the
          order defined by iterating through `engine`
    '''
    # Add input to bindings
    bindings = [0,0]
    outputs = []
    engine = context.engine
    batch_size = inp[0].shape
    inp_idx = engine.get_binding_index("FEATURES")    
    inp_b = inp[0].data_ptr()
    assert(inp[0].is_contiguous())
    bindings[inp_idx] = inp_b
    sh = inp[0].shape
    batch_size = sh[0]
    orig_shape = context.get_binding_shape(inp_idx)
    if orig_shape[0]==-1:
        context.set_binding_shape(inp_idx, trt.Dims([batch_size, sh[1], sh[2]]))

    assert context.all_binding_shapes_specified

    out_idx = engine.get_binding_index("LOGITS")
    # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
    out_shape = context.get_binding_shape(out_idx)
    #print ("Out_shape: ", out_shape)
    h_output = cuda.pagelocked_empty(tuple(out_shape), dtype=np.float32())
    # print ("Out bytes: " , h_output.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings[out_idx] = int(d_output)
    hdm = HostDeviceMem(h_output, d_output)
    outputs.append(hdm)
    return outputs, bindings, out_shape

def get_engine(args):
    '''Get a TRT engine

    If --should_serialize is present, always build from ONNX and store result in --engine_path.
    Else If an engine is provided as an argument (--engine_path) use that one.
    Otherwise, make one from onnx (--onnx_load_path), but don't serialize it.
    '''
    engine = None

    if args.engine_path is not None and args.use_existing_engine:
        engine = deserialize_engine(args.engine_path, args.verbose)
    elif args.engine_path is not None and args.onnx_path is not None:
        # Build a new engine and serialize it.
        print("Building TRT engine ....") 
        engine = build_engine_from_parser(args)
        if engine is not None:
            with open(args.engine_path, 'wb') as f:
                f.write(engine.serialize())
                print("TRT engine saved at " + args.engine_path + " ...") 
    elif args.onnx_path is not None:
        ort_session = ort.InferenceSession(args.onnx_path)
        return ort_session
    else:
        raise Exception("One of the following sets of arguments must be provided:\n"+
                        "<engine_path> + --use_existing_engine\n"+
                        "<engine_path> + <onnx_path>\n"+
                        "in order to construct a TRT engine")
    if engine is None:
        raise Exception("Failed to acquire TRT engine")

    return engine
