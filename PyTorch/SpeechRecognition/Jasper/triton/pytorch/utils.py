# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import tensorrt as trt
import torch
from collections import Counter
import json

import logging

triton_type_to_torch_type = {
    'TYPE_BOOL': torch.bool,
    'TYPE_INT8': torch.int8,
    'TYPE_INT16': torch.int16,
    'TYPE_INT32': torch.int32,
    'TYPE_INT64': torch.int64,
    'TYPE_UINT8': torch.uint8,
    'TYPE_FP16': torch.float16,
    'TYPE_FP32': torch.float32,
    'TYPE_FP64': torch.float64
}

torch_type_to_triton_type = {
    torch.bool: 'TYPE_BOOL',
    torch.int8: 'TYPE_INT8',
    torch.int16: 'TYPE_INT16',
    torch.int32: 'TYPE_INT32',
    torch.int64: 'TYPE_INT64',
    torch.uint8: 'TYPE_UINT8',
    torch.float16: 'TYPE_FP16',
    torch.float32: 'TYPE_FP32',
    torch.float64: 'TYPE_FP64'
}


def build_tensorrt_engine(model_file, shapes, max_workspace_size,
                          max_batch_size, fp16_mode):
    ''' takes a path to an onnx file, and shape information, returns a tensorrt engine
        :: model_file :: path to an onnx model
        :: shapes :: dictionary containing min shape, max shape, opt shape for the tensorrt engine
    '''
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = max_batch_size
    #
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if fp16_mode:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for s in shapes:
        profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
    config.add_optimization_profile(profile)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    #
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
            for i in range(parser.num_errors):
                print("[Converter error]: OnnxParser:", parser.get_error(i))
            engine = builder.build_engine(network, config=config)
    return engine


def get_inputs(dataloader, device, precision):
    ''' load sample inputs to device '''
    inputs = []
    logging.info("Loading sample inputs to device.")
    for idx, batch in enumerate(dataloader):
        if idx % (len(dataloader)//100) == 0:
            logging.info(f"{idx}/{len(dataloader)}")

        if type(batch) is torch.Tensor:
            batch_d = batch.to(device)
            if batch_d.is_floating_point() and precision == 'fp16':
                batch_d = batch_d.to(torch.float16)
            batch_d = (batch_d,)
            inputs.append(batch_d)
        else:
            batch_d = []
            for x in batch:
                assert type(x) is torch.Tensor, "input is not a tensor"
                x = x.to(device)
                if x.is_floating_point() and precision == 'fp16':
                    x = x.to(torch.float16)
                batch_d.append(x)
            batch_d = tuple(batch_d)
            inputs.append(batch_d)
    logging.info("Finished loading sample inputs to device.")
    return inputs


def get_list_of_shapes(l, fun):
    ''' returns the list of min/max shapes, depending on fun
        :: l :: list of tuples of tensors
        :: fun :: min or max
    '''
    tensor_tuple = l[0]
    shapes = [list(x.shape) for x in tensor_tuple]
    for tensor_tuple in l:
        assert len(tensor_tuple) == len(shapes), "tensors with varying shape lengths are not supported"
        for i,x in enumerate(tensor_tuple):
            for j in range(len(x.shape)):
                shapes[i][j] = fun(shapes[i][j], x.shape[j])
    return shapes # a list of shapes


def get_min_shapes(l):
    ''' returns the tuple of min shapes
        :: l :: list of tuples of tensors '''
    shapes = get_list_of_shapes(l, min)
    min_batch = 1
    shapes = [[min_batch,*shape[1:]] for shape in shapes]
    shapes = tuple(shapes)
    return shapes # tuple of min shapes


def get_max_shapes(l):
    ''' returns the tuple of max shapes
        :: l :: list of tuples of tensors '''
    shapes = get_list_of_shapes(l, max)
    max_batch = max(1,shapes[0][0])
    shapes = [[max_batch,*shape[1:]] for shape in shapes]
    shapes = tuple(shapes)
    return shapes # tuple of max shapes


def get_opt_shapes(l):
    ''' returns the tuple of opt shapes
        :: l :: list of tuples of tensors '''
    counter = Counter()
    for tensor_tuple in l:
        shapes = [tuple(x.shape) for x in tensor_tuple]
        shapes = tuple(shapes)
        counter[shapes] += 1
    shapes = counter.most_common(1)[0][0]
    return shapes # tuple of most common occuring shapes


def get_shapes(l, max_batch_size):
    ''' returns a tuple of dynamic shapes: variable tensor dimensions
        (for ex. batch size) occur as -1 in the tuple
        :: l :: list of tuples of tensors '''
    tensor_tuple = l[0]
    shapes = [list(x.shape) for x in tensor_tuple]
    for tensor_tuple in l:
        err_msg = "tensors with varying shape lengths are not supported"
        assert len(tensor_tuple) == len(shapes), err_msg
        for i,x in enumerate(tensor_tuple):
            for j in range(len(x.shape)):
                if shapes[i][j] != x.shape[j] or j == 0 and max_batch_size > 1:
                    shapes[i][j] = -1
    shapes = tuple(shapes)
    return shapes # tuple of dynamic shapes


def get_io_properties(inputs, outputs, max_batch_size):

    # generate input shapes - dynamic tensor shape support
    input_shapes = get_shapes(inputs, max_batch_size)

    # generate output shapes - dynamic tensor shape support
    output_shapes = get_shapes(outputs, max_batch_size)

    # generate input types
    input_types = [torch_type_to_triton_type[x.dtype] for x in inputs[0]]

    # generate output types
    output_types = [torch_type_to_triton_type[x.dtype] for x in outputs[0]]

    # get input names
    rng = range(len(input_types))
    input_names = ["input__" + str(num) for num in rng]

    # get output names
    rng = range(len(output_types))
    output_names = ["output__" + str(num) for num in rng]

    # get indices of dynamic input and output shapes
    dynamic_axes = {}
    for input_name,input_shape in zip(input_names,input_shapes):
        dynamic_axes[input_name] = [i for i,x in enumerate(input_shape) if x == -1]
    for output_name,output_shape in zip(output_names,output_shapes):
        dynamic_axes[output_name] = [i for i,x in enumerate(output_shape) if x == -1]

    # min, opt, max shapes for TensorRT
    min_shapes = get_min_shapes(inputs)
    opt_shapes = get_opt_shapes(inputs)
    max_shapes = get_max_shapes(inputs)

    res = {"input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "input_types": input_types,
            "output_types": output_types,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "min_shapes": min_shapes,
            "opt_shapes": opt_shapes,
            "max_shapes": max_shapes}

    return res


def extract_io_props(model, dataloader, device, precision, max_batch_size):

    # prepare inputs
    inputs = get_inputs(dataloader, device, precision)
    # generate outputs
    outputs = []
    for input in inputs:
        with torch.no_grad():
            output = model(*input)
        if type(output) is torch.Tensor:
            output = [output]
        outputs.append(output)

    # prepare input/output properties
    io_props = get_io_properties(inputs, outputs, max_batch_size)

    return io_props

def save_io_props(io_props, io_props_path):

    with open(io_props_path, "w") as f:
        f.write(json.dumps(io_props))


def load_io_props(io_props_path):

    with open(io_props_path, "r") as f:
        data = json.loads(f.read())
    if "dynamic_axes" not in data.keys():
        return data

    return data
