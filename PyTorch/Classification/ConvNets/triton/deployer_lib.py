#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import sys
import time
import json
import torch
import argparse
import statistics
from collections import Counter

torch_type_to_triton_type = {
    torch.bool: "TYPE_BOOL",
    torch.int8: "TYPE_INT8",
    torch.int16: "TYPE_INT16",
    torch.int32: "TYPE_INT32",
    torch.int64: "TYPE_INT64",
    torch.uint8: "TYPE_UINT8",
    torch.float16: "TYPE_FP16",
    torch.float32: "TYPE_FP32",
    torch.float64: "TYPE_FP64",
}

CONFIG_TEMPLATE = r"""
name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}
input [
    {spec_inputs}
]
output [
    {spec_outputs}
]
{dynamic_batching}
{model_optimizations}
instance_group [
    {{
        count: {engine_count}
        kind: KIND_GPU
        gpus: [ {gpu_list} ]
    }}
]"""

INPUT_TEMPLATE = r"""
{{
    name: "input__{num}"
    data_type: {type}
    dims: {dims}
    {reshape}
}},"""

OUTPUT_TEMPLATE = r""" 
{{
    name: "output__{num}"
    data_type: {type}
    dims: {dims}
    {reshape}
}},"""

MODEL_OPTIMIZATION_TEMPLATE = r"""
optimization {{
  {execution_accelerator}
  cuda {{
    graphs: {capture_cuda_graph}
  }}
}}"""

EXECUTION_ACCELERATOR_TEMPLATE = r"""
  execution_accelerators {{
    gpu_execution_accelerator: [
      {{
        name: "tensorrt"
      }}
    ]
  }},"""


def remove_empty_lines(text):
    """ removes empty lines from text, returns the result """
    ret = "".join([s for s in text.strip().splitlines(True) if s.strip()])
    return ret


def create_deployer(argv):
    """ takes a list of arguments, returns a deployer object and the list of unused arguments """
    parser = argparse.ArgumentParser()
    # required args
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument(
        "--ts-script",
        action="store_true",
        help="convert to torchscript using torch.jit.script",
    )
    method.add_argument(
        "--ts-trace",
        action="store_true",
        help="convert to torchscript using torch.jit.trace",
    )
    method.add_argument(
        "--onnx", action="store_true", help="convert to onnx using torch.onnx.export"
    )
    method.add_argument(
        "--trt", action="store_true", help="convert to trt using tensorrt"
    )
    # triton related args
    arguments = parser.add_argument_group("triton related flags")
    arguments.add_argument(
        "--triton-no-cuda", action="store_true", help="Use the CPU for tracing."
    )
    arguments.add_argument(
        "--triton-model-name",
        type=str,
        default="model",
        help="exports to appropriate directory structure for TRITON",
    )
    arguments.add_argument(
        "--triton-model-version",
        type=int,
        default=1,
        help="exports to appropriate directory structure for TRITON",
    )
    arguments.add_argument(
        "--triton-max-batch-size",
        type=int,
        default=8,
        help="Specifies the 'max_batch_size' in the TRITON model config.\
                                  See the TRITON documentation for more info.",
    )
    arguments.add_argument(
        "--triton-dyn-batching-delay",
        type=float,
        default=0,
        help="Determines the dynamic_batching queue delay in milliseconds(ms) for\
                                  the TRITON model config. Use '0' or '-1' to specify static batching.\
                                  See the TRITON documentation for more info.",
    )
    arguments.add_argument(
        "--triton-engine-count",
        type=int,
        default=1,
        help="Specifies the 'instance_group' count value in the TRITON model config.\
                                  See the TRITON documentation for more info.",
    )
    arguments.add_argument(
        "--save-dir", type=str, default="./triton_models", help="Saved model directory"
    )
    # optimization args
    arguments = parser.add_argument_group("optimization flags")
    arguments.add_argument(
        "--max_workspace_size",
        type=int,
        default=512 * 1024 * 1024,
        help="set the size of the workspace for trt export",
    )
    arguments.add_argument(
        "--trt-fp16",
        action="store_true",
        help="trt flag ---- export model in mixed precision mode",
    )
    arguments.add_argument(
        "--capture-cuda-graph",
        type=int,
        default=1,
        help="capture cuda graph for obtaining speedup. possible values: 0, 1. default: 1. ",
    )

    # remainder args
    arguments.add_argument(
        "model_arguments",
        nargs=argparse.REMAINDER,
        help="arguments that will be ignored by deployer lib and will be forwarded to your deployer script",
    )
    #
    args = parser.parse_args(argv)
    deployer = Deployer(args)
    #
    return deployer, args.model_arguments[1:]


class DeployerLibrary:
    def __init__(self, args):
        self.args = args
        self.platform = None

    def set_platform(self, platform):
        """ sets the platform
            :: platform :: "pytorch_libtorch" or "onnxruntime_onnx" or "tensorrt_plan"
        """
        self.platform = platform

    def build_trt_engine(self, model_file, shapes):
        """ takes a path to an onnx file, and shape information, returns a trt engine
            :: model_file :: path to an onnx model
            :: shapes :: dictionary containing min shape, max shape, opt shape for the trt engine
        """
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        builder.fp16_mode = self.args.trt_fp16
        builder.max_batch_size = self.args.triton_max_batch_size
        #
        config = builder.create_builder_config()
        config.max_workspace_size = self.args.max_workspace_size
        if self.args.trt_fp16:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        for s in shapes:
            profile.set_shape(s["name"], min=s["min"], opt=s["opt"], max=s["max"])
        config.add_optimization_profile(profile)
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        #
        with trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(model_file, "rb") as model:
                parser.parse(model.read())
                for i in range(parser.num_errors):
                    e = parser.get_error(i)
                    print("||||e", e)
                engine = builder.build_engine(network, config=config)
        return engine

    def load_engine(self, engine_filepath):
        """ loads a trt engine from engine_filepath, returns it """
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_filepath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def prepare_inputs(self, dataloader, device):
        """ load sample inputs to device """
        inputs = []
        for batch in dataloader:
            if type(batch) is torch.Tensor:
                batch_d = batch.to(device)
                batch_d = (batch_d,)
                inputs.append(batch_d)
            else:
                batch_d = []
                for x in batch:
                    assert type(x) is torch.Tensor, "input is not a tensor"
                    batch_d.append(x.to(device))
                batch_d = tuple(batch_d)
                inputs.append(batch_d)
        return inputs

    def get_list_of_shapes(self, l, fun):
        """ returns the list of min/max shapes, depending on fun
            :: l :: list of tuples of tensors
            :: fun :: min or max
        """
        tensor_tuple = l[0]
        shapes = [list(x.shape) for x in tensor_tuple]
        for tensor_tuple in l:
            assert len(tensor_tuple) == len(
                shapes
            ), "tensors with varying shape lengths are not supported"
            for i, x in enumerate(tensor_tuple):
                for j in range(len(x.shape)):
                    shapes[i][j] = fun(shapes[i][j], x.shape[j])
        return shapes  # a list of shapes

    def get_tuple_of_min_shapes(self, l):
        """ returns the tuple of min shapes 
            :: l :: list of tuples of tensors """
        shapes = self.get_list_of_shapes(l, min)
        min_batch = 1
        shapes = [[min_batch, *shape[1:]] for shape in shapes]
        shapes = tuple(shapes)
        return shapes  # tuple of min shapes

    def get_tuple_of_max_shapes(self, l):
        """ returns the tuple of max shapes 
            :: l :: list of tuples of tensors """
        shapes = self.get_list_of_shapes(l, max)
        max_batch = max(2, shapes[0][0])
        shapes = [[max_batch, *shape[1:]] for shape in shapes]
        shapes = tuple(shapes)
        return shapes  # tuple of max shapes

    def get_tuple_of_opt_shapes(self, l):
        """ returns the tuple of opt shapes 
            :: l :: list of tuples of tensors """
        counter = Counter()
        for tensor_tuple in l:
            shapes = [tuple(x.shape) for x in tensor_tuple]
            shapes = tuple(shapes)
            counter[shapes] += 1
        shapes = counter.most_common(1)[0][0]
        return shapes  # tuple of most common occuring shapes

    def get_tuple_of_dynamic_shapes(self, l):
        """ returns a tuple of dynamic shapes: variable tensor dimensions 
            (for ex. batch size) occur as -1 in the tuple
            :: l :: list of tuples of tensors """
        tensor_tuple = l[0]
        shapes = [list(x.shape) for x in tensor_tuple]
        for tensor_tuple in l:
            err_msg = "tensors with varying shape lengths are not supported"
            assert len(tensor_tuple) == len(shapes), err_msg
            for i, x in enumerate(tensor_tuple):
                for j in range(len(x.shape)):
                    if shapes[i][j] != x.shape[j] or j == 0:
                        shapes[i][j] = -1
        shapes = tuple(shapes)
        return shapes  # tuple of dynamic shapes

    def run_models(self, models, inputs):
        """ run the models on inputs, return the outputs and execution times """
        ret = []
        for model in models:
            torch.cuda.synchronize()
            time_start = time.time()
            outputs = []
            for input in inputs:
                with torch.no_grad():
                    output = model(*input)
                if type(output) is torch.Tensor:
                    output = [output]
                outputs.append(output)
            torch.cuda.synchronize()
            time_end = time.time()
            t = time_end - time_start
            ret.append(outputs)
            ret.append(t)
        return ret

    def compute_tensor_stats(self, tensor):
        return {
            "std": tensor.std().item(),
            "mean": tensor.mean().item(),
            "max": tensor.max().item(),
            "min": tensor.min().item(),
        }

    def compute_errors(self, outputs_A, outputs_B):
        """ returns dictionary with errors statistics """
        device = outputs_A[0][0][0].device
        dtype = outputs_A[0][0][0].dtype
        x_values = torch.zeros(0, device=device, dtype=dtype)
        y_values = torch.zeros(0, device=device, dtype=dtype)
        d_values = torch.zeros(0, device=device, dtype=dtype)
        for output_A, output_B in zip(outputs_A, outputs_B):
            for x, y in zip(output_A, output_B):
                d = abs(x - y)
                x_values = torch.cat((x_values, x), 0)
                y_values = torch.cat((y_values, y), 0)
                d_values = torch.cat((d_values, d), 0)
        Error_stats = {
            "Original": self.compute_tensor_stats(x_values),
            "Converted": self.compute_tensor_stats(y_values),
            "Absolute difference": self.compute_tensor_stats(d_values),
        }
        return Error_stats

    def print_errors(self, Error_stats):
        """ print various statistcs of Linf errors """
        print()
        print("conversion correctness test results")
        print("-----------------------------------")
        import pandas as pd

        print(pd.DataFrame(Error_stats))

    def write_config(
        self, config_filename, input_shapes, input_types, output_shapes, output_types
    ):
        """ writes TRTIS config file 
            :: config_filename :: the file to write the config file into
            :: input_shapes :: tuple of dynamic shapes of the input tensors
            :: input_types :: tuple of torch types of the input tensors
            :: output_shapes :: tuple of dynamic shapes of the output tensors
            :: output_types :: tuple of torch types of the output tensors
        """
        assert self.platform is not None, "error - platform is not set"

        config_template = CONFIG_TEMPLATE
        input_template = INPUT_TEMPLATE
        optimization_template = MODEL_OPTIMIZATION_TEMPLATE
        accelerator_template = EXECUTION_ACCELERATOR_TEMPLATE

        spec_inputs = r""""""
        for i, (shape, typ) in enumerate(zip(input_shapes, input_types)):
            d = {
                "num": str(i),
                "type": torch_type_to_triton_type[typ],
                "dims": str([1])
                if len(shape) == 1
                else str(list(shape)[1:]),  # first dimension is the batch size
            }
            d["reshape"] = "reshape: { shape: [ ] }" if len(shape) == 1 else ""
            spec_inputs += input_template.format_map(d)
        spec_inputs = spec_inputs[:-1]

        output_template = OUTPUT_TEMPLATE
        spec_outputs = r""""""
        for i, (shape, typ) in enumerate(zip(output_shapes, output_types)):
            d = {
                "num": str(i),
                "type": torch_type_to_triton_type[typ],
                "dims": str([1])
                if len(shape) == 1
                else str(list(shape)[1:]),  # first dimension is the batch size
            }
            d["reshape"] = "reshape: { shape: [ ] }" if len(shape) == 1 else ""
            spec_outputs += output_template.format_map(d)
        spec_outputs = spec_outputs[:-1]

        batching_str = ""
        max_batch_size = self.args.triton_max_batch_size

        if self.args.triton_dyn_batching_delay >= 0:
            # Use only full and half full batches
            pref_batch_size = [int(max_batch_size / 2.0), max_batch_size]

            if self.args.triton_dyn_batching_delay > 0:
                dyn_batch_delay_str = f"max_queue_delay_microseconds: {int(self.args.triton_dyn_batching_delay * 1000.0)}"
            else:
                dyn_batch_delay_str = ""

            batching_str = r"""
dynamic_batching {{
    preferred_batch_size: [{0}]
    {1}
}}""".format(
                ", ".join([str(x) for x in pref_batch_size]), dyn_batch_delay_str
            )

        accelerator_str = ""

        d = {
            "execution_accelerator": accelerator_str,
            "capture_cuda_graph": str(self.args.capture_cuda_graph),
        }
        optimization_str = optimization_template.format_map(d)

        config_values = {
            "model_name": self.args.triton_model_name,
            "platform": self.platform,
            "max_batch_size": max_batch_size,
            "spec_inputs": spec_inputs,
            "spec_outputs": spec_outputs,
            "dynamic_batching": batching_str,
            "model_optimizations": optimization_str,
            "gpu_list": ", ".join([str(x) for x in range(torch.cuda.device_count())]),
            "engine_count": self.args.triton_engine_count,
        }

        # write config
        with open(config_filename, "w") as file:
            final_config_str = config_template.format_map(config_values)
            final_config_str = remove_empty_lines(final_config_str)
            file.write(final_config_str)


class Deployer:
    def __init__(self, args):
        self.args = args
        self.lib = DeployerLibrary(args)

    def deploy(self, dataloader, model):
        """ deploy the model and test for correctness with dataloader """
        if self.args.ts_script or self.args.ts_trace:
            self.lib.set_platform("pytorch_libtorch")
            print(
                "deploying model "
                + self.args.triton_model_name
                + " in format "
                + self.lib.platform
            )
            self.to_triton_torchscript(dataloader, model)
        elif self.args.onnx:
            self.lib.set_platform("onnxruntime_onnx")
            print(
                "deploying model "
                + self.args.triton_model_name
                + " in format "
                + self.lib.platform
            )
            self.to_triton_onnx(dataloader, model)
        elif self.args.trt:
            self.lib.set_platform("tensorrt_plan")
            print(
                "deploying model "
                + self.args.triton_model_name
                + " in format "
                + self.lib.platform
            )
            self.to_triton_trt(dataloader, model)
        else:
            assert False, "error"
        print("done")

    def to_triton_trt(self, dataloader, model):
        """ export the model to trt and test correctness on dataloader """
        import tensorrt as trt

        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate outputs
        outputs = []
        for input in inputs:
            with torch.no_grad():
                output = model(*input)
            if type(output) is torch.Tensor:
                output = [output]
            outputs.append(output)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # get input names
        rng = range(len(input_types))
        input_names = ["input__" + str(num) for num in rng]

        # get output names
        rng = range(len(output_types))
        output_names = ["output__" + str(num) for num in rng]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.plan")

        # get indices of dynamic input and output shapes
        dynamic_axes = {}
        for input_name, shape in zip(input_names, input_shapes):
            dynamic_axes[input_name] = [i for i, x in enumerate(shape) if x == -1]
        for output_name, shape in zip(output_names, output_shapes):
            dynamic_axes[output_name] = [i for i, x in enumerate(shape) if x == -1]

        # export the model to onnx first
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs[0],
                final_model_path,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
            )

        # get shapes
        min_shapes = self.lib.get_tuple_of_min_shapes(inputs)
        opt_shapes = self.lib.get_tuple_of_opt_shapes(inputs)
        max_shapes = self.lib.get_tuple_of_max_shapes(inputs)

        zipped = zip(input_names, min_shapes, opt_shapes, max_shapes)
        shapes = []
        for name, min_shape, opt_shape, max_shape in zipped:
            d = {"name": name, "min": min_shape, "opt": opt_shape, "max": max_shape}
            shapes.append(d)

        # build trt engine
        engine = self.lib.build_trt_engine(final_model_path, shapes)
        assert engine is not None, " trt export failure "

        # write trt engine
        with open(final_model_path, "wb") as f:
            f.write(engine.serialize())

        # load the model
        engine = self.lib.load_engine(final_model_path)

        class TRT_model:
            def __init__(self, engine, input_names, output_names, output_types, device):
                self.engine = engine
                self.context = self.engine.create_execution_context()
                self.input_names = input_names
                self.output_names = output_names
                self.output_types = output_types
                self.device = device

            def is_dimension_dynamic(self, dim):
                return dim is None or dim <= 0

            def is_shape_dynamic(self, shape):
                return any([self.is_dimension_dynamic(dim) for dim in shape])

            def __call__(self, *inputs):
                # get input shapes
                input_shapes = [x.shape for x in inputs]
                # bindings
                bindings = [None] * self.engine.num_bindings
                # set input shapes, bind input tensors
                zipped = zip(self.input_names, inputs)
                for key, input in zipped:
                    idx = self.engine.get_binding_index(key)
                    bindings[idx] = input.data_ptr()
                    if self.engine.is_shape_binding(idx) and self.is_shape_dynamic(
                        self.context.get_shape(idx)
                    ):
                        self.context.set_shape_input(idx, input)
                    elif self.is_shape_dynamic(self.engine.get_binding_shape(idx)):
                        self.context.set_binding_shape(idx, input.shape)
                assert self.context.all_binding_shapes_specified, "trt error"
                assert self.context.all_shape_inputs_specified, "trt error"
                # calculate output shapes, allocate output tensors and bind them
                outputs = []
                zipped = zip(self.output_names, self.output_types)
                for key, dtype in zipped:
                    idx = self.engine.get_binding_index(key)
                    shape = self.context.get_binding_shape(idx)
                    shape = tuple(shape)
                    assert -1 not in shape, "trt error"
                    tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                    outputs.append(tensor)
                    bindings[idx] = outputs[-1].data_ptr()
                # run inference
                self.context.execute_v2(bindings=bindings)
                # return the result
                if len(outputs) == 1:
                    outputs = outputs[0]
                return outputs

        model_trt = TRT_model(engine, input_names, output_names, output_types, device)

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        models = (model, model_trt)
        outputs, time_model, outputs_trt, time_model_trt = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Error_stats = self.lib.compute_errors(outputs, outputs_trt)
        self.lib.print_errors(Error_stats)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of trt model: ", time_model_trt, "seconds")
        print()

        # write TRTIS config
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )

    def name_onnx_nodes(self, model_path):
        """
        Name all unnamed nodes in ONNX model
            parameter model_path: path  ONNX model
            return: none
        """
        model = onnx.load(model_path)
        node_id = 0
        for node in model.graph.node:
            if len(node.name) == 0:
                node.name = "unnamed_node_%d" % node_id
            node_id += 1
        # This check partially validates model
        onnx.checker.check_model(model)
        onnx.save(model, model_path)
        # Only inference really checks ONNX model for some issues
        # like duplicated node names
        onnxruntime.InferenceSession(model_path, None)

    def to_triton_onnx(self, dataloader, model):
        """ export the model to onnx and test correctness on dataloader """
        import onnx as local_onnx

        global onnx
        onnx = local_onnx
        import onnxruntime as local_onnxruntime

        global onnxruntime
        onnxruntime = local_onnxruntime
        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate outputs
        outputs = []
        for input in inputs:
            with torch.no_grad():
                output = model(*input)
            if type(output) is torch.Tensor:
                output = [output]
            outputs.append(output)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # get input names
        rng = range(len(input_types))
        input_names = ["input__" + str(num) for num in rng]

        # get output names
        rng = range(len(output_types))
        output_names = ["output__" + str(num) for num in rng]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.onnx")

        # get indices of dynamic input and output shapes
        dynamic_axes = {}
        for input_name, input_shape in zip(input_names, input_shapes):
            dynamic_axes[input_name] = [i for i, x in enumerate(input_shape) if x == -1]
        for output_name, output_shape in zip(output_names, output_shapes):
            dynamic_axes[output_name] = [
                i for i, x in enumerate(output_shape) if x == -1
            ]

        # export the model
        assert not model.training, "internal error - model should be in eval() mode! "
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs[0],
                final_model_path,
                verbose=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
            )

        # syntactic error check
        converted_model = onnx.load(final_model_path)
        # check that the IR is well formed
        onnx.checker.check_model(converted_model)

        # Name unnamed nodes - it helps for some other processing tools
        self.name_onnx_nodes(final_model_path)
        converted_model = onnx.load(final_model_path)

        # load the model
        session = onnxruntime.InferenceSession(final_model_path, None)

        class ONNX_model:
            def __init__(self, session, input_names, device):
                self.session = session
                self.input_names = input_names

            def to_numpy(self, tensor):
                return (
                    tensor.detach().cpu().numpy()
                    if tensor.requires_grad
                    else tensor.cpu().numpy()
                )

            def __call__(self, *inputs):
                inp = [
                    (input_name, inputs[i])
                    for i, input_name in enumerate(self.input_names)
                ]
                inp = {input_name: self.to_numpy(x) for input_name, x in inp}
                outputs = self.session.run(None, inp)
                outputs = [torch.from_numpy(output) for output in outputs]
                outputs = [output.to(device) for output in outputs]
                if len(outputs) == 1:
                    outputs = outputs[0]
                return outputs

        # switch to eval mode
        model_onnx = ONNX_model(session, input_names, device)

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        models = (model, model_onnx)
        outputs, time_model, outputs_onnx, time_model_onnx = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Error_stats = self.lib.compute_errors(outputs, outputs_onnx)
        self.lib.print_errors(Error_stats)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of onnx model: ", time_model_onnx, "seconds")
        print()

        # write TRTIS config
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )

    def to_triton_torchscript(self, dataloader, model):
        """ export the model to torchscript and test correctness on dataloader """
        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.pt")

        # convert the model
        with torch.no_grad():
            if self.args.ts_trace:  # trace it
                model_ts = torch.jit.trace(model, inputs[0])
            if self.args.ts_script:  # script it
                model_ts = torch.jit.script(model)

        # save the model
        torch.jit.save(model_ts, final_model_path)

        # load the model
        model_ts = torch.jit.load(final_model_path)
        model_ts.eval()  # WAR for bug : by default, model_ts gets loaded in training mode

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        assert (
            not model_ts.training
        ), "internal error - converted model should be in eval() mode! "
        models = (model, model_ts)
        outputs, time_model, outputs_ts, time_model_ts = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Error_stats = self.lib.compute_errors(outputs, outputs_ts)
        self.lib.print_errors(Error_stats)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of ts model: ", time_model_ts, "seconds")
        print()

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # now we build the config for TRTIS
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )
