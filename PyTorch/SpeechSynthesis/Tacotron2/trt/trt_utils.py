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


def is_dimension_dynamic(dim):
    return dim is None or dim <= 0


def is_shape_dynamic(shape):
    return any([is_dimension_dynamic(dim) for dim in shape])


def run_trt_engine(context, engine, tensors):

    bindings = [None]*engine.num_bindings
    for name,tensor in tensors['inputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()
        if engine.is_shape_binding(idx) and is_shape_dynamic(context.get_shape(idx)):
            context.set_shape_input(idx, tensor)
        elif is_shape_dynamic(engine.get_binding_shape(idx)):
            context.set_binding_shape(idx, tensor.shape)

    for name,tensor in tensors['outputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()

    context.execute_v2(bindings=bindings)


def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def engine_info(engine_filepath):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_filepath, TRT_LOGGER)

    binding_template = r"""
{btype} {{
  name: "{bname}"
  data_type: {dtype}
  dims: {dims}
}}"""
    type_mapping = {"DataType.HALF": "TYPE_FP16",
                    "DataType.FLOAT": "TYPE_FP32",
                    "DataType.INT32": "TYPE_INT32",
                    "DataType.BOOL" : "TYPE_BOOL"}

    print("engine name", engine.name)
    print("has_implicit_batch_dimension", engine.has_implicit_batch_dimension)
    start_dim = 0 if engine.has_implicit_batch_dimension else 1
    print("num_optimization_profiles", engine.num_optimization_profiles)
    print("max_batch_size:", engine.max_batch_size)
    print("device_memory_size:", engine.device_memory_size)
    print("max_workspace_size:", engine.max_workspace_size)
    print("num_layers:", engine.num_layers)

    for i in range(engine.num_bindings):
        btype = "input" if engine.binding_is_input(i) else "output"
        bname = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        bdims = engine.get_binding_shape(i)
        config_values = {
            "btype": btype,
            "bname": bname,
            "dtype": type_mapping[str(dtype)],
            "dims": list(bdims[start_dim:])
        }
        final_binding_str = binding_template.format_map(config_values)
        print(final_binding_str)


def build_engine(model_file, shapes, max_ws=512*1024*1024, fp16=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16

    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for s in shapes:
        profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
    config.add_optimization_profile(profile)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            for i in range(parser.num_errors):
                print("TensorRT ONNX parser error:", parser.get_error(i))
            engine = builder.build_engine(network, config=config)

            return engine
