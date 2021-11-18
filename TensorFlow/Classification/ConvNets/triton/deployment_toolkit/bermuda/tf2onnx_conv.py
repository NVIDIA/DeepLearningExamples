# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from collections import Iterable

# pytype: disable=import-error
import onnx
import onnx.shape_inference
import tensorflow as tf
from tf2onnx import optimizer, tfonnx

# pytype: enable=import-error

from ..core import BaseConverter, Format, Model
from ..extensions import converters
from .tf import create_session_config


def _replace_io_names(graph_proto, io_type, name2tensor):
    tensor2name = {v: k for k, v in name2tensor.items()}
    tensor_value_info_list = {"inputs": graph_proto.input, "outputs": graph_proto.output}[io_type]
    for tensor_value_info in tensor_value_info_list:
        old_name = tensor_value_info.name
        new_name = tensor2name.get(old_name)
        if new_name is not None and new_name != old_name:
            tensor_value_info.name = new_name
            # replace other graph nodes I/O
            for node in graph_proto.node:
                if old_name in node.input:
                    idx = list(node.input).index(old_name)
                    node.input[idx] = new_name
                if old_name in node.output:
                    idx = list(node.output).index(old_name)
                    node.output[idx] = new_name


def tfgraph2onnx(graph_def, inputnames2tensornames, outputnames2tensornames, *, onnx_opset, onnx_optimized=True):
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name="")
    session_config = create_session_config(allow_growth=True)
    with tf.compat.v1.Session(graph=tf_graph, config=session_config):
        input_tensor_names = list(inputnames2tensornames.values())
        output_tensor_names = list(outputnames2tensornames.values())
        onnx_graph = tfonnx.process_tf_graph(
            tf_graph,
            input_names=input_tensor_names,
            output_names=output_tensor_names,
            opset=onnx_opset,
        )
    if onnx_optimized:
        onnx_graph = optimizer.optimize_graph(onnx_graph)
    graph_doc: str = "triton export"
    onnx_model = onnx_graph.make_model(graph_doc)

    # to match tensorflow savedmodel signature
    _replace_io_names(onnx_model.graph, "inputs", inputnames2tensornames)
    _replace_io_names(onnx_model.graph, "outputs", outputnames2tensornames)

    onnx.checker.check_model(onnx_model)
    onnx.helper.strip_doc_string(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


class TFGraphDef2ONNXConverter(BaseConverter):
    def __init__(self, *, onnx_opset: int, onnx_optimized: bool = True):
        self._onnx_opset = onnx_opset
        self._onnx_optimized = onnx_optimized

    def convert(self, model: Model, dataloader_fn) -> Model:
        assert isinstance(model.handle, tf.compat.v1.GraphDef)

        inputnames2tensorname = {name: spec.name for name, spec in model.inputs.items()}
        outputnames2tensorname = {name: spec.name for name, spec in model.outputs.items()}
        onnx_model = tfgraph2onnx(
            model.handle,
            inputnames2tensorname,
            outputnames2tensorname,
            onnx_opset=self._onnx_opset,
            onnx_optimized=self._onnx_optimized,
        )
        from .onnx import _infer_graph_precision

        precision = _infer_graph_precision(onnx_model.graph)
        assert precision == model.precision  # for testing precision inference function
        return model._replace(handle=onnx_model)


converters.register_extension(f"{Format.TF_ESTIMATOR.value}--{Format.ONNX.value}", TFGraphDef2ONNXConverter)
converters.register_extension(f"{Format.TF_KERAS.value}--{Format.ONNX.value}", TFGraphDef2ONNXConverter)
converters.register_extension(f"{Format.TF_SAVEDMODEL.value}--{Format.ONNX.value}", TFGraphDef2ONNXConverter)
