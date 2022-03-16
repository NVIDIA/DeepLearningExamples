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

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

# pytype: disable=import-error
import onnx
import onnx.optimizer
import onnx.shape_inference
import onnxruntime
from google.protobuf import text_format
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

# pytype: enable=import-error

from ..core import BaseLoader, BaseRunner, BaseRunnerSession, BaseSaver, Format, Model, Precision, TensorSpec
from ..extensions import loaders, runners, savers
from .utils import infer_precision

LOGGER = logging.getLogger(__name__)


def _value_info2tensor_spec(value_info: onnx.ValueInfoProto):
    onnx_data_type_map = {"float": "float32", "double": "float64"}

    elem_type_name = onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type).lower()
    dtype = onnx_data_type_map.get(elem_type_name, elem_type_name)

    def _get_dim(dim):
        which = dim.WhichOneof("value")
        if which is not None:  # which is None when dim is None
            dim = getattr(dim, which)
        return None if isinstance(dim, (str, bytes)) else dim

    shape = value_info.type.tensor_type.shape
    shape = tuple([_get_dim(d) for d in shape.dim])
    return TensorSpec(value_info.name, dtype=dtype, shape=shape)


def _infer_graph_precision(onnx_graph: onnx.GraphProto) -> Optional[Precision]:
    import networkx as nx

    # build directed graph
    nx_graph = nx.DiGraph()

    def _get_dtype(vi):
        t = vi.type
        if hasattr(t, "tensor_type"):
            type_id = t.tensor_type.elem_type
        else:
            raise NotImplementedError("Not implemented yet")
        return TENSOR_TYPE_TO_NP_TYPE[type_id]

    node_output2type = {vi.name: _get_dtype(vi) for vi in onnx_graph.value_info}

    node_outputs2node = {output_name: node for node in onnx_graph.node for output_name in node.output}
    node_inputs2node = {input_name: node for node in onnx_graph.node for input_name in node.input}

    for node in onnx_graph.node:
        node_dtype = node_output2type.get("+".join(node.output), None)
        nx_graph.add_node(
            node.name,
            op=node.op_type,
            attr={a.name: a for a in node.attribute},
            dtype=node_dtype,
        )
        for input_name in node.input:
            prev_node = node_outputs2node.get(input_name, None)
            if prev_node:
                nx_graph.add_edge(prev_node.name, node.name)

    for input_node in onnx_graph.input:
        input_name = input_node.name
        nx_graph.add_node(input_name, op="input", dtype=_get_dtype(input_node))
        next_node = node_inputs2node.get(input_name, None)
        if next_node:
            nx_graph.add_edge(input_name, next_node.name)

    for output in onnx_graph.output:
        output_name = output.name
        nx_graph.add_node(output_name, op="output", dtype=_get_dtype(output))
        prev_node = node_outputs2node.get(output_name, None)
        if prev_node:
            nx_graph.add_edge(prev_node.name, output_name)
        else:
            LOGGER.warning(f"Could not find previous node for {output_name}")

    input_names = [n.name for n in onnx_graph.input]
    output_names = [n.name for n in onnx_graph.output]
    most_common_dtype = infer_precision(nx_graph, input_names, output_names, lambda node: node.get("dtype", None))
    if most_common_dtype is not None:
        precision = {np.dtype("float32"): Precision.FP32, np.dtype("float16"): Precision.FP16}[most_common_dtype]
    else:
        precision = None
    return precision


class OnnxLoader(BaseLoader):
    def load(self, model_path: Union[str, Path], **_) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()

        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        onnx.helper.strip_doc_string(model)
        model = onnx.shape_inference.infer_shapes(model)

        # TODO: probably modification of onnx model ios causes error on optimize
        # from onnx.utils import polish_model
        # model = polish_model(model)  # run checker, docs strip, optimizer and shape inference

        inputs = {vi.name: _value_info2tensor_spec(vi) for vi in model.graph.input}
        outputs = {vi.name: _value_info2tensor_spec(vi) for vi in model.graph.output}

        precision = _infer_graph_precision(model.graph)

        return Model(model, precision, inputs, outputs)


class OnnxSaver(BaseSaver):
    def __init__(self, as_text: bool = False):
        self._as_text = as_text

    def save(self, model: Model, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        LOGGER.debug(f"Saving ONNX model to {model_path.as_posix()}")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        onnx_model: onnx.ModelProto = model.handle
        if self._as_text:
            with model_path.open("w") as f:
                f.write(text_format.MessageToString(onnx_model))
        else:
            with model_path.open("wb") as f:
                f.write(onnx_model.SerializeToString())


"""
ExecutionProviders on onnxruntime 1.4.0
['TensorrtExecutionProvider',
 'CUDAExecutionProvider',
 'MIGraphXExecutionProvider',
 'NGRAPHExecutionProvider',
 'OpenVINOExecutionProvider',
 'DnnlExecutionProvider',
 'NupharExecutionProvider',
 'VitisAIExecutionProvider',
 'ArmNNExecutionProvider',
 'ACLExecutionProvider',
 'CPUExecutionProvider']
"""


def _check_providers(providers):
    providers = providers or []
    if not isinstance(providers, (list, tuple)):
        providers = [providers]
    available_providers = onnxruntime.get_available_providers()
    unavailable = set(providers) - set(available_providers)
    if unavailable:
        raise RuntimeError(f"Unavailable providers {unavailable}")
    return providers


class OnnxRunner(BaseRunner):
    def __init__(self, verbose_runtime_logs: bool = False):
        self._providers = None
        self._verbose_runtime_logs = verbose_runtime_logs

    def init_inference(self, model: Model):
        assert isinstance(model.handle, onnx.ModelProto)
        return OnnxRunnerSession(
            model=model, providers=self._providers, verbose_runtime_logs=self._verbose_runtime_logs
        )


class OnnxRunnerSession(BaseRunnerSession):
    def __init__(self, model: Model, providers, verbose_runtime_logs: bool = False):
        super().__init__(model)
        self._input_names = None
        self._output_names = None
        self._session = None
        self._providers = providers
        self._verbose_runtime_logs = verbose_runtime_logs
        self._old_env_values = {}

    def __enter__(self):
        self._old_env_values = self._set_env_variables()
        sess_options = onnxruntime.SessionOptions()  # default session options
        if self._verbose_runtime_logs:
            sess_options.log_severity_level = 0
            sess_options.log_verbosity_level = 1
        LOGGER.info(
            f"Starting inference session for onnx model providers={self._providers} sess_options={sess_options}"
        )

        self._input_names = list(self._model.inputs)
        self._output_names = list(self._model.outputs)

        model_payload = self._model.handle.SerializeToString()
        self._session = onnxruntime.InferenceSession(
            model_payload, providers=self._providers, sess_options=sess_options
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._input_names = None
        self._output_names = None
        self._session = None
        self._recover_env_variables(self._old_env_values)

    def __call__(self, x: Dict[str, object]):
        feed_dict = {k: x[k] for k in self._input_names}
        y_pred = self._session.run(self._output_names, feed_dict)
        y_pred = dict(zip(self._output_names, y_pred))

        return y_pred


loaders.register_extension(Format.ONNX.value, OnnxLoader)
runners.register_extension(Format.ONNX.value, OnnxRunner)
savers.register_extension(Format.ONNX.value, OnnxSaver)
