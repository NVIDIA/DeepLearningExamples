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
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional, Union

import torch  # pytype: disable=import-error
import yaml

from ..core import (
    GET_MODEL_FN_NAME,
    BaseConverter,
    BaseLoader,
    BaseRunner,
    BaseRunnerSession,
    BaseSaver,
    Format,
    Model,
    Precision,
    TensorSpec,
    load_from_file,
)
from ..extensions import converters, loaders, runners, savers
from .utils import get_dynamic_axes, get_input_shapes, get_shapes_with_dynamic_axes

LOGGER = logging.getLogger(__name__)


class InputOutputSpec(NamedTuple):
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]


def get_sample_input(dataloader, device):
    for batch in dataloader:
        _, x, _ = batch
        break
    if isinstance(x, dict):
        sample_input = list(x.values())
    elif isinstance(x, list):
        sample_input = x
    else:
        raise TypeError("The first element (x) of batch returned by dataloader must be a list or a dict")

    for idx, s in enumerate(sample_input):
        sample_input[idx] = torch.from_numpy(s).to(device)

    return tuple(sample_input)


def get_model_device(torch_model):
    if next(torch_model.parameters()).is_cuda:
        return "cuda"
    else:
        return "cpu"


def infer_model_precision(model):
    counter = Counter()
    for param in model.parameters():
        counter[param.dtype] += 1
    if counter[torch.float16] > 0:
        return Precision.FP16
    else:
        return Precision.FP32


def _get_tensor_dtypes(dataloader, precision):
    def _get_dtypes(t):
        dtypes = {}
        for k, v in t.items():
            dtype = str(v.dtype)
            if dtype == "float64":
                dtype = "float32"
            if precision == Precision.FP16 and dtype == "float32":
                dtype = "float16"
            dtypes[k] = dtype
        return dtypes

    input_dtypes = {}
    output_dtypes = {}

    for batch in dataloader:
        _, x, y = batch
        input_dtypes = _get_dtypes(x)
        output_dtypes = _get_dtypes(y)
        break

    return input_dtypes, output_dtypes


### TODO assumption: floating point input
### type has same precision as the model
def _get_io_spec(model, dataloader_fn):
    precision = model.precision

    dataloader = dataloader_fn()
    input_dtypes, output_dtypes = _get_tensor_dtypes(dataloader, precision)
    input_shapes, output_shapes = get_shapes_with_dynamic_axes(dataloader)

    inputs = {
        name: TensorSpec(name=name, dtype=input_dtypes[name], shape=tuple(input_shapes[name])) for name in model.inputs
    }
    outputs = {
        name: TensorSpec(name=name, dtype=output_dtypes[name], shape=tuple(output_shapes[name]))
        for name in model.outputs
    }

    return InputOutputSpec(inputs, outputs)


class PyTorchModelLoader(BaseLoader):
    required_fn_name_for_signature_parsing: Optional[str] = GET_MODEL_FN_NAME

    def __init__(self, **kwargs):
        self._model_args = kwargs

    def load(self, model_path: Union[str, Path], **_) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()
        get_model = load_from_file(model_path, "model", GET_MODEL_FN_NAME)
        model, tensor_infos = get_model(**self._model_args)
        io_spec = InputOutputSpec(tensor_infos["inputs"], tensor_infos["outputs"])
        precision = infer_model_precision(model)
        return Model(handle=model, precision=precision, inputs=io_spec.inputs, outputs=io_spec.outputs)


class TorchScriptLoader(BaseLoader):
    def __init__(self, tensor_names_path: str = None, **kwargs):
        self._model_args = kwargs
        self._io_spec = None
        if tensor_names_path is not None:
            with Path(tensor_names_path).open("r") as fh:
                tensor_infos = yaml.load(fh, Loader=yaml.SafeLoader)
                self._io_spec = InputOutputSpec(tensor_infos["inputs"], tensor_infos["outputs"])

    def load(self, model_path: Union[str, Path], **_) -> Model:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        model = torch.jit.load(model_path.as_posix())
        precision = infer_model_precision(model)

        io_spec = self._io_spec
        if not io_spec:
            yaml_path = model_path.parent / f"{model_path.stem}.yaml"
            if not yaml_path.is_file():
                raise ValueError(
                    f"If `--tensor-names-path is not provided, "
                    f"TorchScript model loader expects file {yaml_path} with tensor information."
                )
            with yaml_path.open("r") as fh:
                tensor_info = yaml.load(fh, Loader=yaml.SafeLoader)
                io_spec = InputOutputSpec(tensor_info["inputs"], tensor_info["outputs"])

        return Model(handle=model, precision=precision, inputs=io_spec.inputs, outputs=io_spec.outputs)


class TorchScriptTraceConverter(BaseConverter):
    def __init__(self):
        pass

    def convert(self, model: Model, dataloader_fn) -> Model:
        device = get_model_device(model.handle)
        dummy_input = get_sample_input(dataloader_fn(), device)
        converted_model = torch.jit.trace_module(model.handle, {"forward": dummy_input})
        io_spec = _get_io_spec(model, dataloader_fn)
        return Model(converted_model, precision=model.precision, inputs=io_spec.inputs, outputs=io_spec.outputs)


class TorchScriptScriptConverter(BaseConverter):
    def __init__(self):
        pass

    def convert(self, model: Model, dataloader_fn) -> Model:
        converted_model = torch.jit.script(model.handle)
        io_spec = _get_io_spec(model, dataloader_fn)
        return Model(converted_model, precision=model.precision, inputs=io_spec.inputs, outputs=io_spec.outputs)


class PYT2ONNXConverter(BaseConverter):
    def __init__(self, onnx_opset: int = None):
        self._onnx_opset = onnx_opset

    def convert(self, model: Model, dataloader_fn) -> Model:
        import tempfile

        import onnx  # pytype: disable=import-error

        assert isinstance(model.handle, torch.jit.ScriptModule) or isinstance(
            model.handle, torch.nn.Module
        ), "The model must be of type 'torch.jit.ScriptModule' or 'torch.nn.Module'. Converter aborted."

        dynamic_axes = get_dynamic_axes(dataloader_fn())

        device = get_model_device(model.handle)
        dummy_input = get_sample_input(dataloader_fn(), device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            export_path = os.path.join(tmpdirname, "model.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    model.handle,
                    dummy_input,
                    export_path,
                    do_constant_folding=True,
                    input_names=list(model.inputs),
                    output_names=list(model.outputs),
                    dynamic_axes=dynamic_axes,
                    opset_version=self._onnx_opset,
                    enable_onnx_checker=True,
                )

            onnx_model = onnx.load(export_path)
            onnx.checker.check_model(onnx_model)
            onnx.helper.strip_doc_string(onnx_model)
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

        return Model(
            handle=onnx_model,
            precision=model.precision,
            inputs=model.inputs,
            outputs=model.outputs,
        )


class PYT2TensorRTConverter(BaseConverter):
    def __init__(self, max_batch_size: int, max_workspace_size: int, onnx_opset: int, precision: str):
        self._max_batch_size = max_batch_size
        self._max_workspace_size = max_workspace_size
        self._onnx_opset = onnx_opset
        self._precision = Precision(precision)

    def convert(self, model: Model, dataloader_fn) -> Model:
        from .onnx import _infer_graph_precision
        from .onnx2trt_conv import onnx2trt

        pyt2onnx_converter = PYT2ONNXConverter(self._onnx_opset)
        onnx_model = pyt2onnx_converter.convert(model, dataloader_fn).handle
        precision = _infer_graph_precision(onnx_model.graph)

        input_shapes = get_input_shapes(dataloader_fn(), self._max_batch_size)

        cuda_engine = onnx2trt(
            onnx_model,
            shapes=input_shapes,
            max_workspace_size=self._max_workspace_size,
            max_batch_size=self._max_batch_size,
            model_precision=self._precision.value,
        )

        return Model(
            handle=cuda_engine,
            precision=model.precision,
            inputs=model.inputs,
            outputs=model.outputs,
        )

    @staticmethod
    def required_source_model_precision(requested_model_precision: Precision) -> Precision:
        # TensorRT requires source models to be in FP32 precision
        return Precision.FP32


class TorchScriptSaver(BaseSaver):
    def save(self, model: Model, model_path: Union[str, Path]) -> None:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        if isinstance(model.handle, torch.jit.ScriptModule):
            torch.jit.save(model.handle, model_path.as_posix())
        else:
            print("The model must be of type 'torch.jit.ScriptModule'. Saving aborted.")
            assert False  # temporary error handling

        def _format_tensor_spec(tensor_spec):
            # wrapping shape with list and whole tensor_spec with dict() is required for correct yaml dump
            tensor_spec = tensor_spec._replace(shape=list(tensor_spec.shape))
            tensor_spec = dict(tensor_spec._asdict())
            return tensor_spec

        # store TensorSpecs from inputs and outputs in a yaml file
        tensor_specs = {
            "inputs": {k: _format_tensor_spec(v) for k, v in model.inputs.items()},
            "outputs": {k: _format_tensor_spec(v) for k, v in model.outputs.items()},
        }

        yaml_path = model_path.parent / f"{model_path.stem}.yaml"
        with Path(yaml_path).open("w") as fh:
            yaml.dump(tensor_specs, fh, indent=4)


class PyTorchRunner(BaseRunner):
    def __init__(self):
        pass

    def init_inference(self, model: Model):
        return PyTorchRunnerSession(model=model)


class PyTorchRunnerSession(BaseRunnerSession):
    def __init__(self, model: Model):
        super().__init__(model)

        assert isinstance(model.handle, torch.jit.ScriptModule) or isinstance(
            model.handle, torch.nn.Module
        ), "The model must be of type 'torch.jit.ScriptModule' or 'torch.nn.Module'. Runner aborted."

        self._model = model
        self._output_names = None

    def __enter__(self):
        self._output_names = list(self._model.outputs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._output_names = None
        self._model = None

    def __call__(self, x: Dict[str, object]):
        with torch.no_grad():
            feed_list = [torch.from_numpy(v).cuda() for k, v in x.items()]
            y_pred = self._model.handle(*feed_list)
            if isinstance(y_pred, torch.Tensor):
                y_pred = (y_pred,)
            y_pred = [t.cpu().numpy() for t in y_pred]
            y_pred = dict(zip(self._output_names, y_pred))

        return y_pred


loaders.register_extension(Format.PYT.value, PyTorchModelLoader)
loaders.register_extension(Format.TS_TRACE.value, TorchScriptLoader)
loaders.register_extension(Format.TS_SCRIPT.value, TorchScriptLoader)

converters.register_extension(f"{Format.PYT.value}--{Format.TS_SCRIPT.value}", TorchScriptScriptConverter)
converters.register_extension(f"{Format.PYT.value}--{Format.TS_TRACE.value}", TorchScriptTraceConverter)
converters.register_extension(f"{Format.PYT.value}--{Format.ONNX.value}", PYT2ONNXConverter)
converters.register_extension(f"{Format.PYT.value}--{Format.TRT.value}", PYT2TensorRTConverter)

savers.register_extension(Format.TS_SCRIPT.value, TorchScriptSaver)
savers.register_extension(Format.TS_TRACE.value, TorchScriptSaver)

runners.register_extension(Format.PYT.value, PyTorchRunner)
runners.register_extension(Format.TS_SCRIPT.value, PyTorchRunner)
runners.register_extension(Format.TS_TRACE.value, PyTorchRunner)
