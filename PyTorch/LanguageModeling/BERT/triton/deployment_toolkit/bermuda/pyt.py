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
import typing
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch  # pytype: disable=import-error
import yaml
from model_navigator.model import ModelSignatureConfig
from model_navigator.tensor import TensorSpec
from model_navigator.utils.config import YamlConfigFile

from ..core import (
    GET_MODEL_FN_NAME,
    BaseLoader,
    BaseRunner,
    BaseRunnerSession,
    BaseSaver,
    Format,
    Model,
    Precision,
    load_from_file,
)
from ..extensions import loaders, runners, savers
from .utils import get_dynamic_axes, get_shapes_with_dynamic_axes

LOGGER = logging.getLogger(__name__)


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
        def _get_dtype(v):
            dtype = str(v.dtype)
            if dtype == "float64":
                dtype = "float32"
            if precision == Precision.FP16 and dtype == "float32":
                dtype = "float16"
            return np.dtype(dtype)

        return {k: _get_dtype(v) for k, v in t.items()}

    batch = next(dataloader)
    _, x, y = batch
    input_dtypes = _get_dtypes(x)
    output_dtypes = _get_dtypes(y)

    return input_dtypes, output_dtypes


### TODO assumption: floating point input
### type has same precision as the model
def _get_model_signature(
    inputs_names: typing.List[str],
    outputs_names: typing.List[str],
    precision,
    dataloader_fn,
    batch_size_dim: typing.Optional[int] = None,
):
    dataloader = dataloader_fn()
    input_dtypes, output_dtypes = _get_tensor_dtypes(dataloader, precision)
    input_shapes, output_shapes = get_shapes_with_dynamic_axes(dataloader, batch_size_dim=batch_size_dim)

    inputs = {
        name: TensorSpec(name=name, dtype=input_dtypes[name], shape=tuple(input_shapes[name])) for name in inputs_names
    }
    outputs = {
        name: TensorSpec(name=name, dtype=output_dtypes[name], shape=tuple(output_shapes[name]))
        for name in outputs_names
    }

    return ModelSignatureConfig(inputs, outputs)


class PyTorchModelLoader(BaseLoader):
    required_fn_name_for_signature_parsing: Optional[str] = GET_MODEL_FN_NAME

    def __init__(self, **kwargs):
        self._model_args = kwargs

    def load(self, model_path: Union[str, Path], **kwargs) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()

        get_model = load_from_file(model_path, "model", GET_MODEL_FN_NAME)
        model, io_names_dict = get_model(**self._model_args)

        dataloader_fn = kwargs.get("dataloader_fn", None)
        output_type = kwargs.get("output_type", None)
        precision = infer_model_precision(model)

        batch_axis = getattr(model, "bermuda_batch_axis", 0)  # by default models supports batching; batch_axis=0

        model_signature = _get_model_signature(
            inputs_names=io_names_dict["inputs"],
            outputs_names=io_names_dict["outputs"],
            precision=precision,
            dataloader_fn=dataloader_fn,
            batch_size_dim=batch_axis,
        )

        model = Model(handle=model, precision=precision, inputs=model_signature.inputs, outputs=model_signature.outputs)

        if output_type == Format.TS_TRACE.value:
            return self._trace(model, dataloader_fn)
        elif output_type == Format.TS_SCRIPT.value:
            return self._script(model)
        elif output_type == Format.ONNX.value:
            return model
        else:
            raise ValueError(f"Not supported PyTorch format: {output_type}")

    def _trace(self, model: Model, dataloader_fn) -> Model:
        device = get_model_device(model.handle)
        dummy_input = get_sample_input(dataloader_fn(), device)
        traced_model = torch.jit.trace_module(model.handle, {"forward": dummy_input})
        return Model(traced_model, precision=model.precision, inputs=model.inputs, outputs=model.outputs)

    def _script(self, model: Model) -> Model:
        scripted_model = torch.jit.script(model.handle)
        return Model(scripted_model, precision=model.precision, inputs=model.inputs, outputs=model.outputs)


class TorchScriptLoader(BaseLoader):
    def __init__(self, tensor_names_path: str = None, **kwargs):
        self._model_args = kwargs
        self._io_spec = None
        if tensor_names_path is not None:
            with Path(tensor_names_path).open("r") as fh:
                tensor_infos = yaml.load(fh, Loader=yaml.SafeLoader)
                self._io_spec = ModelSignatureConfig(tensor_infos["inputs"], tensor_infos["outputs"])

    def load(self, model_path: Union[str, Path], **_) -> Model:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        model = torch.jit.load(model_path.as_posix())
        precision = infer_model_precision(model)

        io_spec = self._io_spec
        if not io_spec:
            yaml_path = model_path.parent / f"{model_path.name}.yaml"
            if not yaml_path.is_file():
                raise ValueError(
                    f"If `--tensor-names-path is not provided, "
                    f"TorchScript model loader expects file {yaml_path} with tensor information."
                )
            with yaml_path.open("r") as fh:
                tensor_info = yaml.load(fh, Loader=yaml.SafeLoader)
                io_spec = ModelSignatureConfig(tensor_info["inputs"], tensor_info["outputs"])

        return Model(handle=model, precision=precision, inputs=io_spec.inputs, outputs=io_spec.outputs)


class PYT2ONNXSaver(BaseSaver):
    def __init__(self, onnx_opset: int = None):
        self._onnx_opset = onnx_opset

    def save(self, model: Model, model_path: Union[str, Path], dataloader_fn) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()
        assert isinstance(model.handle, torch.jit.ScriptModule) or isinstance(
            model.handle, torch.nn.Module
        ), "The model must be of type 'torch.jit.ScriptModule' or 'torch.nn.Module'. Converter aborted."

        batch_axis = getattr(model.handle, "bermuda_batch_axis", 0)  # by default models supports batching; batch_axis=0
        dynamic_axes = get_dynamic_axes(dataloader_fn(), batch_size_dim=batch_axis)

        device = get_model_device(model.handle)
        dummy_input = get_sample_input(dataloader_fn(), device)

        with torch.no_grad():
            torch.onnx.export(
                model.handle,
                dummy_input,
                model_path,
                do_constant_folding=True,
                input_names=list(model.inputs),
                output_names=list(model.outputs),
                dynamic_axes=dynamic_axes,
                opset_version=self._onnx_opset,
                enable_onnx_checker=True,
            )


class TorchScriptSaver(BaseSaver):
    def save(self, model: Model, model_path: Union[str, Path], dataloader_fn) -> None:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        if isinstance(model.handle, torch.jit.ScriptModule):
            torch.jit.save(model.handle, model_path.as_posix())
        else:
            raise RuntimeError("The model must be of type 'torch.jit.ScriptModule'. Saving aborted.")

        signature_config = ModelSignatureConfig(inputs=model.inputs, outputs=model.outputs)
        annotation_path = model_path.parent / f"{model_path.name}.yaml"
        with YamlConfigFile(annotation_path) as config_file:
            config_file.save_config(signature_config)


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

savers.register_extension(Format.TS_SCRIPT.value, TorchScriptSaver)
savers.register_extension(Format.TS_TRACE.value, TorchScriptSaver)
savers.register_extension(f"{Format.PYT.value}--{Format.ONNX.value}", PYT2ONNXSaver)

runners.register_extension(Format.PYT.value, PyTorchRunner)
runners.register_extension(Format.TS_SCRIPT.value, PyTorchRunner)
runners.register_extension(Format.TS_TRACE.value, PyTorchRunner)
