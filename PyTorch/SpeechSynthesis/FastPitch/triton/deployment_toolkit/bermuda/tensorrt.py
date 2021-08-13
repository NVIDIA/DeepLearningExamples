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
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import numpy as np

# pytype: disable=import-error
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except (ImportError, Exception) as e:
    logging.getLogger(__name__).warning(f"Problems with importing pycuda package; {e}")
# pytype: enable=import-error

import tensorrt as trt  # pytype: disable=import-error

from ..core import BaseLoader, BaseRunner, BaseRunnerSession, BaseSaver, Format, Model, Precision, TensorSpec
from ..extensions import loaders, runners, savers

LOGGER = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

"""
documentation:
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_samples_section
"""


class TensorRTLoader(BaseLoader):
    def load(self, model_path: Union[str, Path], **_) -> Model:
        model_path = Path(model_path)
        LOGGER.debug(f"Loading TensorRT engine from {model_path}")

        with model_path.open("rb") as fh, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(fh.read())

        if engine is None:
            raise RuntimeError(f"Could not load ICudaEngine from {model_path}")

        inputs = {}
        outputs = {}
        for binding_idx in range(engine.num_bindings):
            name = engine.get_binding_name(binding_idx)
            is_input = engine.binding_is_input(binding_idx)
            dtype = engine.get_binding_dtype(binding_idx)
            shape = engine.get_binding_shape(binding_idx)
            if is_input:
                inputs[name] = TensorSpec(name, dtype, shape)
            else:
                outputs[name] = TensorSpec(name, dtype, shape)

        return Model(engine, None, inputs, outputs)


class TensorRTSaver(BaseSaver):
    def __init__(self):
        pass

    def save(self, model: Model, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        LOGGER.debug(f"Saving TensorRT engine to {model_path.as_posix()}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        engine: "trt.ICudaEngine" = model.handle
        with model_path.open("wb") as fh:
            fh.write(engine.serialize())


class TRTBuffers(NamedTuple):
    x_host: Optional[Dict[str, object]]
    x_dev: Dict[str, object]
    y_pred_host: Dict[str, object]
    y_pred_dev: Dict[str, object]


class TensorRTRunner(BaseRunner):
    def __init__(self):
        pass

    def init_inference(self, model: Model):
        return TensorRTRunnerSession(model=model)


class TensorRTRunnerSession(BaseRunnerSession):
    def __init__(self, model: Model):
        super().__init__(model)
        assert isinstance(model.handle, trt.ICudaEngine)
        self._model = model
        self._has_dynamic_shapes = None

        self._context = None
        self._engine: trt.ICudaEngine = self._model.handle
        self._cuda_context = pycuda.autoinit.context

        self._input_names = None
        self._output_names = None
        self._buffers = None

    def __enter__(self):
        self._context = self._engine.create_execution_context()
        self._context.__enter__()

        self._input_names = [
            self._engine[idx] for idx in range(self._engine.num_bindings) if self._engine.binding_is_input(idx)
        ]
        self._output_names = [
            self._engine[idx] for idx in range(self._engine.num_bindings) if not self._engine.binding_is_input(idx)
        ]
        # all_binding_shapes_specified is True for models without dynamic shapes
        # so initially this variable is False for models with dynamic shapes
        self._has_dynamic_shapes = not self._context.all_binding_shapes_specified

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._context.__exit__(exc_type, exc_value, traceback)
        self._input_names = None
        self._output_names = None

        # TODO: are cuda buffers dealloc automatically?
        self._buffers = None

    def __call__(self, x):
        buffers = self._prepare_buffers_if_needed(x)
        bindings = self._update_bindings(buffers)

        for name in self._input_names:
            cuda.memcpy_htod(buffers.x_dev[name], buffers.x_host[name])
        self._cuda_context.push()
        self._context.execute_v2(bindings=bindings)
        self._cuda_context.pop()
        for name in self._output_names:
            cuda.memcpy_dtoh(buffers.y_pred_host[name], buffers.y_pred_dev[name])

        return buffers.y_pred_host

    def _update_bindings(self, buffers: TRTBuffers):
        bindings = [None] * self._engine.num_bindings
        for name in buffers.y_pred_dev:
            binding_idx: int = self._engine[name]
            bindings[binding_idx] = buffers.y_pred_dev[name]

        for name in buffers.x_dev:
            binding_idx: int = self._engine[name]
            bindings[binding_idx] = buffers.x_dev[name]

        return bindings

    def _set_dynamic_input_shapes(self, x_host):
        def _is_shape_dynamic(input_shape):
            return any([dim is None or dim == -1 for dim in input_shape])

        for name in self._input_names:
            bindings_idx = self._engine[name]
            data_shape = x_host[name].shape  # pytype: disable=attribute-error
            if self._engine.is_shape_binding(bindings_idx):
                input_shape = self._context.get_shape(bindings_idx)
                if _is_shape_dynamic(input_shape):
                    self._context.set_shape_input(bindings_idx, data_shape)
            else:
                input_shape = self._engine.get_binding_shape(bindings_idx)
                if _is_shape_dynamic(input_shape):
                    self._context.set_binding_shape(bindings_idx, data_shape)

        assert self._context.all_binding_shapes_specified and self._context.all_shape_inputs_specified

    def _prepare_buffers_if_needed(self, x_host: Dict[str, object]):
        # pytype: disable=attribute-error
        new_batch_size = list(x_host.values())[0].shape[0]
        current_batch_size = list(self._buffers.y_pred_host.values())[0].shape[0] if self._buffers else 0
        # pytype: enable=attribute-error

        if self._has_dynamic_shapes or new_batch_size != current_batch_size:
            # TODO: are CUDA buffers dealloc automatically?

            self._set_dynamic_input_shapes(x_host)

            y_pred_host = {}
            for name in self._output_names:
                shape = self._context.get_binding_shape(self._engine[name])
                y_pred_host[name] = np.zeros(shape, dtype=trt.nptype(self._model.outputs[name].dtype))

            y_pred_dev = {name: cuda.mem_alloc(data.nbytes) for name, data in y_pred_host.items()}

            x_dev = {
                name: cuda.mem_alloc(host_input.nbytes)
                for name, host_input in x_host.items()
                if name in self._input_names  # pytype: disable=attribute-error
            }

            self._buffers = TRTBuffers(None, x_dev, y_pred_host, y_pred_dev)

        return self._buffers._replace(x_host=x_host)


if "pycuda.driver" in sys.modules:
    loaders.register_extension(Format.TRT.value, TensorRTLoader)
    runners.register_extension(Format.TRT.value, TensorRTRunner)
    savers.register_extension(Format.TRT.value, TensorRTSaver)
else:
    LOGGER.warning("Do not register TensorRT extension due problems with importing pycuda.driver package.")
