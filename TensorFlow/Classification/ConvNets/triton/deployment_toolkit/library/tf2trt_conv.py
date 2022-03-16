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

from typing import Iterable

from ..core import BaseConverter, Format, Model, Precision, ShapeSpec
from ..extensions import converters
from .onnx2trt_conv import onnx2trt
from .tf2onnx_conv import tfgraph2onnx
from .utils import get_input_shapes


class TFGraphDef2TRTConverter(BaseConverter):
    def __init__(
        self,
        *,
        max_batch_size: int,
        max_workspace_size: int,
        onnx_opset: int,
        onnx_optimized: bool = True,
        precision: str,
    ):
        self._max_batch_size = max_batch_size
        self._max_workspace_size = max_workspace_size
        self._onnx_opset = onnx_opset
        self._onnx_optimized = onnx_optimized
        self._precision = Precision(precision)

    def convert(self, model: Model, dataloader_fn) -> Model:
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

        input_shapes = get_input_shapes(dataloader_fn(), self._max_batch_size)
        cuda_engine = onnx2trt(
            onnx_model,
            shapes=input_shapes,
            max_workspace_size=self._max_workspace_size,
            max_batch_size=self._max_batch_size,
            model_precision=self._precision.value,
        )
        return model._replace(handle=cuda_engine)

    @staticmethod
    def required_source_model_precision(requested_model_precision: Precision) -> Precision:
        # TensorRT requires source models to be in FP32 precision
        return Precision.FP32


converters.register_extension(f"{Format.TF_ESTIMATOR.value}--{Format.TRT.value}", TFGraphDef2TRTConverter)
converters.register_extension(f"{Format.TF_KERAS.value}--{Format.TRT.value}", TFGraphDef2TRTConverter)
converters.register_extension(f"{Format.TF_SAVEDMODEL.value}--{Format.TRT.value}", TFGraphDef2TRTConverter)
