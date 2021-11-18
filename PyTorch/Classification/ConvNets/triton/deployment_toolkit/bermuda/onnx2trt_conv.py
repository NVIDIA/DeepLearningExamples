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
from typing import Dict, Iterable, Optional

# pytype: disable=import-error
import onnx
import tensorrt as trt

from ..core import BaseConverter, Format, Model, Precision, ShapeSpec
from ..extensions import converters
from .utils import get_input_shapes

# pytype: enable=import-error


LOGGER = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class Onnx2TRTConverter(BaseConverter):
    def __init__(self, *, max_batch_size: int, max_workspace_size: int, precision: str):
        self._max_batch_size = max_batch_size
        self._max_workspace_size = max_workspace_size
        self._precision = Precision(precision)

    def convert(self, model: Model, dataloader_fn) -> Model:
        input_shapes = get_input_shapes(dataloader_fn(), self._max_batch_size)
        cuda_engine = onnx2trt(
            model.handle,
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


def onnx2trt(
    onnx_model: onnx.ModelProto,
    *,
    shapes: Dict[str, ShapeSpec],
    max_workspace_size: int,
    max_batch_size: int,
    model_precision: str,
) -> "trt.ICudaEngine":
    """
    Converts onnx model to TensorRT ICudaEngine
    Args:
        onnx_model: onnx.Model to convert
        shapes: dictionary containing min shape, max shape, opt shape for each input name
        max_workspace_size: The maximum GPU temporary memory which the CudaEngine can use at execution time.
        max_batch_size: The maximum batch size which can be used at execution time,
                        and also the batch size for which the CudaEngine will be optimized.
        model_precision: precision of kernels (possible values: fp16, fp32)

    Returns: TensorRT ICudaEngine
    """
    # Whether or not 16-bit kernels are permitted.
    # During :class:`ICudaEngine` build fp16 kernels will also be tried when this mode is enabled.
    fp16_mode = "16" in model_precision

    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size

    # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode,
    # meaning that your network definition must be created with the explicitBatch flag set.
    # For more information, see
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        # onnx model parsing
        if not parser.parse(onnx_model.SerializeToString()):
            for i in range(parser.num_errors):
                LOGGER.error(f"OnnxParser error {i}/{parser.num_errors}: {parser.get_error(i)}")
            raise RuntimeError("Error during parsing ONNX model (see logs for details)")

        # OnnxParser produces here FP32 TensorRT engine for FP16 network
        # so we force FP16 here for first input/output
        if fp16_mode:
            network.get_input(0).dtype = trt.DataType.HALF
            network.get_output(0).dtype = trt.DataType.HALF

        # optimization
        config = builder.create_builder_config()
        config.flags |= bool(fp16_mode) << int(trt.BuilderFlag.FP16)
        config.max_workspace_size = max_workspace_size

        profile = builder.create_optimization_profile()
        for name, spec in shapes.items():
            profile.set_shape(name, **spec._asdict())

        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config=config)

    return engine


converters.register_extension(f"{Format.ONNX.value}--{Format.TRT.value}", Onnx2TRTConverter)
