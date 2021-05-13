#!/usr/bin/env python3

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
r"""
`convert_model.py` script allows to convert between model formats with additional model optimizations
for faster inference.
It converts model from results of get_model function.

Currently supported input and output formats are:

  - inputs
    - `tf-estimator` - `get_model` function returning Tensorflow Estimator
    - `tf-keras` - `get_model` function returning Tensorflow Keras Model
    - `tf-savedmodel` - Tensorflow SavedModel binary
    - `pyt` - `get_model` function returning PyTorch Module
  - output
    - `tf-savedmodel` - Tensorflow saved model
    - `tf-trt` - TF-TRT saved model
    - `ts-trace` - PyTorch traced ScriptModule
    - `ts-script` - PyTorch scripted ScriptModule
    - `onnx` - ONNX
    - `trt` - TensorRT plan file

For tf-keras input you can use:
  - --large-model flag - helps loading model which exceeds maximum protobuf size of 2GB
  - --tf-allow-growth flag - control limiting GPU memory growth feature
    (https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth). By default it is disabled.
"""

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import (
    DATALOADER_FN_NAME,
    BaseConverter,
    BaseLoader,
    BaseSaver,
    Format,
    Precision,
    load_from_file,
)
from .deployment_toolkit.extensions import converters, loaders, savers

LOGGER = logging.getLogger("convert_model")

INPUT_MODEL_TYPES = [Format.TF_ESTIMATOR, Format.TF_KERAS, Format.TF_SAVEDMODEL, Format.PYT]
OUTPUT_MODEL_TYPES = [Format.TF_SAVEDMODEL, Format.TF_TRT, Format.ONNX, Format.TRT, Format.TS_TRACE, Format.TS_SCRIPT]


def _get_args():
    parser = argparse.ArgumentParser(description="Script for conversion between model formats.", allow_abbrev=False)
    parser.add_argument("--input-path", help="Path to input model file (python module or binary file)", required=True)
    parser.add_argument(
        "--input-type", help="Input model type", choices=[f.value for f in INPUT_MODEL_TYPES], required=True
    )
    parser.add_argument("--output-path", help="Path to output model file", required=True)
    parser.add_argument(
        "--output-type", help="Output model type", choices=[f.value for f in OUTPUT_MODEL_TYPES], required=True
    )
    parser.add_argument("--dataloader", help="Path to python module containing data loader")
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=False)
    parser.add_argument(
        "--ignore-unknown-parameters",
        help="Ignore unknown parameters (argument often used in CI where set of arguments is constant)",
        action="store_true",
        default=False,
    )

    args, unparsed_args = parser.parse_known_args()

    Loader: BaseLoader = loaders.get(args.input_type)
    ArgParserGenerator(Loader, module_path=args.input_path).update_argparser(parser)

    converter_name = f"{args.input_type}--{args.output_type}"
    Converter: BaseConverter = converters.get(converter_name)
    if Converter is not None:
        ArgParserGenerator(Converter).update_argparser(parser)

    Saver: BaseSaver = savers.get(args.output_type)
    ArgParserGenerator(Saver).update_argparser(parser)

    if args.dataloader is not None:
        get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
        ArgParserGenerator(get_dataloader_fn).update_argparser(parser)

    if args.ignore_unknown_parameters:
        args, unknown_args = parser.parse_known_args()
        LOGGER.warning(f"Got additional args {unknown_args}")
    else:
        args = parser.parse_args()
    return args


def main():
    args = _get_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info(f"args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    requested_model_precision = Precision(args.precision)
    dataloader_fn = None

    # if conversion is required, temporary change model load precision to that required by converter
    # it is for TensorRT converters which require fp32 models for all requested precisions
    converter_name = f"{args.input_type}--{args.output_type}"
    Converter: BaseConverter = converters.get(converter_name)
    if Converter:
        args.precision = Converter.required_source_model_precision(requested_model_precision).value

    Loader: BaseLoader = loaders.get(args.input_type)
    loader = ArgParserGenerator(Loader, module_path=args.input_path).from_args(args)
    model = loader.load(args.input_path)


    LOGGER.info("inputs: %s", model.inputs)
    LOGGER.info("outputs: %s", model.outputs)

    if Converter:  # if conversion is needed
        # dataloader must much source model precision - so not recovering it yet
        if args.dataloader is not None:
            get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
            dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)

    # recover precision to that requested by user
    args.precision = requested_model_precision.value

    if Converter:
        converter = ArgParserGenerator(Converter).from_args(args)
        model = converter.convert(model, dataloader_fn=dataloader_fn)

    Saver: BaseSaver = savers.get(args.output_type)
    saver = ArgParserGenerator(Saver).from_args(args)
    saver.save(model, args.output_path)

    return 0


if __name__ == "__main__":
    main()
