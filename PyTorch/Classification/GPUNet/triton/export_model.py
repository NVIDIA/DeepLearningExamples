#!/usr/bin/env python3

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator  # noqa: E402  module level import not at top of file
from .deployment_toolkit.core import (  # noqa: E402  module level import not at top of file
    DATALOADER_FN_NAME,
    BaseLoader,
    BaseSaver,
    ExportFormat,
    ModelInputType,
    TorchJit,
    load_from_file,
)
from .deployment_toolkit.extensions import loaders, savers  # noqa: E402  module level import not at top of file

LOGGER = logging.getLogger("export_model")

INPUT_MODEL_TYPES = [
    ModelInputType.TF_ESTIMATOR,
    ModelInputType.TF_KERAS,
    ModelInputType.PYT,
]

OUTPUT_MODEL_TYPES = [
    ExportFormat.TF_SAVEDMODEL,
    ExportFormat.TORCHSCRIPT,
    ExportFormat.ONNX,
]

TORCH_JIT_TYPES = [
    TorchJit.NONE,
    TorchJit.TRACE,
    TorchJit.SCRIPT,
]


def _get_args():
    parser = argparse.ArgumentParser(
        description="Script for exporting models from supported frameworks.", allow_abbrev=False
    )
    parser.add_argument("--input-path", help="Path to input python module", required=True)
    parser.add_argument(
        "--input-type", help="Input model type", choices=[f.value for f in INPUT_MODEL_TYPES], required=True
    )
    parser.add_argument("--output-path", help="Path to output model file", required=True)
    parser.add_argument(
        "--output-type", help="Output model type", choices=[f.value for f in OUTPUT_MODEL_TYPES], required=True
    )
    parser.add_argument(
        "--torch-jit",
        help="Torch Jit",
        choices=[f.value for f in TORCH_JIT_TYPES],
        required=False,
        default=None,
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

    if args.input_type == ModelInputType.PYT.value and args.output_type == ExportFormat.ONNX.value:
        saver_type = f"{ModelInputType.PYT.value}--{ExportFormat.ONNX.value}"
    else:
        saver_type = args.output_type
    Saver: BaseSaver = savers.get(saver_type)
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

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    dataloader_fn = None
    if args.dataloader is not None:
        get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
        dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)

    Loader: BaseLoader = loaders.get(args.input_type)
    loader = ArgParserGenerator(Loader, module_path=args.input_path).from_args(args)

    print(args.input_path)
    print(os.path.isfile(args.input_path))
    print(args.output_type)
    model = loader.load(
        args.input_path,
        dataloader_fn=dataloader_fn,
        output_type=args.output_type,
        torch_jit=args.torch_jit,
    )

    LOGGER.info("inputs: %s", model.inputs)
    LOGGER.info("outputs: %s", model.outputs)

    if args.input_type == ModelInputType.PYT.value and args.output_type == ExportFormat.ONNX.value:
        saver_type = f"{ModelInputType.PYT.value}--{ExportFormat.ONNX.value}"
    else:
        saver_type = args.output_type

    Saver: BaseSaver = savers.get(saver_type)
    saver = ArgParserGenerator(Saver).from_args(args)
    saver.save(model, args.output_path, dataloader_fn)


if __name__ == "__main__":
    main()
