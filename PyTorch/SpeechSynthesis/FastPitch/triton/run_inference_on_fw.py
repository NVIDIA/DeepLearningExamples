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
To infer the model on framework runtime, you can use `run_inference_on_fw.py` script.
It infers data obtained from pointed data loader locally and saves received data into npz files.
Those files are stored in directory pointed by `--output-dir` argument.

Example call:

```shell script
python ./triton/run_inference_on_fw.py \
    --input-path /models/exported/model.onnx \
    --input-type onnx \
    --dataloader triton/dataloader.py \
    --data-dir /data/imagenet \
    --batch-size 32 \
    --output-dir /results/dump_local \
    --dump-labels
```
"""

import argparse
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"

from tqdm import tqdm

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import DATALOADER_FN_NAME, BaseLoader, BaseRunner, Format, load_from_file
from .deployment_toolkit.dump import NpzWriter
from .deployment_toolkit.extensions import loaders, runners

LOGGER = logging.getLogger("run_inference_on_fw")


def _verify_and_format_dump(args, ids, x, y_pred, y_real):
    data = {"outputs": y_pred, "ids": {"ids": ids}}
    if args.dump_inputs:
        data["inputs"] = x
    if args.dump_labels:
        if not y_real:
            raise ValueError(
                "Found empty label values. Please provide labels in dataloader_fn or do not use --dump-labels argument"
            )
        data["labels"] = y_real
    return data


def _parse_and_validate_args():
    supported_inputs = set(runners.supported_extensions) & set(loaders.supported_extensions)

    parser = argparse.ArgumentParser(description="Dump local inference output of given model", allow_abbrev=False)
    parser.add_argument("--input-path", help="Path to input model", required=True)
    parser.add_argument("--input-type", help="Input model type", choices=supported_inputs, required=True)
    parser.add_argument("--dataloader", help="Path to python file containing dataloader.", required=True)
    parser.add_argument("--output-dir", help="Path to dir where output files will be stored", required=True)
    parser.add_argument("--dump-labels", help="Dump labels to output dir", action="store_true", default=False)
    parser.add_argument("--dump-inputs", help="Dump inputs to output dir", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=False)

    args, *_ = parser.parse_known_args()

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    ArgParserGenerator(get_dataloader_fn).update_argparser(parser)

    Loader: BaseLoader = loaders.get(args.input_type)
    ArgParserGenerator(Loader, module_path=args.input_path).update_argparser(parser)

    Runner: BaseRunner = runners.get(args.input_type)
    ArgParserGenerator(Runner).update_argparser(parser)

    args = parser.parse_args()

    types_requiring_io_params = []

    if args.input_type in types_requiring_io_params and not all(p for p in [args.inputs, args.outputs]):
        parser.error(f"For {args.input_type} input provide --inputs and --outputs parameters")

    return args


def main():
    args = _parse_and_validate_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info(f"args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    Loader: BaseLoader = loaders.get(args.input_type)
    Runner: BaseRunner = runners.get(args.input_type)

    loader = ArgParserGenerator(Loader, module_path=args.input_path).from_args(args)
    runner = ArgParserGenerator(Runner).from_args(args)
    LOGGER.info(f"Loading {args.input_path}")
    model = loader.load(args.input_path)
    with runner.init_inference(model=model) as runner_session, NpzWriter(args.output_dir) as writer:
        get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
        dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)
        LOGGER.info(f"Data loader initialized; Running inference")
        for ids, x, y_real in tqdm(dataloader_fn(), unit="batch", mininterval=10):
            y_pred = runner_session(x)
            data = _verify_and_format_dump(args, ids=ids, x=x, y_pred=y_pred, y_real=y_real)
            writer.write(**data)
        LOGGER.info(f"Inference finished")


if __name__ == "__main__":
    main()
