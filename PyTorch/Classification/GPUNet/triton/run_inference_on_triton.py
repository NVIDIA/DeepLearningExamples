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

r"""
To infer the model deployed on Triton, you can use `run_inference_on_triton.py` script.
It sends a request with data obtained from pointed data loader and dumps received data into dump files.
Those files are stored in directory pointed by `--output-dir` argument.

Currently, the client communicates with the Triton server asynchronously using GRPC protocol.

Example call:

```shell script
python ./triton/run_inference_on_triton.py \
    --server-url localhost:8001 \
    --model-name ResNet50 \
    --model-version 1 \
    --dump-labels \
    --output-dir /results/dump_triton
```
"""

import argparse
import logging
import time
import traceback
from pathlib import Path

from tqdm import tqdm

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import DATALOADER_FN_NAME, load_from_file
from .deployment_toolkit.dump import JsonDumpWriter
from .deployment_toolkit.triton_inference_runner import TritonInferenceRunner

LOGGER = logging.getLogger("run_inference_on_triton")


def _parse_args():
    parser = argparse.ArgumentParser(description="Infer model on Triton server", allow_abbrev=False)
    parser.add_argument(
        "--server-url", type=str, default="localhost:8001", help="Inference server URL (default localhost:8001)"
    )
    parser.add_argument("--model-name", help="The name of the model used for inference.", required=True)
    parser.add_argument("--model-version", help="The version of the model used for inference.", required=True)
    parser.add_argument("--dataloader", help="Path to python file containing dataloader.", required=True)
    parser.add_argument("--dump-labels", help="Dump labels to output dir", action="store_true", default=False)
    parser.add_argument("--dump-inputs", help="Dump inputs to output dir", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", help="Verbose logs", action="store_true", default=True)
    parser.add_argument("--output-dir", required=True, help="Path to directory where outputs will be saved")
    parser.add_argument(
        "--response-wait-time", required=False, help="Maximal time to wait for response", default=120, type=float
    )
    parser.add_argument(
        "--max-unresponded-requests",
        required=False,
        help="Maximal number of unresponded requests",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--synchronous", help="Enable synchronous calls to Triton Server", action="store_true", default=False
    )

    args, *_ = parser.parse_known_args()

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    ArgParserGenerator(get_dataloader_fn).update_argparser(parser)
    args = parser.parse_args()

    return args


def main():
    args = _parse_args()

    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)
    dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)

    try:
        runner = TritonInferenceRunner(
            server_url=args.server_url,
            model_name=args.model_name,
            model_version=args.model_version,
            dataloader_fn=dataloader_fn,
            verbose=False,
            response_wait_time=args.response_wait_time,
            max_unresponded_requests=args.max_unresponded_requests,
            synchronous=args.synchronous,
        )

    except Exception as e:
        message = traceback.format_exc()
        LOGGER.error(f"Encountered exception \n{message}")
        raise e

    with JsonDumpWriter(output_dir=args.output_dir) as writer:
        start = time.time()
        for ids, x, y_pred, y_real in tqdm(runner, unit="batch", mininterval=10):
            data = _verify_and_format_dump(args, ids, x, y_pred, y_real)
            writer.write(**data)
        stop = time.time()

    LOGGER.info(f"\nThe inference took {stop - start:0.3f}s")


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


if __name__ == "__main__":
    main()
