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
To infer the model on framework runtime, you can use `run_performance_on_fw.py` script.
It infers data obtained from pointed data loader locally and calculate throughput and latency.
Those results are stored in path pointed by `--results-path` in form of CSV file.

Example call:

```shell script
python ./triton/run_performance_on_fw.py \
    --input-path /models/exported/model.onnx \
    --input-type onnx \
    --dataloader triton/dataloader.py \
    --data-dir /data/imagenet \
    --batch-sizes 32 \
    --results-path results.csv
```
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import List

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "0"

from .deployment_toolkit.args import ArgParserGenerator  # noqa: E402  module level import not at top of file
from .deployment_toolkit.core import (  # noqa: E402  module level import not at top of file
    DATALOADER_FN_NAME,
    BaseLoader,
    BaseRunner,
    load_from_file,
)
from .deployment_toolkit.extensions import loaders, runners  # noqa: E402  module level import not at top of file

LOGGER = logging.getLogger("run_performance_on_fw")


def _save_result(results_path: str, results: List):
    LOGGER.info(f"Storing results to {results_path}")
    item = results[0]
    with open(results_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(item.keys()))
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    LOGGER.info("Done")


def _parse_and_validate_args():
    supported_inputs = set(runners.supported_extensions) & set(loaders.supported_extensions)

    parser = argparse.ArgumentParser(
        description="Measure inference performance of given model in framework container", allow_abbrev=False
    )
    parser.add_argument("--input-path", help="Path to input model", required=True)
    parser.add_argument("--input-type", help="Input model type", choices=supported_inputs, required=True)
    parser.add_argument("--dataloader", help="Path to python file containing dataloader.", required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        default=[1],
        help="List of batch sizes to test.",
        nargs="*",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of performance iterations per batch size.",
    )
    parser.add_argument(
        "--results-path",
        help="Path to results file where performance result will be stored",
        required=True,
    )
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

    if args.input_type in types_requiring_io_params and not all(p for p in [args.inputs, args.outptputs]):
        parser.error(f"For {args.input_type} input provide --inputs and --outputs parameters")

    return args


def main():
    args = _parse_and_validate_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    if args.iterations < 10:
        raise ValueError("The minimal number of iterations for performance measurement is 10")

    if not args.results_path.endswith(".csv"):
        raise ValueError("Results path for results is invalid. Please, provide the CSV file name. Example: results.csv")

    Loader: BaseLoader = loaders.get(args.input_type)
    Runner: BaseRunner = runners.get(args.input_type)

    loader = ArgParserGenerator(Loader, module_path=args.input_path).from_args(args)
    runner = ArgParserGenerator(Runner).from_args(args)
    LOGGER.info(f"Loading {args.input_path}")
    model = loader.load(args.input_path)
    get_dataloader_fn = load_from_file(args.dataloader, label="dataloader", target=DATALOADER_FN_NAME)

    results = []
    with runner.init_inference(model=model) as runner_session:
        for batch_size in args.batch_sizes:
            LOGGER.info(f"Running performance measurement for batch size {batch_size}.")
            # WAR - override batch size for dataloader
            args.batch_size = batch_size
            dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)
            LOGGER.debug("Data loader initialized.")
            for _, x, _ in dataloader_fn():
                input = x
                break

            runner_session.start_measurement()
            LOGGER.info("Running measurement")
            for idx in range(args.iterations):
                LOGGER.debug(f"Iteration {idx}")
                runner_session(input)

            throughput, latency = runner_session.stop_measurement(batch_size=batch_size)
            LOGGER.info("Done")

            LOGGER.info(f"Throughput: {throughput:.2f} [infer/s]")
            LOGGER.info(f"Latency: {latency:.2f} [ms]")

            data = {
                "Batch": batch_size,
                "Throughput (infer/sec)": f"{throughput:.2f}",
                "Latency (ms)": f"{latency:.2f}",
            }
            results.append(data)

    if not results:
        raise RuntimeError("No valid measurement performed.")

    _save_result(args.results_path, results)


if __name__ == "__main__":
    main()
