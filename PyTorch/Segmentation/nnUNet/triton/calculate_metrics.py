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
Using `calculate_metrics.py` script, you can obtain model accuracy/error metrics using defined `MetricsCalculator` class.

Data provided to `MetricsCalculator` are obtained from npz dump files
stored in directory pointed by `--dump-dir` argument.
Above files are prepared by `run_inference_on_fw.py` and `run_inference_on_triton.py` scripts.

Output data is stored in csv file pointed by `--csv` argument.

Example call:

```shell script
python ./triton/calculate_metrics.py \
    --dump-dir /results/dump_triton \
    --csv /results/accuracy_results.csv \
    --metrics metrics.py \
    --metric-class-param1 value
```
"""

import argparse
import csv
import logging
import string
from pathlib import Path

import numpy as np

# method from PEP-366 to support relative import in executed modules

if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import BaseMetricsCalculator, load_from_file
from .deployment_toolkit.dump import pad_except_batch_axis

LOGGER = logging.getLogger("calculate_metrics")
TOTAL_COLUMN_NAME = "_total_"


def get_data(dump_dir, prefix):
    """Loads and concatenates dump files for given prefix (ex. inputs, outputs, labels, ids)"""
    dump_dir = Path(dump_dir)
    npz_files = sorted(dump_dir.glob(f"{prefix}*.npz"))
    data = None
    if npz_files:
        # assume that all npz files with given prefix contain same set of names
        names = list(np.load(npz_files[0].as_posix()).keys())
        # calculate target shape
        target_shape = {
            name: tuple(np.max([np.load(npz_file.as_posix())[name].shape for npz_file in npz_files], axis=0))
            for name in names
        }
        # pad and concatenate data
        data = {
            name: np.concatenate(
                [pad_except_batch_axis(np.load(npz_file.as_posix())[name], target_shape[name]) for npz_file in npz_files]
            )
            for name in names
        }
    return data


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run models with given dataloader", allow_abbrev=False)
    parser.add_argument("--metrics", help=f"Path to python module containing metrics calculator", required=True)
    parser.add_argument("--csv", help="Path to csv file", required=True)
    parser.add_argument("--dump-dir", help="Path to directory with dumped outputs (and labels)", required=True)

    args, *_ = parser.parse_known_args()

    MetricsCalculator = load_from_file(args.metrics, "metrics", "MetricsCalculator")
    ArgParserGenerator(MetricsCalculator).update_argparser(parser)

    args = parser.parse_args()

    LOGGER.info(f"args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    MetricsCalculator = load_from_file(args.metrics, "metrics", "MetricsCalculator")
    metrics_calculator: BaseMetricsCalculator = ArgParserGenerator(MetricsCalculator).from_args(args)

    ids = get_data(args.dump_dir, "ids")["ids"]
    x = get_data(args.dump_dir, "inputs")
    y_true = get_data(args.dump_dir, "labels")
    y_pred = get_data(args.dump_dir, "outputs")

    common_keys = list({k for k in (y_true or [])} & {k for k in (y_pred or [])})
    for key in common_keys:
        if y_true[key].shape != y_pred[key].shape:
            LOGGER.warning(
                f"Model predictions and labels shall have equal shapes. "
                f"y_pred[{key}].shape={y_pred[key].shape} != "
                f"y_true[{key}].shape={y_true[key].shape}"
            )

    metrics = metrics_calculator.calc(ids=ids, x=x, y_pred=y_pred, y_real=y_true)
    metrics = {TOTAL_COLUMN_NAME: len(ids), **metrics}

    metric_names_with_space = [name for name in metrics if any([c in string.whitespace for c in name])]
    if metric_names_with_space:
        raise ValueError(f"Metric names shall have no spaces; Incorrect names: {', '.join(metric_names_with_space)}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()
