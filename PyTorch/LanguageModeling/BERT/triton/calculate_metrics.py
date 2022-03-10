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

Data provided to `MetricsCalculator` are obtained from dump files
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

# method from PEP-366 to support relative import in executed modules

if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.args import ArgParserGenerator
from .deployment_toolkit.core import BaseMetricsCalculator, load_from_file
from .deployment_toolkit.dump import JsonDumpReader

LOGGER = logging.getLogger("calculate_metrics")
TOTAL_COLUMN_NAME = "_total_"


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run models with given dataloader", allow_abbrev=False)
    parser.add_argument("--metrics", help="Path to python module containing metrics calculator", required=True)
    parser.add_argument("--csv", help="Path to csv file", required=True)
    parser.add_argument("--dump-dir", help="Path to directory with dumped outputs (and labels)", required=True)

    args, *_ = parser.parse_known_args()

    MetricsCalculator = load_from_file(args.metrics, "metrics", "MetricsCalculator")
    ArgParserGenerator(MetricsCalculator).update_argparser(parser)

    args = parser.parse_args()

    LOGGER.info("args:")
    for key, value in vars(args).items():
        LOGGER.info(f"    {key} = {value}")

    MetricsCalculator = load_from_file(args.metrics, "metrics", "MetricsCalculator")
    metrics_calculator: BaseMetricsCalculator = ArgParserGenerator(MetricsCalculator).from_args(args)

    reader = JsonDumpReader(args.dump_dir)
    for ids, x, y_true, y_pred in reader.iterate_over(["ids", "inputs", "labels", "outputs"]):
        ids = list(ids["ids"]) if ids is not None else None
        metrics_calculator.update(ids=ids, x=x, y_pred=y_pred, y_real=y_true)
    metrics = metrics_calculator.metrics

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
