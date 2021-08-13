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
For models with variable-sized inputs you must provide the --input-shape argument so that perf_analyzer knows
what shape tensors to use. For example, for a model that has an input called IMAGE that has shape [ 3, N, M ],
where N and M are variable-size dimensions, to tell perf_analyzer to send batch-size 4 requests of shape [ 3, 224, 224 ]
`--shape IMAGE:3,224,224`.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = Path(__file__).parent.name

from .deployment_toolkit.report import save_results, show_results, sort_results
from .deployment_toolkit.warmup import warmup


def calculate_average_latency(r):
    avg_sum_fields = [
        "Client Send",
        "Network+Server Send/Recv",
        "Server Queue",
        "Server Compute",
        "Server Compute Input",
        "Server Compute Infer",
        "Server Compute Output",
        "Client Recv",
    ]
    avg_latency = sum([int(r.get(f, 0)) for f in avg_sum_fields])

    return avg_latency


def update_performance_data(results: List, performance_file: str):
    with open(performance_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["avg latency"] = calculate_average_latency(row)

            results.append(row)


def _parse_batch_sizes(batch_sizes: str):
    batches = batch_sizes.split(sep=",")
    return list(map(lambda x: int(x.strip()), batches))


def online_performance(
        model_name: str,
        batch_sizes: List[int],
        result_path: str,
        input_shapes: Optional[List[str]] = None,
        profiling_data: str = "random",
        triton_instances: int = 1,
        triton_gpu_engine_count: int = 1,
        server_url: str = "localhost",
        measurement_window: int = 10000,
        shared_memory: bool = False
):
    print("\n")
    print(f"==== Dynamic batching analysis start ====")
    print("\n")

    input_shapes = " ".join(map(lambda shape: f" --shape {shape}", input_shapes)) if input_shapes else ""

    print(f"Running performance tests for dynamic batching")
    performance_file = f"triton_performance_dynamic_partial.csv"

    max_batch_size = max(batch_sizes)
    max_total_requests = 2 * max_batch_size * triton_instances * triton_gpu_engine_count
    max_concurrency = min(256, max_total_requests)
    batch_size = max(1, max_total_requests // 256)

    step = max(1, max_concurrency // 32)
    min_concurrency = step

    exec_args = f"""-m {model_name} \
        -x 1 \
        -p {measurement_window} \
        -v \
        -i http \
        -u {server_url}:8000 \
        -b {batch_size} \
        -f {performance_file} \
        --concurrency-range {min_concurrency}:{max_concurrency}:{step} \
        --input-data {profiling_data} {input_shapes}"""

    if shared_memory:
        exec_args += " --shared-memory=cuda"

    result = os.system(f"perf_client {exec_args}")
    if result != 0:
        print(f"Failed running performance tests. Perf client failed with exit code {result}")
        sys.exit(1)

    results = list()
    update_performance_data(results=results, performance_file=performance_file)

    results = sort_results(results=results)

    save_results(filename=result_path, data=results)
    show_results(results=results)

    os.remove(performance_file)

    print("Performance results for dynamic batching stored in: {0}".format(result_path))

    print("\n")
    print(f"==== Analysis done ====")
    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to test")
    parser.add_argument(
        "--input-data", type=str, required=False, default="random", help="Input data to perform profiling."
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        required=False,
        help="Input data shape in form INPUT_NAME:<full_shape_without_batch_axis>.",
    )
    parser.add_argument("--batch-sizes", type=str, required=True, help="List of batch sizes to tests. Comma separated.")
    parser.add_argument("--triton-instances", type=int, default=1, help="Number of Triton Server instances")
    parser.add_argument(
        "--number-of-model-instances", type=int, default=1, help="Number of models instances on Triton Server"
    )
    parser.add_argument("--result-path", type=str, required=True, help="Path where result file is going to be stored.")
    parser.add_argument("--server-url", type=str, required=False, default="localhost", help="Url to Triton server")
    parser.add_argument(
        "--measurement-window", required=False, help="Time which perf_analyzer will wait for results", default=10000
    )
    parser.add_argument("--shared-memory", help="Use shared memory for communication with Triton", action="store_true",
                        default=False)

    args = parser.parse_args()

    warmup(
        server_url=args.server_url,
        model_name=args.model_name,
        batch_sizes=_parse_batch_sizes(args.batch_sizes),
        triton_instances=args.triton_instances,
        triton_gpu_engine_count=args.number_of_model_instances,
        profiling_data=args.input_data,
        input_shapes=args.input_shape,
        measurement_window=args.measurement_window,
        shared_memory=args.shared_memory
    )

    online_performance(
        server_url=args.server_url,
        model_name=args.model_name,
        batch_sizes=_parse_batch_sizes(args.batch_sizes),
        triton_instances=args.triton_instances,
        triton_gpu_engine_count=args.number_of_model_instances,
        profiling_data=args.input_data,
        input_shapes=args.input_shape,
        result_path=args.result_path,
        measurement_window=args.measurement_window,
        shared_memory=args.shared_memory
    )


if __name__ == "__main__":
    main()
