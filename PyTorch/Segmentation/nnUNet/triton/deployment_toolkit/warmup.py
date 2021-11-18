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

import os
import sys
from typing import List, Optional


def warmup(
    model_name: str,
    batch_sizes: List[int],
    triton_gpu_engine_count: int = 1,
    triton_instances: int = 1,
    profiling_data: str = "random",
    input_shapes: Optional[List[str]] = None,
    server_url: str = "localhost",
    measurement_window: int = 10000,
    shared_memory: bool = False
):
    print("\n")
    print(f"==== Warmup start ====")
    print("\n")

    input_shapes = " ".join(map(lambda shape: f" --shape {shape}", input_shapes)) if input_shapes else ""

    measurement_window = 6 * measurement_window

    max_batch_size = max(batch_sizes)
    max_total_requests = 2 * max_batch_size * triton_instances * triton_gpu_engine_count
    max_concurrency = min(256, max_total_requests)
    batch_size = max(1, max_total_requests // 256)

    step = max(1, max_concurrency // 2)
    min_concurrency = step

    exec_args = f"""-m {model_name} \
        -x 1 \
        -p {measurement_window} \
        -v \
        -i http \
        -u {server_url}:8000 \
        -b {batch_size} \
        --concurrency-range {min_concurrency}:{max_concurrency}:{step} \
        --input-data {profiling_data} {input_shapes}"""

    if shared_memory:
        exec_args += " --shared-memory=cuda"

    result = os.system(f"perf_client {exec_args}")
    if result != 0:
        print(f"Failed running performance tests. Perf client failed with exit code {result}")
        sys.exit(1)

    print("\n")
    print(f"==== Warmup done ====")
    print("\n")
