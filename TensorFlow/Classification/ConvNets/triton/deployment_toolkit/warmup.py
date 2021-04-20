import os
from typing import List, Optional


def warmup(
    model_name: str,
    batch_sizes: List[int],
    triton_instances: int = 1,
    profiling_data: str = "random",
    input_shapes: Optional[List[str]] = None,
    server_url: str = "localhost",
    measurement_window: int = 10000,
):
    print("\n")
    print(f"==== Warmup start ====")
    print("\n")

    input_shapes = " ".join(map(lambda shape: f" --shape {shape}", input_shapes)) if input_shapes else ""

    bs = set()
    bs.add(min(batch_sizes))
    bs.add(max(batch_sizes))

    measurement_window = 6 * measurement_window

    for batch_size in bs:
        exec_args = f"""-max-threads {triton_instances} \
           -m {model_name} \
           -x 1 \
           -c {triton_instances} \
           -t {triton_instances} \
           -p {measurement_window} \
           -v \
           -i http \
           -u {server_url}:8000 \
           -b {batch_size} \
           --input-data {profiling_data} {input_shapes}
        """

        result = os.system(f"perf_client {exec_args}")
        if result != 0:
            print(f"Failed running performance tests. Perf client failed with exit code {result}")
            exit(1)

    print("\n")
    print(f"==== Warmup done ====")
    print("\n")
