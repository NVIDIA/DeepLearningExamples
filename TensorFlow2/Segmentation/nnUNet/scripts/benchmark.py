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

import subprocess
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"], help="Benchmarking mode")
parser.add_argument("--task", type=str, default="01", help="Task code")
parser.add_argument("--dim", type=int, required=True, choices=[2, 3], help="Dimension of UNet")
parser.add_argument("--gpus", type=int, default=1, help="Number of gpus")
parser.add_argument("--batch-size", "--batch_size", type=int, required=True)
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--bind", action="store_true", help="Bind CPUs for each GPU. Improves throughput for multi-GPU.")
parser.add_argument("--horovod", action="store_true")
parser.add_argument("--xla", action="store_true", help="Enable XLA compiling")
parser.add_argument(
    "--bench-steps", "--bench_steps", type=int, default=200, help="Number of benchmarked steps in total"
)
parser.add_argument(
    "--warmup-steps", "--warmup_steps", type=int, default=100, help="Warmup iterations before collecting statistics"
)
parser.add_argument("--results", type=Path, default=Path("/results"), help="Path to results directory")
parser.add_argument("--logname", type=str, default="perf.json", help="Name of the dlloger output")


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = Path(__file__).resolve().parent.parent / "main.py"
    cmd = f"horovodrun -np {args.gpus} " if args.horovod else ""
    if args.bind:
        cmd += "bindpcie --cpu=exclusive,nosmt "
    cmd += f"python {path_to_main} --benchmark --ckpt-strategy none --seed 0 "
    cmd += f"--exec-mode {args.mode} "
    cmd += f"--task {args.task} "
    cmd += f"--dim {args.dim} "
    cmd += f"--batch-size {args.batch_size} "
    cmd += f"--amp {args.amp} "
    cmd += f"--xla {args.xla} "
    cmd += f"--bench-steps {args.bench_steps} "
    cmd += f"--warmup-steps {args.warmup_steps} "
    cmd += f"--results {args.results} "
    cmd += f"--logname {args.logname} "
    cmd += f"--gpus {args.gpus} "
    subprocess.run(cmd, shell=True)
