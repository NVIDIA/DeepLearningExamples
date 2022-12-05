# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"], help="Benchmarking mode")
parser.add_argument("--task", type=str, default="01", help="Task code")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use")
parser.add_argument("--dim", type=int, required=True, help="Dimension of UNet")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--bind", action="store_true", help="Bind CPUs for each GPU. Improves throughput for multi-GPU.")
parser.add_argument("--train_batches", type=int, default=200, help="Number of batches for training")
parser.add_argument("--test_batches", type=int, default=200, help="Number of batches for inference")
parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations before collecting statistics")
parser.add_argument("--results", type=str, default="/results", help="Path to results directory")
parser.add_argument("--logname", type=str, default="perf.json", help="Name of dlloger output")

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = Path(__file__).resolve().parent.parent / "main.py"
    cmd = ""
    if args.bind:
        cmd += "bindpcie --cpu=exclusive,nosmt "
    cmd += f"python main.py --task {args.task} --benchmark --epochs 2 "
    cmd += f"--results {args.results} "
    cmd += f"--logname {args.logname} "
    cmd += f"--exec_mode {args.mode} "
    cmd += f"--dim {args.dim} "
    cmd += f"--gpus {args.gpus} "
    cmd += f"--nodes {args.nodes} "
    cmd += f"--train_batches {args.train_batches} "
    cmd += f"--test_batches {args.test_batches} "
    cmd += f"--warmup {args.warmup} "
    cmd += "--amp " if args.amp else ""
    if args.mode == "train":
        cmd += f"--batch_size {args.batch_size} "
    else:
        cmd += f"--val_batch_size {args.batch_size} "
    if args.amp and args.dim == 3:
        cmd += "--norm instance_nvfuser --layout NDHWC"
    subprocess.run(cmd, shell=True)
