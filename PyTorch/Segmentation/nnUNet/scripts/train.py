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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from subprocess import run

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, default="01", help="Path to data")
parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs")
parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--dim", type=int, required=True, choices=[2, 3], help="Dimension of UNet")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument("--bind", action="store_true", help="Enable test time augmentation")
parser.add_argument("--resume_training", action="store_true", help="Resume training from checkpoint")
parser.add_argument("--results", type=str, default="/results", help="Path to results directory")
parser.add_argument("--logname", type=str, default="train_logs.json", help="Name of dlloger output")
parser.add_argument("--learning_rate", type=float, default=8e-4, help="Learning rate")

if __name__ == "__main__":
    args = parser.parse_args()
    skip = 100 if args.gpus == 1 else 150
    path_to_main = Path(__file__).resolve().parent.parent / "main.py"
    cmd = ""
    if args.bind:
        cmd += "bindpcie --cpu=exclusive,nosmt "
    cmd = f"python {path_to_main} --exec_mode train --save_ckpt --deep_supervision --skip_first_n_eval {skip} "
    cmd += f"--task {args.task} "
    cmd += f"--results {args.results} "
    cmd += f"--logname {args.logname} "
    cmd += f"--dim {args.dim} "
    cmd += f"--batch_size {2 if args.dim == 3 else 64} "
    cmd += f"--val_batch_size {1 if args.dim == 3 else 64} "
    cmd += f"--norm {'instance_nvfuser' if args.dim == 3 else 'instance'} "
    cmd += f"--layout {'NDHWC' if args.dim == 3 else 'NCDHW'} "
    cmd += f"--fold {args.fold} "
    cmd += f"--gpus {args.gpus} "
    cmd += f"--epochs {300 if args.gpus == 1 else 600} "
    cmd += f"--learning_rate {args.learning_rate} "
    cmd += "--amp " if args.amp else ""
    cmd += "--tta " if args.tta else ""
    cmd += "--resume_training " if args.resume_training else ""
    cmd += f"--seed {args.seed} "
    run(cmd, shell=True)
