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

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import run

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, required=True, help="Path to data")
parser.add_argument("--task", type=str, default="01", help="Path to data")
parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--dim", type=int, required=True, help="Dimension of UNet")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument("--save_preds", action="store_true", help="Save predicted masks")


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = f"python {path_to_main} --exec_mode predict --task {args.task} --gpus 1 "
    cmd += f"--data {args.data} "
    cmd += f"--dim {args.dim} "
    cmd += f"--fold {args.fold} "
    cmd += f"--ckpt_path {args.ckpt_path} "
    cmd += f"--val_batch_size {args.batch_size} "
    cmd += "--amp " if args.amp else ""
    cmd += "--tta " if args.tta else ""
    cmd += "--save_preds " if args.save_preds else ""
    run(cmd, shell=True)
