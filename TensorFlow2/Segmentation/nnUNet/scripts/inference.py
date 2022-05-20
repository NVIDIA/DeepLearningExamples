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

from argparse import ArgumentParser
from pathlib import Path
from subprocess import call

parser = ArgumentParser()
parser.add_argument("--data", type=Path, required=True, help="Path to data")
parser.add_argument("--task", type=str, default="01", help="Task code")
parser.add_argument("--dim", type=int, required=True, choices=[2, 3], help="Dimension of UNet")
parser.add_argument("--batch-size", "--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument("--save-preds", "--save_preds", action="store_true", help="Save predicted masks")
parser.add_argument(
    "--results", type=Path, default=Path("/results"), help="Path to results directory, output for the predicted masks"
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--ckpt-dir", "--ckpt_dir", type=Path, help="Path to checkpoint directory")
group.add_argument("--saved-model-dir", "--saved_model_dir", type=Path, help="Path to saved model directory")

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = Path(__file__).resolve().parent.parent / "main.py"
    cmd = ""
    cmd += f"python {path_to_main} --exec-mode predict "
    cmd += f"--data {args.data} "
    cmd += f"--task {args.task} "
    cmd += f"--dim {args.dim} "
    cmd += f"--batch-size {args.batch_size} "
    cmd += f"--fold {args.fold} "
    cmd += f"--amp {args.amp} "
    cmd += f"--tta {args.tta} "
    cmd += f"--save-preds {args.save_preds} "
    cmd += f"--results {args.results} "

    if args.ckpt_dir:
        cmd += f"--ckpt-dir {args.ckpt_dir} "
    elif args.saved_model_dir:
        cmd += f"--saved-model-dir {args.saved_model_dir} "

    cmd += "--use-wandb false"
    call(cmd, shell=True)
