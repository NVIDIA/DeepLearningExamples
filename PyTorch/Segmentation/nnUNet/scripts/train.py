import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import call

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpus", type=int, required=True, help="Number of GPUs")
parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--dim", type=int, required=True, choices=[2, 3], help="Dimension of UNet")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")

if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = f"python {path_to_main} --exec_mode train --task 01 --deep_supervision --save_ckpt "
    cmd += f"--dim {args.dim} "
    cmd += f"--batch_size {2 if args.dim == 3 else 16} "
    cmd += f"--val_batch_size {4 if args.dim == 3 else 64} "
    cmd += f"--fold {args.fold} "
    cmd += f"--gpus {args.gpus} "
    cmd += "--amp " if args.amp else ""
    call(cmd, shell=True)
