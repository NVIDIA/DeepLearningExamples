import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import dirname
from subprocess import call

parser = ArgumentParser(ArgumentDefaultsHelpFormatter)
parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Fold number")
parser.add_argument("--dim", type=int, required=True, help="Dimension of UNet")
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
parser.add_argument("--tta", action="store_true", help="Enable test time augmentation")
parser.add_argument("--save_preds", action="store_true", help="Save predicted masks")


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_main = os.path.join(dirname(dirname(os.path.realpath(__file__))), "main.py")
    cmd = f"python {path_to_main} --exec_mode evaluate --task 01 --gpus 1 "
    cmd += f"--dim {args.dim} "
    cmd += f"--fold {args.fold} "
    cmd += f"--ckpt_path {args.ckpt_path} "
    cmd += f"--val_batch_size {args.val_batch_size} "
    cmd += "--amp " if args.amp else ""
    cmd += "--tta " if args.tta else ""
    cmd += "--save_preds " if args.save_preds else ""
    call(cmd, shell=True)
