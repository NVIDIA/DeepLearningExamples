import os
import pickle
from functools import wraps
from subprocess import run

import numpy as np
import torch


def rank_zero(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


@rank_zero
def print0(text):
    print(text)


def get_task_code(args):
    return f"{args.task}_{args.dim}d"


def get_config_file(args):
    if args.data != "/data":
        path = os.path.join(args.data, "config.pkl")
    else:
        task_code = get_task_code(args)
        path = os.path.join(args.data, task_code, "config.pkl")
    return pickle.load(open(path, "rb"))


def set_cuda_devices(args):
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)


def verify_ckpt_path(args):
    resume_path = os.path.join(args.results, "checkpoints", "last.ckpt")
    if args.resume_training and os.path.exists(resume_path):
        return resume_path
    return args.ckpt_path


def make_empty_dir(path):
    run(["rm", "-rf", path])
    os.makedirs(path)


def get_stats(pred, targ, class_idx):
    tp = np.logical_and(pred == class_idx, targ == class_idx).sum()
    fn = np.logical_and(pred != class_idx, targ == class_idx).sum()
    fp = np.logical_and(pred == class_idx, targ != class_idx).sum()
    return tp, fn, fp
