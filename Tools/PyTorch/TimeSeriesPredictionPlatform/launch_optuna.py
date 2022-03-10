# SPDX-License-Identifier: Apache-2.0
import argparse
import copy
import logging
import multiprocessing
import os
import warnings
from contextlib import contextmanager
from datetime import datetime

import hydra
import omegaconf
import optuna
import torch
from omegaconf import OmegaConf
from optuna.samplers import TPESampler

from distributed_utils import (
    get_device,
    get_rank,
    init_distributed,
    is_main_process,
    log,
)
from training.trainer import CTLTrainer


def main(args):
    with open(args.config_path, "rb") as f:
        cfg = OmegaConf.load(f)

    if cfg.config.optuna.get("sampler", None):
        sampler = hydra.utils.instantiate(cfg.config.optuna.sampler)
    else:
        sampler = TPESampler(multivariate=True)

    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        direction=cfg.config.optuna.get("direction", "minimize"),
        storage="sqlite:////workspace/{}.db".format(args.study_name),  # XXX we should probably save it in results directory
    )

    import subprocess

    processes = []
    world_size = cfg.config.device.get("world_size", os.environ.get("WORLD_SIZE", 1))
    for i in range(torch.cuda.device_count() // world_size):
        devices = list(range(i * world_size, (i + 1) * world_size))
        command = "export CUDA_VISIBLE_DEVICES={} ; ".format(",".join([str(x) for x in devices]))
        command += "python "
        if world_size > 1:
            command += f'-m torch.distributed.run --nproc_per_node={world_size} --master_port={1234 + i} --master_addr="127.0.0.{1+i}" '
        command += f"hp_search.py --config_path {args.config_path} --study_name {args.study_name}"
        print(command)
        p = subprocess.Popen(command, shell=True)
        processes.append(p)
    for p in processes:
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        required=True,
        help="The path for the configuration file to run this experiement",
    )
    parser.add_argument("--study_name", default="study_" + str(datetime.now()).replace(" ", "_"), type=str)

    args, _ = parser.parse_known_args()
    main(args)
