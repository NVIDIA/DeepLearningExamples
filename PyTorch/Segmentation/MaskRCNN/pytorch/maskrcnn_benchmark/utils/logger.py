# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Iteration: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Epoch: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s
