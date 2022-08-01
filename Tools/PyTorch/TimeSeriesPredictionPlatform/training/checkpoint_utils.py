# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import shutil

import dllogger
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from loggers.log_helper import restart_logger


def save_checkpoint(trainer, filename="checkpoint.zip", checkpoint_dir="."):
    if trainer.ema:
        module_to_save = trainer.ema.module
    elif isinstance(trainer.model, DDP):
        module_to_save = trainer.model.module
    else:
        module_to_save = trainer.model
    state = {
        "epoch": trainer.epoch + 1,
        "global_step": trainer.global_step,
        "model_state_dict": module_to_save.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    trainer.logger.log(step='event', data={"String": f"Saving checkpoint to {filename}"}, verbosity=dllogger.Verbosity.DEFAULT)
    torch.save(state, checkpoint_path)


def maybe_restore_checkpoint(trainer, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        trainer.logger.log(
            step='event',
            data={"String": f"Restoring checkpoint from {checkpoint_path}"},
            verbosity=dllogger.Verbosity.DEFAULT
        )
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        trainer.epoch = checkpoint["epoch"]


def trim_json_log(log_path):
    """
    Loads dllogger's json log and returns its lines without unfinished epochs.
    Does not modify the logfile
    """
    if os.path.isfile(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # In case log file is newly created
            if not lines:
                return lines

            for i, l in enumerate(reversed(lines)):
                d = json.loads(l[4:])
                if d.get('step') == []:
                    return lines
                if 'data' in d and 'String' in d['data'] and 'Epoch' in d['data']['String']:
                    break
            lines = lines[:-i-1]

        return lines

    return []


def detect_duplicated_run():
    """
    Returns list of paths of the runs with the same config as provided
    """
    # This is meant to be called in a trainer class, which means that this doesn't have access to the top level config
    current_config = OmegaConf.load('.hydra/config.yaml')
    rel = os.path.relpath(os.getcwd(), get_original_cwd())
    rel = next(x for x in rel.split(os.path.sep))
    result_dir = os.path.join(get_original_cwd(), rel)

    duplicated = []
    for p, s, f in os.walk(result_dir):
        if '.hydra' in s:
            c = OmegaConf.load(os.path.join(p, '.hydra/config.yaml'))
            if hash(c) == hash(current_config):
                duplicated.append(p)
    # Don't take into account runs that ended before any checkpoint had been saved
    # or current run (at this point hydra's config has already been saved)
    duplicated = [p for p in duplicated if os.path.exists(os.path.join(p,'last_checkpoint.zip'))]

    return duplicated


def get_most_advanced_run(paths, logfile_name):
    adv = 0
    path = ''
    for p in paths:
        log_path = os.path.join(p, logfile_name)
        log_lines = trim_json_log(log_path)
        if len(log_lines) > adv:
            adv = len(log_lines)
            path = p
    return path


def maybe_continue_run(trainer):
    duplicates = detect_duplicated_run()
    if not duplicates:
        return

    logfile_name = trainer.config.get('logfile_name', 'log.json')
    unfinished_run_path = get_most_advanced_run(duplicates, logfile_name)
    checkpoint_path = os.path.join(unfinished_run_path, 'last_checkpoint.zip')
    best_checkpoint_path = os.path.join(unfinished_run_path, 'best_checkpoint.zip')
    maybe_restore_checkpoint(trainer, checkpoint_path)
    log_lines = trim_json_log(os.path.join(unfinished_run_path, logfile_name))

    # Reinitialize the logger. This will cause it to append to the copied log file.
    with open(logfile_name, 'w') as f:
        f.writelines(log_lines)
    trainer.logger = restart_logger(trainer.config, trainer.logger)
    trainer.logger.log(
        step='event',
        data={"String": f"Resuming run: {unfinished_run_path}"},
        verbosity=dllogger.Verbosity.DEFAULT
    )

    shutil.copyfile(checkpoint_path, 'last_checkpoint.zip')
    shutil.copyfile(best_checkpoint_path, 'best_checkpoint.zip')
