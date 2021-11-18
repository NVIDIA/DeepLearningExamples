# SPDX-License-Identifier: Apache-2.0
import logging
import os

import dgl
import dllogger
import numpy as np
import torch


def save_checkpoint(trainer, filename="checkpoint.pth.tar", checkpoint_dir="./"):
    if trainer.ema:
        module_to_save = trainer.ema.module if trainer.world_size == 1 else trainer.ema
    else:
        module_to_save = trainer.model
    state = {
        "epoch": trainer.epoch + 1,
        "global_step": trainer.global_step,
        "model_state_dict": module_to_save.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    trainer.logger.log(step=[], data={"String": f"Saving checkpoint to {filename}"}, verbosity=dllogger.Verbosity.DEFAULT)
    torch.save(state, checkpoint_path)


def maybe_restore_checkpoint(trainer, checkpoint_path):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        trainer.logger.log(
            step=[], data={"String": f"Restoring checkpoint from {checkpoint_path}"}, verbosity=dllogger.Verbosity.DEFAULT
        )
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        trainer.epoch = checkpoint["epoch"]


def round_dict(input_data, decimal=2):
    rounded_data = {
        key: (np.around(value, decimal) if isinstance(value, (np.floating, float)) else value)
        for key, value in input_data.items()
    }
    return rounded_data


def to_device(batch, device=None):
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    if isinstance(batch, dict):
        return {k: t.to(device=device) if t.numel() else None for k, t in batch.items()}
    elif isinstance(batch, dgl.DGLGraph):
        return batch.to(device=device)
