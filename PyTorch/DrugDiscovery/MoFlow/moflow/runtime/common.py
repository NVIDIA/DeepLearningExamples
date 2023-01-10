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


from glob import glob
import logging
import os
from typing import List, Optional, Tuple
import torch

from moflow.model.model import MoFlow


CHECKPOINT_PATTERN = 'model_snapshot_epoch_%s'


def _sort_checkpoints(paths: List[str]) -> List[str]:
    return sorted(paths, key=lambda x: int(x.split('_')[-1]))


def save_state(dir: str, model: MoFlow, optimizer: torch.optim.Optimizer, ln_var: float, epoch: int, keep: int = 1) -> None:
    """Save training state in a given dir. This checkpoint can be used to resume training or run inference
    with the trained model. This function will keep up to <keep> newest checkpoints and remove the oldest ones.
    """
    save_path = os.path.join(dir, CHECKPOINT_PATTERN % (epoch + 1))
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ln_var': ln_var,
        'epoch': epoch,
    }
    torch.save(state, save_path)

    if keep > 0:
        filenames = glob(os.path.join(dir, CHECKPOINT_PATTERN % '*'))
        if len(filenames) <= keep:
            return

        to_del = _sort_checkpoints(filenames)[:-keep]
        for path in to_del:
            os.remove(path)


def load_state(path: str, model: MoFlow, device: torch.device, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
    """Load model's and optimizer's state from a given file.
    This function returns the number of epochs the model was trained for and natural logarithm of variance
    the for the distribution of the latent space.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return state['epoch'], state['ln_var']


def get_newest_checkpoint(model_dir: str, validate: bool = True) -> str:
    """Find newest checkpoint in a given directory.
    If validate is set to True, this function will also verify that the file can be loaded and
    select older checkpoint if neccessary.
    """
    filenames = glob(os.path.join(model_dir, CHECKPOINT_PATTERN % '*'))
    if len(filenames) == 0:
        logging.info(f'No checkpoints available')
        return None

    paths = _sort_checkpoints(filenames)
    if validate:
        for latest_path in paths[::-1]:
            try:
                torch.load(latest_path, map_location='cpu')
                break
            except:
                logging.info(f'Checkpoint {latest_path} is corrupted')
        else:
            logging.info(f'All available checkpoints were corrupted')
            return None

    else:
        latest_path = paths[-1]

    logging.info(f'Found checkpoint {latest_path}')
    return latest_path
