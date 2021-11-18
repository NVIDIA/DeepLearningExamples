# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import torch

from copy import deepcopy
import logging
import logging.handlers
from collections import OrderedDict

_logger = logging.getLogger(__name__)

class ModelEma:
    """ Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """
    def __init__(self, model, decay=0.9999, device='', resume='', remove_params=[]):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.remove_params = remove_params
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # Check if key k is in the remove_params list
                if any(remove_str in k for remove_str in self.remove_params):
                    continue
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            if len(self.remove_params) > 0:
                this_dict = self.ema.state_dict()
                this_dict.update(new_state_dict)
                self.ema.load_state_dict(this_dict)
            else:
                self.ema.load_state_dict(new_state_dict)
            _logger.info("Loaded state_dict_ema")
        else:
            _logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        x = []
        y = []
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                x.append(ema_v.type(torch.float32))
                if self.device:
                    model_v = model_v.detach().to(device=self.device)
                y.append(model_v.type(torch.float32))
            torch._foreach_mul_(x, self.decay)
            torch._foreach_add_(x, y, alpha=1.-self.decay)
            for ind, ema_v in enumerate(self.ema.state_dict().values()):
                ema_v.copy_(x[ind])