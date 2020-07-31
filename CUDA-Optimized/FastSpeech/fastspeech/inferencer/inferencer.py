# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pathlib
import time
import abc

import numpy as np
import torch
from tensorboardX import SummaryWriter
import glob
from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import to_device_async, to_cpu_numpy
import torch.nn as nn

    
class Inferencer(object):
    """
    set seed
    load model
    logging
    """
    def __init__(self, model_name, model, data_loader=None, ckpt_path=None, ckpt_file=None, log_path=None, device='cuda', use_fp16=False, seed=None):
        self.data_loader = data_loader
        self.model_name = model_name
        self.model = model
        self.ckpt_path = ckpt_path
        self.log_path = log_path
        self.device = device
        self.seed = seed
        self.step = 0
        self.ckpt_file = ckpt_file
        self.use_fp16 = use_fp16

        # model
        self.model.eval()
        to_device_async(self.model, self.device)
        num_param = sum(param.numel() for param in model.parameters())
        tprint('The number of {} parameters: {}'.format(self.model_name, num_param))

        # precision
        if self.use_fp16:
            self.model = self.model.half()

        # data parallel
        self.model = nn.DataParallel(self.model)

        # set seed
        if seed is None:
            seed = np.random.randint(2**16)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.data_loader_iter = iter(self.data_loader)

        # logging
        if log_path:
            # tensorboard log path : {log_path}/YYYYMMDD-HHMMMSS
            log_path = os.path.join(log_path, time.strftime('%Y%m%d-%H%M%S'))
            self.tbwriter = SummaryWriter(log_dir=log_path, flush_secs=10)

        # checkpoint path
        if self.ckpt_path:
            self.ckpt_path = os.path.join(self.ckpt_path, self.model_name)
            pathlib.Path(self.ckpt_path).mkdir(parents=True, exist_ok=True)

            # load checkpoint
            self.load(ckpt_file)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    @abc.abstractmethod
    def infer(self):
        return NotImplemented

    def load(self, ckpt_file):
        # load latest checkpoint file if not defined.
        if not ckpt_file:
            files_exist = glob.glob(os.path.join(self.ckpt_path, '*'))
            if files_exist:
                ckpt_file = max(files_exist, key=os.path.getctime)

        if ckpt_file:
            state_dict = torch.load(ckpt_file, map_location=self.device)

            self.step = state_dict['step']
            self.model.load_state_dict(state_dict['model'])

            tprint('[Load] Checkpoint \'{}\'. Step={}'.format(ckpt_file, self.step))
        else:
            tprint('No checkpoints in {}. Load skipped.'.format(self.ckpt_path))
            raise Exception("No checkpoints found.")

    def log(self, output):
        output = {k: to_cpu_numpy(v) for k, v in output.items()}
        self.console_log('infer', output)
        if self.log_path:
            self.tensorboard_log('infer', output)

    @abc.abstractmethod
    def console_log(self, tag, output):
        raise NotImplemented

    @abc.abstractmethod
    def tensorboard_log(self, tag, output):
        raise NotImplemented

