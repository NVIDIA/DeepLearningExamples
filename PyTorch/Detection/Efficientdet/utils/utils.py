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

import os
import shutil
from collections import OrderedDict
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

import torch
from torch import distributed as dist

import logging
import logging.handlers
from .model_ema import ModelEma

_logger = logging.getLogger(__name__)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model



def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()



def setup_dllogger(rank, enabled=True, filename='log.json'):
    if enabled and rank == 0:
        backends = [
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(
                Verbosity.VERBOSE,
                filename,
                ),
            ]
        DLLogger.init(backends)
    else:
        DLLogger.init([])


def get_latest_file(files):
    prefix = files[0].split("checkpoint")[0]
    max_checkpoint_number =  max([int(f.split("checkpoint_")[1].split('.')[0]) for f in files])  # checkpoint_name_convention = checkpoint_ + number + .pth.tar
    return prefix + "checkpoint_" + str(max_checkpoint_number) + ".pth.tar"

def get_latest_checkpoint(dir_path):
    if not os.path.exists(dir_path):
        print("{} does not exist to load checkpoint".format(dir_path))
        return None
    files = [os.path.join(dir_path, f) for f in sorted(os.listdir(dir_path)) if "checkpoint" in f]
    print("... Looking inside {}".format(dir_path))
    if len(files) > 0:
        return get_latest_file(files)
    return None



class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)



class CheckpointSaver:
    def __init__(
            self,
            args=None,
            checkpoint_dir='',
            unwrap_fn=unwrap_model):

        # objects to save state_dicts of
        self.args = args

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None

        # config
        self.checkpoint_dir = checkpoint_dir
        self.extension = '.pth.tar'
        self.unwrap_fn = unwrap_fn

    def save_checkpoint(self, model, optimizer, epoch, scaler=None, model_ema=None, metric=None, is_best=False):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, "tmp" + self.extension)
        actual_save_path = os.path.join(self.checkpoint_dir, "checkpoint_" + str(epoch) + self.extension)
        self._save(model, optimizer, tmp_save_path, actual_save_path, epoch, scaler, model_ema, metric, is_best)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, model, optimizer, tmp_save_path, save_path, epoch, scaler=None, model_ema=None, metric=None, is_best=False):
        save_state = {
            'epoch': epoch,
            'arch': type(model).__name__.lower(),
            'state_dict': get_state_dict(model, self.unwrap_fn),
            'optimizer': optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if scaler is not None:
            save_state['scaler'] = scaler.state_dict()
        if model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, tmp_save_path)
        os.rename(tmp_save_path, save_path)

        if is_best:
            shutil.copyfile(
                save_path, os.path.join(self.checkpoint_dir, "model_best" + self.extension)
            )
            self.best_epoch = epoch
            self.best_metric = metric
        print("Checkpoint saving for {} epoch is done...".format(epoch))



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def freeze_layers_fn(model, freeze_layers=[]):
    for name, param in model.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False


def load_state_dict(checkpoint_path, has_module, use_ema=False, remove_params=[]):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            ckpt = checkpoint[state_dict_key]
            _logger.info('Restoring model state from checkpoint...')
        else:
            ckpt = checkpoint
            _logger.info('Restoring model state from stete_dict ...')
        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            if any(remove_str in k for remove_str in remove_params):
                continue
            # strip `module.` prefix
            if not has_module and k.startswith('module'):
                name = k[7:]
            elif k.startswith('model'):
                name = k[6:]
            elif has_module and not k.startswith('module'):
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        state_dict = new_state_dict
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict, checkpoint
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()



def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True, remove_params=[]):
    has_module = hasattr(model, 'module')
    if has_module:
        _logger.info('model has attribute module...')
    else:
        _logger.info('model does not have attribute module...')
    
    state_dict, checkpoint = load_state_dict(checkpoint_path, has_module, use_ema, remove_params)

    if len(remove_params) > 0:
        this_dict = model.state_dict()
        this_dict.update(state_dict)
        model.load_state_dict(this_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)
    return checkpoint



def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True, remove_params=[]):
    resume_epoch = None
    checkpoint = load_checkpoint(model, checkpoint_path=checkpoint_path, use_ema=False, strict=False, remove_params=remove_params)

    resume_epoch = 0
    if 'epoch' in checkpoint:
        resume_epoch = checkpoint['epoch'] + 1

    if log_info:
        _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

    return checkpoint, resume_epoch