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

#  MIT License
#
#  Copyright (c) 2020 Jungil Kong
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# The following functions/classes were based on code from https://github.com/jik876/hifi-gan:
# plot_spectrogram, init_weights, get_padding, AttrDict

import ctypes
import glob
import os
import re
import shutil
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional

import soundfile  # flac

import matplotlib

import numpy as np
import torch
import torch.distributed as dist


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def reduce_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    rt = tensor.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(world_size)


def adjust_fine_tuning_lr(args, ckpt_d):
    assert args.fine_tuning
    if args.fine_tune_lr_factor == 1.:
        return
    for k in ['optim_d', 'optim_g']:
        for param_group in ckpt_d[k]['param_groups']:
            old_v = param_group['lr']
            new_v = old_v * args.fine_tune_lr_factor
            print(f'Init fine-tuning: changing {k} lr: {old_v} --> {new_v}')
            param_group['lr'] = new_v


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print(f"{args.local_rank}: Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(backend=('nccl' if args.cuda else 'gloo'),
                            init_method='env://')
    print(f"{args.local_rank}: Done initializing distributed training")


def load_wav(full_path, torch_tensor=False):
    data, sampling_rate = soundfile.read(full_path, dtype='int16')
    if torch_tensor:
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    else:
        return data, sampling_rate


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        raise NotImplementedError
    return load_wav(full_path, True)


def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|"):
    def split_line(root, line):
        parts = line.strip().split(split)
        if len(parts) == 1:
            paths, non_paths = parts, []
        else:
            if has_speakers:
                paths, non_paths = parts[:-2], parts[-2:]
            else:
                paths, non_paths = parts[:-1], parts[-1:]
        return tuple(str(Path(root, p)) for p in paths) + tuple(non_paths)

    fpaths_and_text = []
    for fname in fnames:
        with open(fname, encoding='utf-8') as f:
            fpaths_and_text += [split_line(dataset_path, line) for line in f]
    return fpaths_and_text


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def l2_promote():
    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def prepare_tmp(path):
    if path is None:
        return
    p = Path(path)
    if p.is_dir():
        warnings.warn(f'{p} exists. Removing...')
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=False, exist_ok=False)


def print_once(*msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*msg)


def plot_spectrogram(spectrogram):
    matplotlib.use("Agg")
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DefaultAttrDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(DefaultAttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, item):
        return self[item]


class Checkpointer:

    def __init__(self, save_dir,
                 keep_milestones=[1000, 2000, 3000, 4000, 5000, 6000]):
        self.save_dir = save_dir
        self.keep_milestones = keep_milestones

        find = lambda name: {int(re.search('_(\d+).pt', fn).group(1)): fn
                             for fn in glob.glob(f'{save_dir}/{name}_checkpoint_*.pt')}

        saved_g = find('hifigan_gen')
        saved_d = find('hifigan_discrim')

        common_epochs = sorted(set(saved_g.keys()) & set(saved_d.keys()))
        self.tracked = OrderedDict([(ep, (saved_g[ep], saved_d[ep]))
                                    for ep in common_epochs])

    def maybe_load(self, gen, mpd, msd, optim_g, optim_d, scaler_g, scaler_d,
                   train_state, args, gen_ema=None, mpd_ema=None, msd_ema=None):

        fpath_g = args.checkpoint_path_gen
        fpath_d = args.checkpoint_path_discrim

        assert (fpath_g is None) == (fpath_d is None)

        if fpath_g is not None:
            ckpt_paths = [(fpath_g, fpath_d)]
            self.tracked = OrderedDict()  # Do not track/delete prev ckpts
        elif args.resume:
            ckpt_paths = list(reversed(self.tracked.values()))[:2]
        else:
            return

        ckpt_g = None
        ckpt_d = None
        for fpath_g, fpath_d in ckpt_paths:
            if args.local_rank == 0:
                print(f'Loading models from {fpath_g} {fpath_d}')
            try:
                ckpt_g = torch.load(fpath_g, map_location='cpu')
                ckpt_d = torch.load(fpath_d, map_location='cpu')
                break
            except:
                print(f'WARNING: Cannot load {fpath_g} and {fpath_d}')

        if ckpt_g is None or ckpt_d is None:
            return

        ep_g = ckpt_g.get('train_state', ckpt_g).get('epoch', None)
        ep_d = ckpt_d.get('train_state', ckpt_d).get('epoch', None)
        assert ep_g == ep_d, \
            f'Mismatched epochs of gen and discrim ({ep_g} != {ep_d})'

        train_state.update(ckpt_g['train_state'])

        fine_tune_epoch_start = train_state.get('fine_tune_epoch_start')
        if args.fine_tuning and fine_tune_epoch_start is None:
            # Fine-tuning just began
            train_state['fine_tune_epoch_start'] = train_state['epoch'] + 1
            train_state['fine_tune_lr_factor'] = args.fine_tune_lr_factor
            adjust_fine_tuning_lr(args, ckpt_d)

        unwrap = lambda m: getattr(m, 'module', m)
        unwrap(gen).load_state_dict(ckpt_g.get('gen', ckpt_g['generator']))
        unwrap(mpd).load_state_dict(ckpt_d['mpd'])
        unwrap(msd).load_state_dict(ckpt_d['msd'])
        optim_g.load_state_dict(ckpt_d['optim_g'])
        optim_d.load_state_dict(ckpt_d['optim_d'])
        if 'scaler_g' in ckpt_d:
            scaler_g.load_state_dict(ckpt_d['scaler_g'])
            scaler_d.load_state_dict(ckpt_d['scaler_d'])
        else:
            warnings.warn('No grad scaler state found in the checkpoint.')

        if gen_ema is not None:
            gen_ema.load_state_dict(ckpt_g['gen_ema'])
        if mpd_ema is not None:
            mpd_ema.load_state_dict(ckpt_d['mpd_ema'])
        if msd_ema is not None:
            msd_ema.load_state_dict(ckpt_d['msd_ema'])

    def maybe_save(self, gen, mpd, msd, optim_g, optim_d, scaler_g, scaler_d,
                   epoch, train_state, args, gen_config, train_setup,
                   gen_ema=None, mpd_ema=None, msd_ema=None):

        rank = 0
        if dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()

        if rank != 0:
            return

        if epoch == 0:
            return

        if epoch < args.epochs and (args.checkpoint_interval == 0
                or epoch % args.checkpoint_interval > 0):
            return

        unwrap = lambda m: getattr(m, 'module', m)

        fpath_g = Path(self.save_dir, f'hifigan_gen_checkpoint_{epoch}.pt')
        ckpt_g = {
            'generator': unwrap(gen).state_dict(),
            'gen_ema': gen_ema.state_dict() if gen_ema is not None else None,
            'config': gen_config,
            'train_setup': train_setup,
            'train_state': train_state,
        }

        fpath_d = Path(self.save_dir, f'hifigan_discrim_checkpoint_{epoch}.pt')
        ckpt_d = {
            'mpd': unwrap(mpd).state_dict(),
            'msd': unwrap(msd).state_dict(),
            'mpd_ema': mpd_ema.state_dict() if mpd_ema is not None else None,
            'msd_ema': msd_ema.state_dict() if msd_ema is not None else None,
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'scaler_g': scaler_g.state_dict(),
            'scaler_d': scaler_d.state_dict(),
            'train_state': train_state,
            # compat with original code
            'steps': train_state['iters_all'],
            'epoch': epoch,
        }

        print(f"Saving model and optimizer state to {fpath_g} and {fpath_d}")
        torch.save(ckpt_g, fpath_g)
        torch.save(ckpt_d, fpath_d)

        # Remove old checkpoints; keep milestones and the last two
        self.tracked[epoch] = (fpath_g, fpath_d)
        for epoch in set(list(self.tracked)[:-2]) - set(self.keep_milestones):
            try:
                os.remove(self.tracked[epoch][0])
                os.remove(self.tracked[epoch][1])
                del self.tracked[epoch]
            except:
                pass
