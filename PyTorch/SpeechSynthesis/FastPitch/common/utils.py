# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
# init_weights, get_padding, AttrDict

import ctypes
import glob
import os
import re
import shutil
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

import torch
import torch.distributed as dist
from scipy.io.wavfile import read


def mask_from_lens(lens, max_len: Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


def load_wav(full_path, torch_tensor=False):
    import soundfile  # flac
    data, sampling_rate = soundfile.read(full_path, dtype='int16')
    if torch_tensor:
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    else:
        return data, sampling_rate


def load_wav_to_torch(full_path, force_sampling_rate=None):
    if force_sampling_rate is not None:
        data, sampling_rate = librosa.load(full_path, sr=force_sampling_rate)
    else:
        sampling_rate, data = read(full_path)

    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(dataset_path, fnames, has_speakers=False, split="|"):
    def split_line(root, line):
        parts = line.strip().split(split)
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


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_pretrained_weights(model, ckpt_fpath):
    model = getattr(model, "module", model)
    weights = torch.load(ckpt_fpath, map_location="cpu")["state_dict"]
    weights = {re.sub("^module.", "", k): v for k, v in weights.items()}

    ckpt_emb = weights["encoder.word_emb.weight"]
    new_emb = model.state_dict()["encoder.word_emb.weight"]

    ckpt_vocab_size = ckpt_emb.size(0)
    new_vocab_size = new_emb.size(0)
    if ckpt_vocab_size != new_vocab_size:
        print("WARNING: Resuming from a checkpoint with a different size "
              "of embedding table. For best results, extend the vocab "
              "and ensure the common symbols' indices match.")
        min_len = min(ckpt_vocab_size, new_vocab_size)
        weights["encoder.word_emb.weight"] = ckpt_emb if ckpt_vocab_size > new_vocab_size else new_emb
        weights["encoder.word_emb.weight"][:min_len] = ckpt_emb[:min_len]

    model.load_state_dict(weights)


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


class BenchmarkStats:
    """ Tracks statistics used for benchmarking. """
    def __init__(self):
        self.num_frames = []
        self.losses = []
        self.mel_losses = []
        self.took = []

    def update(self, num_frames, losses, mel_losses, took):
        self.num_frames.append(num_frames)
        self.losses.append(losses)
        self.mel_losses.append(mel_losses)
        self.took.append(took)

    def get(self, n_epochs):
        frames_s = sum(self.num_frames[-n_epochs:]) / sum(self.took[-n_epochs:])
        return {'frames/s': frames_s,
                'loss': np.mean(self.losses[-n_epochs:]),
                'mel_loss': np.mean(self.mel_losses[-n_epochs:]),
                'took': np.mean(self.took[-n_epochs:]),
                'benchmark_epochs_num': n_epochs}

    def __len__(self):
        return len(self.losses)


class Checkpointer:

    def __init__(self, save_dir, keep_milestones=[]):
        self.save_dir = save_dir
        self.keep_milestones = keep_milestones

        find = lambda name: [
            (int(re.search("_(\d+).pt", fn).group(1)), fn)
            for fn in glob.glob(f"{save_dir}/{name}_checkpoint_*.pt")]

        tracked = sorted(find("FastPitch"), key=lambda t: t[0])
        self.tracked = OrderedDict(tracked)

    def last_checkpoint(self, output):

        def corrupted(fpath):
            try:
                torch.load(fpath, map_location="cpu")
                return False
            except:
                warnings.warn(f"Cannot load {fpath}")
                return True

        saved = sorted(
            glob.glob(f"{output}/FastPitch_checkpoint_*.pt"),
            key=lambda f: int(re.search("_(\d+).pt", f).group(1)))

        if len(saved) >= 1 and not corrupted(saved[-1]):
            return saved[-1]
        elif len(saved) >= 2:
            return saved[-2]
        else:
            return None

    def maybe_load(self, model, optimizer, scaler, train_state, args,
                   ema_model=None):

        assert args.checkpoint_path is None or args.resume is False, (
            "Specify a single checkpoint source")

        fpath = None
        if args.checkpoint_path is not None:
            fpath = args.checkpoint_path
            self.tracked = OrderedDict()  # Do not track/delete prev ckpts
        elif args.resume:
            fpath = self.last_checkpoint(args.output)

        if fpath is None:
            return

        print_once(f"Loading model and optimizer state from {fpath}")
        ckpt = torch.load(fpath, map_location="cpu")
        train_state["epoch"] = ckpt["epoch"] + 1
        train_state["total_iter"] = ckpt["iteration"]

        no_pref = lambda sd: {re.sub("^module.", "", k): v for k, v in sd.items()}
        unwrap = lambda m: getattr(m, "module", m)

        unwrap(model).load_state_dict(no_pref(ckpt["state_dict"]))

        if ema_model is not None:
            unwrap(ema_model).load_state_dict(no_pref(ckpt["ema_state_dict"]))

        optimizer.load_state_dict(ckpt["optimizer"])

        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        else:
            warnings.warn("AMP scaler state missing from the checkpoint.")

    def maybe_save(self, args, model, ema_model, optimizer, scaler, epoch,
                   total_iter, config):

        intermediate = (args.epochs_per_checkpoint > 0
                        and epoch % args.epochs_per_checkpoint == 0)
        final = epoch == args.epochs

        if not intermediate and not final and epoch not in self.keep_milestones:
            return

        rank = 0
        if dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()

        if rank != 0:
            return

        unwrap = lambda m: getattr(m, "module", m)
        ckpt = {"epoch": epoch,
                "iteration": total_iter,
                "config": config,
                "train_setup": args.__dict__,
                "state_dict": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()}
        if ema_model is not None:
            ckpt["ema_state_dict"] = unwrap(ema_model).state_dict()

        fpath = Path(args.output, f"FastPitch_checkpoint_{epoch}.pt")
        print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
        torch.save(ckpt, fpath)

        # Remove old checkpoints; keep milestones and the last two
        self.tracked[epoch] = fpath
        for epoch in set(list(self.tracked)[:-2]) - set(self.keep_milestones):
            try:
                os.remove(self.tracked[epoch])
            except:
                pass
            del self.tracked[epoch]
