# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch.nn as nn
import math
import librosa
from .perturb import AudioAugmentor
from .segment import AudioSegment
from apex import amp


def audio_from_file(file_path, offset=0, duration=0, trim=False, target_sr=16000):
    audio = AudioSegment.from_file(file_path,
                                   target_sr=target_sr,
                                   int_values=False,
                                   offset=offset, duration=duration, trim=trim)
    samples=torch.tensor(audio.samples, dtype=torch.float).cuda()
    num_samples = torch.tensor(samples.shape[0]).int().cuda()
    return (samples.unsqueeze(0), num_samples.unsqueeze(0))

class WaveformFeaturizer(object):
    def __init__(self, input_cfg, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.cfg = input_cfg

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path, offset=0, duration=0, trim=False):
        audio = AudioSegment.from_file(file_path,
                                       target_sr=self.cfg['sample_rate'],
                                       int_values=self.cfg.get('int_values', False),
                                       offset=offset, duration=duration, trim=trim)
        return self.process_segment(audio)

    def process_segment(self, audio_segment):
        self.augmentor.perturb(audio_segment)
        return torch.tensor(audio_segment.samples, dtype=torch.float)

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        return cls(input_config, augmentor=aa)


# @torch.jit.script
# def normalize_batch_per_feature(x, seq_len):
#     x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
#     x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)

#     for i in range(x.shape[0]):
#         x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
#         x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
#     # make sure x_std is not zero
#     x_std += 1e-5
#     return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)

# @torch.jit.script
# def normalize_batch_all_features(x, seq_len):
#     x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
#     x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
#     for i in range(x.shape[0]):
#         x_mean[i] = x[i, :, :int(seq_len[i])].mean()
#         x_std[i] = x[i, :, :int(seq_len[i])].std()
#     # make sure x_std is not zero
#     x_std += 1e-5
#     return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)

@torch.jit.script
def normalize_batch(x, seq_len, normalize_type: str):
#    print ("normalize_batch: x, seq_len, shapes: ", x.shape, seq_len, seq_len.shape)
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                                 device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                                device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :int(seq_len[i])].mean()
            x_std[i] = x[i, :, :int(seq_len[i])].std()
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x

@torch.jit.script
def splice_frames(x, frame_splicing: int):
    """ Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    # TORCHSCRIPT: JIT doesnt like range(start, stop)
    for n in range(frame_splicing - 1):
        seq.append(torch.cat([x[:, :, :n + 1], x[:, :, n + 1:]], dim=2))
    return torch.cat(seq, dim=1)

class SpectrogramFeatures(nn.Module):
    # For JIT. See https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length", "log", "frame_splicing", "window", "normalize", "pad_to", "max_duration", "do_normalize"]

    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                       n_fft=None,
                       window="hamming", normalize="per_feature", log=True, 
                       dither=1e-5, pad_to=8, max_duration=16.7,
                       frame_splicing=1):
        super(SpectrogramFeatures, self).__init__()
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        self.window = window_tensor

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.pad_to = pad_to
        self.frame_splicing = frame_splicing

        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)

    @torch.no_grad()
    def forward(self, x, seq_len):
        dtype = x.dtype

        seq_len = self.get_seq_len(seq_len)

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if hasattr(self,'preemph') and self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                                        dim=1)

        # get spectrogram
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       win_length=self.win_length,
                       window=self.window.to(torch.float))
        x = torch.sqrt(x.pow(2).sum(-1))

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype).to(seq_len.device).expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        
        # TORCHSCRIPT: Is this del important? It breaks scripting
        #del mask
    
        pad_to = self.pad_to
    
        # TORCHSCRIPT: Cant have mixed types. Using pad_to < 0 for "max"
        if pad_to < 0:
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)))
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt))
        return x.to(dtype)

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg['sample_rate'], window_size=cfg['window_size'],
                   window_stride=cfg['window_stride'],
                   n_fft=cfg['n_fft'], window=cfg['window'],
                   normalize=cfg['normalize'],
                   max_duration=cfg.get('max_duration', 16.7),
                   dither=cfg.get('dither', 1e-5), pad_to=cfg.get("pad_to", 0),
                   frame_splicing=cfg.get("frame_splicing", 1), log=log)
constant=1e-5
class FilterbankFeatures(nn.Module):
    # For JIT. See https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length", "center", "log", "frame_splicing", "window", "normalize", "pad_to", "max_duration", "max_length"]

    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                       window="hamming", normalize="per_feature", n_fft=None,
                       preemph=0.97,
                       nfilt=64, lowfreq=0, highfreq=None, log=True, dither=constant,
                       pad_to=8,
                       max_duration=16.7,
                       frame_splicing=1):
        super(FilterbankFeatures, self).__init__()

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = int(sample_rate * window_size) # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        #TORCHSCRIPT: Check whether or not we need this
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq,
                                                    fmax=highfreq), dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)
        # Calculate maximum sequence length (# frames)
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)

    # do stft
    # TORCHSCRIPT: center removed due to bug
    def  stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(dtype=torch.float))
    def forward(self, x, seq_len):
        dtype = x.dtype

        seq_len = self.get_seq_len(seq_len)
        
        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                                        dim=1)
            
        x  = self.stft(x)
            
            # get power spectrum
        x = x.pow(2).sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype).to(x.device).expand(x.size(0),
                                                                              max_len) >= seq_len.unsqueeze(1)

        x = x.masked_fill(mask.unsqueeze(1), 0)
        # TORCHSCRIPT: Is this del important? It breaks scripting
        # del mask
        # TORCHSCRIPT: Cant have mixed types. Using pad_to < 0 for "max"
        if self.pad_to < 0:
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)))
        elif self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            #            if pad_amt != 0:
            x = nn.functional.pad(x, (0, self.pad_to - pad_amt))
        
        return x # .to(dtype)

    @classmethod
    def from_config(cls, cfg, log=False):
        return cls(sample_rate=cfg['sample_rate'], window_size=cfg['window_size'],
                   window_stride=cfg['window_stride'], n_fft=cfg['n_fft'],
                   nfilt=cfg['features'], window=cfg['window'],
                   normalize=cfg['normalize'],
                   max_duration=cfg.get('max_duration', 16.7),
                   dither=cfg['dither'], pad_to=cfg.get("pad_to", 0),
                   frame_splicing=cfg.get("frame_splicing", 1), log=log)

class FeatureFactory(object):
    featurizers = {
        "logfbank": FilterbankFeatures,
        "fbank": FilterbankFeatures,
        "stft": SpectrogramFeatures,
        "logspect": SpectrogramFeatures,
        "logstft": SpectrogramFeatures
    }

    def __init__(self):
        pass

    @classmethod
    def from_config(cls, cfg):
        feat_type = cfg.get('feat_type', "logspect")
        featurizer = cls.featurizers[feat_type]
        #return featurizer.from_config(cfg, log="log" in cfg['feat_type'])
        return featurizer.from_config(cfg, log="log" in feat_type)
