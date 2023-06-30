import math
import random

import librosa
import torch
import torch.nn as nn


class BaseFeatures(nn.Module):
    """Base class for GPU accelerated audio preprocessing."""
    __constants__ = ["pad_align", "pad_to_max_duration", "max_len"]

    def __init__(self, pad_align, pad_to_max_duration, max_duration,
                 sample_rate, window_size, window_stride, spec_augment=None,
                 cutout_augment=None):
        super(BaseFeatures, self).__init__()

        self.pad_align = pad_align
        self.pad_to_max_duration = pad_to_max_duration
        self.win_length = int(sample_rate * window_size) # frame size
        self.hop_length = int(sample_rate * window_stride)

        # Calculate maximum sequence length (# frames)
        if pad_to_max_duration:
            self.max_len = 1 + math.ceil(
                (max_duration * sample_rate - self.win_length) / self.hop_length
            )

        if spec_augment is not None:
            self.spec_augment = SpecAugment(**spec_augment)
        else:
            self.spec_augment = None

        if cutout_augment is not None:
            self.cutout_augment = CutoutAugment(**cutout_augment)
        else:
            self.cutout_augment = None

    @torch.no_grad()
    def calculate_features(self, audio, audio_lens):
        return audio, audio_lens

    def __call__(self, audio, audio_lens):
        dtype = audio.dtype
        audio = audio.float()
        feat, feat_lens = self.calculate_features(audio, audio_lens)

        feat = self.apply_padding(feat)

        if self.cutout_augment is not None:
            feat = self.cutout_augment(feat)

        if self.spec_augment is not None:
            feat = self.spec_augment(feat)

        feat = feat.to(dtype)
        return feat, feat_lens

    def apply_padding(self, x):
        if self.pad_to_max_duration:
            x_size = max(x.size(-1), self.max_len)
        else:
            x_size = x.size(-1)

        if self.pad_align > 0:
            pad_amt = x_size % self.pad_align
        else:
            pad_amt = 0

        padded_len = x_size + (self.pad_align - pad_amt if pad_amt > 0 else 0)
        return nn.functional.pad(x, (0, padded_len - x.size(-1)))


class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self, freq_masks=0, min_freq=0, max_freq=10, time_masks=0,
                 min_time=0, max_time=10):
        super(SpecAugment, self).__init__()
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.freq_masks = freq_masks
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.time_masks = time_masks
        self.min_time = min_time
        self.max_time = max_time

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):
            for _ in range(self.freq_masks):
                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                f0 = torch.randint(0, max(1, sh[1] - w), size=(1,))
                mask[idx, f0:f0+w] = 1

            for _ in range(self.time_masks):
                w = torch.randint(self.min_time, self.max_time + 1, size=(1,)).item()
                t0 = torch.randint(0, max(1, sh[2] - w), size=(1,))
                mask[idx, :, t0:t0+w] = 1

        return x.masked_fill(mask, 0)


class CutoutAugment(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, masks=0, min_freq=20, max_freq=20, min_time=5, max_time=5):
        super(CutoutAugment, self).__init__()
        assert 0 <= min_freq <= max_freq
        assert 0 <= min_time <= max_time

        self.masks = masks
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_time = min_time
        self.max_time = max_time

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape
        mask = torch.zeros(x.shape, dtype=torch.bool, device=x.device)

        for idx in range(sh[0]):
            for i in range(self.masks):

                w = torch.randint(self.min_freq, self.max_freq + 1, size=(1,)).item()
                h = torch.randint(self.min_time, self.max_time + 1, size=(1,)).item()

                f0 = int(random.uniform(0, sh[1] - w))
                t0 = int(random.uniform(0, sh[2] - h))

                mask[idx, f0:f0+w, t0:t0+h] = 1

        return x.masked_fill(mask, 0)


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
def stack_subsample_frames(x, x_lens, stacking: int = 1, subsampling: int = 1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]

    if subsampling > 1:
        x_lens = torch.ceil(x_lens.float() / subsampling).int()

        if x.size(2) > x_lens.max().item():
            assert abs(x.size(2) - x_lens.max().item()) <= 1
            x = x[:,:,:x_lens.max().item()]

    return x, x_lens


class FilterbankFeatures(BaseFeatures):
    # For JIT, https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length",
                     "log", "frame_splicing", "normalize"]
    # torchscript: "center" removed due to a bug

    def __init__(self, spec_augment=None, cutout_augment=None,
                 sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hamming", normalize="per_feature", n_fft=None,
                 preemph=0.97, n_filt=64, lowfreq=0, highfreq=None, log=True,
                 dither=1e-5, pad_align=8, pad_to_max_duration=False,
                 max_duration=float('inf'), frame_splicing=1):
        super(FilterbankFeatures, self).__init__(
            pad_align=pad_align, pad_to_max_duration=pad_to_max_duration,
            max_duration=max_duration, sample_rate=sample_rate,
            window_size=window_size, window_stride=window_stride,
            spec_augment=spec_augment, cutout_augment=cutout_augment)

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        #TORCHSCRIPT: Check whether or not we need this
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.n_filt = n_filt
        self.preemph = preemph
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=n_filt,
                                fmin=lowfreq, fmax=highfreq),
            dtype=torch.float).unsqueeze(0)
        # torchscript
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)

    # TORCHSCRIPT: center removed due to bug
    def stft(self, x):
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(dtype=torch.float),
                          return_complex=True)
        return torch.view_as_real(spec)

    @torch.no_grad()
    def calculate_features(self, x, seq_len):
        dtype = x.dtype

        seq_len = self.get_seq_len(seq_len)

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
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
            raise ValueError('Frame splicing not supported')

        # normalize if required
        x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch,
        # pad to multiple of `pad_align` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype, device=x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), 0)

        # TORCHSCRIPT: Is this del important? It breaks scripting
        # del mask

        return x.to(dtype), seq_len
