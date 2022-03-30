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
# ResBlock1, ResBlock2, Generator, DiscriminatorP, DiscriminatorS, MultiScaleDiscriminator,
# MultiPeriodDiscriminator, feature_loss, discriminator_loss, generator_loss,
# init_weights, get_padding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from common.stft import STFT
from common.utils import AttrDict, init_weights, get_padding

LRELU_SLOPE = 0.1


class NoAMPConv1d(Conv1d):
    def __init__(self, *args, no_amp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_amp = no_amp

    def _cast(self, x, dtype):
        if isinstance(x, (list, tuple)):
            return [self._cast(t, dtype) for t in x]
        else:
            return x.to(dtype)

    def forward(self, *args):
        if not self.no_amp:
            return super().forward(*args)

        with torch.cuda.amp.autocast(enabled=False):
            return self._cast(
                super().forward(*self._cast(args, torch.float)), args[0].dtype)


class ResBlock1(nn.Module):
    __constants__ = ['lrelu_slope']

    def __init__(self, conf, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.conf = conf
        self.lrelu_slope = LRELU_SLOPE

        ch, ks = channels, kernel_size
        self.convs1 = nn.Sequential(*[
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(ks, dilation[0]), dilation[0])),
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(ks, dilation[1]), dilation[1])),
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(ks, dilation[2]), dilation[2])),
        ])

        self.convs2 = nn.Sequential(*[
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(ks, 1))),
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(ks, 1))),
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(ks, 1))),
        ])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    __constants__ = ['lrelu_slope']

    def __init__(self, conf, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.conf = conf

        ch, ks = channels, kernel_size
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(kernel_size, dilation[0]), dilation[0])),
            weight_norm(Conv1d(ch, ch, ks, 1, get_padding(kernel_size, dilation[1]), dilation[1])),
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    __constants__ = ['lrelu_slope', 'num_kernels', 'num_upsamples']

    def __init__(self, conf):
        super().__init__()
        conf = AttrDict(conf)
        self.conf = conf
        self.num_kernels = len(conf.resblock_kernel_sizes)
        self.num_upsamples = len(conf.upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(80, conf.upsample_initial_channel, 7, 1, padding=3))

        self.lrelu_slope = LRELU_SLOPE

        resblock = ResBlock1 if conf.resblock == '1' else ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(conf.upsample_rates,
                                       conf.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(conf.upsample_initial_channel // (2 ** i),
                                conf.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k-u)//2)))

        self.ups = nn.Sequential(*self.ups)

        self.resblocks = []
        for i in range(len(self.ups)):
            resblock_list = []

            ch = conf.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(conf.resblock_kernel_sizes,
                                           conf.resblock_dilation_sizes)):
                resblock_list.append(resblock(conf, ch, k, d))
            resblock_list = nn.Sequential(*resblock_list)
            self.resblocks.append(resblock_list)
        self.resblocks = nn.Sequential(*self.resblocks)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def load_state_dict(self, state_dict, strict=True):
        # Fallback for old checkpoints (pre-ONNX fix)
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            if 'resblocks' in k:
                parts = k.split(".")
                # only do this is the checkpoint type is older
                if len(parts) == 5:
                    layer = int(parts[1])
                    new_layer = f"{layer//3}.{layer%3}"
                    new_k = f"resblocks.{new_layer}.{'.'.join(parts[2:])}"
            new_sd[new_k] = v

        # Fix for conv1d/conv2d/NHWC
        curr_sd = self.state_dict()
        for key in new_sd:
            len_diff = len(new_sd[key].size()) - len(curr_sd[key].size())
            if len_diff == -1:
                new_sd[key] = new_sd[key].unsqueeze(-1)
            elif len_diff == 1:
                new_sd[key] = new_sd[key].squeeze(-1)

        super().load_state_dict(new_sd, strict=strict)

    def forward(self, x):
        x = self.conv_pre(x)

        for upsample_layer, resblock_group in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsample_layer(x)
            xs = 0
            for resblock in resblock_group:
                xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)

        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('HiFi-GAN: Removing weight norm.')
        for l in self.ups:
            remove_weight_norm(l)
        for group in self.resblocks:
            for block in group:
                block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Denoiser(nn.Module):
    """ Removes model bias from audio produced with hifigan """

    def __init__(self, hifigan, filter_length=1024, n_overlap=4,
                 win_length=1024, mode='zeros', **infer_kw):
        super().__init__()
        self.stft = STFT(filter_length=filter_length,
                         hop_length=int(filter_length/n_overlap),
                         win_length=win_length).cuda()

        for name, p in hifigan.named_parameters():
            if name.endswith('.weight'):
                dtype = p.dtype
                device = p.device
                break

        mel_init = {'zeros': torch.zeros, 'normal': torch.randn}[mode]
        mel_input = mel_init((1, 80, 88), dtype=dtype, device=device)

        with torch.no_grad():
            bias_audio = hifigan(mel_input, **infer_kw).float()

            if len(bias_audio.size()) > 2:
                bias_audio = bias_audio.squeeze(0)
            elif len(bias_audio.size()) < 2:
                bias_audio = bias_audio.unsqueeze(0)
            assert len(bias_audio.size()) == 2

            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        ks = kernel_size
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (ks, 1), (stride, 1), (get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (ks, 1), (stride, 1), (get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (ks, 1), (stride, 1), (get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (ks, 1), (stride, 1), (get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (ks, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def share_params_of(self, dp):
        assert len(self.convs) == len(dp.convs)
        for c1, c2 in zip(self.convs, dp.convs):
            c1.weight = c2.weight
            c1.bias = c2.bias


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods, concat_fwd=False):
        super().__init__()
        layers = [DiscriminatorP(p) for p in periods]
        self.discriminators = nn.ModuleList(layers)
        self.concat_fwd = concat_fwd

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if self.concat_fwd:
                y_ds, fmaps = d(concat_discr_input(y, y_hat))
                y_d_r, y_d_g, fmap_r, fmap_g = split_discr_output(y_ds, fmaps)
            else:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False, no_amp_grouped_conv=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(NoAMPConv1d(128, 256, 41, 2, groups=16, padding=20, no_amp=no_amp_grouped_conv)),
            norm_f(NoAMPConv1d(256, 512, 41, 4, groups=16, padding=20, no_amp=no_amp_grouped_conv)),
            norm_f(NoAMPConv1d(512, 1024, 41, 4, groups=16, padding=20, no_amp=no_amp_grouped_conv)),
            norm_f(NoAMPConv1d(1024, 1024, 41, 1, groups=16, padding=20, no_amp=no_amp_grouped_conv)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            # x = l(x.unsqueeze(-1)).squeeze(-1)
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, no_amp_grouped_conv=False, concat_fwd=False):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True, no_amp_grouped_conv=no_amp_grouped_conv),
            DiscriminatorS(no_amp_grouped_conv=no_amp_grouped_conv),
            DiscriminatorS(no_amp_grouped_conv=no_amp_grouped_conv),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=1),
            AvgPool1d(4, 2, padding=1)
        ])
        self.concat_fwd = concat_fwd

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if self.concat_fwd:
                ys = concat_discr_input(y, y_hat)
                if i != 0:
                    ys = self.meanpools[i-1](ys)
                y_ds, fmaps = d(ys)
                y_d_r, y_d_g, fmap_r, fmap_g = split_discr_output(y_ds, fmaps)
            else:
                if i != 0:
                    y = self.meanpools[i-1](y)
                    y_hat = self.meanpools[i-1](y_hat)
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def concat_discr_input(y, y_hat):
    return torch.cat((y, y_hat), dim=0)


def split_discr_output(y_ds, fmaps):
    y_d_r, y_d_g = torch.chunk(y_ds, 2, dim=0)
    fmap_r, fmap_g = zip(*(torch.chunk(f, 2, dim=0) for f in fmaps))
    return y_d_r, y_d_g, fmap_r, fmap_g


def feature_loss(fmap_r, fmap_g):
    loss = 0

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
