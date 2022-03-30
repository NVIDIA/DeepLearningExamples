import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, ConvTranspose2d, AvgPool2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from common.utils import init_weights, get_padding, print_once

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv2d(channels, channels, (kernel_size, 1), 1, dilation=(dilation[0], 1),
                               padding=(get_padding(kernel_size, dilation[0]), 0))),
            weight_norm(Conv2d(channels, channels, (kernel_size, 1), 1, dilation=(dilation[1], 1),
                               padding=(get_padding(kernel_size, dilation[1]), 0))),
            weight_norm(Conv2d(channels, channels, (kernel_size, 1), 1, dilation=(dilation[2], 1),
                               padding=(get_padding(kernel_size, dilation[2]), 0)))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv2d(channels, channels, (kernel_size, 1), 1, dilation=1,
                               padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(channels, channels, (kernel_size, 1), 1, dilation=1,
                               padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(Conv2d(channels, channels, (kernel_size, 1), 1, dilation=1,
                               padding=(get_padding(kernel_size, 1), 0)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv2d(80, h.upsample_initial_channel, (7,1), (1,1), padding=(3,0)))
        assert h.resblock == '1', 'Only ResBlock1 currently supported for NHWC'
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                # ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                #                 k, u, padding=(k-u)//2)))
                ConvTranspose2d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                (k, 1), (u, 1), padding=((k-u)//2, 0))))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv2d(ch, 1, (7,1), (1,1), padding=(3,0)))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = x.unsqueeze(-1).to(memory_format=torch.channels_last)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            # x = self.ups[i](x.unsqueeze(-1)).squeeze(-1)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = x.squeeze(-1)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t, unit = x.shape
        assert unit == 1

        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, 0, 0, n_pad), "reflect")
            t = t + n_pad
        # print_once('x pre channels last:', x.is_contiguous(memory_format=torch.channels_last))
        x = x.view(b, c, t // self.period, self.period)
        # print_once('x post channels last:', x.is_contiguous(memory_format=torch.channels_last))

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        # x = torch.flatten(x, 1, -1)

        return x, fmap

    def share_params_of(self, dp):
        assert len(self.convs) == len(dp.convs)
        for c1, c2 in zip(self.convs, dp.convs):
            c1.weight = c2.weight
            c1.bias = c2.bias


class DiscriminatorPConv1d(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorPConv1d, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0), dilation=(period, 1))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0), dilation=(period, 1))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0), dilation=(period, 1))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0), dilation=(period, 1))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0), dilation=(period, 1))),
        ])
        # self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1, dilation=period))
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0), dilation=(period, 1)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t, unit = x.shape
        assert unit == 1
        # if t % self.period != 0: # pad first
        #     n_pad = self.period - (t % self.period)
        #     x = F.pad(x, (0, n_pad), "reflect")
        #     t = t + n_pad
        # x = x.view(b, c, t // self.period, self.period)

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


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods, use_conv1d=False, shared=False):
        super(MultiPeriodDiscriminator, self).__init__()
        print('MPD PERIODS:', periods)
        if use_conv1d:
            print('Constructing dilated MPD')
            layers = [DiscriminatorPConv1d(p) for p in periods]
        else:
            layers = [DiscriminatorP(p) for p in periods]

        if shared:
            print('MPD HAS SHARED PARAMS')
            for l in layers[1:]:
                l.share_params_of(layers[0])

        self.discriminators = nn.ModuleList(layers)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, amp_groups=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # self.convs = nn.ModuleList([
        #     norm_f(Conv1d(1, 128, 15, 1, padding=7)),
        #     norm_f(Conv1d(128, 128, 41, 2, groups=1 if amp_groups else 4, padding=20)),   # was: groups=4
        #     norm_f(Conv1d(128, 256, 41, 2, groups=1 if amp_groups else 16, padding=20)),  # was: groups=16
        #     norm_f(Conv1d(256, 512, 41, 4, groups=1 if amp_groups else 16, padding=20)),  # was: groups=16
        #     norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
        #     norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
        #     norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        # ])
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1,     128, (15,1), (1,1),                                 padding=(7 , 0))),
            norm_f(Conv2d(128,   128, (41,1), (2,1), groups=1 if amp_groups else  4, padding=(20, 0))),   # was: groups=4
            norm_f(Conv2d(128,   256, (41,1), (2,1), groups=1 if amp_groups else 16, padding=(20, 0))),  # was: groups=16
            norm_f(Conv2d(256,   512, (41,1), (4,1), groups=1 if amp_groups else 16, padding=(20, 0))),  # was: groups=16
            norm_f(Conv2d(512,  1024, (41,1), (4,1), groups=16                     , padding=(20, 0))),
            norm_f(Conv2d(1024, 1024, (41,1), (1,1), groups=16                     , padding=(20, 0))),
            norm_f(Conv2d(1024, 1024, ( 5,1), (1,1),                                 padding=(2 , 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3,1), (1,1), padding=(1,0)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        # x = x.squeeze(-1)
        # x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, amp_groups=False):
        super(MultiScaleDiscriminator, self).__init__()
        if amp_groups:
            print('MSD: AMP groups')
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True, amp_groups=amp_groups),
            DiscriminatorS(amp_groups=amp_groups),
            DiscriminatorS(amp_groups=amp_groups),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool2d((4, 1), (2, 1), padding=(1, 0)),
            AvgPool2d((4, 1), (2, 1), padding=(1, 0))
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
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


def feature_loss(fmap_r, fmap_g, keys=[]):
    loss = 0
    meta = {}
    assert len(keys) == len(fmap_r)

    for key, dr, dg in zip(keys, fmap_r, fmap_g):

        k = 'loss_gen_feat_' + key
        meta[k] = 0

        for rl, gl in zip(dr, dg):
            # loss += torch.mean(torch.abs(rl - gl))
            diff = torch.mean(torch.abs(rl - gl))
            loss += diff
            meta[k] += diff.item()

    return loss*2, meta


def discriminator_loss(disc_real_outputs, disc_generated_outputs, keys=[]):
    loss = 0
    r_losses = []
    g_losses = []
    meta = {}
    assert len(keys) == len(disc_real_outputs)

    for key, dr, dg in zip(keys, disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

        meta['loss_disc_real_' + key] = r_loss.item()
        meta['loss_disc_gen_' + key] = g_loss.item()

    return loss, r_losses, g_losses, meta


def generator_loss(disc_outputs, keys=[]):
    loss = 0
    gen_losses = []
    meta = {}
    assert len(keys) == len(disc_outputs)

    for key, dg in zip(keys, disc_outputs):
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l
        meta['loss_gen_' + key] = l.item()

    return loss, gen_losses, meta

